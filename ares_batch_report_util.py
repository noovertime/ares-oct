# ARES 보고서 생성
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

import config
from config import KEY_CR, KEY_AF, KEY_AR, JUDGE_TYPES


# ===================================================================
# 상수
# ===================================================================

CI_ALPHA: float = 0.05  # 95% 신뢰구간
CI_Z_SCORE: float = float(norm.ppf(1 - CI_ALPHA / 2))  # 약 1.96


# ===================================================================
# 내부 유틸 함수
# ===================================================================

def _load_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """JSONL 파일을 안전하게 로드한다."""
    lines: List[Dict[str, Any]] = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data: Dict[str, Any] = json.loads(line)
                    lines.append(data)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return lines


def _safe_parse_label(value: Any) -> Optional[int]:
    """0 또는 1의 유효한 정수만 반환."""
    try:
        label = int(value)
        if label in (0, 1):
            return label
    except (ValueError, TypeError):
        pass
    return None


def _init_score_dict() -> Dict[str, List[Any]]:
    """각 축별 빈 리스트 딕셔너리 초기화."""
    return {KEY_CR: [], KEY_AF: [], KEY_AR: []}


# ===================================================================
# 통계 계산
# ===================================================================

def calculate_ppi_asymptotic_ci(
    machine_preds: List[int],
    rectifiers: List[float],
    total_n: int,
    labeled_n: int
) -> float:
    """PPI Asymptotic CI Half-width 계산."""
    if labeled_n <= 1 or total_n <= 0:
        return 0.0

    y_hat_array: NDArray[np.float64] = np.array(machine_preds, dtype=np.float64)
    rectifier_array: NDArray[np.float64] = np.array(rectifiers, dtype=np.float64)

    sigma2_f: float = float(np.var(y_hat_array))
    sigma2_rec: float = float(np.var(rectifier_array))

    variance: float = (sigma2_f / float(total_n)) + (sigma2_rec / float(labeled_n))
    half_width: float = CI_Z_SCORE * math.sqrt(variance)
    return round(half_width, 3)


# ===================================================================
# PPI 파일 분석
# ===================================================================

def analyze_ppi_file(
    filepath: str,
    ppi_correction_active: bool,
    gold_fields: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """단일 PPI 파일을 분석하고 요약 결과를 반환."""
    if not ppi_correction_active:
        raise RuntimeError("PPI 보정을 위한 골든 라벨이 로드되지 않았습니다.")

    records: List[Dict[str, Any]] = _load_json_lines(filepath)
    if not records:
        return None

    all_scores: Dict[str, List[int]] = _init_score_dict()  # type: ignore
    rectifier_terms: Dict[str, List[float]] = _init_score_dict()  # type: ignore
    gold_label_counts: Dict[str, int] = {k: 0 for k in JUDGE_TYPES}

    model_name_raw: str = os.path.basename(filepath)
    name_parts: List[str] = model_name_raw.split('_')
    model_name: str = "_".join(name_parts[:-1]).split(".jsonl")[0] or model_name_raw.split(".jsonl")[0]

    for data in records:
        for key in JUDGE_TYPES:
            if key in data:
                value = data[key]
                if isinstance(value, (int, float)):
                    all_scores[key].append(int(value))

        for machine_key, gold_key in gold_fields.items():
            machine_pred: Optional[int] = _safe_parse_label(data.get(machine_key))
            gold_label: Optional[int] = _safe_parse_label(data.get(gold_key))

            if machine_pred is None or gold_label is None:
                continue

            rectifier_terms[machine_key].append(float(machine_pred - gold_label))
            gold_label_counts[machine_key] += 1

    total_n: int = len(all_scores[KEY_CR])
    if total_n == 0:
        return None

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "n": total_n,
        "ppi_active": ppi_correction_active,
        "labeled_n_rep": gold_label_counts[KEY_CR],
    }

    overall_corrected_scores: List[float] = []

    for axis in JUDGE_TYPES:
        scores: List[int] = all_scores[axis]
        if not scores:
            continue

        machine_mean: float = sum(scores) / float(total_n)
        labeled_n_axis: int = gold_label_counts[axis]
        rectifier: float = (
            sum(rectifier_terms[axis]) / labeled_n_axis if labeled_n_axis > 0 else 0.0
        )
        corrected_mean: float = max(0.0, min(1.0, machine_mean - rectifier))

        # CI 계산 코드 유지 (출력은 제외)
        margin: float = calculate_ppi_asymptotic_ci(scores, rectifier_terms[axis], total_n, labeled_n_axis)

        summary[axis] = {
            "machine_mean": round(machine_mean, 2),
            "corrected_mean": round(corrected_mean, 2),
            "applied_rectifier": round(rectifier, 3),
            "ci": round(margin, 3)
        }
        overall_corrected_scores.append(corrected_mean)

    if not overall_corrected_scores:
        return None

    summary["overall"] = round(sum(overall_corrected_scores) / len(overall_corrected_scores), 2)
    return summary


# ===================================================================
# 보고서 생성
# ===================================================================

def generate_summary_report(model_summaries: List[Dict[str, Any]]) -> str:
    """Markdown 형식 보고서 생성 (원본 모든 설명 포함)."""
    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없습니다."

    model_summaries.sort(key=lambda x: float(x["overall"]), reverse=True)
    current_time: str = time.strftime("%Y-%m-%d %H:%M:%S")
    total_golden_set_count: int = int(model_summaries[0]["labeled_n_rep"])
    model_list: str = "\n".join([f"   - {m['model_name']}" for m in model_summaries])

    report_content: str = f"""
## 🧭 ARES 결과 보고서
평가 일자: {current_time}

--- 
### 1️⃣ 프로젝트 개요 
- 프로젝트명: ARES 심사관 로컬 배치 평가
- 평가 프레임워크: Stanford ARES (골든셋 기반 PPI 보정 로직 통합)
- 평가 대상 : (q, c, a) 트리플 셋으로 구성 {model_list}
- 골든셋 유효 개수 (n) : {total_golden_set_count}

--- 
### 2️⃣ 평가 
- Context Relevance (CR, 문맥 적합성) : 검색된 문서가 질문과 얼마나 관련 있는가
- Answer Faithfulness (AF, 응답 충실도) : 생성된 답변이 검색 문서 내용에 충실한가 
- Answer Relevance (AR, 응답 적절성) : 답변이 질문에 직접적이고 구체적인가

--- 
### 3️⃣ PPI 보정 점수 요약 

| 평가대상 | **CR (보정)** | AF (보정) | AR (보정) | **종합 점수** | 기계 예측 평균 | **CR 편향** | **AF 편향** | **AR 편향** | 총 샘플 수 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for summary in model_summaries:
        cr = summary[KEY_CR]
        af = summary[KEY_AF]
        ar = summary[KEY_AR]

        report_content += (
            f"| **{summary['model_name']}** "
            f"| {cr['corrected_mean']:.2f} "
            f"| {af['corrected_mean']:.2f} "
            f"| {ar['corrected_mean']:.2f} "
            f"| **{summary['overall']:.2f}** "
            f"| {cr['machine_mean']:.2f} "
            f"| **{cr['applied_rectifier']:.3f}** "
            f"| **{af['applied_rectifier']:.3f}** "
            f"| **{ar['applied_rectifier']:.3f}** "
            f"| {summary['n']} |\n"
        )

    report_content += f"""
#### 💡 점수 요약 의미 설명
- CR, AF, AR (보정) : PPI 보정 로직을 거쳐 **편향이 제거된** 각 축의 최종 성능 추정치. 1에 가까울수록 좋은 성능 (0.0 ~ 1.0)
- 종합 점수 : 심사관의 예측 평균에서 골든셋 기반 예측 편향을 제거하여 계산된 신뢰할 수 있는 성능 추정치
- 심사관 예측 평균 : ARES 심사관이 예측한 점수($\\hat{{Y}}$)의 단순 평균 (보정 전 점수)
- CR/AF/AR 편향 : 모델의 예측 평균($\\hat{{Y}}$) - 각 축의 편향값. ($\\hat{{Y}} - Y$)
- 총 샘플 수 : 평가에 사용된 Q-C-A 트리플의 전체 개수

---
### 4️⃣ 보정 로직 및 해석

* PPI 보정 적용: PPI(Prediction-Powered Inference) 방식에 따라 보정 평균 (Corrected Mean)을 산출했습니다.
* 보정 방법: 골든 라벨(Gold Label) 기반의 실제 편향(Rectifier Mean)이 적용되었습니다.
"""
    return report_content


# ===================================================================
# 실행 파이프라인
# ===================================================================

def run_summary_generation_pipeline(
    ppi_correction_active: bool,
    gold_fields: Dict[str, str]
) -> None:
    """보고서 생성 파이프라인."""
    if not ppi_correction_active:
        raise RuntimeError("PPI 보정을 위한 골든 라벨이 없습니다.")

    ppi_dir: str = config.DATA_OUT_DIR
    report_dir: str = config.DATA_REPORT_DIR
    os.makedirs(report_dir, exist_ok=True)

    ppi_files: List[str] = [
        os.path.join(ppi_dir, f)
        for f in os.listdir(ppi_dir)
        if f.endswith(".jsonl") and "_" in f
    ]
    if not ppi_files:
        print(f"[WARN] {ppi_dir}에 분석할 PPI 파일이 없습니다.")
        return

    print(f"[INFO] 총 {len(ppi_files)}개 모델 분석 중...")

    model_summaries: List[Dict[str, Any]] = []

    for path in ppi_files:
        try:
            summary = analyze_ppi_file(path, True, gold_fields)
            if summary:
                model_summaries.append(summary)
                print(f"   [OK] '{summary['model_name']}' 완료.")
        except Exception as e:
            print(f"   [ERROR] '{os.path.basename(path)}' 분석 실패: {e}")

    if not model_summaries:
        print("[WARN] 분석 가능한 데이터가 없습니다.")
        return

    report_content: str = generate_summary_report(model_summaries)
    timestamp: str = time.strftime("%Y%m%d_%H%M%S")
    output_path: str = os.path.join(report_dir, f"summary_{timestamp}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[완료] ARES 통계 보고서 생성됨 → {output_path}")
