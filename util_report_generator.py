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
    #name_parts: List[str] = model_name_raw.split('_')
    #model_name: str = "_".join(name_parts[:-1]).split(".jsonl")[0] or model_name_raw.split(".jsonl")[0]
    model_name: str = model_name_raw.replace(".jsonl", "")

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
# 보고서 생성 (수정됨)
# ===================================================================

def generate_summary_report(model_summaries: List[Dict[str, Any]]) -> str:
    """Markdown 형식 보고서 생성 (요청된 마크다운/HTML 테이블 형식 적용)."""
    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없습니다."

    # 'overall' 점수를 기준으로 내림차순 정렬
    model_summaries.sort(key=lambda x: float(x["overall"]), reverse=True)
    current_time: str = time.strftime("%Y-%m-%d %H:%M:%S")
    # 모든 모델이 동일한 골든셋을 사용한다고 가정하고 첫 번째 모델의 labeled_n_rep 사용
    total_golden_set_count: int = int(model_summaries[0]["labeled_n_rep"])
    model_list: str = "\n".join([f"   - {m['model_name']}" for m in model_summaries])

    # -------------------------------------------------------------
    # 1. 보고서 기본 정보 섹션
    # -------------------------------------------------------------
    report_content: str = f"""
## 🧭 ARES 결과 보고서
평가 일자: {current_time}

--- 
### 1️⃣ 프로젝트 개요 
- 프로젝트명: ARES 심사관 로컬 배치 평가
- 평가 프레임워크: Stanford ARES (골든셋 기반 PPI 보정 로직 통합)
- 평가 대상 : (q, c, a) 트리플 셋으로 구성 <br>
{model_list}
- 골든셋 유효 개수 (n) : {total_golden_set_count}

--- 
### 2️⃣ 평가 
- Context Relevance (CR, 문맥 적합성) : 검색된 문서가 질문과 얼마나 관련 있는가
- Answer Faithfulness (AF, 응답 충실도) : 생성된 답변이 검색 문서 내용에 충실한가 
- Answer Relevance (AR, 응답 적절성) : 답변이 질문에 직접적이고 구체적인가

--- 
### 3️⃣ PPI 추정 성능 점수
#### 🎯 성능점수 요약 

| 순번 | 평가대상 | 종합 점수 | CR(보정) | AF(보정) | AR(보정)|
|:--|:---:|:---:|:---:|:---:|:---:|
"""
    # -------------------------------------------------------------
    # 2. 요약 테이블 (마크다운)
    # -------------------------------------------------------------
    for i, summary in enumerate(model_summaries):
        report_content += (
            f"| {i + 1} "
            f"| {summary['model_name']} "
            f"| {summary['overall']:.2f} "
            f"| {summary[KEY_CR]['corrected_mean']:.2f} "
            f"| {summary[KEY_AF]['corrected_mean']:.2f} "
            f"| {summary[KEY_AR]['corrected_mean']:.2f} |\n"
        )

    report_content += """
    📝 의미 요약 
    * 종합점수 : 평가 대상 모델의 전반적인 성능 추정치 
    * CR/AR/AF(보정) : 문맥 적합성(CR), 응답 충실도(AF), 응답 적절성(AR) 성능 추정치


<br>

#### 🎯 성능 점수 세부 항목 값\n
    """

    report_content += """
<table>
  <thead>
    <tr>
        <td rowspan="2">순번</td>
        <td rowspan="2">평가대상</td>
        <td colspan="3" align="center">CR</td>
        <td colspan="3" align="center">AF</td>
        <td colspan="3" align="center">AR</td>
        <td rowspan="2">심사관 예측 평균</td>
        <td rowspan="2">총 샘플 수</td>
    </tr>
    <tr>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
    </tr>
  </thead>
  <tbody>
"""
    # -------------------------------------------------------------
    # 3. 세부내용 테이블 (HTML)
    # -------------------------------------------------------------
    for i, summary in enumerate(model_summaries):
        cr = summary[KEY_CR]
        af = summary[KEY_AF]
        ar = summary[KEY_AR]

        # 모델의 모든 기계 예측 평균 (CR 축의 machine_mean 사용)
        model_machine_mean: float = cr['machine_mean']

        report_content += "    <tr>\n"
        report_content += f"        <td>{i + 1}</td>\n"
        report_content += f"        <td>{summary['model_name']}</td>\n"

        # CR
        report_content += f"        <td>{cr['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{cr['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{cr['ci']:.2f}</td>\n"

        # AF
        report_content += f"        <td>{af['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{af['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{af['ci']:.2f}</td>\n"

        # AR
        report_content += f"        <td>{ar['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{ar['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{ar['ci']:.2f}</td>\n"

        # 모델 예측 및 샘플 수
        report_content += f"        <td>{model_machine_mean:.2f}</td>\n"
        report_content += f"        <td>{summary['n']}</td>\n"
        report_content += "    </tr>\n"

    report_content += """
  </tbody>
</table>


    📝 의미 요약 
    * CR/AR/AF(보정) : 문맥 적합성(CR), 응답 충실도(AF), 응답 적절성(AR) 성능 추정치 ( 0 ~ 1 : 최고점 ) 
    * 편향 : ARES 심사관의 예측과 골든라벨의 평균 차이 ( 작을수록 좋음 )
    * CI : CR/AR/AF(보정) 값의 신뢰구간 ( 작을수록 좋음 )
    * 심사관 예측평균 : ARES 심사관이 부여한 평균 점수 
    * 총 샘플 수 : 평가에 사용된 Q-C-A 트리플의 전체 개수

---
### 4️⃣ PPI 보정 의미 
* PPI(Prediction-Powered Inference) 역할 
  * PPI는 ARES 심사관 모델($\hat{Y}$)의 예측 편향을 제거하여 평가 결과의 신뢰도와 통계적 효율성을 높이는 방법론 
  * 효율성 결합 
    * ARES 심사관이 수행한 대규모 예측 ($\hat{Y}$)의 정보력과 고비용으로 얻은 소규모 골든셋 ($Y$)의 정확성을 결합 
  * 편향계산 (Rectifier)
    * 소수의 골든셋에서 심사관 예측과 골든라벨의 평균 차이를 계산해 편향 수정자로 사용
* PPI 보정 값의 의미 
  * PPI 보정 값은 PPI 방법론을 통해 산출된 모델의 성능 추정치 
  * 참된 성능 추정 (true performance)
    * ARES 심사관의 예측 점수에서 골든셋 기반의 편향이 제거된 모델의 참된 성능을 신뢰할 수 있게 추정한 값 
  * 계산공식 
    * 보정값 = ARES 심사관 모델 예측평균 - 편향 
  * 보고서 표시
      * CR(보정), AF(보정), AR(보정) 값에 해당 
* 보고서 값의 의미 상세
  * $\text{종합 점수} = \frac{\text{CR}(\text{보정}) + \text{AF}(\text{보정}) + \text{AR}(\text{보정})}{3}$
  * 보정 : 예측값에서 편향을 제거한 성능 추정치 
  * 편향 : 골든셋과 심사관 예측 사이의 격차 ( -1 ~ 1 이상 )
  * CI (Confidence Interval, 신뢰구간)
    * PPI 보정 값(CR보정, AF보정, AR보정)의 신뢰도와 정밀도를 나타냄 
    * 값이 작을수록 추정된 보정값에 대한 오차 범위가 좁아져 신뢰도가 높다는 것을 의미 
  * 심사관예측점수 
    * CR/AF/AR 항목을 평가한 점수의 평균
    * PPI 보정의 입력값이며 편향이 포함되어 있어 이 값만으로는 모델 성능을 대표하지 않음
    * 보정을 거친 후에 CR/AF/AR 보정값으로 표현됨 
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