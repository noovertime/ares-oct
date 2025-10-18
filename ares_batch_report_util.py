# ares_batch_report_util.py (상수 제거 및 config import 버전)

import os
import json
import time
import math
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy.stats import norm
from scipy.optimize import brentq

import config
# 🚨 핵심 수정: config.py에서 필요한 상수를 직접 import
from config import (
    KEY_CR,
    KEY_AF,
    KEY_AR,
    JUDGE_TYPES,
    GOLD_LABEL_FIELDS
)

# ===================================================================
# 0. 전역 상수
# ===================================================================

CI_ALPHA = 0.05  # 95% 신뢰구간 (alpha=0.05)
CI_Z_SCORE = norm.ppf(1 - CI_ALPHA / 2)  # Z-score: 약 1.96


# ===================================================================
# 1. 통계 계산 유틸리티
# ===================================================================

def calculate_ppi_asymptotic_ci(machine_preds: List[int], rectifiers: List[float], total_n: int,
                                labeled_n: int) -> float:
    """
    PPI Asymptotic CI Half-width (반폭)을 계산합니다.
    """

    if labeled_n <= 1:
        # CI 계산을 위한 샘플 수가 부족한 경우
        return 0.0

    y_hat_array = np.array(machine_preds)
    rectifier_array = np.array(rectifiers)

    sigma2_f = np.var(y_hat_array)
    sigma2_rec = np.var(rectifier_array)

    # PPI Asymptotic CI Variance 공식 적용
    variance = (sigma2_f / total_n) + (sigma2_rec / labeled_n)

    half_width = CI_Z_SCORE * math.sqrt(variance)

    return round(half_width, 3)


def analyze_ppi_file(filepath: str, ppi_correction_active: bool,
                     gold_fields: Dict[str, str]) -> Dict[str, Any]:
    """
    단일 PPI 파일을 읽어 PPI 보정 평균 및 축별 편향을 계산하고 요약합니다.
    """
    if not ppi_correction_active:
        raise RuntimeError("PPI 보정을 위한 골든 라벨이 로드되지 않았습니다.")

    model_name_raw = os.path.basename(filepath)
    all_scores: Dict[str, List[int]] = {
        KEY_CR: [],
        KEY_AF: [],
        KEY_AR: []
    }

    rectifier_terms: Dict[str, List[float]] = {KEY_CR: [], KEY_AF: [], KEY_AR: []}
    gold_label_counts: Dict[str, int] = {KEY_CR: 0, KEY_AF: 0, KEY_AR: 0}

    model_name_parts = model_name_raw.split('_')
    # 파일 이름에서 확장자 및 시간 부분을 제거하고 모델 이름 추출
    model_name = "_".join(model_name_parts[:-1]).split(".jsonl")[0]
    if not model_name or len(model_name_parts) < 2:
        model_name = model_name_raw.split(".jsonl")[0]

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())

                    # 기계 예측 점수 수집
                    for key in JUDGE_TYPES:
                        if key in data:
                            all_scores[key].append(data[key])

                    # Rectifier Term 계산을 위한 골든 라벨 추출 및 검증
                    for machine_key, gold_key in gold_fields.items():
                        machine_pred = data.get(machine_key)
                        gold_label_raw = data.get(gold_key)

                        try:
                            gold_label = int(gold_label_raw)
                        except (ValueError, TypeError):
                            continue

                            # Rectifier Term 계산 (유효한 [0, 1] 값인 경우만)
                        if gold_label in [0, 1] and machine_pred in [0, 1]:
                            rectifier_terms[machine_key].append(machine_pred - gold_label)
                            gold_label_counts[machine_key] += 1

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return None

    total_n = len(all_scores[KEY_CR])
    if total_n == 0:
        return None


    summary = {
        'model_name': model_name,
        'n': total_n,
        'ppi_active': ppi_correction_active,
        'labeled_n_rep':  gold_label_counts[KEY_CR] # 대표 골든셋 개수는 CR 축을 기준으로
    }

    overall_corrected_scores = []

    # CR, AF, AR 세 가지 축에 대해 반복
    for axis in JUDGE_TYPES:
        scores = all_scores.get(axis, [])
        if not scores:
            continue

        # 1. 기계 예측 평균 (tilde_theta_f)
        machine_mean = sum(scores) / total_n

        # 2. Rectifier Mean (실제 편향) 계산: E[Y_hat - Y]
        labeled_n_for_axis = gold_label_counts[axis]

        if rectifier_terms[axis] and labeled_n_for_axis > 0:
            rectifier = sum(rectifier_terms[axis]) / labeled_n_for_axis
        else:
            rectifier = 0.0

            # 3. PPI 보정 평균 (theta_hat_PP = machine_mean - rectifier)
        corrected_mean = max(0.0, min(1.0, machine_mean - rectifier))

        # 4. CI 마진 계산 (CI는 출력되지 않지만, 계산 코드는 유지)
        # if labeled_n_for_axis <= 1:
        #    margin = 0.0
        # else:
        #    margin = calculate_ppi_asymptotic_ci(scores, rectifier_terms[axis], total_n, labeled_n_for_axis)

        # 결과 저장
        summary[axis] = {
            'machine_mean': round(machine_mean, 2),
            'corrected_mean': round(corrected_mean, 2),
            'applied_rectifier': round(rectifier, 3)  # 각 축의 편향을 개별적으로 저장
            # 'ci': round(margin, 3)
        }
        overall_corrected_scores.append(corrected_mean)

    summary['overall'] = round(sum(overall_corrected_scores) / len(overall_corrected_scores), 2)

    return summary


def generate_summary_report(model_summaries: List[Dict[str, Any]]):
    """
    분석된 모델 요약을 바탕으로 최종 Markdown 형식의 보고서를 생성합니다. (축별 편향 출력)
    """

    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없어 보고서를 생성할 수 없습니다."

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # 종합 점수 기준으로 정렬
    model_summaries.sort(key=lambda x: x['overall'], reverse=True)

    # 모델 리스트 생성
    model_list_markdown = ""
    for summary in model_summaries:
        model_list_markdown += f"\n   - {summary['model_name']}\n"

    # 골든셋 갯수
    total_golden_set_count = model_summaries[0]['labeled_n_rep']

    report_content = f"""
## 🧭 ARES 결과 보고서
평가 일자: {current_time}

--- 
### 1️⃣ 프로젝트 개요 
- 프로젝트명: ARES 심사관 로컬 배치 평가
- 평가 프레임워크: Stanford ARES (골든셋 기반 PPI 보정 로직 통합)
- 평가 대상 : (q, c, a) 트리플 셋으로 구성 {model_list_markdown}
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

    # 각 축별 편향을 출력하도록 테이블 데이터 구성
    for summary in model_summaries:
        # KEY_CR 등의 상수를 사용하여 summary 딕셔너리의 값을 안전하게 참조
        cr_str = f"{summary[KEY_CR]['corrected_mean']:.2f}"
        af_str = f"{summary[KEY_AF]['corrected_mean']:.2f}"
        ar_str = f"{summary[KEY_AR]['corrected_mean']:.2f}"

        # 축별 편향 (개별적으로 추출)
        cr_bias = f"{summary[KEY_CR]['applied_rectifier']:.3f}"
        af_bias = f"{summary[KEY_AF]['applied_rectifier']:.3f}"
        ar_bias = f"{summary[KEY_AR]['applied_rectifier']:.3f}"

        report_content += (
            f"| **{summary['model_name']}** "
            f"| {cr_str} "
            f"| {af_str} "
            f"| {ar_str} "
            f"| **{summary['overall']:.2f}** "
            f"| {summary[KEY_CR]['machine_mean']:.2f} "  # CR 축의 기계 예측 평균 사용 (대표값)
            f"| **{cr_bias}** "
            f"| **{af_bias}** "
            f"| **{ar_bias}** "
            f"| {summary['n']} |\n"
        )

    report_content += f"""

#### 💡 점수 요약 의미 설명
- CR, AF, AR (보정) : PPI 보정 로직을 거쳐 **편향이 제거된** 각 축의 최종 성능 추정치. 1에 가까울수록 좋은 성능 (0.0 ~ 1.0)
- 종합 점수 : 심사관의 예측 평균에서 골든셋 기반 예측 편향을 제거하여 계산된 신뢰할 수 있는 성능 추정치
- 심사관 예측 평균 : ARES 심사관이 예측한 점수($\hat{{Y}}$)의 단순 평균 (보정 전 점수)
- CR/AF/AR 편향 : 모델의 예측 평균($\hat{{Y}}$) - 각 축의 편향값. ($\hat{{Y}} - Y$)
- 총 샘플 수 : 평가에 사용된 Q-C-A 트리플의 전체 개수

---
### 4️⃣ 보정 로직 및 해석

* PPI 보정 적용: PPI(Prediction-Powered Inference) 방식에 따라 보정 평균 (Corrected Mean)을 산출했습니다.
* 보정 방법: 골든 라벨(Gold Label) 기반의 실제 편향(Rectifier Mean)이 적용되었습니다.
"""

    return report_content


def run_summary_generation_pipeline(ppi_correction_active: bool, gold_fields: Dict[str, str]):
    """메인 실행 함수를 위해 분석/저장 로직만 분리합니다."""

    if not ppi_correction_active:
        raise RuntimeError("PPI 보정을 위한 골든 라벨이 로드되지 않아 보고서 생성을 진행할 수 없습니다.")

    # config 모듈의 DATA_OUT_DIR, DATA_REPORT_DIR 참조
    INPUT_DIR_SUM = config.DATA_OUT_DIR
    OUTPUT_DIR_SUM = config.DATA_REPORT_DIR

    os.makedirs(OUTPUT_DIR_SUM, exist_ok=True)

    ppi_files = [
        os.path.join(INPUT_DIR_SUM, f)
        for f in os.listdir(INPUT_DIR_SUM)
        if f.endswith('.jsonl') and len(f.split('_')) >= 2
    ]

    if not ppi_files:
        print(f"[WARN] {INPUT_DIR_SUM}에서 분석할 PPI 파일을 찾을 수 없습니다. 보고서 생성을 건너뜁니다.")
        return

    print(f"[INFO] 총 {len(ppi_files)}개 모델의 PPI 파일을 분석합니다.")

    model_summaries = []

    for file_path in ppi_files:
        try:
            summary = analyze_ppi_file(file_path, True, gold_fields)
            if summary:
                model_summaries.append(summary)
                print(f"   [SUCCESS] 모델 '{summary['model_name']}' 분석 완료.")
        except Exception as e:
            # 유효성 검증 오류 시 오류 발생
            print(f"   [ERROR] 모델 '{os.path.basename(file_path)}' 분석 실패: {e}")
            continue

    if not model_summaries:
        print("[WARN] 분석 가능한 모델 데이터가 없어 보고서 생성을 건너뜁니다.")
        return

    report_content = generate_summary_report(model_summaries)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"summary_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR_SUM, output_filename)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(report_content)

    print("\n\n=============== ARES 통계 보고서 생성 완료 (축별 편향 반영) ===============")
    print(f"**통계 보고서 저장 경로:** {output_path}")
    print("==========================================================")