# ares_batch_report_util.py

import os
import json
import time
from typing import Dict, List, Any, Tuple
from scipy.stats import binom
from scipy.optimize import brentq

# config는 메인 스크립트에서 설정된 환경을 사용합니다.
import config

# ===================================================================
# 0. 전역 상수 (메인 스크립트에서 정의된 것을 사용하거나 재정의)
# ===================================================================

CI_ALPHA = 0.05  # 95% 신뢰구간 (alpha=0.05)
# PPI 보정 시연을 위한 가상 상수 (골든 라벨 제공 시 실제 값으로 대체될 예정)
HYPOTHETICAL_RECTIFIER_MEAN = 0.20


# ===================================================================
# 1. 통계 계산 유틸리티 (PPI 로직 기반)
# ===================================================================

def calculate_binomial_ci(n: int, k: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    ppi.py의 binomial_iid 로직을 사용하여 95% 이항 신뢰구간을 계산합니다.
    """
    if n == 0:
        return 0.0, 1.0

    muhat = k / n

    def invert_lower_tail(mu):
        return binom.cdf(k, n, mu) - (1 - alpha / 2)

    def invert_upper_tail(mu):
        return binom.cdf(k, n, mu) - (alpha / 2)

    if k == n:
        u = 1.0
    else:
        try:
            u = brentq(invert_upper_tail, muhat, 1.0)
        except ValueError:
            u = 1.0

    if k == 0:
        l = 0.0
    else:
        try:
            l = brentq(invert_lower_tail, 0.0, muhat)
        except ValueError:
            l = 0.0

    return l, u


def analyze_ppi_file(filepath: str) -> Dict[str, Any]:
    """
    단일 PPI 파일을 읽어 PPI 보정 평균 및 신뢰구간을 계산하고 요약합니다.
    """
    model_name_raw = os.path.basename(filepath)
    all_scores: Dict[str, List[int]] = {
        'contextrelevance': [],
        'answerfaithfulness': [],
        'answerrelevance': []
    }

    # 입력 파일명에서 모델명 추출 (예: 'inputfilename_timestamp.jsonl' -> 'inputfilename')
    model_name_parts = model_name_raw.split('_')
    model_name = "_".join(model_name_parts[:-1])
    if not model_name:
        model_name = model_name_raw.replace(".jsonl", "")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for key in all_scores.keys():
                        if key in data:
                            all_scores[key].append(data[key])
                except json.JSONDecodeError:
                    print(f"[WARN] 파일 {model_name_raw}에서 JSON 디코딩 오류 발생. 라인 건너뜀.")
                    continue
    except FileNotFoundError:
        print(f"[ERROR] 파일 {filepath}을 찾을 수 없습니다. 건너뜁니다.")
        return None

    total_n = len(all_scores['contextrelevance'])
    if total_n == 0:
        print(f"[WARN] 파일 {model_name}에 유효한 샘플이 없습니다.")
        return None

    summary = {
        'model_name': model_name,
        'n': total_n,
        'rectifier_mean': HYPOTHETICAL_RECTIFIER_MEAN
    }

    overall_corrected_scores = []

    for axis in all_scores.keys():
        scores = all_scores[axis]
        success_k = sum(scores)

        # 1. 기계 예측 평균 (tilde_theta_f)
        machine_mean = success_k / total_n

        # 2. PPI 보정 평균 (theta_hat_PP = machine_mean - rectifier_mean)
        corrected_mean = max(0.0, min(1.0, machine_mean - HYPOTHETICAL_RECTIFIER_MEAN))

        # 3. 신뢰구간 계산 (이항 CI)
        l, u = calculate_binomial_ci(total_n, success_k)
        margin = (u - l) / 2

        summary[axis] = {
            'machine_mean': round(machine_mean, 2),
            'corrected_mean': round(corrected_mean, 2),
            'ci': round(margin, 3)
        }
        overall_corrected_scores.append(corrected_mean)

    # 종합 점수 계산 (보정된 3축 평균)
    summary['overall'] = round(sum(overall_corrected_scores) / 3, 2)

    return summary


def generate_summary_report(model_summaries: List[Dict[str, Any]]):
    """
    분석된 모델 요약을 바탕으로 최종 Markdown 형식의 보고서를 생성합니다.
    """

    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없어 보고서를 생성할 수 없습니다."

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # 모델 이름 정렬 (종합 점수가 높은 순서로)
    model_summaries.sort(key=lambda x: x['overall'], reverse=True)

    report_content = f"""
# 🧭 ARES 자동 평가 결과 보고서 (Prediction-Powered Inference 보정)

프로젝트명: ARES 심사관 로컬 배치 평가
평가 프레임워크: Stanford ARES (PPI 보정 로직 통합)
평가 일자: {current_time}
평가 대상 모델: {', '.join([s['model_name'] for s in model_summaries])}

### 1️⃣ 평가 개요 (PPI 보정 기반)

| 평가 축 | 미세부 설명 |
| :--- | :--- |
| **Context Relevance (CR)** | 검색된 문서가 질문과 얼마나 관련 있는가 (문맥 적합성) |
| **Answer Faithfulness (AF)** | 생성된 답변이 검색 문서 내용에 충실한가 (응답 충실도) |
| **Answer Relevance (AR)** | 답변이 질문에 직접적이고 구체적인가 (응답 적절성) |

### 2️⃣ PPI 보정 점수 및 신뢰구간 요약 (95% CI)

| 모델명 | **CR (보정)** | AF (보정) | AR (보정) | **종합 점수** | 기계 예측 평균 | CI 마진 (±) | 총 샘플 수 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for summary in model_summaries:
        cr_str = f"{summary['contextrelevance']['corrected_mean']:.2f}"
        af_str = f"{summary['answerfaithfulness']['corrected_mean']:.2f}"
        ar_str = f"{summary['answerrelevance']['corrected_mean']:.2f}"
        ci_margin = f"±{summary['contextrelevance']['ci']:.3f}"

        report_content += (
            f"| **{summary['model_name']}** "
            f"| {cr_str} "
            f"| {af_str} "
            f"| {ar_str} "
            f"| **{summary['overall']:.2f}** "
            f"| {summary['contextrelevance']['machine_mean']:.2f} "
            f"| {ci_margin} "
            f"| {summary['n']} |\n"
        )

    report_content += f"""
---
### 3️⃣ 보정 로직 및 해석

* **PPI 보정 적용:** PPI(Prediction-Powered Inference) 방식에 따라 **보정 평균 (Corrected Mean)**을 산출했습니다.
* **PPI 분산 계산 방식:** ARES는 PPI 평균 추정의 **점근적 분산 공식**을 사용합니다. 보고서의 CI 마진은 현재 **이항 신뢰구간 근사치**를 사용하고 있으며, PPI의 엄격한 분산 계산(기계 예측 분산 및 보정항 분산)은 골든 라벨 제공 후 추가 통합될 예정입니다.
* **보정 방법:** 이 보고서에서는 {HYPOTHETICAL_RECTIFIER_MEAN:.2f}의 **가상 편향(Rectifier Mean)**이 모든 모델의 예측 평균에서 차감되었습니다. (골든 라벨 제공 시 실제 편향으로 대체됩니다.)
"""

    return report_content


def run_summary_generation_pipeline():
    """메인 실행 함수를 위해 분석/저장 로직만 분리합니다."""

    INPUT_DIR_SUM = config.DATA_OUT_DIR
    OUTPUT_DIR_SUM = config.DATA_REPORT_DIR

    os.makedirs(OUTPUT_DIR_SUM, exist_ok=True)

    # 입력 파일명_타임스탬프.jsonl 형식 파일을 검색
    ppi_files = [
        os.path.join(INPUT_DIR_SUM, f)
        for f in os.listdir(INPUT_DIR_SUM)
        if f.endswith('.jsonl') and len(f.split('_')) >= 2
    ]

    if not ppi_files:
        print(f"[WARN] {INPUT_DIR_SUM}에서 분석할 PPI 파일을 찾을 수 없습니다. 보고서 생성을 건너뜕니다.")
        return

    print(f"[INFO] 총 {len(ppi_files)}개 모델의 PPI 파일을 분석합니다.")

    model_summaries = []

    for file_path in ppi_files:
        summary = analyze_ppi_file(file_path)
        if summary:
            model_summaries.append(summary)
            print(f"   [SUCCESS] 모델 '{summary['model_name']}' 분석 완료.")

    if not model_summaries:
        print("[WARN] 분석 가능한 모델 데이터가 없어 보고서 생성을 건너뜁니다.")
        return

    report_content = generate_summary_report(model_summaries)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"summary_report_ppi_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR_SUM, output_filename)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(report_content)

    print("\n\n=============== ARES 통계 보고서 생성 완료 ===============")
    print(f"**통계 보고서 저장 경로:** {output_path}")
    print("==========================================================")

# run_summary_generation_pipeline 함수는 메인 스크립트에서 호출됩니다.