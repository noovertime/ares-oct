import os
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple # <-- Tuple 추가
from scipy.stats import binom, norm
from scipy.optimize import brentq


# ===================================================================
# 0. 전역 상수 및 설정
# ===================================================================

INPUT_DIR = "./out"  # PPI 파일 (report_*.jsonl)이 저장된 디렉토리
OUTPUT_DIR = "./report"  # 최종 통계 보고서가 저장될 디렉토리
CI_ALPHA = 0.05  # 95% 신뢰구간 (alpha=0.05)


# ===================================================================
# 1. 통계 계산 유틸리티 (PPI 로직 기반)
# ===================================================================
def calculate_binomial_ci(n: int, k: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    ppi.py의 binomial_iid 로직을 사용하여 95% 이항 신뢰구간 (Clopper-Pearson 근사)을 계산합니다.

    Args:
        n (int): 총 시행 횟수 (Total Samples)
        k (int): 성공 횟수 (Success Count)
        alpha (float): 유의 수준 (기본값 0.05 for 95% CI)

    Returns:
        Tuple[float, float]: (하한, 상한)
    """
    if n == 0:
        return 0.0, 1.0  # 샘플 없으면 신뢰구간 정의 불가능

    # muhat (관측된 성공률)
    muhat = k / n

    # 하한(l) 계산을 위한 함수: invert_lower_tail
    # (ppi.py의 invert_lower_tail 로직)
    def invert_lower_tail(mu):
        return binom.cdf(k, n, mu) - (1 - alpha / 2)

    # 상한(u) 계산을 위한 함수: invert_upper_tail
    # (ppi.py의 invert_upper_tail 로직)
    def invert_upper_tail(mu):
        return binom.cdf(k, n, mu) - (alpha / 2)

    # 1. 상한 (u) 계산
    # k == n (모두 성공)이면 상한은 1.0
    if k == n:
        u = 1.0
    else:
        # brentq를 사용하여 [muhat, 1.0] 범위에서 근을 찾습니다.
        try:
            u = brentq(invert_upper_tail, muhat, 1.0)
        except ValueError:
            u = 1.0

            # 2. 하한 (l) 계산
    # k == 0 (모두 실패)이면 하한은 0.0
    if k == 0:
        l = 0.0
    else:
        # brentq를 사용하여 [0.0, muhat] 범위에서 근을 찾습니다.
        try:
            l = brentq(invert_lower_tail, 0.0, muhat)
        except ValueError:
            l = 0.0

    return l, u



def analyze_ppi_file(filepath: str) -> Dict[str, Any]:
    """
    단일 PPI 파일을 읽어 CR, AF, AR의 평균 및 신뢰구간을 계산합니다.
    """
    model_name_raw = os.path.basename(filepath)
    all_scores: Dict[str, List[int]] = {
        'contextrelevance': [],
        'answerfaithfulness': [],
        'answerrelevance': []
    }

    # 파일명에서 모델명 추출 (예: 'report_gpt-rag.jsonl' -> 'gpt-rag')
    model_name = model_name_raw.replace("report_", "").replace(".jsonl", "")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for key in all_scores.keys():
                        if key in data:
                            all_scores[key].append(data[key])
                except json.JSONDecodeError:
                    # 파일명과 라인 정보 출력 추가 (디버깅 용이)
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
        'n': total_n
    }

    overall_scores = []

    for axis in all_scores.keys():
        scores = all_scores[axis]
        success_k = sum(scores)

        # 신뢰구간 (하한, 상한) 계산
        l, u = calculate_binomial_ci(total_n, success_k)

        mean_score = success_k / total_n

        # 신뢰구간의 마진 (± CI) 계산
        # margin = (상한 - 하한) / 2
        margin = (u - l) / 2

        summary[axis] = {
            'mean': round(mean_score, 2),
            'ci': round(margin, 3)  # 소수점 셋째 자리까지 반올림
        }
        overall_scores.append(mean_score)

    # 종합 점수 계산 (3축 평균)
    summary['overall'] = round(sum(overall_scores) / 3, 2)

    return summary


# ===================================================================
# 2. 보고서 생성 및 출력
# ===================================================================

def generate_summary_report(model_summaries: List[Dict[str, Any]]):
    # (내용 유지)
    """
    분석된 모델 요약을 바탕으로 최종 Markdown 형식의 보고서를 생성합니다.
    """

    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없어 보고서를 생성할 수 없습니다."

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    report_name = "RAG 자동 평가 결과 (ARES PPI 요약)"

    # 모델 이름 정렬 (보고서 가독성 향상)
    model_summaries.sort(key=lambda x: x['overall'], reverse=True)

    # 템플릿 작성
    report_content = f"""
# 🧭 ARES 자동 평가 결과 보고서 (Prediction-Powered Inference 요약)

프로젝트명: ARES 심사관 로컬 배치 평가
평가 프레임워크: Stanford ARES (PPI 기반 이항 신뢰구간)
평가 일자: {current_time}
평가 대상 모델: {', '.join([s['model_name'] for s in model_summaries])}

### 1️⃣ 평가 개요

| 평가 축 | 미세부 설명 |
| :--- | :--- |
| **Context Relevance (CR)** | 검색된 문서가 질문과 얼마나 관련 있는가 (문맥 적합성) |
| **Answer Faithfulness (AF)** | 생성된 답변이 검색 문서 내용에 충실한가 (응답 충실도) |
| **Answer Relevance (AR)** | 답변이 질문에 직접적이고 구체적인가 (응답 적절성) |

### 2️⃣ 자동 평가 점수 및 신뢰구간 요약 (95% CI)

| 모델명 | CR (±95% CI) | AF (±95% CI) | AR (±95% CI) | 종합 점수 | 총 샘플 수 |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    # 데이터 행 추가
    for summary in model_summaries:
        cr_str = f"{summary['contextrelevance']['mean']:.2f} ±{summary['contextrelevance']['ci']:.3f}"
        af_str = f"{summary['answerfaithfulness']['mean']:.2f} ±{summary['answerfaithfulness']['ci']:.3f}"
        ar_str = f"{summary['answerrelevance']['mean']:.2f} ±{summary['answerrelevance']['ci']:.3f}"

        report_content += (
            f"| **{summary['model_name']}** "
            f"| {cr_str} "
            f"| {af_str} "
            f"| {ar_str} "
            f"| **{summary['overall']:.2f}** "
            f"| {summary['n']} |\n"
        )

    report_content += """
---
### 해석 및 결론 (자동 생성)

* **PPI 적용:** ARES의 PPI(Prediction-Powered Inference) 방법론을 기반으로, 각 모델의 성능은 95% 신뢰구간(CI) 내에 존재한다고 통계적으로 추정됩니다.
* **통계적 신뢰도:** **신뢰구간이 겹치지 않는 모델 간**에는 95% 신뢰 수준에서 통계적으로 유의미한 성능 차이가 존재합니다.
* **평가 일관성:** 신뢰구간 폭이 좁은 모델일수록 평가 샘플에 대한 예측 일관성이 높습니다.
"""

    return report_content


# ===================================================================
# 3. 메인 실행 로직
# ===================================================================

def run_summary_generation():
    """
    메인 함수: ./out 디렉토리의 PPI 파일을 분석하고 요약 보고서를 생성합니다.
    """
    print(f"\n>> ARES 통계 보고서 생성 시작")

    # 1. 디렉토리 구조 확인
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(INPUT_DIR):
        print(f"[FATAL] 입력 디렉토리 {INPUT_DIR}을 찾을 수 없습니다. PPI 파일 생성 단계를 먼저 완료하세요.")
        return

    # 2. 입력 PPI 파일 목록 검색
    ppi_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith('.jsonl') and f.startswith('report_')
    ]

    if not ppi_files:
        print(f"[WARN] {INPUT_DIR} 디렉토리에서 분석할 'report_*.jsonl' PPI 파일을 찾을 수 없습니다.")
        return

    print(f"[INFO] 총 {len(ppi_files)}개 모델의 PPI 파일을 분석합니다.")

    model_summaries = []

    # 3. 파일별 분석 수행
    for file_path in ppi_files:
        summary = analyze_ppi_file(file_path)
        if summary:
            model_summaries.append(summary)
            print(f"   [SUCCESS] 모델 '{summary['model_name']}' 분석 완료 (샘플 수: {summary['n']}개).")

    if not model_summaries:
        print("[WARN] 분석 가능한 모델 데이터가 없어 보고서 생성을 건너뜁니다.")
        return

    # 4. 최종 보고서 생성
    report_content = generate_summary_report(model_summaries)

    # 5. 파일 저장
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"summary_report_{timestamp}.md"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(report_content)

    print("\n\n=============== 보고서 생성 완료 ===============")
    print(f"분석된 모델 수: {len(model_summaries)}개")
    print(f"**통계 보고서 저장 경로:** {output_path}")
    print("==============================================")


if __name__ == "__main__":
    run_summary_generation()