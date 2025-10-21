# 02_generate_ppi.py (상수 최적화 최종 버전)
# 1. 표준 라이브러리
import json
import logging
import math
import os
import time
import sys  # sys가 없으므로 생략

from typing import Dict, Any, List, Tuple
from numpy.typing import NDArray

# 2. 서드파티 라이브러리
import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 3. 로컬/애플리케이션 고유 라이브러리
import config
import report_util
from config import (
    KEY_CR,
    KEY_AF,
    KEY_AR,
    JUDGE_TYPES,
    JUDGE_PREDICTION_FIELDS,
    GOLD_LABEL_FIELDS,
    TOKEN_TYPE_ID_OFF
)
from json_util import _load_json_lines

# ===================================================================
# 0. 전역 상수 및 환경 설정
# ===================================================================

# 1. Transformers 경고 메시지 비활성화
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 경로 설정: config 파일에서 불러오기
MODEL_DIR_BASE = config.MODEL_DIR
MODEL_NAME = config.MODEL_NAME

# CPU 환경 설정
DEVICE = torch.device("cpu")
MAX_LENGTH = 128

# ARES 심사관 타입 및 폴더 매핑
FOLDER_MAPPING = {
    config.KEY_CR: 'context_relevance',
    config.KEY_AF: 'answer_faithfulness',
    config.KEY_AR: 'answer_relevance'
}

CI_ALPHA: float = 0.05  # 95% 신뢰구간
CI_Z_SCORE: float = float(norm.ppf(1 - CI_ALPHA / 2))  # 약 1.96


# ===================================================================
# 1. ARES 평가 및 유틸리티 함수
# ===================================================================

def _find_model_path(judge_type: str) -> str:
    """고정된 심사관 이름 폴더 경로를 반환합니다."""
    # judge_type은 KEY_CR, KEY_AF, KEY_AR 중 하나
    target_folder = FOLDER_MAPPING.get(judge_type)

    if not target_folder:
        raise ValueError(f"정의되지 않은 심사관 타입: {judge_type}")

    model_path = os.path.join(MODEL_DIR_BASE, target_folder)

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {model_path}")

    return model_path


def load_ares_judges() -> Tuple[AutoTokenizer, Dict[str, AutoModelForSequenceClassification]]:
    """CR, AF, AR 세 가지 심사관 모델과 토크나이저를 로드합니다."""
    print("\n>> ARES 심사관 로딩 시작 (CPU 환경)...")
    tokenizer = None
    judges = {}

    # 1. 토크나이저 초기화
    try:
        # KEY_CR을 사용하여 경로를 찾음
        cr_path = _find_model_path(KEY_CR)
        tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
        print(f"   [INFO] {cr_path} 에서 토크나이저 로드 성공.")
    except Exception as e:
        print(f"   [WARN] 저장 경로에서 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        except Exception as fallback_e:
            print(f"   [FATAL] 토크나이저 로드 최종 실패: {fallback_e}")
            raise fallback_e

    if TOKEN_TYPE_ID_OFF:
        # DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화합니다.
        tokenizer.model_input_names = [
            name for name in tokenizer.model_input_names if name != 'token_type_ids'
        ]
        print("   [INFO] DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화했습니다.")

    # 2. 모델 로드 (AutoModelForSequenceClassification 사용)
    for judge_type in JUDGE_TYPES:  # JUDGE_TYPES 리스트 사용
        try:
            model_path = _find_model_path(judge_type)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                trust_remote_code=True
            )
            model.to(DEVICE)
            model.eval()
            judges[judge_type] = model  # judges 딕셔너리에 KEY_CR, KEY_AF, KEY_AR 키로 저장
            print(f"   [SUCCESS] {judge_type} Judge 로드 완료 ( {model_path} )")

        except Exception as e:
            print(f"   [ERROR] {judge_type} Judge 로드 실패: {e}. 이 모델은 건너뜁니다.")

    if len(judges) != 3:
        raise RuntimeError(f"총 {len(judges)}개만 로드됨. ARES 평가를 위해 3개 모델이 모두 필요합니다.")

    print(f">> ARES 심사관 {MODEL_NAME} 로드 완료. 총 {len(judges)}개 심사관 활성화.")
    return tokenizer, judges


def evaluate_triple(tokenizer_obj: AutoTokenizer, judges: Dict[str, AutoModelForSequenceClassification],
                    query: str, context: str, answer: str) -> Dict[str, int]:
    """하나의 Q-C-A 쌍에 대해 3가지 ARES 점수 (0 또는 1)를 계산합니다."""

    results = {}

    # JUDGE_PREDICTION_FIELDS 상수를 사용하여 judge_inputs 구성
    judge_inputs = {
        JUDGE_PREDICTION_FIELDS[KEY_CR]: (query, context, judges[KEY_CR]),
        JUDGE_PREDICTION_FIELDS[KEY_AF]: (context, answer, judges[KEY_AF]),
        JUDGE_PREDICTION_FIELDS[KEY_AR]: (query, answer, judges[KEY_AR])
    }

    with torch.no_grad():
        # name은 JUDGE_PREDICTION_FIELDS의 값 ('contextrelevance' 등)
        for name, (text_a, text_b, model) in judge_inputs.items():
            # 1. 입력 토큰화
            inputs = tokenizer_obj(
                text_a, text_b,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH
            ).to(DEVICE)

            # 2. 예측 수행 및 결과 산출
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

            results[name] = prediction  # 결과는 JUDGE_PREDICTION_FIELDS의 값으로 저장됨

    return results


def load_gold_labels_map(filepath: str, gold_field_mapping: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    골든 라벨 파일을 로드하여 ID(string)를 키로 하는 맵을 생성합니다. (ID 기반 매칭)
    """
    gold_map = {}
    print(f"\n>> 골든 라벨 로딩 시작: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())

                    # ID 필드를 고유 키로 사용
                    sample_id = data.get('id')

                    if not sample_id:
                        print(f"[WARN] 골든 라벨 파일 {filepath}의 {i + 1}번째 줄: 'id' 필드 누락. 건너뜀.")
                        continue

                    labels = {}
                    # gold_field_mapping을 사용하여 L_CR 등의 필드만 추출
                    for machine_key, gold_key_name in gold_field_mapping.items():
                        if gold_key_name in data:
                            labels[gold_key_name] = data[gold_key_name]
                        else:
                            labels[gold_key_name] = -1

                    gold_map[sample_id] = labels

                except json.JSONDecodeError:
                    print(f"[WARN] 골든 라벨 파일 {filepath}의 {i + 1}번째 줄: JSON 오류. 건너ntd.")
                    continue

    except FileNotFoundError:
        print(f"[ERROR] 골든 라벨 파일 {filepath}을 찾을 수 없습니다.")
        return {}

    print(f">> 골든 라벨 로딩 완료. 총 {len(gold_map)}개 샘플.")
    return gold_map


def cleanup_evaluation_data():
    """
    config에 정의된 PPI 출력 디렉토리와 최종 보고서 디렉토리 내의 모든 파일을 삭제합니다.
    """

    dirs_to_clean = [config.DATA_OUT_DIR, config.DATA_REPORT_DIR]

    print("\n>> 평가 및 보고서 출력 파일 정리 시작...")

    for target_dir in dirs_to_clean:
        if os.path.isdir(target_dir):
            files_deleted = 0
            for filename in os.listdir(target_dir):
                file_path = os.path.join(target_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_deleted += 1
                except Exception as e:
                    print(f"   [ERROR] 파일 삭제 실패: {file_path} - {e}")

            print(f"   [SUCCESS] '{target_dir}' 디렉토리에서 총 {files_deleted}개 파일 삭제 완료.")
        else:
            print(f"   [INFO] 디렉토리 '{target_dir}'가 존재하지 않아 정리할 파일이 없습니다.")


# --- 02_generate_ppi.py 파일 내 evaluate_golden_set 함수 ---

def generate_golden_set_stat(golden_set_stats) -> str:
    if golden_set_stats:

        # LM 예측 평균 값 추출
        cr_mean = golden_set_stats.get(KEY_CR, {}).get('machine_mean', 'N/A')
        af_mean = golden_set_stats.get(KEY_AF, {}).get('machine_mean', 'N/A')
        ar_mean = golden_set_stats.get(KEY_AR, {}).get('machine_mean', 'N/A')

        # --- 새로운 로직: 심사관 정답 비율 (정확도) 계산 ---

        # 정확도 = 1 - (LM 예측 오류 비율)
        # LM 예측 오류 비율 = |Yhat - Y|의 평균 = |rectifier_terms|의 평균

        def calculate_accuracy(stats):
            """rectifier_terms를 사용하여 심사관의 정확도를 계산합니다."""
            terms = stats.get('rectifier_terms')
            labeled_n = stats.get('labeled_n')

            if not terms or labeled_n == 0:
                return 'N/A'

            # 오류 횟수 = (rectifier_terms 리스트에서 0이 아닌 값의 개수)
            error_count = sum(1 for term in terms if term != 0.0)
            accuracy = 1.0 - (error_count / labeled_n)
            return f"{accuracy:13.3f}"

        cr_acc = calculate_accuracy(golden_set_stats.get(KEY_CR, {}))
        af_acc = calculate_accuracy(golden_set_stats.get(KEY_AF, {}))
        ar_acc = calculate_accuracy(golden_set_stats.get(KEY_AR, {}))

        # ----------------------------------------------------

        # 1. 헤더 (Header) - ^12, ^13 중앙 정렬
        header = f"| {'구분':^12} | {'CR':^13} | {'AF':^13} | {'AR':^13} |"

        # 2. 구분선 (Separator)
        separator = f"+----------------+---------------+---------------+---------------+"

        # 3. 예측 평균 (Mean Row) - ^12 중앙 정렬, >13.3f 오른쪽 정렬
        mean_row = (
            f"| {'예측평균':^12} | {cr_mean:>13.3f} | {af_mean:>13.3f} | {ar_mean:>13.3f} |"
            f"  심사관이 1이라고 예측한 비율 (긍정 편향)"
        )

        # 4. 정답 비율 (Accuracy Row) - ^12 중앙 정렬, 값들은 >13 오른쪽 정렬 (문자열 가정)
        acc_row = (
            f"| {'정답비율':^12} | {cr_acc:>13} | {af_acc:>13} | {ar_acc:>13} |"
            f"  심사관 예측의 정확도"
        )

        # 모든 행을 합쳐 마크다운 코드 블록 문자열을 생성
        markdown_content = "\n".join([header, separator, mean_row, acc_row])
        markdown_string = f"```\n{markdown_content}\n```"
        return markdown_string
    else:
        print("[FAIL] 골든셋 평가 결과가 비어있거나 실패했습니다.")
        return "골든셋 평가 결과 없음"


def evaluate_golden_set(tokenizer_obj: AutoTokenizer, judges: Dict[str, AutoModelForSequenceClassification],
                        golden_set_filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    골든셋 데이터를 심사관이 평가하고, PPI 편향 계산에 필요한 통계 정보를 반환합니다.

    Args:
        tokenizer_obj: 로드된 토크나이저 객체.
        judges: 로드된 3개의 LM 심사관 모델 딕셔너리.
        golden_set_filepath: 골든셋 Q-C-A와 Y_labeled를 포함하는 JSONL 파일 경로.

    Returns:
        { 'KEY_CR': { 'labeled_n': int, 'rectifier_terms': List[float], 'machine_mean': float }, ... }
    """
    golden_stats = {key: {'labeled_n': 0, 'rectifier_terms': [], 'machine_mean_sum': 0.0} for key in JUDGE_TYPES}

    # 1. 골든셋 파일 로드: 인수로 받은 경로를 사용합니다.
    golden_records = _load_json_lines(golden_set_filepath)

    if not golden_records:
        print(f"[WARN] 골든셋 파일 {golden_set_filepath}에 평가할 데이터가 없습니다.")
        return {}

    print(f"\n>> 골든셋 평가 시작. {golden_set_filepath}, 총 {len(golden_records)}개 샘플 심사관 예측 중...")

    for data in tqdm(golden_records, desc="골든셋 심사관 평가 중"):
        try:
            # Q, C, A 추출 (골든셋 파일에도 'q', 'c', 'a'가 있다고 가정)
            query = ' '.join(data.get('q', '').split()).strip()
            context = ' '.join(data.get('c', '').split()).strip()
            answer = ' '.join(data.get('a', '').split()).strip()

            if not all([query, context, answer]):
                continue

            # 1. LM 심사관 예측 (Yhat_labeled)
            scores = evaluate_triple(tokenizer_obj, judges, query, context, answer)

            # 2. LM 예측값과 인간 주석값 비교 (Y_labeled는 골든셋 파일에서 직접 추출)
            for axis in JUDGE_TYPES:
                machine_pred = scores.get(JUDGE_PREDICTION_FIELDS[axis])
                gold_key = GOLD_LABEL_FIELDS[axis]
                gold_label = data.get(gold_key)  # 골든셋 파일에서 직접 Y_labeled 로드

                if machine_pred is None or gold_label is None:
                    continue

                # 통계 업데이트
                machine_pred = int(machine_pred)
                gold_label = int(gold_label)

                rectifier_term = float(machine_pred - gold_label)

                stats = golden_stats[axis]
                stats['labeled_n'] += 1
                stats['rectifier_terms'].append(rectifier_term)
                stats['machine_mean_sum'] += machine_pred

        except Exception as e:
            # print(f"[WARN] 골든셋 평가 중 오류 발생: {e}")
            continue

    # 3. 최종 통계 계산
    final_golden_stats = {}
    for axis, stats in golden_stats.items():
        if stats['labeled_n'] > 0:
            final_golden_stats[axis] = {
                'labeled_n': stats['labeled_n'],
                'rectifier_terms': stats['rectifier_terms'],
                # LM 심사관이 골든셋에 대해 예측한 평균 (Yhat_labeled 평균)
                'machine_mean': stats['machine_mean_sum'] / stats['labeled_n']
            }

    return final_golden_stats


# ---

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


def _calculate_ppi_summary(
        file_base_name: str,
        current_lm_scores: Dict[str, List[int]],
        total_n: int,
        golden_set_stats: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    LM 예측 결과와 골든셋 통계를 결합하여 PPI 요약 결과를 계산합니다.
    """

    model_name_report = file_base_name  # 파일 이름을 모델 이름으로 사용

    summary: Dict[str, Any] = {
        "model_name": model_name_report,
        "n": total_n,
        "ppi_active": True,
        "labeled_n_rep": golden_set_stats[KEY_CR]['labeled_n'],  # CR 기준으로 대표값 사용
    }

    overall_corrected_scores: List[float] = []

    for axis in JUDGE_TYPES:
        # 1. LM 예측 결과 (Yhat_unlabeled)
        scores: List[int] = current_lm_scores[axis]

        # 2. 골든셋 통계 (Rectifier Terms, labeled_n)
        golden_axis_stats = golden_set_stats.get(axis, {})
        if not golden_axis_stats: continue  # 통계가 없는 축은 건너뜀

        labeled_n_axis: int = golden_axis_stats['labeled_n']
        rectifier_terms: List[float] = golden_axis_stats['rectifier_terms']

        # 3. 통계 계산

        # 3-1. 심사관 예측 평균 (Avg(Yhat_unlabeled))
        machine_mean: float = sum(scores) / float(total_n)

        # 3-2. 편향 (Rectifier) = Avg(Yhat_labeled - Y_labeled)
        rectifier: float = (
            sum(rectifier_terms) / labeled_n_axis if labeled_n_axis > 0 else 0.0
        )

        # 3-3. 보정된 성능 (Corrected Mean)
        corrected_mean: float = max(0.0, min(1.0, machine_mean - rectifier))

        # 3-4. 신뢰 구간 (CI)
        margin: float = calculate_ppi_asymptotic_ci(scores, rectifier_terms, total_n, labeled_n_axis)

        # 4. 최종 요약에 추가
        summary[axis] = {
            "machine_mean": round(machine_mean, 2),
            "corrected_mean": round(corrected_mean, 2),
            "applied_rectifier": round(rectifier, 3),
            "ci": round(margin, 2)
        }
        overall_corrected_scores.append(corrected_mean)

    if overall_corrected_scores:
        summary["overall"] = round(sum(overall_corrected_scores) / len(overall_corrected_scores), 2)

    return summary


def run_ares_pipeline():
    """
    ARES 전체 파이프라인 실행: 골든셋 평가를 메모리에서 처리하고,
    LM 예측 파일 생성 및 최종 보고서를 즉시 생성합니다. (단일 모듈)
    """

    # --- 1. 환경 설정 및 초기화 단계 ---
    INPUT_DIR = config.DATA_IN_DIR
    OUTPUT_DIR = config.DATA_OUT_DIR
    REPORT_DIR = config.DATA_REPORT_DIR

    # 필수 디렉토리 생성 (DATA_STATS_DIR 저장은 제거됨)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"\n[SETUP] QCA 입력 디렉토리: {INPUT_DIR}, LM 예측 출력 디렉토리: {OUTPUT_DIR}")

    # LM 심사관 로드 (재사용됨)
    try:
        tokenizer, judges = load_ares_judges()
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] ARES 심사관 시스템 초기화 실패: {e}")
        return

    # 1-2. 골든 라벨 로드 및 LM 심사관 평가 (메모리 처리)
    GOLD_LABEL_PATH = os.path.join(config.DATA_GOLDEN_DIR, config.DATA_GOLDEN_FILE_NAME)

    try:
        # 1. 골든셋 평가 수행 및 통계 반환 (golden_set_stats는 메모리에 유지)
        golden_set_stats = evaluate_golden_set(tokenizer, judges, GOLD_LABEL_PATH)
        if not golden_set_stats:
            print("\n[FATAL ERROR] PPI 보정을 위한 골든 라벨 데이터 평가 실패. 파이프라인을 중단합니다.")
            return
    except Exception as e:
        print(f"\n[FATAL ERROR] 골든셋 평가 중 오류 발생: {e}. 파이프라인을 중단합니다.")
        return

    # --- 2. 입력 파일 검색 및 LM 예측 생성 루프 단계 ---
    input_files = [
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl')
    ]

    if not input_files:
        print(f"\n[WARN] QCA `.jsonl` 파일을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return

    print(f"\n[INFO] 총 {len(input_files)}개의 `.jsonl` 파일을 평가합니다.")
    print("   [LIST] 평가 대상 파일:")
    for file_path in input_files:
        print(f"       - {os.path.basename(file_path)}")

    # 골든셋 결과를 문자열로 생성
    golden_set_markdown:str = generate_golden_set_stat(golden_set_stats)

    # 출력
    print(f"골든셋 평가 결과 ---- ")
    print(f"{golden_set_markdown}")

    total_successful_evals = 0
    full_start_time = time.time()

    # 각 rag(out파일) 결과를 담을 자료구조
    model_summaries: List[Dict[str, Any]] = []

    # 2-2. 파일별 평가, LM 예측 파일 생성 (Yhat_unlabeled) 및 결과 집계
    for file_path in input_files:
        start_time = time.time()
        file_base_name = os.path.basename(file_path).replace('.jsonl', '')
        all_results_for_file = []

        # LM 예측값(Yhat_unlabeled)을 축별로 저장할 리스트
        current_lm_scores = {k: [] for k in JUDGE_TYPES}

        print(f"\n--- 대규모 평가 시작: {file_base_name} ---")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_samples_in_file = len(lines)

            for line in tqdm(lines, desc=f"평가 중 [{file_base_name}]"):
                try:
                    data = json.loads(line.strip())

                    # Q, C, A 정규화
                    query = ' '.join(data.get('q', '').split()).strip()
                    context = ' '.join(data.get('c', '').split()).strip()
                    answer = ' '.join(data.get('a', '').split()).strip()

                    if not all([query, context, answer]):
                        continue

                    # ARES 심사관 예측 수행 (Yhat_unlabeled 생성)
                    scores = evaluate_triple(tokenizer, judges, query, context, answer)
                    data.update(scores)

                    # LM 예측 결과를 현재 파일의 집계 리스트에 추가
                    for axis in JUDGE_TYPES:
                        pred_key = JUDGE_PREDICTION_FIELDS[axis]
                        current_lm_scores[axis].append(scores.get(pred_key, 0))

                    all_results_for_file.append(data)
                    total_successful_evals += 1

                except json.JSONDecodeError:
                    print(f"[SKIP] JSON 구문 오류 - 파일: {file_base_name}. 라인 건너뜀.")
                except Exception as e:
                    print(f"[ERROR] 처리 중 알 수 없는 오류 발생: {e} - 파일: {file_base_name}. 라인 건너뜀.")

        end_time = time.time()
        print(f"[INFO] 경과시간 : {end_time - start_time:,.2f}초 ")

        # 2-3. 심사관 예측이 있다면 보고서 생성 준비
        processed_count_in_file = len(all_results_for_file)
        if processed_count_in_file > 0:
            # 3. 보고서 생성을 위한 통계 계산 및 집계
            summary = _calculate_ppi_summary(
                file_base_name,
                current_lm_scores,
                processed_count_in_file,
                golden_set_stats  # 메모리에 있는 통계값 사용
            )
            model_summaries.append(summary)
            print(f"   [집계 완료] '{file_base_name}' 결과 집계 완료.")
        else:
            print(f"   [ERROR] 심사관의 평가 겨로가 없음")
            return

    # --- 3. 최종 보고서 생성 단계 ---
    full_end_time = time.time()
    full_elapsed_time = full_end_time - full_start_time

    # 최종 보고서 생성
    if model_summaries:
        report_content: str = report_util.generate_summary_report(golden_set_markdown, model_summaries)
        timestamp: str = time.strftime("%Y%m%d_%H%M%S")
        output_path: str = os.path.join(REPORT_DIR, f"summary_{timestamp}.md")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n[최종 완료] ARES 통계 보고서 생성됨 → {output_path}")

    print("\n\n=============== LM 예측 생성 최종 요약 ===============")
    print(f"총 LM 예측 샘플 수: {total_successful_evals}개")
    print(f"총 소요 시간: {full_elapsed_time:.2f}초")
    print("==================================================")


if __name__ == "__main__":
    run_ares_pipeline()  # 전체 파이프라인 실행
