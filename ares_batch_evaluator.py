# ares_batch_evaluator.py (상수 최적화 최종 버전)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import time
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import logging

import config
import ares_batch_report_util
from config import (
    KEY_CR,
    KEY_AF,
    KEY_AR,
    JUDGE_TYPES,
    JUDGE_PREDICTION_FIELDS,
    GOLD_LABEL_FIELDS
)


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
        print(f"   [INFO] {cr_path}에서 토크나이저 로드 성공.")
    except Exception as e:
        print(f"   [WARN] 저장 경로에서 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        except Exception as fallback_e:
            print(f"   [FATAL] 토크나이저 로드 최종 실패: {fallback_e}")
            raise fallback_e

    # DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화합니다.
    #tokenizer.model_input_names = [
    #    name for name in tokenizer.model_input_names if name != 'token_type_ids'
    #]
    #print("   [INFO] DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화했습니다.")

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
            print(f"   [SUCCESS] {judge_type.upper()} Judge 로드 완료.")

        except Exception as e:
            print(f"   [ERROR] {judge_type.upper()} Judge 로드 실패: {e}. 이 모델은 건너뜁니다.")

    if len(judges) != 3:
        raise RuntimeError(f"총 {len(judges)}개만 로드됨. ARES 평가를 위해 3개 모델이 모두 필요합니다.")

    print(f">> ARES 심사관 로드 완료. 총 {len(judges)}개 활성화.")
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


# ===================================================================
# 4. 메인 실행 함수 (평가 및 보고서 생성 통합)
# ===================================================================

def run_ares_pipeline():
    """
    ARES 전체 파이프라인 실행: PPI 파일 생성 후 통계 보고서 생성까지 자동 실행합니다.
    """

    # --- 1. PPI 파일 생성 단계 ---

    INPUT_DIR = config.DATA_IN_DIR
    OUTPUT_DIR = config.DATA_OUT_DIR

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.DATA_REPORT_DIR, exist_ok=True)
    print(f"\n[SETUP] QCA 입력 디렉토리: {INPUT_DIR}, PPI 출력 디렉토리: {OUTPUT_DIR}")

    # 1-1. 모델 로드
    try:
        tokenizer, judges = load_ares_judges()
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] ARES 심사관 시스템 초기화 실패: {e}")
        return

    # 1-2. 골든 라벨 로드 (PPI 보정을 위해)
    GOLD_LABEL_PATH = os.path.join(config.DATA_GOLDEN_DIR, config.DATA_GOLDEN_FILE_NAME)
    gold_label_map = load_gold_labels_map(GOLD_LABEL_PATH, GOLD_LABEL_FIELDS)

    # PPI 보정 활성화 여부 검증
    if not gold_label_map:
        print("\n[FATAL ERROR] PPI 보정을 위한 골든 라벨 데이터가 없습니다. 파이프라인을 중단합니다.")
        return

    ppi_correction_active = True

    # 1-3. 입력 파일 목록 검색 및 처리 루프
    input_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith('.jsonl')
    ]

    if not input_files:
        print(f"\n[WARN] QCA `.jsonl` 파일을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return

    print(f"\n[INFO] 총 {len(input_files)}개의 `.jsonl` 파일을 평가합니다.")

    total_processed_samples = 0
    total_successful_evals = 0
    full_start_time = time.time()

    for file_path in input_files:
        start_time = time.time()
        file_base_name = os.path.basename(file_path).replace('.jsonl', '')
        all_results_for_file = []
        total_samples_in_file = 0

        print(f"\n--- 파일 평가 시작: {file_base_name} ---")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_samples_in_file = len(lines)
            total_processed_samples += total_samples_in_file

            for line in tqdm(lines, desc=f"평가 중 [{file_base_name}]"):
                try:
                    data = json.loads(line.strip())

                    # Q, C, A 정규화: 매칭 정확도를 위해 입력 데이터도 정규화
                    query = ' '.join(data.get('q', '').split()).strip()
                    context = ' '.join(data.get('c', '').split()).strip()
                    answer = ' '.join(data.get('a', '').split()).strip()

                    if not all([query, context, answer]):
                        continue

                    # QCA 정규화된 값을 원본 data 딕셔너리에 덮어쓰기 (PPI 파일의 일관성 유지)
                    data['q'] = query
                    data['c'] = context
                    data['a'] = answer

                    # 1. ARES 예측 수행
                    scores = evaluate_triple(tokenizer, judges, query, context, answer)

                    # scores의 키는 JUDGE_PREDICTION_FIELDS의 값 ('contextrelevance' 등)
                    data.update(scores)

                    # 2. 골든 라벨 추가 (ID를 키로 사용)
                    sample_id = data.get('id')

                    if not sample_id:
                        # ID가 없으면 골든 라벨 매칭 불가능. 경고만 주고 넘어갑니다.
                        pass
                    elif sample_id in gold_label_map:
                        # gold_label_map의 키는 GOLD_LABEL_FIELDS의 값 ('L_CR' 등)
                        data.update(gold_label_map[sample_id])

                    all_results_for_file.append(data)
                    total_successful_evals += 1

                except json.JSONDecodeError:
                    print(f"[SKIP] JSON 구문 오류 - 파일: {file_base_name}. 라인 건너뜀.")
                except Exception as e:
                    print(f"[ERROR] 처리 중 알 수 없는 오류 발생: {e} - 파일: {file_base_name}. 라인 건너뜀.")

        end_time = time.time()

        processed_count_in_file = len(all_results_for_file)
        if processed_count_in_file > 0:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{file_base_name}_{timestamp}.jsonl"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for result in all_results_for_file:
                    outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

            elapsed_time = end_time - start_time
            print(f"\n--- {file_base_name} PPI 생성 완료 ---")
            print(f"  평가 성공 샘플 수: {processed_count_in_file} / {total_samples_in_file}개")
            print(f"  소요 시간: {elapsed_time:.2f}초")
            print(f"  저장 경로: {output_path}")

    full_end_time = time.time()
    full_elapsed_time = full_end_time - full_start_time

    print("\n\n=============== PPI 생성 최종 요약 ===============")
    print(f"총 PPI 생성 샘플 수: {total_successful_evals}개")
    print(f"총 소요 시간: {full_elapsed_time:.2f}초")
    print("==================================================")

    # --- 2. 통계 보고서 생성 단계 (유틸리티 함수 호출) ---
    print("\n>> ARES 통계 보고서 생성 시작")
    # ares_batch_report_util.py에 PPI 보정 활성화와 GOLD_LABEL_FIELDS 전달
    ares_batch_report_util.run_summary_generation_pipeline(True, GOLD_LABEL_FIELDS)


if __name__ == "__main__":
    run_ares_pipeline()