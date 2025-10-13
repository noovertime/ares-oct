import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import time
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import logging

import config
import ares_batch_report_util  # 유틸리티 파일 임포트

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
JUDGE_TYPES = ['contextrelevance', 'answerfaithfulness', 'answerrelevance']
FOLDER_MAPPING = {
    'contextrelevance': 'context_relevance',
    'answerfaithfulness': 'answer_faithfulness',
    'answerrelevance': 'answer_relevance'
}


# ===================================================================
# 1. ARES 평가 및 유틸리티 함수
# ===================================================================

def _find_model_path(judge_type: str) -> str:
    """고정된 심사관 이름 폴더 경로를 반환합니다."""
    target_folder = FOLDER_MAPPING.get(judge_type.lower())

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
        cr_path = _find_model_path('contextrelevance')
        tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
        print(f"   [INFO] 토크나이저 로드 성공 (저장된 경로에서).")
    except Exception as e:
        print(f"   [WARN] 저장 경로에서 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        except Exception as fallback_e:
            print(f"   [FATAL] 토크나이저 로드 최종 실패: {fallback_e}")
            raise fallback_e

    # DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화합니다.
    tokenizer.model_input_names = [
        name for name in tokenizer.model_input_names if name != 'token_type_ids'
    ]
    print("   [INFO] DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화했습니다.")

    # 2. 모델 로드 (AutoModelForSequenceClassification 사용)
    for judge_type in JUDGE_TYPES:
        try:
            model_path = _find_model_path(judge_type)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                trust_remote_code=True
            )
            model.to(DEVICE)
            model.eval()
            judges[judge_type] = model
            print(f"   [SUCCESS] {judge_type.upper()} Judge 로드 완료.")

        except Exception as e:
            print(f"   [ERROR] {judge_type.upper()} Judge 로드 실패: {e}. 이 모델은 건너뜁니다.")

    if len(judges) != 3:
        raise RuntimeError(f"총 {len(judges)}개만 로드됨. ARES 평가를 위해 3개 모델이 모두 필요합니다.")

    print(f">> ARES 심사관 로드 완료. 총 {len(judges)}개 활성화.")
    return tokenizer, judges


# *** 함수 인자 이름 수정: tokenizer -> tokenizer_obj ***
def evaluate_triple(tokenizer_obj: AutoTokenizer, judges: Dict[str, AutoModelForSequenceClassification],
                    query: str, context: str, answer: str) -> Dict[str, int]:
    """하나의 Q-C-A 쌍에 대해 3가지 ARES 점수 (0 또는 1)를 계산합니다."""

    results = {}

    judge_inputs = {
        'contextrelevance': (query, context, judges['contextrelevance']),
        'answerfaithfulness': (context, answer, judges['answerfaithfulness']),
        'answerrelevance': (query, answer, judges['answerrelevance'])
    }

    with torch.no_grad():
        for name, (text_a, text_b, model) in judge_inputs.items():
            # 1. 입력 토큰화: tokenizer_obj 사용으로 변경
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

            results[name] = prediction

    return results


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
    os.makedirs(config.DATA_REPORT_DIR, exist_ok=True)  # 보고서 디렉토리 생성
    print(f"\n[SETUP] QCA 입력 디렉토리: {INPUT_DIR}, PPI 출력 디렉토리: {OUTPUT_DIR}")

    # load_ares_judges 호출
    try:
        tokenizer, judges = load_ares_judges()
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] ARES 심사관 시스템 초기화 실패: {e}")
        return

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

                    query = data.get('q')
                    context = data.get('c')
                    answer = data.get('a')

                    if not all([query, context, answer]):
                        print(f"[SKIP] 데이터 형식 오류 (Q, C, A 중 누락) - 파일: {file_base_name}, 라인: {line.strip()[:50]}...")
                        continue

                    # evaluate_triple 호출 시 인자 이름 변경 적용
                    scores = evaluate_triple(tokenizer, judges, query, context, answer)
                    data.update(scores)
                    all_results_for_file.append(data)
                    total_successful_evals += 1

                except json.JSONDecodeError:
                    print(f"[SKIP] JSON 구문 오류 발생 - 파일: {file_base_name}, 라인 건너뜀: {line.strip()[:50]}...")
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
    ares_batch_report_util.run_summary_generation_pipeline()
    # (run_summary_generation_pipeline 내에서 최종 완료 로그 출력)


if __name__ == "__main__":
    run_ares_pipeline()