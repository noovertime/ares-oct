import torch
# AutoModelForSequenceClassification 임포트
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
import time
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import logging

# ===================================================================
# 0. 전역 상수 및 환경 설정
# ===================================================================

# 1. Transformers 경고 메시지 비활성화 (로그 깔끔하게 유지)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# 다른 잠재적 경고도 포함
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 경로 설정: 이 경로를 다운로드한 모델 폴더(answer_faithfulness 등)들이 모여있는 상위 디렉터리로 수정하세요.
MODEL_DIR_BASE = r"C:\dev\workspaces\pycharm\ares-oct\model"

# CPU 환경 설정
DEVICE = torch.device("cpu")
MAX_LENGTH = 128  # 학습 시 사용했던 최대 길이 유지
MODEL_NAME = "monologg/distilkobert"  # 토크나이저 fallback용 원본 모델 이름

# ARES 심사관 타입 및 폴더 매핑
JUDGE_TYPES = ['contextrelevance', 'answerfaithfulness', 'answerrelevance']
FOLDER_MAPPING = {
    'contextrelevance': 'context_relevance',
    'answerfaithfulness': 'answer_faithfulness',
    'answerrelevance': 'answer_relevance'
}


# ===================================================================
# 1. 모델 로드 및 유틸리티 함수
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


# AutoModelForSequenceClassification 사용으로 반환 타입 변경
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

    # 2. 모델 로드 (AutoModelForSequenceClassification 사용)
    for judge_type in JUDGE_TYPES:
        try:
            model_path = _find_model_path(judge_type)
            # *** 핵심 수정: AutoModelForSequenceClassification 사용 ***
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
            # 모델 구조 불일치 오류 시에도 상세 로그를 출력
            print(f"   [ERROR] {judge_type.upper()} Judge 로드 실패: {e}. 이 모델은 건너뜁니다.")

    if len(judges) != 3:
        raise RuntimeError(f"총 {len(judges)}개만 로드됨. ARES 평가를 위해 3개 모델이 모두 필요합니다.")

    print(f">> ARES 심사관 로드 완료. 총 {len(judges)}개 활성화.")
    return tokenizer, judges


# 모델 타입 힌트를 AutoModelForSequenceClassification으로 변경
def evaluate_triple(tokenizer: AutoTokenizer, judges: Dict[str, AutoModelForSequenceClassification],
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

            # 1. 입력 토큰화
            inputs = tokenizer(
                text_a, text_b,
                return_tensors="pt",
                truncation=True,
                padding='max_length',
                max_length=MAX_LENGTH
            ).to(DEVICE)

            # AutoModel을 사용하므로, KoBERT/DistilBERT 호환성을 위해 token_type_ids 처리 로직을 그대로 둡니다.
            # AutoModel이 BertForSequenceClassification을 로드하면 token_type_ids가 필요하고,
            # DistilBertForSequenceClassification을 로드하면 필요하지 않으므로, 충돌 방지를 위해 이 코드를 포함합니다.
            if 'token_type_ids' in inputs and model.config.model_type == 'distilbert':
                del inputs['token_type_ids']

            # 2. 예측 수행 및 결과 산출
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

            results[name] = prediction

    return results


# ===================================================================
# 2. 배치 처리 및 보고서 생성 로직 (로직 변경 없음)
# ===================================================================

def run_batch_evaluation():
    """입력 디렉토리의 파일을 처리하고 PPI 보고서를 생성합니다."""

    INPUT_DIR = "./in"
    OUTPUT_DIR = "./out"

    # 1. 디렉토리 구조 확인 및 생성
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[SETUP] 입력 디렉토리: {INPUT_DIR}, 출력 디렉토리: {OUTPUT_DIR}")

    # 2. 모델 및 토크나이저 로드 (전역 초기화)
    try:
        tokenizer, judges = load_ares_judges()
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] 모델 로드 실패: {e}")
        return

    # 3. 입력 파일 목록 검색
    input_files = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith('.jsonl')
    ]

    if not input_files:
        print(f"\n[WARN] {INPUT_DIR} 디렉토리에서 처리할 `.jsonl` 파일을 찾을 수 없습니다. 작업을 종료합니다.")
        return

    print(f"\n[INFO] 총 {len(input_files)}개의 `.jsonl` 파일을 평가합니다.")

    all_results: List[Dict[str, Any]] = []
    total_samples = 0
    start_time = time.time()

    # 4. 파일별 배치 평가 진행
    for file_path in input_files:
        print(f"\n--- 파일 평가 시작: {os.path.basename(file_path)} ---")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_samples_in_file = len(lines)
            total_samples += total_samples_in_file

            # tqdm을 사용하여 진행률 표시
            for line in tqdm(lines, desc=f"평가 중 [{os.path.basename(file_path)}]"):
                try:
                    data = json.loads(line.strip())

                    # 필수 필드 확인 (q, c, a 필드 사용)
                    query = data.get('q')
                    context = data.get('c')
                    answer = data.get('a')

                    if not all([query, context, answer]):
                        print(
                            f"[SKIP] 데이터 형식 오류 (Q, C, A 중 누락) - 파일: {os.path.basename(file_path)}, 라인: {line.strip()[:50]}...")
                        continue

                    # ARES 평가 수행
                    scores = evaluate_triple(tokenizer, judges, query, context, answer)

                    # 원본 데이터에 점수 추가 (PPI 포맷)
                    data.update(scores)
                    all_results.append(data)

                except json.JSONDecodeError:
                    print(f"[SKIP] JSON 구문 오류 발생 - 파일: {os.path.basename(file_path)}, 라인 건너뜀: {line.strip()[:50]}...")
                except Exception as e:
                    print(f"[ERROR] 처리 중 알 수 없는 오류 발생: {e} - 파일: {os.path.basename(file_path)}. 라인 건너뜀.")

    end_time = time.time()

    # 5. PPI 보고서 저장
    processed_count = len(all_results)
    if processed_count > 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"report_{timestamp}.jsonl"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for result in all_results:
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

        # 6. 결과 요약 출력
        elapsed_time = end_time - start_time

        print("\n\n=============== 배치 평가 완료 요약 ===============")
        print(f"처리된 총 샘플 수: {total_samples}개")
        print(f"평가 성공 샘플 수: {processed_count}개")
        print(f"총 소요 시간: {elapsed_time:.2f}초")
        print(f"샘플 당 평균 처리 시간: {(elapsed_time / processed_count):.4f}초 (CPU 환경)")
        print(f"**PPI 보고서 저장 경로:** {output_path}")
        print("====================================================")
    else:
        print("\n[INFO] 평가할 샘플이 없어 보고서가 생성되지 않았습니다.")


if __name__ == "__main__":
    run_batch_evaluation()