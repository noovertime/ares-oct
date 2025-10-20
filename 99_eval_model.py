import json
from transformers import AutoTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification, \
    pipeline
import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
import os
import logging
import config

# --- 로깅 설정: 토큰화 경고 메시지 (Warning) 숨기기 ---
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# 모델 설정 (전역 상수라고 가정)
NUM_TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512

JUDGE_TASKS = [
    ('context_relevance', 'q', 'c', 'L_CR'),
    ('answer_faithfulness', 'c', 'a', 'L_AF'),
    ('answer_relevance', 'q', 'a', 'L_AR'),
]


def load_jsonl_to_list(file_path):
    """지정된 JSONL 파일 경로로부터 데이터를 로드합니다."""
    all_data = []
    if not os.path.exists(file_path):
        file_name = os.path.basename(file_path)
        print(f"[ERROR] 파일이 존재하지 않습니다: {file_name} at {file_path}")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line))
        file_name = os.path.basename(file_path)
        print(f"[INFO] {file_name} 파일에서 총 {len(all_data)}개의 샘플 로드 완료.")
        return all_data
    except Exception as e:
        print(f"[ERROR] JSONL 파일 로드 중 오류 발생: {e}")
        return []


def prepare_dataset(data_list, text_a_key, text_b_key, label_key):
    """
    로드된 리스트 데이터를 Hugging Face Dataset 형식으로 변환합니다.
    (pandas 사용 없이, Dataset.from_dict 사용)
    """

    dataset_dict = {
        'text_a': [],
        'text_b': [],
        'labels': []
    }

    for item in data_list:
        dataset_dict['text_a'].append(item.get(text_a_key))
        dataset_dict['text_b'].append(item.get(text_b_key))
        dataset_dict['labels'].append(item.get(label_key))

    return Dataset.from_dict(dataset_dict)


# 성능 지표 함수 (F1 스코어 및 정확도) - 유지
def compute_metrics(y_true, y_pred):
    """실제 레이블과 예측 레이블을 기반으로 F1 및 Accuracy를 계산합니다."""
    accuracy = np.mean(y_true == y_pred)
    # 이진 분류를 위한 'binary' average 사용
    f1 = f1_score(y_true, y_pred, average='binary')

    return {
        'accuracy': accuracy,
        'f1': f1,
    }


def evaluate_judges():
    """
    config 모듈의 경로를 사용하여, 모델 디렉토리 내의 세 가지 Judge 모델을 찾아
    pipeline 모드로 성능 평가를 수행합니다.
    (CPU Only 환경에서 accelerate 오류를 우회하기 위한 방식)
    """
    print("\n\n=============================================")
    print("✨ 심사관 모델 성능 평가 시작 (Pipeline 모드) ✨")
    print("=============================================")

    all_results = []

    # 모델 이름으로 클래스 결정 (Pipeline은 내부적으로 AutoModel을 사용하지만, 정보 제공 목적)
    model_name_lower = config.MODEL_NAME.lower()
    if "distil" in model_name_lower:
        MODEL_CLS = DistilBertForSequenceClassification
        print(f"[INFO] 모델 클래스: DistilBertForSequenceClassification")
    elif "bert" in model_name_lower or "klue" in model_name_lower:
        MODEL_CLS = AutoModelForSequenceClassification
        print(f"[INFO] 모델 클래스: AutoModelForSequenceClassification")
    else:
        MODEL_CLS = AutoModelForSequenceClassification
        print(f"[INFO] 모델 클래스: AutoModelForSequenceClassification (기본값 설정)")

    # 1. 평가용 파일 경로 설정 및 확인 (모든 Judge가 동일한 파일을 사용한다고 가정)
    eval_json_path = os.path.join(config.DATA_EVAL_DIR, config.DATA_EVAL_FILE_NAME)
    if not os.path.exists(eval_json_path):
        print(f"[ERROR] 평가에 사용될 파일 {eval_json_path}가 없어서 진행할 수 없습니다.")
        return

    # 2. 평가 데이터셋 로드 (전체 원본)
    raw_eval_data_all = load_jsonl_to_list(eval_json_path)

    for judge_type, text_a_key, text_b_key, label_key in JUDGE_TASKS:
        # 3. 모델 경로 생성 및 확인 (규칙: config.MODEL_DIR/judge_type)
        model_path = os.path.join(config.MODEL_DIR, judge_type)
        if not os.path.exists(model_path):
            print(f"[SKIP] 모델 폴더를 찾을 수 없어 건너뜱니다: {model_path}")
            continue

        print(f"\n--- {judge_type} Judge 평가 ---")
        print(f"[INFO] 모델 경로: {model_path}")

        try:
            # 4. Pipeline 로드 (CPU 명시: accelerate 오류 회피)
            qa_pipeline = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=-1,  # CPU Only 환경 명시
            )

            # 5. 평가 데이터 준비 및 필터링
            eval_dataset_full = prepare_dataset(raw_eval_data_all, text_a_key, text_b_key, label_key)

            # 필터링 로직 (유효하지 않은 문자열 제거)
            def is_valid_text(example):
                text_a_val = example.get('text_a')
                text_b_val = example.get('text_b')
                return isinstance(text_a_val, str) and isinstance(text_b_val,
                                                                  str) and text_a_val.strip() != "" and text_b_val.strip() != ""

            # 필터링된 데이터셋 (유효한 데이터만 남음)
            eval_dataset = eval_dataset_full.filter(is_valid_text)

            initial_eval_size = len(eval_dataset_full)
            print(
                f"[INFO] 평가 데이터 필터링: {initial_eval_size}개 -> {len(eval_dataset)}개 (제거됨: {initial_eval_size - len(eval_dataset)}개)")

            # 6. 예측 입력 생성 및 예측 실행
            test_texts = [
                (row['text_a'], row['text_b'])
                for row in eval_dataset
            ]

            # 예측 수행 (Pipeline)
            print(f"[INFO] {judge_type} 예측 시작...")
            predictions = qa_pipeline(
                test_texts,
                truncation=True,
                padding='max_length',
                max_length=MAX_SEQ_LENGTH,
                batch_size=EVAL_BATCH_SIZE  # 배치 사이즈 사용
            )

            # 7. 예측 결과와 레이블 개수 보정 (핵심 수정)

            # 예측 결과 (y_pred) 생성
            y_pred_labels = [int(p['label'].split('_')[-1]) for p in predictions]
            y_pred = np.array(y_pred_labels)

            # 정답 레이블 (y_true) 추출
            y_true_full = np.array(eval_dataset['labels'])

            # 🌟 [레이블 재구성 로직: 개수 불일치 문제 해결] 🌟
            len_true = len(y_true_full)
            len_pred = len(y_pred)

            min_len = min(len_true, len_pred)

            # 두 배열의 길이를 더 작은 쪽에 맞춰 자릅니다.
            y_true_adjusted = y_true_full[:min_len]
            y_pred_adjusted = y_pred[:min_len]

            if len_true != len_pred:
                print(f"[WARN] 예측/레이블 개수 불일치 발생 ({len_true} vs {len_pred}). {min_len}개에 맞춰 평가합니다.")

            # 8. 지표 계산
            metrics = compute_metrics(y_true_adjusted, y_pred_adjusted)

            result = {
                'Judge': judge_type,
                'F1-Score': metrics.get('f1'),
                'Accuracy': metrics.get('accuracy'),
                'Path': model_path
            }
            all_results.append(result)

            print(f"[RESULT] {judge_type} F1 Score: {result['F1-Score']:.4f}, Accuracy: {result['Accuracy']:.4f}")

        except Exception as e:
            print(f"[ERROR] {judge_type} 모델 평가 중 오류 발생: {e}")
            all_results.append({'Judge': judge_type, 'Error': str(e), 'Path': model_path})

    # 9. 최종 결과 요약 출력
    print("\n=============================================")
    print("✅ 디렉토리 평가 최종 요약")
    print("=============================================")
    for res in all_results:
        if 'F1-Score' in res:
            print(f"[{res['Judge']}]: F1 Score = {res['F1-Score']:.4f}, Accuracy = {res['Accuracy']:.4f}")
        else:
            print(f"[{res['Judge']}]: 평가 실패 - {res['Error']}")
    print("=============================================")

    return all_results


if __name__ == "__main__":
    final_metrics = evaluate_judges()

    if final_metrics:
        print("\n[SUCCESS] 단일 모델 평가 성공적으로 완료.")