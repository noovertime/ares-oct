import json
import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score
import os
import logging
# ⚠️ NOTE: config 모듈이 외부에 있다고 가정합니다.
import config
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

# --- 로깅 설정: 토큰화 경고 메시지 (Warning) 숨기기 ---
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# ===================================================================
# 0. 전역 상수 설정 (config 모듈에서 가져온다고 가정)
# ===================================================================
# ⚠️ NOTE: config.MODEL_NAME 등은 외부 config 파일에 정의되어 있어야 합니다.
# 예시:
# MODEL_NAME = "monologg/distilkobert"
# DATA_EVAL_DIR = "/data/eval"
# DATA_EVAL_FILE_NAME = "eval_data.jsonl"
# MODEL_DIR = "/models"

NUM_TRAIN_EPOCHS = 5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512

JUDGE_TASKS = [
    ('context_relevance', 'q', 'c', 'L_CR'),
    ('answer_faithfulness', 'c', 'a', 'L_AF'),
    ('answer_relevance', 'q', 'a', 'L_AR'),
]


# ===================================================================
# 1. 보조 함수
# ===================================================================

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
        # NOTE: .get()을 사용하여 키가 없을 때 오류 방지
        dataset_dict['text_a'].append(item.get(text_a_key))
        dataset_dict['text_b'].append(item.get(text_b_key))
        dataset_dict['labels'].append(item.get(label_key))

    return Dataset.from_dict(dataset_dict)


def compute_metrics(y_true, y_pred):
    """실제 레이블과 예측 레이블을 기반으로 F1 및 Accuracy를 계산합니다."""
    accuracy = np.mean(y_true == y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')

    return {
        'accuracy': accuracy,
        'f1': f1,
    }


# ===================================================================
# 2. 평가 함수 (PyTorch 수동 추론)
# ===================================================================

def evaluate_judges():
    """
    CPU Only 환경에서 Trainer/accelerate 없이 PyTorch DataLoader를 사용하여
    모든 Judge 모델의 성능을 평가합니다.
    """
    print("\n\n=============================================")
    print("✨ 심사관 모델 성능 평가 시작 (PyTorch 수동 모드) ✨")
    print("=============================================")

    all_results = []

    # 모델 클래스 결정 (MODEL_NAME에 따라 동적 결정)
    model_name_lower = config.MODEL_NAME.lower()
    if "distil" in model_name_lower:
        MODEL_CLS = DistilBertForSequenceClassification
    elif "bert" in model_name_lower or "klue" in model_name_lower:
        MODEL_CLS = AutoModelForSequenceClassification
    else:
        MODEL_CLS = AutoModelForSequenceClassification
    print(f"[INFO] 모델 클래스: {MODEL_CLS.__name__}")

    # 1. 평가용 파일 경로 설정 및 확인
    eval_json_path = os.path.join(config.DATA_EVAL_DIR, config.DATA_EVAL_FILE_NAME)
    if not os.path.exists(eval_json_path):
        print(f"[ERROR] 평가에 사용될 파일 {eval_json_path}가 없어서 진행할 수 없습니다.")
        return []

    raw_eval_data_all = load_jsonl_to_list(eval_json_path)

    # 2. PyTorch device 설정 (CPU Only)
    device = torch.device("cpu")
    print(f"[INFO] 평가 장치: {device}")

    for judge_type, text_a_key, text_b_key, label_key in JUDGE_TASKS:
        judge_lower = judge_type.lower()

        # 3. 모델 경로 생성 및 확인 (규칙: config.MODEL_DIR/judge_type)
        model_path = os.path.join(config.MODEL_DIR, judge_type)
        if not os.path.exists(model_path):
            print(f"[SKIP] 모델 폴더를 찾을 수 없어 건너뜱니다: {model_path}")
            continue

        print(f"\n--- {judge_type} Judge 평가 ---")
        print(f"[INFO] 모델 경로: {model_path}")

        try:
            # 4. 모델 및 토크나이저 로드
            model = MODEL_CLS.from_pretrained(model_path, num_labels=2, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model.to(device)  # 모델을 CPU로 이동
            model.eval()  # 평가 모드로 설정 (필수)

            # 5. 평가 데이터 준비 및 필터링
            eval_dataset_full = prepare_dataset(raw_eval_data_all, text_a_key, text_b_key, label_key)

            # 필터링 로직 (유효하지 않은 문자열 제거)
            def is_valid_text(example):
                text_a_val = example.get('text_a')
                text_b_val = example.get('text_b')
                return isinstance(text_a_val, str) and isinstance(text_b_val,
                                                                  str) and text_a_val.strip() != "" and text_b_val.strip() != ""

            eval_dataset = eval_dataset_full.filter(is_valid_text)

            initial_eval_size = len(eval_dataset_full)
            valid_size = len(eval_dataset)
            print(f"[INFO] 평가 데이터 필터링: {initial_eval_size}개 -> {valid_size}개 (제거됨: {initial_eval_size - valid_size}개)")

            # 6. 토큰화 및 DataLoader 준비

            # 토큰화 함수 (PyTorch Tensor 반환)
            def tokenize_data(examples):
                tokenized = tokenizer(
                    examples['text_a'], examples.get('text_b', None),
                    truncation=True, max_length=MAX_SEQ_LENGTH, padding='max_length',
                    return_tensors='pt'
                )
                # 토큰화 결과의 [1, MAX_SEQ_LENGTH] 형태를 [MAX_SEQ_LENGTH]로 squeeze 합니다.
                return {k: v.squeeze(0) for k, v in tokenized.items()}

            # 데이터셋을 토큰화 (map의 batched=False는 데이터 무결성을 유지하는 데 유리)
            eval_tokenized = eval_dataset.map(tokenize_data, batched=False, remove_columns=['text_a', 'text_b'])

            # 🌟 [수정 시작: NumPy를 통한 안전한 텐서 추출] 🌟

            # 1. datasets.Dataset을 NumPy 배열로 변환하여 Column 객체를 해제
            input_ids_np = np.array(eval_tokenized['input_ids'])
            attention_mask_np = np.array(eval_tokenized['attention_mask'])
            labels_np = np.array(eval_tokenized['labels'])

            # 2. NumPy 배열을 PyTorch 텐서로 변환
            input_ids = torch.tensor(input_ids_np).to(torch.long)
            attention_mask = torch.tensor(attention_mask_np).to(torch.long)
            labels = torch.tensor(labels_np).to(torch.long)

            # 3. token_type_ids 처리
            if 'token_type_ids' in eval_tokenized.column_names:
                token_type_ids_np = np.array(eval_tokenized['token_type_ids'])
                token_type_ids = torch.tensor(token_type_ids_np).to(torch.long)

                # PyTorch Dataset 객체 생성 (4개의 텐서)
                eval_data = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
            else:
                # PyTorch Dataset 객체 생성 (3개의 텐서)
                eval_data = TensorDataset(input_ids, attention_mask, labels)

            # 🌟 [수정 끝] 🌟

            eval_dataloader = DataLoader(
                eval_data,
                sampler=SequentialSampler(eval_data),
                batch_size=EVAL_BATCH_SIZE
            )

            # 7. 추론 실행 (수동 배치 처리)
            y_pred_list = []
            y_true_list = []

            print(f"[INFO] {judge_type} 예측 시작 ({len(eval_dataloader)} 배치)")

            for batch in eval_dataloader:
                # 데이터를 CPU로 이동 (GPU가 없으므로)
                batch = tuple(t.to(device) for t in batch)

                # token_type_ids가 있을 경우 inputs에 추가
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                }

                # TensorDataset에 token_type_ids가 포함되면 batch[2]에, 아니면 labels는 batch[2]
                if len(batch) == 4:
                    inputs['token_type_ids'] = batch[2]
                    labels = batch[3].cpu().numpy()
                else:
                    labels = batch[2].cpu().numpy()

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits.cpu().numpy()
                predictions = np.argmax(logits, axis=1)

                y_pred_list.extend(predictions)
                y_true_list.extend(labels)

            # 8. 지표 계산 (개수 불일치 오류 발생 가능성 없음)
            y_true_final = np.array(y_true_list)
            y_pred_final = np.array(y_pred_list)

            # 최종 유효 데이터 개수 확인
            if len(y_true_final) != valid_size:
                print(f"[WARN] 예측/레이블 개수 불일치 발생! (예측: {len(y_pred_final)}, 기대: {valid_size})")

            metrics = compute_metrics(y_true_final, y_pred_final)

            result = {
                'Judge': judge_type,
                'F1-Score': metrics.get('f1'),
                'Accuracy': metrics.get('accuracy'),
                'Path': model_path
            }
            all_results.append(result)

            print(f"[RESULT] {judge_type} F1 Score: {result['F1-Score']:.4f}, Accuracy = {result['Accuracy']:.4f}")

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
    # ⚠️ NOTE: 이 섹션은 config 모듈이 정의되어 있고,
    # 필요한 데이터와 모델이 경로에 있다고 가정하고 실행됩니다.

    final_metrics = evaluate_judges()
    if not final_metrics:
        print(f"{final_metrics}")

    print("최종 평가 함수가 정의되었습니다. config 모듈과 함께 실행해주세요.")
