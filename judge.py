import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification
import os
import json
import time
import config

# ===================================================================
# 0. 전역 상수 설정
# ===================================================================

# CPU 환경 설정 (PC 환경이므로 GPU 사용하지 않음)
DEVICE = torch.device("cpu")
MAX_LENGTH = 128  # 학습 시 사용했던 최대 길이 유지
MODEL_NAME = config.MODEL_NAME  # 토크나이저 fallback용 원본 모델 이름


class ARESJudgeSystem:
    """
    PC 환경에서 학습된 ARES 심사관 3개를 로드하고 RAG 응답을 평가하는 클래스
    """

    def __init__(self, model_dir_base: str):
        self.model_dir_base = model_dir_base
        self.judges = {}
        self.tokenizer = None
        self._load_all_judges()

    def _find_latest_model_path(self, judge_type: str) -> str:
        """
        [수정] 타임스탬프 대신, 고정된 심사관 이름 폴더 경로를 반환합니다.
        (사용자 디렉터리 구조: answer_faithfulness, answer_relevance, context_relevance)
        """
        judge_lower = judge_type.lower()

        # 사용자 디렉터리 구조에 맞춘 고정 폴더 이름 매핑
        folder_mapping = {
            'contextrelevance': 'context_relevance',
            'answerfaithfulness': 'answer_faithfulness',
            'answerrelevance': 'answer_relevance'
        }

        target_folder = folder_mapping.get(judge_lower)

        if not target_folder:
            raise ValueError(f"정의되지 않은 심사관 타입: {judge_type}")

        model_path = os.path.join(self.model_dir_base, target_folder)

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {model_path}")

        return model_path

    def _load_all_judges(self):
        """CR, AF, AR 세 가지 심사관 모델을 로드합니다."""
        print(">> ARES 심사관 로딩 시작 (CPU 환경)...")

        judge_types = ['contextrelevance', 'answerfaithfulness', 'answerrelevance']

        # 1. 토크나이저 초기화
        try:
            # 첫 번째 모델 경로 (Context Relevance)에서 토크나이저 로드 시도
            cr_path = self._find_latest_model_path('contextrelevance')
            # trust_remote_code=True는 KoBERT 토크나이저 사용에 필수입니다.
            self.tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
            print(f"   [INFO] 토크나이저 로드 성공 (저장된 경로에서).")
        except Exception as e:
            # 토크나이저 파일이 저장 폴더에 없거나 로드 오류 발생 시, 원본 HuggingFace 모델에서 로드 시도
            print(f"   [WARN] 저장된 경로에서 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # 2. 모델 로드
        for judge_type in judge_types:
            try:
                model_path = self._find_latest_model_path(judge_type)

                # 모델 로드 (수동 저장했으므로 pytorch_model.bin과 config.json을 사용)
                model = DistilBertForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=2,
                    trust_remote_code=True
                )
                model.to(DEVICE)
                model.eval()
                self.judges[judge_type] = model
                print(f"   [SUCCESS] {judge_type.upper()} Judge 로드 완료: {model_path}")

            except Exception as e:
                print(f"   [ERROR] {judge_type.upper()} Judge 로드 실패: {e}")
                # 로드 실패 시 활용 단계에서 오류 나지 않도록 judges 딕셔너리에서 제외합니다.
                if judge_type in self.judges:
                    del self.judges[judge_type]

        print(f">> ARES 심사관 로드 완료. 총 {len(self.judges)}개 활성화.")

    def evaluate_single_triple(self, query: str, context: str, answer: str) -> dict:
        """
        [수정 반영] 하나의 Q-C-A 쌍에 대해 3가지 ARES 점수를 계산합니다.
        점수는 0 (실패) 또는 1 (성공)입니다.
        """
        if len(self.judges) < 3:
            return {"error": "모든 심사관 모델이 로드되지 않았습니다."}

        results = {}

        judge_inputs = {
            'contextrelevance': (query, context, self.judges['contextrelevance']),
            'answerfaithfulness': (context, answer, self.judges['answerfaithfulness']),
            'answerrelevance': (query, answer, self.judges['answerrelevance'])
        }

        with torch.no_grad():
            for name, (text_a, text_b, model) in judge_inputs.items():

                # 1. 입력 토큰화
                inputs = self.tokenizer(
                    text_a, text_b,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=MAX_LENGTH
                ).to(DEVICE)

                # **** CRITICAL FIX: DistilBERT는 token_type_ids를 받지 않으므로 제거합니다. ****
                if 'token_type_ids' in inputs:
                    del inputs['token_type_ids']
                # *************************************************************************

                # 2. 예측 수행
                outputs = model(**inputs)

                # 3. 이진 분류 결과 (가장 높은 로짓 값의 인덱스가 예측값)
                prediction = torch.argmax(outputs.logits, dim=1).item()

                results[name] = prediction

        return results


# --- PC 환경 실행 예시 ---

if __name__ == "__main__":
    # --- 1. 경로 설정 (PC 환경에 맞게 반드시 수정하세요) ---
    # 다운로드한 모델 폴더(answer_faithfulness 등)들이 모여있는 상위 디렉터리 경로입니다.
    MODEL_DIR_BASE = config.MODEL_DIR

    # --- 2. 심사관 시스템 초기화 ---
    try:
        ares_evaluator = ARESJudgeSystem(MODEL_DIR_BASE)
    except Exception as e:
        print(f"\n[FATAL ERROR] ARES 심사관 시스템 초기화 실패: {e}")
        exit()

    # --- 3. 평가할 RAG 출력 예시 (학습 데이터 기반 예시) ---
    test_query = "관세청장이 품목별 원산지 기준을 정할 때 누구와 협의해야 하나요?"
    test_context = "관세법 시행규칙 제74조 ⑤ 관세청장은 품목별 원산지기준을 정하는 때에는 기획재정부장관 및 해당 물품의 관계부처의 장과 협의하여야 한다."
    test_answer = "관세청장은 기획재정부장관 및 해당 물품의 관계부처의 장과 협의해야 합니다."

    test_query_bad = "알제리 여행자가 반입할 수 있는 주류는 얼마나 됩니까?"
    test_context_bad = "관세법 시행령 제245조 (반입명령) ③ 관세청장 또는 세관장은 명령서를 받을 자의 주소 또는 거소가 불분명한 때에는 공시할 수 있다."
    test_answer_bad = "세관장이 명령서를 받을 자의 주소나 거소가 불분명할 경우 공시 송달할 수 있습니다. (답변 무관)"

    print("\n\n=============== RAG 출력 평가 시작 ===============")

    # Good RAG 응답 평가
    start_time = time.time()
    result_good = ares_evaluator.evaluate_single_triple(test_query, test_context, test_answer)
    end_time = time.time()

    print("\n--- [GOOD EXAMPLE] ---")
    print(f"  질문: {test_query}")
    print(f"  답변: {test_answer}")
    print("  --- ARES JUDGE SCORES ---")
    print(f"  CR (Context Relevance): {result_good.get('contextrelevance', 'N/A')}")
    print(f"  AF (Answer Faithfulness): {result_good.get('answerfaithfulness', 'N/A')}")
    print(f"  AR (Answer Relevance): {result_good.get('answerrelevance', 'N/A')}")
    print(f"  TOTAL SCORE: {sum(result_good.values()) if 'error' not in result_good else 'N/A'}/3")
    print(f"  CPU 추론 시간: {end_time - start_time:.4f}초\n")

    # Bad RAG 응답 평가
    start_time = time.time()
    result_bad = ares_evaluator.evaluate_single_triple(test_query_bad, test_context_bad, test_answer_bad)
    end_time = time.time()

    print("\n--- [BAD EXAMPLE] ---")
    print(f"  질문: {test_query_bad}")
    print(f"  답변: {test_answer_bad}")
    print("  --- ARES JUDGE SCORES ---")
    print(f"  CR (Context Relevance): {result_bad.get('contextrelevance', 'N/A')}")
    print(f"  AF (Answer Faithfulness): {result_bad.get('answerfaithfulness', 'N/A')}")
    print(f"  AR (Answer Relevance): {result_bad.get('answerrelevance', 'N/A')}")
    print(f"  TOTAL SCORE: {sum(result_bad.values()) if 'error' not in result_bad else 'N/A'}/3")
    print(f"  CPU 추론 시간: {end_time - start_time:.4f}초")