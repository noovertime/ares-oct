# 02_generate_ppi.py (클래스 기반 및 다중 골든셋 확장 버전)
# 1. 표준 라이브러리
import abc
import json
import logging
import math
import os
import time
from typing import Dict, Any, List, Tuple

# 2. 서드파티 라이브러리
import numpy as np
from numpy.typing import NDArray
import torch
from scipy.stats import norm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ONNX Runtime 및 Transformers 라이브러리
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer

# Transformers 경고 메시지 비활성화
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

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
from json_util import _load_json_lines  # 외부 모듈 가정

# ===================================================================
# 0. 전역 상수 및 환경 설정
# ===================================================================

# Transformers 경고 메시지 비활성화
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 경로 설정
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


# 평가 배치 크기 설정 (클수록 속도 향상, 메모리 사용 증가)
BATCH_SIZE = 32
#
ENGINE_TYPE_ONNX_RUNTIME = "ONNX Runtime"

# ===================================================================
# 1. 유틸리티 함수 (클래스에 포함시키지 않는 범용 기능)
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

    print(f"모델 : {judge_type} -> {model_path}")

    return model_path


def cleanup_evaluation_data():
    """
    config에 정의된 PPI 출력 디렉토리와 최종 보고서 디렉토리 내의 모든 파일을 삭제합니다.
    """
    dirs_to_clean = [ config.DATA_REPORT_DIR]
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
# 2. ARES 평가 클래스 (추상화 버전)
# ===================================================================
class AresJudge(abc.ABC):
    """모델 로딩 및 평가를 위한 추상 슈퍼 클래스."""

    def __init__(self, device: torch.device, max_length: int, model_type_name: str, engine_type:str) -> None:
        self.tokenizer: AutoTokenizer | None = None
        self.judges: Dict[str, Any] = {}  # PyTorch 모델 또는 ONNX InferenceSession을 담음
        self.device = device
        self.max_length = max_length
        self.model_type_name = model_type_name
        self.engine_type: str = engine_type
        self._load_models()

    @abc.abstractmethod
    def _load_models(self) -> None:
        """각 서브 클래스에서 엔진에 맞게 모델을 로드하는 필수 메서드."""
        pass

    @abc.abstractmethod
    def _get_logits(self, judge_type: str, inputs: Dict[str, Any],
                    tokenized_data: Dict[str, Any]) -> np.ndarray | torch.Tensor:
        """엔진 타입에 따라 추론을 실행하고 Logits를 반환하는 필수 메서드."""
        pass

    def evaluate_triple(self, query: str, context: str, answer: str) -> Dict[str, Dict[str, float | int]]:
        """
        하나의 Q-C-A 쌍에 대해 CR, AF, AR 점수(0/1)와 Softmax 확률을 계산하는 공통 로직.
        """
        results: Dict[str, Dict[str, float | int]] = {}

        # 각각이 어떤 것을 평가할지 맵핑 관계
        judge_inputs_map = {
            KEY_CR: (query, context),
            KEY_AF: (context, answer),
            KEY_AR: (query, answer)
        }

        for judge_type in JUDGE_TYPES:
            text_a, text_b = judge_inputs_map[judge_type]

            # 1. 토큰화 (NumPy 텐서 또는 PyTorch 텐서)
            return_tensors = "np" if self.engine_type == ENGINE_TYPE_ONNX_RUNTIME else "pt"

            tokenized_data = self.tokenizer(
                text_a, text_b,
                return_tensors=return_tensors,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

            # 2. Logits 계산 (서브 클래스 호출)
            logits_tensor = self._get_logits(judge_type, tokenized_data, tokenized_data)

            # 3. 결과 산출 (공통 후처리)
            probabilities = torch.softmax(logits_tensor, dim=1).squeeze().cpu().numpy()
            prediction = int(torch.argmax(logits_tensor, dim=1).item())

            prob_neg = round(float(probabilities[0]), 4)
            prob_pos = round(float(probabilities[1]), 4)

            results[JUDGE_PREDICTION_FIELDS[judge_type]] = {
                'machine_pred': prediction,
                'prob_neg': prob_neg,
                'prob_pos': prob_pos
            }

        return results


class AresPytorchJudge(AresJudge):
    """BERT 및 DistilBERT와 같은 PyTorch 기반 모델 로딩 및 평가 담당."""

    def __init__(self, device, max_length, model_type_name) -> None:
        super().__init__(device, max_length, model_type_name, "PyTorch")

    def _get_model_path_base(self, judge_type: str) -> str:
        """모델 폴더 경로를 반환합니다."""
        target_folder = FOLDER_MAPPING.get(judge_type)
        if not target_folder: raise ValueError(f"정의되지 않은 심사관 타입: {judge_type}")
        model_path = os.path.join(MODEL_DIR_BASE, target_folder)
        if not os.path.isdir(model_path): raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {model_path}")
        return model_path

    def _load_models(self) -> None:
        """PyTorch 모델 로딩 로직."""
        print(f"\n>> ARES 심사관 로딩 시작 ({self.model_type_name} / {self.engine_type} 엔진)...")

        # 1. 토크나이저 로드
        cr_path = self._get_model_path_base(KEY_CR)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
            print(f"[INFO] 토크나이저 로드 성공: {cr_path}")
        except Exception:
            print(f"[WARN] 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            print(f"[INFO] 기본 모델에서 토크나이저 로드 완료.")

        # DistilBERT 호환: token_type_ids 제거
        if TOKEN_TYPE_ID_OFF and self.tokenizer:
            self.tokenizer.model_input_names = [
                n for n in self.tokenizer.model_input_names if n != "token_type_ids"
            ]
            print("[INFO] 'token_type_ids' 비활성화 완료")

        # 2. PyTorch 모델 로드
        for judge_type in JUDGE_TYPES:
            try:
                model_folder_path = self._get_model_path_base(judge_type)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_folder_path, num_labels=2, trust_remote_code=True
                ).to(self.device)
                model.eval()
                self.judges[judge_type] = model
                print(f"[SUCCESS] {judge_type} Judge 로드 완료 ({self.engine_type})")
            except Exception as e:
                print(f"[ERROR] {judge_type} Judge 로드 실패: {e}")

        if len(self.judges) != len(JUDGE_TYPES):
            raise RuntimeError(f"필요한 PyTorch 모델 {len(JUDGE_TYPES)}개 중 {len(self.judges)}개만 로드됨.")

        print(f">> ARES 심사관 로드 완료. 총 {len(self.judges)}개 심사관 활성화.")

    def _get_logits(self, judge_type: str, inputs: Dict[str, Any], tokenized_data: Dict[str, Any]) -> torch.Tensor:
        """PyTorch 추론 로직: 모델을 실행하고 Logits를 반환."""
        model = self.judges[judge_type]

        with torch.no_grad():
            # PyTorch 텐서를 GPU/CPU 장치로 이동
            pt_inputs = {k: v.to(self.device) for k, v in tokenized_data.items()}
            outputs = model.forward(**pt_inputs)
            return outputs.logits

    # evaluate_triple 메서드는 AresJudge 슈퍼 클래스에서 상속 및 재활용됨


class AresSbertOnnxJudge(AresJudge):
    """SBERT ONNX Runtime 기반 모델 로딩 및 평가 담당."""
    def __init__(self, device: torch.device,  max_length: int, model_type_name) -> None:
        super().__init__(device, max_length, model_type_name, ENGINE_TYPE_ONNX_RUNTIME)

    def _get_model_path_base(self, judge_type: str) -> str:
        """모델 폴더 경로를 반환합니다."""
        target_folder = FOLDER_MAPPING.get(judge_type)
        if not target_folder: raise ValueError(f"정의되지 않은 심사관 타입: {judge_type}")
        model_path = os.path.join(MODEL_DIR_BASE, target_folder)
        if not os.path.isdir(model_path): raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {model_path}")
        return model_path

    def _load_models(self) -> None:
        """ONNX InferenceSession 로딩 로직."""
        print(f"\n>> ARES 심사관 로딩 시작 ({self.model_type_name} / {self.engine_type} 엔진)...")

        # 1. 토크나이저 로드 (기존 로직 유지)
        cr_path = self._get_model_path_base(KEY_CR)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
            print(f"[INFO] 토크나이저 로드 성공: {cr_path}")
        except Exception as e:
            print(f"[WARN] 토크나이저 로드 실패 : {e}. 원본 모델 ({MODEL_NAME}) 로드 시도.")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            print(f"[INFO] 기본 모델에서 토크나이저 로드 완료.")

        # 2. ONNX InferenceSession 로드
        options = SessionOptions()
        providers = ['CPUExecutionProvider']

        for judge_type in JUDGE_TYPES:
            try:
                # model_folder_path는 PyTorch와 동일한 디렉토리 구조를 사용합니다.
                model_folder_path = self._get_model_path_base(judge_type)
                # onnx 파일은 모델 폴더에 같이 들어있음
                quant_model_path = os.path.join(model_folder_path, "model_quant.onnx")
                if not os.path.exists(quant_model_path):
                    raise FileNotFoundError(f"ONNX 파일 없음. 존재하지 않는 경로 : {quant_model_path}")

                session = InferenceSession(quant_model_path, sess_options=options, providers=providers)
                self.judges[judge_type] = session
                print(f"[SUCCESS] {judge_type} Judge 로드 완료 (ONNX)")
            except Exception as e:
                print(f"[ERROR] {judge_type} 로드 실패: {e}")

        if len(self.judges) != len(JUDGE_TYPES):
            raise RuntimeError(f"필요한 ONNX 모델 {len(JUDGE_TYPES)}개 중 {len(self.judges)}개만 로드됨.")

        print(f">> ARES 심사관 로드 완료. 총 {len(self.judges)}개 심사관 활성화.")

    def _get_logits(self, judge_type: str, inputs: Dict[str, Any], tokenized_data: Dict[str, Any]) -> torch.Tensor:
        """ONNX 추론 로직: Logits를 NumPy로 얻은 후 PyTorch Tensor로 변환."""
        session: InferenceSession = self.judges[judge_type]

        # ONNX 추론 (NumPy 텐서 사용)
        input_names = [input.name for input in session.get_inputs()]

        # tokenized_data (NumPy 배열)에서 ONNX 세션이 필요로 하는 입력만 추출
        session_input = {
            name: tokenized_data[name] for name in input_names if name in tokenized_data
        }

        outputs = session.run(None, session_input)
        logits_array = outputs[0]

        # Logits를 PyTorch 텐서로 변환하여 슈퍼 클래스의 후처리에 사용
        return torch.from_numpy(logits_array)

    # evaluate_triple 메서드는 AresJudge 슈퍼 클래스에서 상속 및 재활용됨


# ===================================================================
# 3. PPI 통계 및 보정 계산 클래스
# ===================================================================
# 02_generate_ppi.py (PPICalculator 클래스 내부)

class PPICalculator:
    """PPI 보정 및 통계 계산을 담당하는 클래스."""

    def __init__(self, ci_z_score: float = CI_Z_SCORE):
        self.ci_z_score = ci_z_score

    def calculate_accuracy(self, stats: Dict[str, Any]) -> float | str:
        """rectifier_terms를 사용하여 심사관의 정확도를 계산하고 float 또는 'N/A'를 반환합니다."""
        terms = stats.get('rectifier_terms')
        labeled_n = stats.get('labeled_n')

        if not terms or labeled_n == 0:
            return 'N/A'

        # 오류 횟수 = (rectifier_terms 리스트에서 0이 아닌 값의 개수)
        error_count = sum(1 for term in terms if term != 0.0)
        accuracy = 1.0 - (error_count / labeled_n)
        return accuracy

    def generate_golden_set_stat(self, golden_set_stats: Dict[str, Dict[str, Any]]) -> Dict[
        str, Dict[str, float | str]]:
        """
        골든셋 통계 결과를 순수한 데이터 딕셔너리 형태로 반환합니다.
        (report_util에서 MD 포맷팅 담당)
        """
        if not golden_set_stats:
            print("[FAIL] 골든셋 평가 결과가 비어있거나 실패했습니다.")
            return {}

        report_data = {}

        for axis in JUDGE_TYPES:
            stats = golden_set_stats.get(axis, {})

            # LM 예측 평균 값
            mean_pred = stats.get('machine_mean', 'N/A')

            # 심사관 정확도 값
            accuracy = self.calculate_accuracy(stats)

            report_data[axis] = {
                'mean_pred': mean_pred,  # LM 예측 평균 (float | 'N/A')
                'accuracy': accuracy  # 심사관 정확도 (float | 'N/A')
            }

        return report_data

    def evaluate_golden_set(self, judge: AresJudge, golden_set_filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        특정 골든셋 파일을 심사관이 평가하고, PPI 편향 계산에 필요한 통계 정보를 반환합니다.

        반환 딕셔너리에 'prob_pos_list' (골든셋의 P_pos 리스트)를 추가합니다.
        """
        golden_stats = {key: {
            'labeled_n': 0,
            'rectifier_terms': [],
            'machine_mean_sum': 0.0,
            'prob_pos_list': []  # 🚨 추가: 골든셋의 P_pos 리스트
        } for key in JUDGE_TYPES}

        golden_records = _load_json_lines(golden_set_filepath)

        if not golden_records:
            print(f"[WARN] 골든셋 파일 {golden_set_filepath}에 평가할 데이터가 없습니다.")
            return {}

        total_samples = len(golden_records)
        print(f"\n>> 골든셋 평가 시작. {golden_set_filepath}, 총 {total_samples}개 예측 시작 ")

        progress_points = [0, 10, 30, 50, 70, 90, 100]
        next_progress_index = 0

        print(f"[PROGRESS] 0%. ", end=' ')

        for i, data in enumerate(golden_records):
            current_progress = (i * 100) // total_samples

            # 다음 목표 %에 도달했는지 확인
            if next_progress_index < len(progress_points) - 1 and current_progress >= progress_points[
                next_progress_index + 1]:
                next_progress_index += 1
                print(f" {progress_points[next_progress_index]}%. ", end=' ')

            try:
                # Q, C, A 추출 및 정규화
                query = ' '.join(data.get('q', '').split()).strip()
                context = ' '.join(data.get('c', '').split()).strip()
                answer = ' '.join(data.get('a', '').split()).strip()

                if not all([query, context, answer]):
                    continue

                scores_with_probs = judge.evaluate_triple(query, context, answer)

                for axis in JUDGE_TYPES:
                    pred_key = JUDGE_PREDICTION_FIELDS[axis]
                    axis_scores = scores_with_probs.get(pred_key)

                    if axis_scores is None: continue

                    machine_pred = axis_scores.get('machine_pred')
                    prob_pos = axis_scores.get('prob_pos')

                    gold_key = GOLD_LABEL_FIELDS[axis]
                    gold_label = data.get(gold_key)

                    if machine_pred is None or prob_pos is None or gold_label is None:
                        continue

                    # 통계 업데이트
                    machine_pred = int(machine_pred)
                    gold_label = int(gold_label)

                    rectifier_term = float(machine_pred - gold_label)

                    stats = golden_stats[axis]
                    stats['labeled_n'] += 1
                    stats['rectifier_terms'].append(rectifier_term)
                    stats['machine_mean_sum'] += machine_pred
                    stats['prob_pos_list'].append(prob_pos)

            except Exception as e:
                print(f"[예외] {e}")
                continue

        # 100% 완료 로그 출력 (루프 내에서 이미 출력되었을 수 있으나, 안전장치)
        if total_samples > 0:
            print(f"100% 완료.")

        # 3. 최종 통계 계산
        final_golden_stats = {}
        for axis, stats in golden_stats.items():
            if stats['labeled_n'] > 0:
                final_golden_stats[axis] = {
                    'labeled_n': stats['labeled_n'],
                    'rectifier_terms': stats['rectifier_terms'],
                    'machine_mean': stats['machine_mean_sum'] / stats['labeled_n'],
                    'prob_pos_list': stats['prob_pos_list']
                }

        return final_golden_stats


    def calculate_ppi_asymptotic_ci(
            self,
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
        half_width: float = self.ci_z_score * math.sqrt(variance)
        return round(half_width, 3)

    def _calculate_binned_probs(self, probs_pos: List[float], total_n: int) -> List[Dict[str, float | int | str]]:
        """Softmax P_pos 값을 10개 구간으로 나누어 빈도와 비율을 계산합니다."""
        probs_array = np.array(probs_pos)
        # 0.0 to 1.0, 10 bins. range=(0.0, 1.000001)을 사용하여 1.0을 포함합니다.
        counts, _ = np.histogram(probs_array, bins=10, range=(0.0, 1.000001))

        binned_data = []

        for i in range(10):
            low = i * 0.1
            high = (i + 1) * 0.1
            count = int(counts[i])
            percentage = round((count / total_n) * 100, 1) if total_n > 0 else 0.0

            binned_data.append({
                'range': f"{low:.1f} - {high:.1f}",
                'count': count,
                'percentage': percentage
            })

        # 시각화를 위해 최대 비율을 계산 (report_util에서 막대 길이 정규화용)
        max_perc = max(item['percentage'] for item in binned_data)

        # 모든 빈에 max_perc 값을 추가
        for item in binned_data:
            item['max_perc'] = max_perc

        return binned_data

    def _calculate_confidence_stats(self, scores: List[int], probs_pos: List[float]) -> Dict[str, float]:
        """Softmax 긍정 확률을 기반으로 확신도 통계를 계산합니다. (중앙값 포함)"""

        scores_array = np.array(scores)
        probs_pos_array = np.array(probs_pos)
        probs_neg_array = 1.0 - probs_pos_array

        # 1. 기본 분포 통계 계산 (P_pos 및 P_neg 기준)
        stats = {
            'prob_pos_min': round(float(np.min(probs_pos_array)), 4),
            'prob_pos_avg': round(float(np.mean(probs_pos_array)), 4),
            'prob_pos_median': round(float(np.median(probs_pos_array)), 4),
            'prob_pos_max': round(float(np.max(probs_pos_array)), 4),

            'prob_neg_min': round(float(np.min(probs_neg_array)), 4),
            'prob_neg_avg': round(float(np.mean(probs_neg_array)), 4),
            'prob_neg_median': round(float(np.median(probs_neg_array)), 4),
            'prob_neg_max': round(float(np.max(probs_neg_array)), 4),
        }

        # 2. 조건부 확신도 및 마진 계산 (파생 지표)

        margin_array = np.abs(probs_pos_array - probs_neg_array)
        stats['mean_margin'] = round(float(np.mean(margin_array)), 4)

        pos_mask = scores_array == 1
        num_pos_preds = np.sum(pos_mask)
        neg_mask = scores_array == 0
        num_neg_preds = np.sum(neg_mask)

        stats['conf_pos_avg'] = round(float(np.mean(probs_pos_array[pos_mask])), 4) if num_pos_preds > 0 else 0.0
        stats['conf_neg_avg'] = round(float(np.mean(probs_neg_array[neg_mask])), 4) if num_neg_preds > 0 else 0.0

        return stats


    def calculate_ppi_summary(
            self,
            file_base_name: str,
            current_lm_scores: Dict[str, List[int]],
            current_lm_probs: Dict[str, List[float]],
            total_n: int,
            golden_set_stats: Dict[str, Dict[str, Any]],
            golden_set_name: str
    ) -> Dict[str, Any]:
        """
        LM 예측 결과와 골든셋 통계를 결합하여 PPI 요약 결과를 계산합니다.
        (확신도 통계, 평가셋 Binning 데이터, 골든셋 Binning 데이터를 포함)
        """

        summary: Dict[str, Any] = {
            "model_name": file_base_name,
            "golden_set_name": golden_set_name,
            "n": total_n,
            "ppi_active": True,
            "labeled_n_rep": golden_set_stats.get(KEY_CR, {}).get('labeled_n', 0),
        }

        overall_corrected_scores: List[float] = []

        for axis in JUDGE_TYPES:
            scores: List[int] = current_lm_scores[axis]
            probs_pos: List[float] = current_lm_probs[axis]

            golden_axis_stats = golden_set_stats.get(axis, {})
            labeled_n_axis: int = golden_axis_stats.get('labeled_n', 0)

            # 필수 통계 누락 검사 (이전 단계 수정 로직 유지)
            if labeled_n_axis == 0 or not golden_axis_stats:
                raise RuntimeError(
                    f"[{axis} 축 통계 누락] 골든셋 '{golden_set_name}'에 '{axis}' 축의 유효한 라벨링 데이터가 부족합니다. "
                    f"Labeled N: {labeled_n_axis}."
                )

            rectifier_terms: List[float] = golden_axis_stats['rectifier_terms']

            # 1. PPI 통계 계산
            machine_mean: float = sum(scores) / float(total_n)
            rectifier: float = sum(rectifier_terms) / labeled_n_axis
            corrected_mean: float = max(0.0, min(1.0, machine_mean - rectifier))
            margin: float = self.calculate_ppi_asymptotic_ci(scores, rectifier_terms, total_n, labeled_n_axis)

            # 2. 확신도 통계 계산 (기술 통계량)
            confidence_stats = self._calculate_confidence_stats(scores, probs_pos)

            # 3. Binning 데이터 계산
            # 평가 대상셋 Binning
            prob_bins = self._calculate_binned_probs(probs_pos, total_n)

            # 🚨 추가: 골든셋 Binning 데이터 계산
            golden_probs_pos: List[float] = golden_axis_stats['prob_pos_list']
            # labeled_n_axis는 골든셋의 총 샘플 수(N) 역할을 합니다.
            golden_prob_bins = self._calculate_binned_probs(golden_probs_pos, labeled_n_axis)

            # 4. 최종 요약에 추가
            summary[axis] = {
                # PPI 통계
                "machine_mean": round(machine_mean, 2),
                "corrected_mean": round(corrected_mean, 2),
                "applied_rectifier": round(rectifier, 3),
                "ci": round(margin, 2),
                # 확신도 통계 합병
                **confidence_stats,
                # Softmax Binning 데이터
                'prob_bins': prob_bins,  # 평가셋 히스토그램용
                'golden_prob_bins': golden_prob_bins  # 🚨 추가: 골든셋 히스토그램용
            }
            overall_corrected_scores.append(corrected_mean)

        if overall_corrected_scores:
            summary["overall"] = round(sum(overall_corrected_scores) / len(overall_corrected_scores), 2)

        return summary


# ===================================================================
# 4. 메인 파이프라인 함수
# ===================================================================
def _load_judges_and_calc() -> Tuple[AresJudge, PPICalculator]:
    """심사관과 계산기 인스턴스를 로드하고 반환합니다."""

    lower_model_name = MODEL_NAME.lower()
    if "onnx" in lower_model_name and "sbert" in lower_model_name:
        # SBERT ONNX 모델인 경우
        judge = AresSbertOnnxJudge(DEVICE, MAX_LENGTH, model_type_name="SBert-Onnx")
    elif "distil" in lower_model_name:
        # DistilBERT 모델인 경우
        judge = AresPytorchJudge(DEVICE, MAX_LENGTH, model_type_name="DistilBERT")
    else:
        # 일반 BERT 모델인 경우
        judge = AresPytorchJudge(DEVICE, MAX_LENGTH, model_type_name="BERT")

    calculator = PPICalculator()
    return judge, calculator

# 골든셋 평가
def _evaluate_golden_sets(judge: AresJudge, calculator: PPICalculator) -> Tuple[Dict, Dict]:
    """
    골든셋 디렉토리를 탐색하고, 각 골든셋을 평가하여 통계 맵을 반환합니다.
    (골든셋 통계가 없으면 Fatal Error 발생)
    """
    GOLDEN_DIR = config.DATA_GOLDEN_DIR

    # 1. 골든 라벨 파일 검색
    golden_files: List[Tuple[str, str]] = []
    if os.path.isdir(GOLDEN_DIR):
        for filename in os.listdir(GOLDEN_DIR):
            if filename.endswith('.jsonl'):
                golden_name = filename.replace('.jsonl', '')
                file_path = os.path.join(GOLDEN_DIR, filename)
                golden_files.append((golden_name, file_path))

        print(f"\n[INFO] {GOLDEN_DIR}에서 총 {len(golden_files)}개의 골든셋 파일을 찾았습니다.")
    else:
        print(f"\n[WARN] 골든셋 디렉토리 '{GOLDEN_DIR}'가 존재하지 않습니다.")

    if not golden_files:
        raise RuntimeError("PPI 보정을 위한 골든 라벨 파일이 없습니다.")

    # 2. 모든 골든셋 평가 및 통계 저장
    golden_stats_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    # 🚨 수정: 순수 데이터 딕셔너리로 변경 (Markdown 아님)
    golden_report_data: Dict[str, Dict[str, Dict[str, float | str]]] = {}

    for golden_name, path in golden_files:
        try:
            stats = calculator.evaluate_golden_set(judge, path)
            if stats:
                golden_stats_map[golden_name] = stats

                # 🚨 수정: 순수 데이터 딕셔너리 반환 및 저장
                report_data = calculator.generate_golden_set_stat(stats)
                golden_report_data[golden_name] = report_data

                print(f"[SUCCESS] 골든셋 '{golden_name}' 평가 완료.")
            else:
                print(f"\n   [WARN] 골든셋 '{golden_name}' 평가 결과가 비어있습니다. 건너뜁니다.")
        except Exception as e:
            print(f"\n[FATAL ERROR] 골든셋 '{golden_name}' 평가 중 오류 발생: {e}. 건너뜁니다.")

    if not golden_stats_map:
        raise RuntimeError("PPI 보정을 위한 유효한 골든 라벨 데이터 평가 실패.")

    # 🚨 수정: golden_report_data 반환
    return golden_stats_map, golden_report_data


# 대규모 평가 루프
def _process_input_files(judge: AresJudge, calculator: PPICalculator, golden_stats_map: Dict) -> Tuple[
    List, int, float]:
    """
    입력 파일을 순회하며 LM 예측을 수행하고, PPI 보정을 적용하여 model_summaries를 반환합니다.
    """
    INPUT_DIR = config.DATA_IN_DIR
    input_files = [
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl')
    ]

    if not input_files:
        print(f"\n[WARN] QCA `.jsonl` 파일을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return [], 0, 0.0

    print(f"\n[INFO] 평가 대상 파일 갯수 : {len(input_files)}")

    total_successful_evals = 0
    model_summaries: List[Dict[str, Any]] = []
    full_start_time = time.time()

    # 2-2. 파일별 평가, LM 예측 파일 생성 및 결과 집계
    for file_path in input_files:
        start_time = time.time()
        file_base_name = os.path.basename(file_path).replace('.jsonl', '')

        # QCA 평가 (Judge)는 파일 당 한 번만 수행
        print(f"\n--- 평가 시작: {file_base_name} ---")
        current_lm_scores = {k: [] for k in JUDGE_TYPES}
        current_lm_probs = {k: [] for k in JUDGE_TYPES}  # 🚨 수정: Softmax 긍정 확률 집계용
        # all_results_for_file = [] # 저장 로직 비활성화 시 불필요
        processed_count_in_file = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            total_samples = len(lines)
            progress_points = [0, 10, 30, 50, 70, 90, 100]
            next_progress_index = 0

            if total_samples > 0:
                print(f"[PROGRESS] 0%. ", end=' ')

            for i, line in enumerate(lines):
                current_progress = (i * 100) // total_samples

                if next_progress_index < len(progress_points) - 1 and current_progress >= progress_points[
                    next_progress_index + 1]:
                    next_progress_index += 1
                    print(f"{progress_points[next_progress_index]}%  ", end=' ')

                try:
                    data = json.loads(line.strip())
                    query = ' '.join(data.get('q', '').split()).strip()
                    context = ' '.join(data.get('c', '').split()).strip()
                    answer = ' '.join(data.get('a', '').split()).strip()

                    if not all([query, context, answer]): continue

                    scores_with_probs = judge.evaluate_triple(query, context, answer)

                    for axis in JUDGE_TYPES:
                        pred_key = JUDGE_PREDICTION_FIELDS[axis]
                        axis_scores = scores_with_probs.get(pred_key)

                        if axis_scores is None: continue

                        current_lm_scores[axis].append(axis_scores.get('machine_pred', 0))
                        current_lm_probs[axis].append(axis_scores.get('prob_pos', 0.0))

                    total_successful_evals += 1
                    processed_count_in_file += 1

                except Exception as e:
                    print(f"[ERROR] 평가 중 예외발생 : {e}")
                    continue

            if processed_count_in_file > 0:
                print(f"100% 완료.")  # 100% 완료 명시

            end_time = time.time()
            print(f"[INFO] 평가 소요시간 : {end_time - start_time:,.2f}초 ")

            # 2-3. 평가셋당 모든 골든셋에 대해 PPI 통계 계산 및 집계
            if processed_count_in_file > 0:
                for golden_name, golden_stats in golden_stats_map.items():
                    summary = calculator.calculate_ppi_summary(
                        file_base_name,
                        current_lm_scores,
                        current_lm_probs,
                        processed_count_in_file,
                        golden_stats,
                        golden_name
                    )
                    model_summaries.append(summary)

                print(f"[집계 완료] '{file_base_name}' 결과 집계 완료. ({len(golden_stats_map)}개 골든셋 적용)")
            else:
                print(f"[ERROR] 심사관의 평가 결과 없음 - 파일: {file_base_name}")

        full_elapsed_time = time.time() - full_start_time

    return model_summaries, total_successful_evals, full_elapsed_time


# 보고서 및 요약
def _generate_report_and_summary(golden_markdown_map: Dict, model_summaries: List, total_successful_evals: int,
                                 full_elapsed_time: float) -> None:
    """
    최종 보고서를 생성하고 콘솔에 실행 요약을 출력합니다.
    """
    REPORT_DIR = config.DATA_REPORT_DIR
    MODEL_NAME = config.MODEL_NAME

    if model_summaries:
        report_content: str = report_util.generate_summary_report(golden_markdown_map, model_summaries)
        timestamp: str = time.strftime("%Y%m%d_%H%M%S")

        # 파일명 수정 반영
        report_filename: str = f"{MODEL_NAME}_{timestamp}.md"
        output_path: str = os.path.join(REPORT_DIR, report_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n[최종 완료] ARES 통계 보고서 생성됨 → {output_path}")

    print("\n\n=============== LM 예측 생성 최종 요약 ===============")
    print(f"총 LM 예측 샘플 수: {total_successful_evals}개")
    print(f"총 소요 시간: {full_elapsed_time:.2f}초")
    print("==================================================")



# 실행
# 02_generate_ppi.py (run_ares_pipeline 함수)

def run_ares_pipeline():
    """
    ARES 전체 파이프라인 실행을 관리하는 메인 함수.
    """
    try:
        # 0. 환경 설정
        os.makedirs(config.DATA_IN_DIR, exist_ok=True)
        os.makedirs(config.DATA_REPORT_DIR, exist_ok=True)
        print(f"\n[SETUP] 평가대상인 QCA 입력 디렉토리: {config.DATA_IN_DIR}, 보고서 디렉토리: {config.DATA_REPORT_DIR}")

        # 1. 초기화 및 골든셋 평가
        current_judge, calculator = _load_judges_and_calc()
        # 🚨 수정: golden_markdown_map 대신 golden_report_data 변수명 사용
        golden_stats_map, golden_report_data = _evaluate_golden_sets(current_judge, calculator)

        # 2. 대규모 평가 실행
        model_summaries, total_successful_evals, full_elapsed_time = _process_input_files(
            current_judge, calculator, golden_stats_map
        )

        # 3. 보고서 생성 및 최종 요약
        # 🚨 수정: golden_report_data 인수로 전달 (report_util에서 MD 생성)
        _generate_report_and_summary(golden_report_data, model_summaries, total_successful_evals, full_elapsed_time)

    except RuntimeError as e:
        print(f"\n[FATAL ERROR] 파이프라인 실행 중 치명적인 오류 발생: {e}")
        return

if __name__ == "__main__":
    run_ares_pipeline()

