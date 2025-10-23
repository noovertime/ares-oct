# 02_generate_ppi.py (클래스 기반 및 다중 골든셋 확장 버전)
# 1. 표준 라이브러리
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
            # ... (기존 cleanup_evaluation_data 로직 유지)
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
# 2. ARES 평가 클래스
# ===================================================================

class AresJudge:
    """ARES 심사관(토크나이저 및 3개 모델) 로딩 및 평가를 담당하는 클래스."""

    def __init__(self, device: torch.device = DEVICE, max_length: int = MAX_LENGTH):
        self.tokenizer: AutoTokenizer = None
        self.judges: Dict[str, AutoModelForSequenceClassification] = {}
        self.device = device
        self.max_length = max_length
        self._load_models()

    def _load_models(self):
        """CR, AF, AR 세 가지 심사관 모델과 토크나이저를 로드합니다."""
        print("\n>> ARES 심사관 로딩 시작 (CPU 환경)...")

        # 1. 토크나이저 초기화
        try:
            cr_path = _find_model_path(KEY_CR)
            self.tokenizer = AutoTokenizer.from_pretrained(cr_path, trust_remote_code=True)
            print(f"   [INFO] {cr_path} 에서 토크나이저 로드 성공.")
        except Exception as e:
            print(f"   [WARN] 저장 경로에서 토크나이저 로드 실패. 원본 모델 ({MODEL_NAME}) 로드 시도.")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            except Exception as fallback_e:
                print(f"   [FATAL] 토크나이저 로드 최종 실패: {fallback_e}")
                raise fallback_e

        if TOKEN_TYPE_ID_OFF:
            self.tokenizer.model_input_names = [
                name for name in self.tokenizer.model_input_names if name != 'token_type_ids'
            ]
            print("   [INFO] DistilBERT 호환성을 위해 토크나이저의 'token_type_ids' 생성을 비활성화했습니다.")

        # 2. 모델 로드 (AutoModelForSequenceClassification 사용)
        for judge_type in JUDGE_TYPES:
            try:
                model_path = _find_model_path(judge_type)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=2, trust_remote_code=True
                )
                model.to(self.device)
                model.eval()
                self.judges[judge_type] = model
                print(f"   [SUCCESS] {judge_type} Judge 로드 완료 ( {model_path} )")

            except Exception as e:
                print(f"   [ERROR] {judge_type} Judge 로드 실패: {e}. 이 모델은 건너뜁니다.")

        if len(self.judges) != 3:
            raise RuntimeError(f"총 {len(self.judges)}개만 로드됨. ARES 평가를 위해 3개 모델이 모두 필요합니다.")

        print(f">> ARES 심사관 {MODEL_NAME} 로드 완료. 총 {len(self.judges)}개 심사관 활성화.")

    def evaluate_triple(self, query: str, context: str, answer: str) -> Dict[str, int]:
        """하나의 Q-C-A 쌍에 대해 3가지 ARES 점수 (0 또는 1)를 계산합니다."""

        results = {}

        judge_inputs = {
            JUDGE_PREDICTION_FIELDS[KEY_CR]: (query, context, self.judges[KEY_CR]),
            JUDGE_PREDICTION_FIELDS[KEY_AF]: (context, answer, self.judges[KEY_AF]),
            JUDGE_PREDICTION_FIELDS[KEY_AR]: (query, answer, self.judges[KEY_AR])
        }

        with torch.no_grad():
            for name, (text_a, text_b, model) in judge_inputs.items():
                # 1. 입력 토큰화
                inputs = self.tokenizer(
                    text_a, text_b,
                    return_tensors="pt",
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length
                ).to(self.device)

                # 2. 예측 수행 및 결과 산출
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                results[name] = prediction

        return results


# ===================================================================
# 3. PPI 통계 및 보정 계산 클래스
# ===================================================================

class PPICalculator:
    """PPI 보정 및 통계 계산을 담당하는 클래스."""

    def __init__(self, ci_z_score: float = CI_Z_SCORE):
        self.ci_z_score = ci_z_score

    def calculate_accuracy(self, stats: Dict[str, Any]) -> str:
        """rectifier_terms를 사용하여 심사관의 정확도를 계산하고 문자열로 반환합니다."""
        terms = stats.get('rectifier_terms')
        labeled_n = stats.get('labeled_n')

        if not terms or labeled_n == 0:
            return 'N/A'

        # 오류 횟수 = (rectifier_terms 리스트에서 0이 아닌 값의 개수)
        error_count = sum(1 for term in terms if term != 0.0)
        accuracy = 1.0 - (error_count / labeled_n)
        return f"{accuracy:13.3f}"

    def generate_golden_set_stat(self, golden_set_stats: Dict[str, Dict[str, Any]]) -> str:
        """골든셋 통계 결과를 마크다운 문자열로 변환합니다."""
        if not golden_set_stats:
            print("[FAIL] 골든셋 평가 결과가 비어있거나 실패했습니다.")
            return "골든셋 평가 결과 없음"

        # ... (기존 generate_golden_set_stat 로직 유지)
        cr_mean = golden_set_stats.get(KEY_CR, {}).get('machine_mean', 'N/A')
        af_mean = golden_set_stats.get(KEY_AF, {}).get('machine_mean', 'N/A')
        ar_mean = golden_set_stats.get(KEY_AR, {}).get('machine_mean', 'N/A')

        cr_acc = self.calculate_accuracy(golden_set_stats.get(KEY_CR, {}))
        af_acc = self.calculate_accuracy(golden_set_stats.get(KEY_AF, {}))
        ar_acc = self.calculate_accuracy(golden_set_stats.get(KEY_AR, {}))

        header = f"| {'구분':^12} | {'CR':^13} | {'AF':^13} | {'AR':^13} |"
        separator = f"+----------------+---------------+---------------+---------------+"
        mean_row = (
            f"| {'예측평균':^12} | {cr_mean:>13.3f} | {af_mean:>13.3f} | {ar_mean:>13.3f} |"
            f"  심사관이 1이라고 예측한 비율 (긍정 편향)"
        )
        acc_row = (
            f"| {'정답비율':^12} | {cr_acc:>13} | {af_acc:>13} | {ar_acc:>13} |"
            f"  심사관 예측의 정확도"
        )
        markdown_content = "\n".join([header, separator, mean_row, acc_row])
        markdown_string = f"```\n{markdown_content}\n```"
        return markdown_string

    def evaluate_golden_set(self, judge: AresJudge, golden_set_filepath: str) -> Dict[str, Dict[str, Any]]:
        """
        특정 골든셋 파일을 심사관이 평가하고, PPI 편향 계산에 필요한 통계 정보를 반환합니다.
        (기존의 load_gold_labels_map 로직은 이 함수에 통합되어 ID 기반 매칭 없이 직접 평가)
        """
        golden_stats = {key: {'labeled_n': 0, 'rectifier_terms': [], 'machine_mean_sum': 0.0} for key in JUDGE_TYPES}
        golden_records = _load_json_lines(golden_set_filepath)

        if not golden_records:
            print(f"[WARN] 골든셋 파일 {golden_set_filepath}에 평가할 데이터가 없습니다.")
            return {}

        print(f"\n>> 골든셋 평가 시작. {golden_set_filepath}, 총 {len(golden_records)}개 샘플 심사관 예측 중...")

        for data in tqdm(golden_records, desc="골든셋 심사관 평가 중"):
            try:
                # Q, C, A 추출 및 정규화
                query = ' '.join(data.get('q', '').split()).strip()
                context = ' '.join(data.get('c', '').split()).strip()
                answer = ' '.join(data.get('a', '').split()).strip()

                if not all([query, context, answer]):
                    continue

                # 1. LM 심사관 예측 (Yhat_labeled)
                scores = judge.evaluate_triple(query, context, answer)

                # 2. LM 예측값과 인간 주석값 비교 (Y_labeled는 골든셋 파일에서 직접 추출)
                for axis in JUDGE_TYPES:
                    machine_pred = scores.get(JUDGE_PREDICTION_FIELDS[axis])
                    gold_key = GOLD_LABEL_FIELDS[axis]
                    gold_label = data.get(gold_key)

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

            except Exception:
                continue

        # 3. 최종 통계 계산
        final_golden_stats = {}
        for axis, stats in golden_stats.items():
            if stats['labeled_n'] > 0:
                final_golden_stats[axis] = {
                    'labeled_n': stats['labeled_n'],
                    'rectifier_terms': stats['rectifier_terms'],
                    'machine_mean': stats['machine_mean_sum'] / stats['labeled_n']
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

    # PPICalculator 클래스 내부의 calculate_ppi_summary 메서드

    def calculate_ppi_summary(
            self,
            file_base_name: str,
            current_lm_scores: Dict[str, List[int]],
            total_n: int,
            golden_set_stats: Dict[str, Dict[str, Any]],
            golden_set_name: str
    ) -> Dict[str, Any]:
        """
        LM 예측 결과와 골든셋 통계를 결합하여 PPI 요약 결과를 계산합니다.
        (골든셋 통계가 누락되면 명시적 오류 발생)
        """

        summary: Dict[str, Any] = {
            "model_name": file_base_name,
            "golden_set_name": golden_set_name,
            "n": total_n,
            "ppi_active": True,
            # CR 기준으로 대표값 사용. 통계가 없는 경우를 대비해 기본값 0 설정
            "labeled_n_rep": golden_set_stats.get(KEY_CR, {}).get('labeled_n', 0),
        }

        overall_corrected_scores: List[float] = []

        for axis in JUDGE_TYPES:
            # 1. LM 예측 결과 (Yhat_unlabeled)
            scores: List[int] = current_lm_scores[axis]

            # 2. 골든셋 통계 (Rectifier Terms, labeled_n)
            golden_axis_stats = golden_set_stats.get(axis, {})

            labeled_n_axis: int = golden_axis_stats.get('labeled_n', 0)

            # 🚨 핵심 수정: 필수 통계 누락 시 명시적 오류 발생
            if labeled_n_axis == 0 or not golden_axis_stats:
                raise RuntimeError(
                    f"[{axis} 축 통계 누락] 골든셋 '{golden_set_name}'에 '{axis}' 축의 유효한 라벨링 데이터가 부족합니다. "
                    f"Labeled N: {labeled_n_axis}."
                )

            rectifier_terms: List[float] = golden_axis_stats['rectifier_terms']

            # 3. 통계 계산
            machine_mean: float = sum(scores) / float(total_n)

            # 3-2. 편향 (Rectifier) = Avg(Yhat_labeled - Y_labeled)
            # labeled_n_axis는 0이 아님을 위에서 보장했음
            rectifier: float = sum(rectifier_terms) / labeled_n_axis

            # 3-3. 보정된 성능 (Corrected Mean)
            corrected_mean: float = max(0.0, min(1.0, machine_mean - rectifier))

            # 3-4. 신뢰 구간 (CI)
            margin: float = self.calculate_ppi_asymptotic_ci(scores, rectifier_terms, total_n, labeled_n_axis)

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


# ===================================================================
# 4. 메인 파이프라인 함수
# ===================================================================

def run_ares_pipeline():
    """
    ARES 전체 파이프라인 실행: 다중 골든셋 처리 및 LM 예측 파일 생성/보고서 생성.
    """

    # --- 1. 환경 설정 및 초기화 단계 ---
    INPUT_DIR = config.DATA_IN_DIR
    REPORT_DIR = config.DATA_REPORT_DIR
    GOLDEN_DIR = config.DATA_GOLDEN_DIR

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"\n[SETUP] 평가대상인 QCA 입력 디렉토리: {INPUT_DIR}")

    # 클래스 인스턴스 생성
    try:
        judge = AresJudge()
        calculator = PPICalculator()
    except RuntimeError as e:
        print(f"\n[FATAL ERROR] ARES 심사관 시스템 초기화 실패: {e}")
        return

    # 1-2. 골든 라벨 파일 검색 (다중 골든셋을 처리하도록 확장)
    # 결과: golden_files = [ ('golden_set_name_1', '/path/to/file1.jsonl'), ... ]
    golden_files: List[Tuple[str, str]] = []
    if os.path.isdir(GOLDEN_DIR):
        for filename in os.listdir(GOLDEN_DIR):
            if filename.endswith('.jsonl'):
                # 파일명을 골든셋 이름으로 사용합니다 (확장자 제거)
                golden_name = filename.replace('.jsonl', '')
                file_path = os.path.join(GOLDEN_DIR, filename)
                golden_files.append((golden_name, file_path))

        print(f"\n[INFO] {GOLDEN_DIR}에서 총 {len(golden_files)}개의 골든셋 파일을 찾았습니다.")
    else:
        print(f"\n[WARN] 골든셋 디렉토리 '{GOLDEN_DIR}'가 존재하지 않습니다.")

    if not golden_files:
        print("\n[FATAL ERROR] PPI 보정을 위한 골든 라벨 파일이 없습니다. 파이프라인을 중단합니다.")
        return

    # 1-3. 모든 골든셋 평가 및 통계 저장 (메모리 처리)
    # golden_stats_map: { 'golden_set_name': { 'KEY_CR': { stats } }, ... }
    golden_stats_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    golden_markdown_map: Dict[str, str] = {}

    for golden_name, path in golden_files:
        try:
            stats = calculator.evaluate_golden_set(judge, path)
            if stats:
                golden_stats_map[golden_name] = stats
                golden_markdown_map[golden_name] = calculator.generate_golden_set_stat(stats)
                print(f"\n   [SUCCESS] 골든셋 '{golden_name}' 평가 완료.")
            else:
                print(f"\n   [WARN] 골든셋 '{golden_name}' 평가 결과가 비어있습니다. 건너뜁니다.")
        except Exception as e:
            print(f"\n[FATAL ERROR] 골든셋 '{golden_name}' 평가 중 오류 발생: {e}. 건너뜁니다.")

    if not golden_stats_map:
        print("\n[FATAL ERROR] PPI 보정을 위한 유효한 골든 라벨 데이터 평가 실패. 파이프라인을 중단합니다.")
        return

    # --- 2. 입력 파일 검색 및 LM 예측 생성 루프 단계 (다중 골든셋으로 확장됨) ---
    input_files = [
        os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.jsonl')
    ]

    if not input_files:
        print(f"\n[WARN] QCA `.jsonl` 파일을 찾을 수 없습니다. 평가를 건너뜁니다.")
        return

    print(f"\n[INFO] 평가 대상 파일 갯수 : {len(input_files)}")

    total_successful_evals = 0
    full_start_time = time.time()

    # 최종 보고서에 들어갈 요약 리스트 (다중 골든셋 * 다중 평가셋 조합)
    model_summaries: List[Dict[str, Any]] = []

    # 2-2. 파일별 평가, LM 예측 파일 생성 및 결과 집계
    for file_path in input_files:
        start_time = time.time()  # 파일별 시간 측정 시작
        file_base_name = os.path.basename(file_path).replace('.jsonl', '')

        # QCA 평가 (Judge)는 한 번만 수행
        print(f"\n--- 대규모 평가 시작: {file_base_name} ---")
        current_lm_scores = {k: [] for k in JUDGE_TYPES}
        all_results_for_file = []  # 예측 결과를 JSONL 파일로 출력할 경우를 대비하여 유지
        processed_count_in_file = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"평가 중 [{file_base_name}]"):
                try:
                    data = json.loads(line.strip())
                    query = ' '.join(data.get('q', '').split()).strip()
                    context = ' '.join(data.get('c', '').split()).strip()
                    answer = ' '.join(data.get('a', '').split()).strip()

                    if not all([query, context, answer]): continue

                    # ARES 심사관 예측 수행 (Yhat_unlabeled 생성)
                    scores = judge.evaluate_triple(query, context, answer)
                    data.update(scores)

                    # LM 예측 결과를 현재 파일의 집계 리스트에 추가
                    for axis in JUDGE_TYPES:
                        pred_key = JUDGE_PREDICTION_FIELDS[axis]
                        current_lm_scores[axis].append(scores.get(pred_key, 0))

                    all_results_for_file.append(data)
                    total_successful_evals += 1
                    processed_count_in_file += 1

                except Exception as e:
                    print(f"[ERROR] 평가 중 예외발생 : {e}")
                    continue

        end_time = time.time()
        print(f"[INFO] 평가 소요시간 : {end_time - start_time:,.2f}초 ")

        # 2-3. 평가셋당 모든 골든셋에 대해 PPI 통계 계산 및 집계
        if processed_count_in_file > 0:
            for golden_name, golden_stats in golden_stats_map.items():
                summary = calculator.calculate_ppi_summary(
                    file_base_name,
                    current_lm_scores,
                    processed_count_in_file,
                    golden_stats,
                    golden_name  # 골든셋 이름 전달
                )
                model_summaries.append(summary)

            print(f"   [집계 완료] '{file_base_name}' 결과 집계 완료. (모든 {len(golden_stats_map)}개 골든셋 적용)")

            # --- LM 예측 파일 (Yhat_unlabeled) 저장 로직 (필요하다면 활성화) ---
            # output_filename = f"{file_base_name}_lm_preds.jsonl"
            # output_path = os.path.join(OUTPUT_DIR, output_filename)
            # with open(output_path, 'w', encoding='utf-8') as outfile:
            #     for result in all_results_for_file:
            #         outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
            # print(f"   [SUCCESS] LM 예측 결과 저장 완료 → {output_path}")

        else:
            print(f"   [ERROR] 심사관의 평가 결과 없음 - 파일: {file_base_name}")

    # --- 3. 최종 보고서 생성 단계 ---
    full_end_time = time.time()
    full_elapsed_time = full_end_time - full_start_time

    # --- 3. 최종 보고서 생성 단계 ---
    # 🚨 디버깅을 위해 이 부분의 출력 결과를 알려주세요.
    print(f"\n[DEBUG] Model Summaries 첫 번째 요소: {model_summaries[0]}")

    # 최종 보고서 생성
    if model_summaries:
        # report_util.generate_summary_report 함수는 이제 golden_markdown_map을 받도록 변경되어야 합니다.
        report_content: str = report_util.generate_summary_report(golden_markdown_map, model_summaries)
        timestamp: str = time.strftime("%Y%m%d_%H%M%S")
        report_filename: str = f"{MODEL_NAME}_{timestamp}.md"
        output_path: str = os.path.join(REPORT_DIR, report_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"\n[최종 완료] ARES 통계 보고서 생성됨 → {output_path}")

    print("\n\n=============== LM 예측 생성 최종 요약 ===============")
    print(f"총 LM 예측 샘플 수: {total_successful_evals}개")
    print(f"총 소요 시간: {full_elapsed_time:.2f}초")
    print("==================================================")


if __name__ == "__main__":
    run_ares_pipeline()