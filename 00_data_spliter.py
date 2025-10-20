#
# 학습 데이터 균형있게 나누기
#
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Literal

# ===================================================================
# 0. 전역 상수 및 경로 설정
# ===================================================================
# 이전 단계에서 생성된 1:1 균형 데이터셋 경로
INPUT_DIR = "./data/learn/source"
# 분할된 train/val/test 파일이 저장될 경로
OUTPUT_DIR = "./data/learn/processed"

# 분석 및 분할할 레이블 키 정의 (각 파일명 접두사와 일치)
JUDGE_TASK_LABELS = [
    'L_AF',  # balanced_train_L_AF.jsonl 파일을 처리합니다.
    'L_AR',
    'L_CR',
]

# 통계 키 정의
StatKey = Literal["single_neg", "double_neg", "triple_neg", "all_pos", "unclassified"]


# --- 1. 통계 분석 헬퍼 함수 정의 ---

def _init_stats() -> Dict[str, int]:
    """통계 카운터 초기화"""
    return {
        "single_neg": 0, "double_neg": 0, "triple_neg": 0, "all_pos": 0, "unclassified": 0,
        "total": 0, "skipped": 0,
        # L_CR, L_AF, L_AR의 0/1 카운터
        "L_CR_0": 0, "L_CR_1": 0, "L_AF_0": 0, "L_AF_1": 0, "L_AR_0": 0, "L_AR_1": 0,
    }


def _analyze_negation(item: Dict[str, Any]) -> StatKey:
    """아이템을 분석하여 부정 유형을 반환합니다."""
    try:
        features = [
            item['L_CR'],
            item['L_AF'],
            item['L_AR']
        ]

        if not all(isinstance(f, int) and f in (0, 1) for f in features):
            return "unclassified"

        sum_of_features = sum(features)

        # 0: triple_neg, 1: double_neg, 2: single_neg, 3: all_pos (기존 로직 따름)
        if sum_of_features == 1:
            return "double_neg"
        elif sum_of_features == 2:
            return "single_neg"
        elif sum_of_features == 0:
            return "triple_neg"
        elif sum_of_features == 3:
            return "all_pos"

        return "unclassified"

    except Exception:
        return "unclassified"


# --- 2. 통계 계산 및 출력 함수 정의 ---

def _calculate_and_print_stats(df_data: pd.DataFrame, label_key: str, split_name: str, total_items_in_task: int):
    """DataFrame으로부터 통계를 계산하고 간결한 인라인 형식으로 결과를 출력합니다."""

    stats = _init_stats()
    count = len(df_data)

    # 1. 항목 순회하며 통계 산출
    for _, item in df_data.iterrows():
        # 부정 유형 분석
        negation_type = _analyze_negation(item.to_dict())  # 딕셔너리로 변환하여 전달
        stats[negation_type] += 1
        stats["total"] += 1

        # 개별 피처 0/1 카운트
        for feature in ['L_CR', 'L_AF', 'L_AR']:
            value = item.get(feature)
            if value in (0, 1):
                stats[f"{feature}_{value}"] += 1

    # 2. 통계 출력

    # 총 항목 수
    print(f"\n[{label_key.upper()}_{split_name.upper()} 파일] - 총 항목 {count:,d}개")

    # L_XX 0/1 분포
    count_0 = stats.get(f"{label_key}_0", 0)
    count_1 = stats.get(f"{label_key}_1", 0)
    total_feature_count = count_0 + count_1

    perc_0 = (count_0 / total_feature_count) * 100 if total_feature_count > 0 else 0.00
    perc_1 = (count_1 / total_feature_count) * 100 if total_feature_count > 0 else 0.00

    print(f"  - 주요 분포 ({label_key}): 0 ({perc_0:.2f}%) / 1 ({perc_1:.2f}%)")

    # 부정/긍정 유형 분포 (단일 라인 요약)
    STAT_LABELS = {
        "single_neg": "단일 부정", "double_neg": "이중 부정",
        "triple_neg": "셋 다 부정", "all_pos": "모두 긍정",
    }

    parts = []
    for stat_key, label in STAT_LABELS.items():
        stat_count = stats.get(stat_key, 0)
        percentage = (stat_count / count) * 100 if count > 0 else 0.0
        parts.append(f"{label}: {percentage:.2f}%")

    print(f"  - 유형 분포: {', '.join(parts)}")
    print("-" * 60)


# --- 3. 데이터 분할 및 저장 함수 (메인 로직) ---

def split_and_save_dataset(label_key: str):
    """
    1:1 균형 데이터셋을 8:1:1 비율로 분할하고 JSONL 파일로 저장합니다.
    """
    input_file = os.path.join(INPUT_DIR, f"balanced_train_{label_key}.jsonl")

    if not os.path.exists(input_file):
        print(f"[ERROR] 파일이 존재하지 않습니다: {input_file}")
        return

    # 1. 데이터 로드
    try:
        df = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"[ERROR] {input_file} 로드 실패: {e}")
        return

    total_count = len(df)
    print(f"\n[INFO] {label_key} 총 항목 수: {total_count}개")

    # 2. Train (80%) / Temp (20%) 분할
    df_train, df_temp = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )

    # 3. Validation (10%) / Test (10%) 분할 (Temp 데이터를 50:50으로 분할)
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, shuffle=True
    )

    # 4. 결과 출력 및 저장
    df_splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test
    }

    # JSONL 파일로 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def save_df_to_jsonl(df_data, file_suffix):
        output_path = os.path.join(OUTPUT_DIR, f"{label_key}_{file_suffix}.jsonl")
        df_data.to_json(output_path, orient='records', lines=True, force_ascii=False)
        print(f"[SUCCESS] 저장 완료: {output_path}")

    # 총 항목 수 (분모)는 total_count를 사용
    total_items_in_task = total_count

    for split_name, df_data in df_splits.items():
        save_df_to_jsonl(df_data, split_name)

        # 5. 분할된 파일의 통계 계산 및 출력
        _calculate_and_print_stats(df_data, label_key, split_name, total_items_in_task)

    print("-" * 60)


# --- 4. 메인 실행 ---

if __name__ == "__main__":
    start_time = time.time()

    print("=" * 60)
    print("### 데이터셋 분할 및 통계 분석 시작 (8:1:1 비율) ###")
    print("=" * 60)

    for label_key in JUDGE_TASK_LABELS:
        split_and_save_dataset(label_key)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 60)
    print(f"총 소요 시간: {elapsed_time:.4f} 초")
    print("데이터 분할 및 통계 분석 완료.")
    print("=" * 60)