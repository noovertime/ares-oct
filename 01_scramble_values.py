#
# 지정된 파일에서 다음을 생성
# 1. 골든셋
# 2. 평가대상 RAG이 대답한 값인 척 하는 것
#
# 그러므로 테스트용(train, val말고 test)에서 추출해야 적당
#
# 2는 질문/답, 근거/답을 일정 비율 섞어서 오답 생성하고, 판단라벨 삭제함
#
import os
import json
import random
import math
import uuid
import sys
from enum import Enum  # Enum 모듈 추가
from typing import List, Dict, Any, Tuple
from config import PREPARE_DIR, PREPARE_FILE_NAME, PREPARE_OUT_PREFIX

# =========================================================================
# 상수 정의
# =========================================================================
# 하위 디렉토리 상수 정의
GOLDEN_SUBDIR = "golden"
IN_SUBDIR = "in"


# 추출 스타일 정의
class ExtractType(Enum):
    """RAG 평가 대상 샘플의 긍정/부정 비율을 결정하는 추출 스타일"""
    BALANCE = 1  # 긍정:부정 = 50:50
    POSITIVE = 2  # 긍정 최대화 (부정 최소 10% 보장)
    NEGATIVE = 3  # 부정 최대화 (긍정 최소 0%~5%만 포함 가능성 있음)
    ALL_POSITIVE = 4  # L_CR, L_AF, L_AR이 모두 1인 값만 추출
    ALL_NEGATIVE = 5  # L_CR, L_AF, L_AR이 모두 0인 값만 추출


# 추출 스타일
EXTRACT_STYLE = ExtractType.ALL_NEGATIVE
# 추출 비율
GOLDEN_RATIO = 0.05
EXTRACT_RATIO = 0.50  # RAG 샘플 추출 비율을 50%로 상향 조정
MIN_GOLDEN_COUNT = 100  # 골든셋 최소 확보 목표
MIN_NEGATIVE_RATIO_POS = 0.10  # POSITIVE 스타일에서 최소 부정 비율

# 오답 생성비율
SWAP_A_MIN_RATE = 0.0
SWAP_A_MAX_RATE = 0.0
SWAP_C_MIN_RATE = 0.0
SWAP_C_MAX_RATE = 0.0

# README 파일 이름 상수화
README_GOLDEN_FILE = "README_golden.txt"
README_RAG_FILE = "README_rag.txt"


# =========================================================================
# 유틸리티 함수 (load_data, save_data, swap_values, print_statistics)
# =========================================================================

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
    os.makedirs(PREPARE_DIR, exist_ok=True)
    full_path = os.path.join(PREPARE_DIR, file_path)
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"JSON 디코딩 오류 발생: {e}")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로: {full_path}")
        return []
    return data


def save_data(data: List[Dict[str, Any]], file_path: str, keys_to_keep: List[str] = None, subdir: str = None):
    """데이터를 JSONL 형식으로 지정된 하위 디렉토리에 저장합니다."""

    # 1. 대상 디렉토리 설정 및 생성
    target_dir = PREPARE_DIR
    if subdir:
        target_dir = os.path.join(PREPARE_DIR, subdir)

    os.makedirs(target_dir, exist_ok=True)

    # 2. 전체 파일 경로 구성
    full_path = os.path.join(target_dir, file_path)
    print(f"총 {len(data)}개의 데이터를 {full_path}에 저장합니다.")

    with open(full_path, 'w', encoding='utf-8') as f:
        for item in data:
            if keys_to_keep:
                filtered_item = {k: v for k, v in item.items() if k in keys_to_keep}
                f.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def swap_values(data: List[Dict[str, Any]], key: str, min_rate: float, max_rate: float) -> float:
    """
    데이터 리스트 내에서 특정 키(key)의 값을 무작위로 추출된 비율만큼 서로 교환합니다.
    (단, 두 값이 서로 다른 경우에만 교환을 진행합니다.)

    Returns:
        float: 실제로 교환된 항목 수 / 전체 데이터 수 (오류 주입 비율)
    """
    data_size = len(data)
    if data_size < 2:
        return 0.0

    if max_rate <= 0.0:
        print(f"오류 주입 없음")
        return 0.0

    swap_rate = random.uniform(min_rate, max_rate)

    num_pairs_to_attempt = math.floor(data_size * swap_rate / 2) * 2
    num_pairs_to_attempt = min(num_pairs_to_attempt, data_size)

    if num_pairs_to_attempt < 2:
        print(f"  > '{key}' 값 교환 시도 ({swap_rate * 100:.2f}%): 교환 가능한 항목이 부족합니다 (0쌍).")
        return 0.0

    swap_indices = random.sample(range(data_size), num_pairs_to_attempt)
    actual_swaps = 0

    print(f"  > '{key}' 값 교환 시도 ({swap_rate * 100:.2f}%):")
    for i in range(0, num_pairs_to_attempt, 2):
        idx1 = swap_indices[i]
        idx2 = swap_indices[i + 1]

        val1 = data[idx1].get(key)
        val2 = data[idx2].get(key)

        if val1 is not None and val2 is not None and val1 != val2:
            data[idx1][key], data[idx2][key] = val2, val1
            actual_swaps += 1

    print(f"  > '{key}' 값 {actual_swaps}쌍 (총 {actual_swaps * 2}개 항목)이 실제로 교환되었습니다.")

    return (actual_swaps * 2) / data_size if data_size > 0 else 0.0


def print_statistics(title: str, data: List[Dict[str, Any]], swap_rate_info: Dict[str, float] = None):
    """
    제공된 데이터 리스트의 평가 유형별 통계를 화면에 출력하고 파일로 저장합니다.
    """
    if not data:
        print(f"\n[{title}] - 데이터가 없어 통계를 출력/저장할 수 없습니다.")
        return

    stats = {
        'L_CR': {'0': 0, '1': 0}, 'L_AF': {'0': 0, '1': 0}, 'L_AR': {'0': 0, '1': 0},
    }
    all_zero_count = 0
    all_one_count = 0

    for item in data:
        for key in stats.keys():
            value = str(item.get(key, -1))
            if value in stats[key]:
                stats[key][value] += 1

        cr, af, ar = item.get('L_CR'), item.get('L_AF'), item.get('L_AR')
        if cr == 0 and af == 0 and ar == 0:
            all_zero_count += 1
        elif cr == 1 and af == 1 and ar == 1:
            all_one_count += 1

    total_data_count = len(data)
    at_least_one_zero_count = total_data_count - all_one_count

    if "GOLDEN" in title:
        file_name, subdir = README_GOLDEN_FILE, GOLDEN_SUBDIR
    else:
        file_name, subdir = README_RAG_FILE, IN_SUBDIR

    target_dir = os.path.join(PREPARE_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)

    output_lines = []
    output_lines.append("\n" + "=" * 50)
    output_lines.append(f"{title}")
    output_lines.append(f"총 데이터 수: {total_data_count}")
    output_lines.append(f"추출 스타일: {EXTRACT_STYLE.name}")

    # 오류 주입 비율 정보 추가
    if swap_rate_info:
        output_lines.append("-" * 50)
        output_lines.append("[오류 주입 비율 (Swap Rate)]")
        total_swapped_items = sum(total_data_count * rate for rate in swap_rate_info.values())
        total_swap_rate = total_swapped_items / total_data_count if total_data_count > 0 else 0.0

        output_lines.append(f"- '답변(a)' 변경률: {swap_rate_info.get('a', 0.0) * 100:.2f}%")
        output_lines.append(f"- '근거(c)' 변경률: {swap_rate_info.get('c', 0.0) * 100:.2f}%")
        output_lines.append(f"- 최종 오답 주입률: {total_swap_rate * 100:.2f}% (중복 포함, {math.floor(total_swapped_items)}개 항목)")

    output_lines.append("=" * 50)

    # 비율 계산 및 출력 포맷팅
    output_lines.append("| 평가 유형 | '0' 개수 (부정) | '1' 개수 (긍정) |")
    output_lines.append("|:---------|:----------------|:----------------|")
    CELL_WIDTH = 15

    for key, label in [('L_CR', 'C'), ('L_AF', 'F'), ('L_AR', 'R')]:
        count_0 = stats[key]['0']
        count_1 = stats[key]['1']
        ratio_0 = (count_0 / total_data_count) * 100 if total_data_count else 0
        ratio_1 = (count_1 / total_data_count) * 100 if total_data_count else 0
        line_0 = f"{count_0} ({ratio_0:.2f}%)"
        line_1 = f"{count_1} ({ratio_1:.2f}%)"
        output_lines.append(f"| {key} ({label}) | {line_0:>{CELL_WIDTH}} | {line_1:>{CELL_WIDTH}} |")

    output_lines.append("-" * 50)
    output_lines.append("종합 판별 결과:")
    output_lines.append(f"- 3개 유형 모두 '0'인 항목 수 (C=0 & F=0 & R=0): {all_zero_count}개")
    output_lines.append(f"- 3개 유형 모두 '1'인 항목 수 (C=1 & F=1 & R=1): {all_one_count}개")
    output_lines.append(f"- 1개라도 '0'인 항목 수 : {at_least_one_zero_count}개")
    output_lines.append("=" * 50 + "\n")

    output_text = "\n".join(output_lines)
    print(output_text)

    full_path = os.path.join(target_dir, file_name)
    try:
        with open(full_path, 'a', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[알림] 통계 정보를 '{full_path}'에 저장했습니다.")
    except Exception as e:
        print(f"[오류] 통계 정보 파일 저장 실패: {e}")


def categorize_data(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    데이터를 긍정(Positive) 및 부정(Negative)의 세 가지 유형으로 분류합니다.
    """
    categorized = {
        'positive': [], 'neg_type_3': [], 'neg_type_2': [], 'neg_type_1': [],
    }

    for item in data:
        cr, af, ar = item.get('L_CR', 0), item.get('L_AF', 0), item.get('L_AR', 0)
        num_negatives = (1 - cr) + (1 - af) + (1 - ar)

        if num_negatives == 0:
            categorized['positive'].append(item)
        elif num_negatives == 3:
            categorized['neg_type_3'].append(item)
        elif num_negatives == 2:
            categorized['neg_type_2'].append(item)
        elif num_negatives == 1:
            categorized['neg_type_1'].append(item)
    return categorized


def calculate_sample_counts(rag_target_size: int, categorized_data: Dict[str, List[Dict[str, Any]]]):
    """추출 스타일에 따라 긍정/부정 샘플 목표 개수를 계산합니다."""

    pos_count = len(categorized_data['positive'])
    neg_total_count = sum(len(v) for k, v in categorized_data.items() if k.startswith('neg'))

    # ALL_NEGATIVE는 L_CR=0, L_AF=0, L_AR=0 인 경우만 추출하므로 neg_type_3 개수만 사용합니다.
    neg_type_3_count = len(categorized_data['neg_type_3'])

    if EXTRACT_STYLE == ExtractType.BALANCE:
        target_pos = min(rag_target_size // 2, pos_count)
        target_neg = min(rag_target_size - target_pos, neg_total_count)

    elif EXTRACT_STYLE == ExtractType.POSITIVE:
        min_neg_count = max(1, math.floor(rag_target_size * MIN_NEGATIVE_RATIO_POS))
        target_neg = min(min_neg_count, neg_total_count)
        target_pos = min(rag_target_size - target_neg, pos_count)

    elif EXTRACT_STYLE == ExtractType.NEGATIVE:
        target_neg = min(rag_target_size, neg_total_count)
        target_pos = min(rag_target_size - target_neg, pos_count)

    # --- 새로 추가된 스타일 ---
    elif EXTRACT_STYLE == ExtractType.ALL_POSITIVE:
        target_pos = min(rag_target_size, pos_count)  # 긍정 최대화
        target_neg = 0  # 부정 0

    elif EXTRACT_STYLE == ExtractType.ALL_NEGATIVE:
        target_neg = min(rag_target_size, neg_type_3_count)  # L_CR=0, L_AF=0, L_AR=0인 데이터로만 부정 최대화
        target_pos = 0  # 긍정 0
    # -------------------------

    else:
        # 정의되지 않은 스타일의 경우 기본값(BALANCE)으로 폴백
        return calculate_sample_counts(rag_target_size, categorized_data)

        # 최종적으로 target_pos와 target_neg의 합이 rag_target_size를 초과하지 않도록 보정
    if target_pos + target_neg > rag_target_size:
        if EXTRACT_STYLE in [ExtractType.ALL_POSITIVE, ExtractType.POSITIVE, ExtractType.BALANCE]:
            # 긍정 스타일 우선: 목표 크기에 맞춰 부정 개수를 줄임
            target_neg = rag_target_size - target_pos
        else:  # NEGATIVE, ALL_NEGATIVE 스타일 우선: 목표 크기에 맞춰 긍정 개수를 줄임
            target_pos = rag_target_size - target_neg

    return max(0, target_pos), max(0, target_neg)


def sample_negative_data(target_neg: int, categorized_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    부정 샘플을 target_neg 개수만큼 3-2-1 순서로 우선 추출합니다.
    """
    sampled_neg = []
    neg_keys = ['neg_type_3', 'neg_type_2', 'neg_type_1']

    for key in neg_keys:
        if len(sampled_neg) >= target_neg:
            break

        remaining_needed = target_neg - len(sampled_neg)
        available_data = categorized_data[key]
        num_to_sample = min(remaining_needed, len(available_data))

        if num_to_sample > 0:
            sampled_neg.extend(random.sample(available_data, num_to_sample))

    return sampled_neg


# =========================================================================
# 분리된 핵심 함수
# =========================================================================

def _initialize_and_calculate_targets() -> Tuple[
    List[Dict[str, Any]], int, int, Tuple[int, int], Dict[str, List[Dict[str, Any]]]]:
    """초기화, 데이터 로드, 크기/비율 계산, README 파일 정리 및 유효성 검사를 수행합니다."""

    raw_data = load_data(PREPARE_FILE_NAME)
    if not raw_data:
        sys.exit(1)

    total_count = len(raw_data)

    if total_count < MIN_GOLDEN_COUNT:
        print(f"\n[오류] 전체 데이터 수({total_count}개)가 최소 골든셋 요구치({MIN_GOLDEN_COUNT}개) 미만입니다.")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    # 크기 계산
    golden_size = max(int(total_count * GOLDEN_RATIO), MIN_GOLDEN_COUNT)
    golden_size = min(golden_size, total_count)
    rag_extract_size = max(int(total_count * EXTRACT_RATIO), golden_size)
    rag_extract_size = min(rag_extract_size, total_count)

    print(f"전체 데이터 수: {total_count}개")
    print(f"RAG 추출 목표 수: {rag_extract_size}개")
    print(f"GOLDEN 샘플 목표 수: {golden_size}개")

    # README 파일 초기화
    readme_files_with_dir = [(GOLDEN_SUBDIR, README_GOLDEN_FILE), (IN_SUBDIR, README_RAG_FILE)]
    print("\n--- 기존 README 파일 삭제 시작 ---")
    for subdir, file_name in readme_files_with_dir:
        full_path = os.path.join(PREPARE_DIR, subdir, file_name)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"  > 파일 삭제 완료: {subdir}/{file_name}")
        else:
            print(f"  > 파일 없음: {subdir}/{file_name} (삭제 건너뛰기)")
    print("--- 기존 README 파일 삭제 완료 ---\n")

    # 데이터 분류 및 목표 개수 계산
    categorized = categorize_data(raw_data)
    target_pos, target_neg = calculate_sample_counts(rag_extract_size, categorized)

    print(f"\n[추출 목표] 스타일={EXTRACT_STYLE.name} 적용 결과:")
    print(f"- 목표 긍정 샘플 수: {target_pos}개")
    print(f"- 목표 부정 샘플 수: {target_neg}개")

    return raw_data, golden_size, rag_extract_size, (target_pos, target_neg), categorized


def _prepare_common_rag_samples(target_pos: int, target_neg: int, categorized: Dict[str, List[Dict[str, Any]]]) -> \
Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """공통 RAG 평가셋을 추출하고, ID 주입 및 오류 주입을 수행합니다."""

    # 1. RAG 평가 대상 항목 추출
    sampled_pos = random.sample(categorized['positive'], target_pos)
    sampled_neg = sample_negative_data(target_neg, categorized)

    common_rag_samples = sampled_pos + sampled_neg
    random.shuffle(common_rag_samples)

    # 2. RAG 데이터에 ID 부여
    for item in common_rag_samples:
        item['id'] = str(uuid.uuid4())
    print(f"\n[공통] RAG 평가 대상 샘플 {len(common_rag_samples)}개에 고유 ID(UUID)를 부여했습니다.")

    # 3. 오답 주입
    print("\n--- 공통 RAG 샘플 오류 주입 시작 ---")
    swap_rate_a = swap_values(common_rag_samples, key='a', min_rate=SWAP_A_MIN_RATE, max_rate=SWAP_A_MAX_RATE)
    swap_rate_c = swap_values(common_rag_samples, key='c', min_rate=SWAP_C_MIN_RATE, max_rate=SWAP_C_MAX_RATE)
    swap_info = {'a': swap_rate_a, 'c': swap_rate_c}
    print("--- 공통 RAG 샘플 오류 주입 완료 ---\n")

    # 4. 통계 출력
    print_statistics("[공통 RAG 샘플 (오류 주입 후) 통계 정보]", common_rag_samples, swap_info)

    return common_rag_samples, swap_info


def _generate_subset_scenario(common_rag_samples: List[Dict[str, Any]], golden_size: int):
    """시나리오 2: Subset 데이터셋을 생성하고 저장합니다."""

    print("\n\n" + "#" * 50)
    print("## [1/2] Subset 데이터셋 생성 시작 (RAG 평가셋 공유) ##")
    print("#" * 50)

    # 1. GOLDEN 샘플 추출 (Subset)
    golden_samples_subset = random.sample(common_rag_samples, golden_size)

    # 2. 통계 출력 및 저장
    print_statistics("[Subset GOLDEN 데이터셋 통계 정보]", golden_samples_subset)
    golden_output_name = f"{EXTRACT_STYLE.name}_golden_subset.jsonl"
    save_data(golden_samples_subset, golden_output_name, subdir=GOLDEN_SUBDIR)


def _generate_mutually_exclusive_golden(raw_data: List[Dict[str, Any]], golden_size: int, target_pos: int,
                                        target_neg: int, categorized: Dict[str, List[Dict[str, Any]]]):
    """시나리오 1: Mutually Exclusive 골든셋을 생성하고 저장합니다."""

    print("\n\n" + "#" * 50)
    print("## [2/2] Mutually Exclusive 데이터셋 생성 시작 (RAG 평가셋 공유) ##")
    print("#" * 50)

    # 1. GOLDEN 샘플 (Mutually Exclusive)의 긍정/부정 개수 계산 (RAG 추출 스타일 비율을 따라감)
    rag_extract_size = target_pos + target_neg
    if rag_extract_size > 0:
        ratio_to_rag = golden_size / rag_extract_size
        golden_target_pos = math.floor(target_pos * ratio_to_rag)
        golden_target_neg = golden_size - golden_target_pos
    else:
        golden_target_pos = golden_size // 2
        golden_target_neg = golden_size - golden_target_pos

    # 2. 원본 데이터에서 긍정/부정 개수를 맞춰 추출
    golden_target_pos = min(golden_target_pos, len(categorized['positive']))
    golden_target_neg = min(golden_target_neg, sum(len(v) for k, v in categorized.items() if k.startswith('neg')))

    # RAG 평가셋에 포함되지 않은 데이터를 사용해야 하지만, 통계 비율을 맞추기 위해
    # 원본 분류 데이터에서 다시 추출합니다. (통계적 비율 일치 우선)
    sampled_pos_ex = random.sample(categorized['positive'], golden_target_pos)
    sampled_neg_ex = sample_negative_data(golden_target_neg, categorized)

    golden_ex_candidates = sampled_pos_ex + sampled_neg_ex
    random.shuffle(golden_ex_candidates)

    # 3. 통계 출력 및 저장
    print_statistics("[Mutually Exclusive GOLDEN 데이터셋 통계 정보]", golden_ex_candidates)
    golden_output_name_ex = f"{EXTRACT_STYLE.name}_golden_mutually_ex.jsonl"
    save_data(golden_ex_candidates, golden_output_name_ex, subdir=GOLDEN_SUBDIR)


def _save_final_rag_file(common_rag_samples: List[Dict[str, Any]]):
    """최종 RAG 평가셋을 단일 파일로 저장합니다."""

    rag_samples = common_rag_samples
    rag_output_name = f"{EXTRACT_STYLE.name}_rag.jsonl"

    print("\n" + "=" * 50)
    print(f"**RAG 평가셋 최종 저장: {rag_output_name}**")
    print(f"**주의:** 이 파일은 '{EXTRACT_STYLE.name}_golden_subset.jsonl'의 상위셋이며,")
    print(f"두 시나리오의 평가셋 파일 내용이 동일함을 대표합니다.")
    print("=" * 50)

    save_data(rag_samples, rag_output_name, keys_to_keep=['id', 'q', 'c', 'a'], subdir=IN_SUBDIR)
    print("RAG 평가셋 데이터 저장이 완료되었습니다.")


def generate_samples():
    """요청된 샘플링 및 오류 주입 로직을 실행합니다. (Mutually Exclusive & Subset 방식 모두 생성)"""

    # 1. 초기화 및 크기 계산
    raw_data, golden_size, rag_extract_size, target_counts, categorized = _initialize_and_calculate_targets()
    target_pos, target_neg = target_counts

    # 2. 공통 RAG 샘플 준비 (오류 주입 및 ID 포함)
    common_rag_samples, swap_info = _prepare_common_rag_samples(
        target_pos, target_neg, categorized
    )

    # 3. 시나리오 2: Subset 데이터셋 생성
    _generate_subset_scenario(common_rag_samples, golden_size)

    # 4. 시나리오 1: Mutually Exclusive 골든셋 생성
    _generate_mutually_exclusive_golden(
        raw_data, golden_size, target_pos, target_neg, categorized
    )

    # 5. 최종 RAG 평가셋 저장 (단일 파일)
    _save_final_rag_file(common_rag_samples)

    print("\n**모든 데이터셋 생성이 완료되었습니다.**")


if __name__ == "__main__":
    generate_samples()