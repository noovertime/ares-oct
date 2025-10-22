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
from typing import List, Dict, Any
from config import PREPARE_DIR, PREPARE_FILE_NAME, PREPARE_OUT_PREFIX


# 추출 스타일 정의
class ExtractType(Enum):
    """RAG 평가 대상 샘플의 긍정/부정 비율을 결정하는 추출 스타일"""
    BALANCE = 1  # 긍정:부정 = 50:50
    POSITIVE = 2  # 긍정 최대화 (부정 최소 10% 보장)
    NEGATIVE = 3  # 부정 최대화 (긍정 최소 0%~5%만 포함 가능성 있음)


# 추출 스타일
EXTRACT_STYLE = ExtractType.POSITIVE
# 추출 비율
GOLDEN_RATIO = 0.05
EXTRACT_RATIO = 0.50  # RAG 샘플 추출 비율을 50%로 상향 조정
MIN_GOLDEN_COUNT = 100  # 골든셋 최소 확보 목표
MIN_NEGATIVE_RATIO_POS = 0.10  # POSITIVE 스타일에서 최소 부정 비율

# 오답 생성비율
SWAP_A_MIN_RATE = 0.10
SWAP_A_MAX_RATE = 0.10
SWAP_C_MIN_RATE = 0.20
SWAP_C_MAX_RATE = 0.20

# **수정: README 파일 이름 상수화**
README_GOLDEN_FILE = "README_golden.txt"
README_RAG_FILE = "README_rag.txt"


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


def save_data(data: List[Dict[str, Any]], file_path: str, keys_to_keep: List[str] = None):
    """데이터를 JSONL 형식으로 저장합니다."""
    full_path = os.path.join(PREPARE_DIR, file_path)
    print(f"총 {len(data)}개의 데이터를 {full_path}에 저장합니다.")

    with open(full_path, 'w', encoding='utf-8') as f:
        for item in data:
            if keys_to_keep:
                filtered_item = {k: v for k, v in item.items() if k in keys_to_keep}
                f.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def swap_values(data: List[Dict[str, Any]], key: str, min_rate: float, max_rate: float):
    """
    데이터 리스트 내에서 특정 키(key)의 값을 무작위로 추출된 비율만큼 서로 교환합니다.
    (단, 두 값이 서로 다른 경우에만 교환을 진행합니다.)
    """
    data_size = len(data)
    if data_size < 2:
        return

    # 무작위 비율(0.0~1.0)을 사용
    swap_rate = random.uniform(min_rate, max_rate)

    # 시도할 횟수는 짝수 (교환 쌍)
    num_pairs_to_attempt = math.floor(data_size * swap_rate / 2) * 2
    num_pairs_to_attempt = min(num_pairs_to_attempt, data_size)

    if num_pairs_to_attempt < 2:
        print(f"  > '{key}' 값 교환 시도 ({swap_rate * 100:.2f}%): 교환 가능한 항목이 부족합니다 (0쌍).")
        return

    # 교환에 시도할 무작위 인덱스 추출 (비복원 추출)
    swap_indices = random.sample(range(data_size), num_pairs_to_attempt)

    actual_swaps = 0

    # 값 교환
    print(f"  > '{key}' 값 교환 시도 ({swap_rate * 100:.2f}%):")
    for i in range(0, num_pairs_to_attempt, 2):
        idx1 = swap_indices[i]
        idx2 = swap_indices[i + 1]

        val1 = data[idx1].get(key)
        val2 = data[idx2].get(key)

        # 두 값이 서로 다른 경우에만 교환을 실행
        if val1 is not None and val2 is not None and val1 != val2:
            data[idx1][key], data[idx2][key] = val2, val1
            actual_swaps += 1

    print(f"  > '{key}' 값 {actual_swaps}쌍 (총 {actual_swaps * 2}개 항목)이 실제로 교환되었습니다.")


def print_statistics(title: str, data: List[Dict[str, Any]]):
    """
    제공된 데이터 리스트의 평가 유형별 통계를 화면에 출력하고 파일로 저장합니다.
    (파일 저장 시 기존 내용에 덧붙여(append) 저장하도록 수정)
    """
    if not data:
        print(f"\n[{title}] - 데이터가 없어 통계를 출력/저장할 수 없습니다.")
        return

    stats = {
        'L_CR': {'0': 0, '1': 0},  # Context Relevance (CR)
        'L_AF': {'0': 0, '1': 0},  # Answer Faithfulness (AF)
        'L_AR': {'0': 0, '1': 0},  # Answer Relevance (AR)
    }
    all_zero_count = 0
    all_one_count = 0

    for item in data:
        # 각 유형별 카운트
        for key in stats.keys():
            value = str(item.get(key, -1))
            if value in stats[key]:
                stats[key][value] += 1

        # 종합 판별 카운트
        cr = item.get('L_CR')
        af = item.get('L_AF')
        ar = item.get('L_AR')

        if cr == 0 and af == 0 and ar == 0:
            all_zero_count += 1
        elif cr == 1 and af == 1 and ar == 1:
            all_one_count += 1

    total_data_count = len(data)
    # 1개라도 0인 항목 수 계산: 전체 항목 수 - 모두 1인 항목 수
    at_least_one_zero_count = total_data_count - all_one_count

    # 1. 통계 문자열 구성 (비율 추가)

    # 출력 제목에 따라 저장할 파일 이름을 결정합니다.
    if "GOLDEN" in title:
        file_name = README_GOLDEN_FILE  # 상수 사용
    else:  # 오류 주입된 RAG 샘플
        file_name = README_RAG_FILE  # 상수 사용

    output_lines = []

    output_lines.append("\n" + "=" * 50)
    output_lines.append(f"{title}")
    output_lines.append(f"총 데이터 수: {total_data_count}")
    output_lines.append(f"추출 스타일: {EXTRACT_STYLE.name}")
    output_lines.append("=" * 50)

    # 비율 계산 및 출력 포맷팅
    output_lines.append("| 평가 유형 | '0' 개수 (부정) | '1' 개수 (긍정) |")
    output_lines.append("|:---------|:----------------|:----------------|")

    # 각 셀의 너비를 15자로 고정합니다.
    CELL_WIDTH = 15

    for key, label in [('L_CR', 'C'), ('L_AF', 'F'), ('L_AR', 'R')]:
        count_0 = stats[key]['0']
        count_1 = stats[key]['1']

        # 비율 계산
        ratio_0 = (count_0 / total_data_count) * 100 if total_data_count else 0
        ratio_1 = (count_1 / total_data_count) * 100 if total_data_count else 0

        # 출력 문자열 포맷팅 (오른쪽 정렬 적용)
        line_0 = f"{count_0} ({ratio_0:.2f}%)"
        line_1 = f"{count_1} ({ratio_1:.2f}%)"

        # 오른쪽 정렬 포맷을 사용하여 셀 너비를 맞춥니다.
        output_lines.append(f"| {key} ({label}) | {line_0:>{CELL_WIDTH}} | {line_1:>{CELL_WIDTH}} |")

    output_lines.append("-" * 50)

    output_lines.append("종합 판별 결과:")
    output_lines.append(f"- 3개 유형 모두 '0'인 항목 수 (C=0 & F=0 & R=0): {all_zero_count}개")
    output_lines.append(f"- 3개 유형 모두 '1'인 항목 수 (C=1 & F=1 & R=1): {all_one_count}개")
    output_lines.append(f"- 1개라도 '0'인 항목 수 : {at_least_one_zero_count}개")
    output_lines.append("=" * 50 + "\n")

    output_text = "\n".join(output_lines)

    # 2. 화면 출력
    print(output_text)

    # 3. 파일 저장 (PREPARE_DIR에 저장)
    full_path = os.path.join(PREPARE_DIR, file_name)
    try:
        # 'a' (append) 모드로 변경하여 기존 내용에 덧붙여 저장
        with open(full_path, 'a', encoding='utf-8') as f:
            f.write(output_text)
        print(f"[알림] 통계 정보를 '{full_path}'에 저장했습니다.")
    except Exception as e:
        print(f"[오류] 통계 정보 파일 저장 실패: {e}")


def categorize_data(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    데이터를 긍정(Positive) 및 부정(Negative)의 세 가지 유형으로 분류합니다.
    - Positive: L_CR=1, L_AF=1, L_AR=1
    - Negative_Type_3: 부정 라벨 3개 (000)
    - Negative_Type_2: 부정 라벨 2개 (001, 010, 100)
    - Negative_Type_1: 부정 라벨 1개 (011, 101, 110)
    """
    categorized = {
        'positive': [],
        'neg_type_3': [],
        'neg_type_2': [],
        'neg_type_1': [],
    }

    for item in data:
        cr = item.get('L_CR', 0)
        af = item.get('L_AF', 0)
        ar = item.get('L_AR', 0)

        num_negatives = (1 - cr) + (1 - af) + (1 - ar)  # 0의 개수

        if num_negatives == 0:
            categorized['positive'].append(item)
        elif num_negatives == 3:
            categorized['neg_type_3'].append(item)
        elif num_negatives == 2:
            categorized['neg_type_2'].append(item)
        elif num_negatives == 1:
            categorized['neg_type_1'].append(item)
        # else: 데이터가 이상한 경우 무시

    return categorized


def calculate_sample_counts(rag_target_size: int, categorized_data: Dict[str, List[Dict[str, Any]]]):
    """추출 스타일에 따라 긍정/부정 샘플 목표 개수를 계산합니다."""

    # 가용 데이터 크기
    pos_count = len(categorized_data['positive'])
    neg_total_count = sum(len(v) for k, v in categorized_data.items() if k.startswith('neg'))

    # 1. BALANCE (50:50)
    if EXTRACT_STYLE == ExtractType.BALANCE:
        target_pos = min(rag_target_size // 2, pos_count)
        target_neg = min(rag_target_size - target_pos, neg_total_count)
        # 긍정 샘플이 부족하면, 남은 공간을 부정 샘플로 채웁니다.
        target_neg = min(rag_target_size - target_pos, neg_total_count)

        # 2. POSITIVE (긍정 최대화, 부정 최소 10% 보장)
    elif EXTRACT_STYLE == ExtractType.POSITIVE:
        min_neg_count = max(1, math.floor(rag_target_size * MIN_NEGATIVE_RATIO_POS))

        target_neg = min(min_neg_count, neg_total_count)
        target_pos = min(rag_target_size - target_neg, pos_count)
        # 긍정 샘플이 부족하면, 남은 공간을 부정 샘플로 채웁니다.
        target_neg = min(rag_target_size - target_pos, neg_total_count)

        # 3. NEGATIVE (부정 최대화)
    elif EXTRACT_STYLE == ExtractType.NEGATIVE:
        target_neg = min(rag_target_size, neg_total_count)
        target_pos = min(rag_target_size - target_neg, pos_count)
        # 부정 샘플이 부족하면, 남은 공간을 긍정 샘플로 채웁니다.
        target_pos = min(rag_target_size - target_neg, pos_count)

    else:  # 기본은 BALANCE
        return calculate_sample_counts(rag_target_size, categorized_data)

    return target_pos, target_neg


def sample_negative_data(target_neg: int, categorized_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    부정 샘플을 target_neg 개수만큼 3-2-1 순서로 우선 추출합니다.
    """
    sampled_neg = []

    # 순위: 3 -> 2 -> 1
    neg_keys = ['neg_type_3', 'neg_type_2', 'neg_type_1']

    for key in neg_keys:
        if len(sampled_neg) >= target_neg:
            break

        remaining_needed = target_neg - len(sampled_neg)
        available_data = categorized_data[key]

        # 남은 필요 개수와 가용 데이터 중 더 작은 개수만큼 추출
        num_to_sample = min(remaining_needed, len(available_data))

        if num_to_sample > 0:
            # 비복원 추출하여 리스트에 추가
            sampled_neg.extend(random.sample(available_data, num_to_sample))

    return sampled_neg


def generate_samples():
    """요청된 샘플링 및 오류 주입 로직을 실행합니다. (Mutually Exclusive & Subset 방식 모두 생성)"""

    # 1. 파일 로드 및 초기 설정
    raw_data = load_data(PREPARE_FILE_NAME)
    if not raw_data:
        return

    total_count = len(raw_data)

    # **원본 데이터 부족 시 프로그램 종료**
    if total_count < MIN_GOLDEN_COUNT:
        print(f"\n[오류] 전체 데이터 수({total_count}개)가 최소 골든셋 요구치({MIN_GOLDEN_COUNT}개) 미만입니다.")
        print("프로그램을 종료합니다.")
        sys.exit(1)

    # 2. 골든셋 및 RAG 추출 크기 결정 (최소 100건 보장 로직 포함)
    golden_size = max(int(total_count * GOLDEN_RATIO), MIN_GOLDEN_COUNT)
    golden_size = min(golden_size, total_count)

    rag_extract_size_by_ratio = int(total_count * EXTRACT_RATIO)
    rag_extract_size = max(rag_extract_size_by_ratio, golden_size)
    rag_extract_size = min(rag_extract_size, total_count)

    print(f"전체 데이터 수: {total_count}개")
    print(f"RAG 추출 목표 수 (최소 50% 보장 및 GOLDEN 포함): {rag_extract_size}개")
    print(f"GOLDEN 샘플 목표 수 (최소 {MIN_GOLDEN_COUNT}개 보장): {golden_size}개")

    # 3. 데이터 분류 (추출 스타일을 위한 전처리)
    categorized = categorize_data(raw_data)

    # 4. RAG 샘플의 긍정/부정 개수 계산 (스타일 적용)
    target_pos, target_neg = calculate_sample_counts(rag_extract_size, categorized)

    print(f"\n[추출 목표] 스타일={EXTRACT_STYLE.name} 적용 결과:")
    print(f"- 목표 긍정 샘플 수: {target_pos}개 (가용 {len(categorized['positive'])}개)")
    print(f"- 목표 부정 샘플 수: {target_neg}개 (가용 {sum(len(v) for k, v in categorized.items() if k.startswith('neg'))}개)")

    # =========================================================================
    # [공통 단계] RAG 평가 대상 항목 추출 및 오답 주입 (두 시나리오에 공통 사용)
    # =========================================================================

    # **수정: README 파일 초기화 로직 추가**
    readme_files = [README_GOLDEN_FILE, README_RAG_FILE]

    print("\n--- 기존 README 파일 삭제 시작 ---")
    for file_name in readme_files:
        full_path = os.path.join(PREPARE_DIR, file_name)
        if os.path.exists(full_path):
            os.remove(full_path)
            print(f"  > 파일 삭제 완료: {file_name}")
        else:
            print(f"  > 파일 없음: {file_name} (삭제 건너뛰기)")
    print("--- 기존 README 파일 삭제 완료 ---\n")

    # 5. RAG 평가 대상 항목 추출 (temp_rag_samples 생성)
    sampled_pos = random.sample(categorized['positive'], target_pos)
    sampled_neg = sample_negative_data(target_neg, categorized)

    # RAG 평가 대상 리스트 (두 시나리오에 공통 사용될 원본)
    common_rag_samples = sampled_pos + sampled_neg
    random.shuffle(common_rag_samples)

    # 6. RAG 데이터에 ID 부여
    for item in common_rag_samples:
        item['id'] = str(uuid.uuid4())
    print(f"\n[공통] RAG 평가 대상 샘플 {len(common_rag_samples)}개에 고유 ID(UUID)를 부여했습니다.")

    # 7. 오답 주입 (두 시나리오의 RAG 평가셋에 모두 적용될 최종 상태)
    print("\n--- 공통 RAG 샘플 오류 주입 시작 ---")
    swap_values(common_rag_samples, key='a', min_rate=SWAP_A_MIN_RATE, max_rate=SWAP_A_MAX_RATE)
    swap_values(common_rag_samples, key='c', min_rate=SWAP_C_MIN_RATE, max_rate=SWAP_C_MAX_RATE)
    print("--- 공통 RAG 샘플 오류 주입 완료 ---\n")

    # RAG 평가셋 통계 출력 (오류 주입 완료된 최종 상태)
    print_statistics("[공통 RAG 샘플 (오류 주입 후) 통계 정보]", common_rag_samples)

    # =========================================================================
    # [시나리오 2] 서브셋 (Subset) 데이터셋 생성 (기존 로직 유지)
    # =========================================================================
    print("\n\n" + "#" * 50)
    print("## [1/2] Subset 데이터셋 생성 시작 (RAG 평가셋 공유) ##")
    print("#" * 50)

    # 8. GOLDEN 샘플 추출 (Subset)
    # ID와 오답 주입이 완료된 common_rag_samples에서 golden_size만큼 추출
    golden_samples_subset = random.sample(common_rag_samples, golden_size)

    # 9. GOLDEN 데이터셋 통계 출력 및 저장
    print_statistics("[Subset GOLDEN 데이터셋 통계 정보]", golden_samples_subset)
    golden_output_name = f"{EXTRACT_STYLE.name}_golden_subset.jsonl"
    save_data(golden_samples_subset, golden_output_name)

    # 10. RAG 데이터 최종 저장 (공통 샘플을 rag_subset.jsonl로 저장)
    rag_samples_subset = common_rag_samples
    rag_output_name = f"{EXTRACT_STYLE.name}_rag.jsonl"
    save_data(rag_samples_subset, rag_output_name, keys_to_keep=['id', 'q', 'c', 'a'])
    print("Subset 데이터 저장이 완료되었습니다. (오류 없음)")

    # =========================================================================
    # [시나리오 1] 상호 배제 (Mutually Exclusive) 데이터셋 생성
    # =========================================================================
    print("\n\n" + "#" * 50)
    print("## [2/2] Mutually Exclusive 데이터셋 생성 시작 (RAG 평가셋 공유) ##")
    print("#" * 50)

    # 11. GOLDEN 샘플 (Mutually Exclusive) 추출
    # 원본 raw_data에서 golden_size만큼 추출
    # raw_data에서 ID가 없는 원본 항목을 추출합니다. (RAG 평가셋과 겹칠 수 있으나, ID가 없어 추출 자체는 가능)
    golden_ex_candidates = random.sample(raw_data, golden_size)

    # 12. 통계 및 저장
    print_statistics("[Mutually Exclusive GOLDEN 데이터셋 통계 정보]", golden_ex_candidates)
    golden_output_name_ex = f"{EXTRACT_STYLE.name}_golden_mutually_ex.jsonl"
    save_data(golden_ex_candidates, golden_output_name_ex)

    #
    # # 13. RAG 데이터 최종 저장 (공통 샘플을 rag_mutually_ex.jsonl로 저장)
    # # rag_subset과 동일한 내용을 저장하여 "평가셋은 같은 내용"이라는 요구사항을 충족합니다.
    # rag_samples_ex = common_rag_samples
    # rag_output_name_ex = f"{EXTRACT_STYLE.name}_rag_mutually_ex.jsonl"
    # save_data(rag_samples_ex, rag_output_name_ex, keys_to_keep=['id', 'q', 'c', 'a'])
    # print("Mutually Exclusive 데이터 저장이 완료되었습니다. (오류 없음)")
    #print("\n**두 시나리오의 RAG 평가셋 (rag_subset.jsonl, rag_mutually_ex.jsonl) 내용이 동일합니다.**")


if __name__ == "__main__":
    generate_samples()