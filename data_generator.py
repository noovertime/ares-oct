#
# 지정된 파일의 질문/답, 근거/답을 일정 비율 섞어서 오답 생성
# 라벨 삭제
# 결국 q, c, a 남음
#
import os
import json
import random
import math
import uuid
from typing import List, Dict, Any

# config.py가 현재 디렉토리에 있다고 가정하고 임포트합니다.
try:
    from config import PREPARE_DIR, PREPARE_FILE_NAME, PREPARE_OUT_PREFIX
except ImportError:
    print("경고: config.py에서 필요한 상수를 가져올 수 없습니다. 임시 설정 사용.")
    DATA_ROOT = os.path.dirname(os.path.abspath(__file__))
    PREPARE_DIR = os.path.join(DATA_ROOT, "data", "prepare")
    PREPARE_FILE_NAME = "qna_1_train.jsonl"
    PREPARE_OUT_PREFIX = "random_sample"
    if not os.path.exists(PREPARE_DIR):
        os.makedirs(PREPARE_DIR, exist_ok=True)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """JSONL 파일에서 데이터를 로드합니다."""
    data = []
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

    swap_rate = random.uniform(min_rate, max_rate)
    num_pairs_to_attempt = math.floor(data_size * swap_rate / 2) * 2  # 시도할 횟수

    if num_pairs_to_attempt < 2:
        return

    # 교환에 시도할 무작위 인덱스 추출 (비복원 추출)
    swap_indices = random.sample(range(data_size), num_pairs_to_attempt)

    actual_swaps = 0

    # 값 교환
    print(f"  > '{key}' 값 교환 시도 ({swap_rate * 100:.2f}%):")
    for i in range(0, num_pairs_to_attempt, 2):
        idx1 = swap_indices[i]
        idx2 = swap_indices[i + 1]

        val1 = data[idx1][key]
        val2 = data[idx2][key]

        # 두 값이 서로 다른 경우에만 교환을 실행
        if val1 != val2:
            data[idx1][key], data[idx2][key] = val2, val1
            actual_swaps += 1

    print(f"  > '{key}' 값 {actual_swaps}쌍 (총 {actual_swaps * 2}개 항목)이 실제로 교환되었습니다.")


def print_statistics(data: List[Dict[str, Any]]):
    """
    제공된 데이터 리스트의 평가 유형별 통계를 화면에 출력합니다.
    """
    if not data:
        print("\n[통계 정보] - 데이터가 없어 통계를 출력할 수 없습니다.")
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
            value = str(item.get(key, -1))  # 값이 없으면 -1로 처리하여 카운트 방지
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

    print("\n" + "=" * 50)
    print("[GOLDEN 샘플 저장 전 통계 정보]")
    print(f"총 데이터 수: {len(data)}")
    print("=" * 50)

    print("| 평가 유형 | '0' 개수 (부정) | '1' 개수 (긍정) |")
    print("|:---------|:---------------------|:---------------------|")
    print(f"| L_CR (C) | {stats['L_CR']['0']:<16}개  | {stats['L_CR']['1']:<16}개  |")
    print(f"| L_AF (F) | {stats['L_AF']['0']:<16}개  | {stats['L_AF']['1']:<16}개  |")
    print(f"| L_AR (R) | {stats['L_AR']['0']:<16}개  | {stats['L_AR']['1']:<16}개  |")
    print("-" * 50)

    print("종합 판별 결과:")
    print(f"- 3개 유형 모두 '0'인 항목 수 (C=0 & F=0 & R=0): {all_zero_count}개")
    print(f"- 3개 유형 모두 '1'인 항목 수 (C=1 & F=1 & R=1): {all_one_count}개")
    print("=" * 50 + "\n")


def generate_samples():
    """요청된 샘플링 및 오류 주입 로직을 실행합니다."""

    # 1. 파일 로드 및 초기 설정
    raw_data = load_data(PREPARE_FILE_NAME)
    if not raw_data:
        return

    total_count = len(raw_data)
    golden_size = int(total_count * 0.05)
    rag_extract_size = int(total_count * 0.30)

    print(f"전체 데이터 수: {total_count}개")
    print(f"RAG 추출 목표 수 (30%): {rag_extract_size}개")
    print(f"GOLDEN 샘플 목표 수 (전체의 5%): {golden_size}개")

    # 2. RAG 샘플 1차 추출 (전체의 30%)
    if rag_extract_size > total_count:
        rag_extract_size = total_count

    temp_rag_samples = random.sample(raw_data, rag_extract_size)

    # 3. RAG 데이터에 ID 부여
    for item in temp_rag_samples:
        item['id'] = str(uuid.uuid4())
    print(f"\n[준비] RAG 초기 샘플 {len(temp_rag_samples)}개에 고유 ID(UUID)를 부여했습니다.")

    # 4. 통계 정보 화면 출력 (GOLDEN 파일 저장 전)
    # 통계 분석 대상은 ID가 부여된 30% 데이터입니다.
    print_statistics(temp_rag_samples)

    # 5. GOLDEN 샘플 추출 및 파일 저장
    if golden_size > rag_extract_size:
        golden_size = rag_extract_size

    golden_samples = random.sample(temp_rag_samples, golden_size)

    # GOLDEN 데이터는 오류 주입 전, 원본 상태 그대로 저장됩니다.
    golden_output_name = f"{PREPARE_OUT_PREFIX}_golden.jsonl"
    save_data(golden_samples, golden_output_name)
    print("GOLDEN 데이터 저장이 완료되었습니다. (오류 없음)")

    # 6. RAG 샘플 오류 주입 (30% 전체 대상)
    # golden은 이미 파일로 저장되어 있으므로, 오류 주입은 파일에 영향을 주지 않습니다.
    print("\n--- RAG 샘플 오류 주입 시작 (Negative Sampling) ---")

    # 오류 응답을 생성하기 위한 값 바꾸기 (a 값 교환: 3% ~ 10%)
    swap_values(temp_rag_samples, key='a', min_rate=0.03, max_rate=0.10)

    # 오류 응답을 생성하기 위한 값 바꾸기 (c 값 교환: 0% ~ 10%)
    swap_values(temp_rag_samples, key='c', min_rate=0.00, max_rate=0.10)

    print("--- RAG 샘플 오류 주입 완료 ---\n")

    # 7. RAG 데이터 최종 저장
    rag_samples = temp_rag_samples

    # RAG 데이터 저장 (id, q, c, a 만 남김)
    rag_output_name = f"{PREPARE_OUT_PREFIX}_rag.jsonl"
    save_data(rag_samples, rag_output_name, keys_to_keep=['id', 'q', 'c', 'a'])


if __name__ == "__main__":
    generate_samples()