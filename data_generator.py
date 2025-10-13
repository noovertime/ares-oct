# data_generator.py

import os
import json
import random
import math
from typing import List, Dict, Any

# config.py가 현재 디렉토리에 있다고 가정하고 임포트합니다.
try:
    # 전제조건: config.py에 다음과 같은 상수가 있음
    from config import PREPARE_DIR, PREPARE_FILE_NAME, PREPARE_OUT_PREFIX
except ImportError:
    print("경고: config.py에서 필요한 상수를 가져올 수 없습니다.")
    print("PREPARE_DIR, PREPARE_FILE_NAME, PREPARE_OUT_PREFIX를 임시로 설정합니다.")
    # 테스트를 위한 임시 상수 설정
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
                    print(f"JSON 디코딩 오류 발생: {e} - 라인: {line.strip()[:50]}...")
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로: {full_path}")
        return []
    return data


def save_data(data: List[Dict[str, Any]], file_path: str, keys_to_keep: List[str] = None):
    """데이터를 JSONL 형식으로 저장합니다."""
    full_path = os.path.join(PREPARE_DIR, file_path)
    print(f"총 {len(data)}개의 데이터를 {full_path}에 저장합니다.")

    # 디렉토리가 없으면 생성 (PREPARE_DIR이 존재한다는 전제)

    with open(full_path, 'w', encoding='utf-8') as f:
        for item in data:
            if keys_to_keep:
                # 지정된 키만 남기도록 필터링합니다.
                filtered_item = {k: v for k, v in item.items() if k in keys_to_keep}
                f.write(json.dumps(filtered_item, ensure_ascii=False) + '\n')
            else:
                # 모든 키를 저장합니다.
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def swap_values(data: List[Dict[str, Any]], key: str, min_rate: float, max_rate: float):
    """
    데이터 리스트 내에서 특정 키(key)의 값을 무작위로 추출된 비율(min_rate ~ max_rate)만큼 서로 교환합니다.
    """
    data_size = len(data)
    if data_size < 2:
        return  # 교환할 데이터가 충분하지 않음

    # 1. 교환할 비율 결정 (min_rate와 max_rate 사이의 균일 분포)
    swap_rate = random.uniform(min_rate, max_rate)

    # 2. 교환할 항목 수 계산 (짝수 개로 보장하기 위해 / 2 후 * 2)
    # 총 교환할 데이터 쌍의 수
    num_pairs = math.floor(data_size * swap_rate / 2)
    num_items_to_swap = num_pairs * 2  # 실제 교환에 사용될 데이터 항목 수

    if num_items_to_swap < 2:
        return

    # 3. 교환에 사용할 무작위 인덱스 추출 (비복원 추출)
    swap_indices = random.sample(range(data_size), num_items_to_swap)

    # 4. 값 교환
    print(f"  > '{key}' 값 {swap_rate * 100:.2f}% (총 {num_pairs}쌍) 교환 시작.")
    for i in range(num_pairs):
        # 쌍으로 묶어 교환
        idx1 = swap_indices[2 * i]
        idx2 = swap_indices[2 * i + 1]

        # 값 교환
        data[idx1][key], data[idx2][key] = data[idx2][key], data[idx1][key]


def generate_samples():
    """요청된 비율에 따라 중복 없이 데이터를 샘플링하고 오류를 주입하여 저장합니다."""

    input_file_name = PREPARE_FILE_NAME

    # 1. 파일 로드
    raw_data = load_data(input_file_name)
    if not raw_data:
        print("로딩된 데이터가 없어 처리를 중단합니다.")
        return

    total_count = len(raw_data)

    # 목표 샘플 크기 계산
    golden_size = int(total_count * 0.05)  # 5%
    rag_size = int(total_count * 0.30)  # 30%

    print(f"전체 데이터 수: {total_count}개")
    print(f"GOLDEN 샘플 목표 수 (5%): {golden_size}개")
    print(f"RAG 샘플 목표 수 (30%): {rag_size}개")

    # 2. 데이터 분할: 중복 방지를 위해 인덱스를 무작위로 섞습니다.
    indices = list(range(total_count))
    random.shuffle(indices)

    # 3. GOLDEN (5%) 데이터 추출
    golden_indices = indices[:golden_size]
    golden_samples = [raw_data[i] for i in golden_indices]

    # 4. RAG (30%) 데이터 추출 (GOLDEN과 겹치지 않음)
    remaining_indices = indices[golden_size:]

    rag_indices = remaining_indices[:rag_size]
    rag_samples = [raw_data[i] for i in rag_indices]

    print("\n--- RAG 샘플 오류 주입 시작 (Negative Sampling) ---")

    # 5. RAG 샘플에 오류 주입 (a 값 교환)
    # 데이터의 3 ~ 10%를 추출해서 서로의 a 값을 바꿔주세요.
    # 오류 응답을 생성하기 위한 값 바꾸기
    swap_values(rag_samples, key='a', min_rate=0.03, max_rate=0.10)

    # 6. RAG 샘플에 오류 주입 (c 값 교환)
    # 데이터의 0 ~ 10%를 추출해서 서로의 c 값을 바꿔주세요.
    # 오류 응답을 생성하기 위한 값 바꾸기
    swap_values(rag_samples, key='c', min_rate=0.00, max_rate=0.10)

    print("--- RAG 샘플 오류 주입 완료 ---\n")

    # 7. RAG 데이터 저장 (q, c, a 만 남김)
    rag_output_name = f"{PREPARE_OUT_PREFIX}_rag.jsonl"
    save_data(rag_samples, rag_output_name, keys_to_keep=['q', 'c', 'a'])

    # 8. GOLDEN 데이터 저장 (모든 속성 유지)
    golden_output_name = f"{PREPARE_OUT_PREFIX}_golden.jsonl"
    save_data(golden_samples, golden_output_name)


if __name__ == "__main__":
    generate_samples()