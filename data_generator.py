# data_generator.py

import os
import json
import random
import math
import uuid
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
    golden_size = int(total_count * 0.05)  # 전체 데이터의 5%
    rag_extract_size = int(total_count * 0.30)  # 전체 데이터의 30%를 RAG로 추출

    print(f"전체 데이터 수: {total_count}개")
    print(f"RAG 초기 추출 목표 수 (30%): {rag_extract_size}개")
    print(f"GOLDEN 샘플 목표 수 (전체의 5%): {golden_size}개")

    # 2. RAG 샘플 1차 추출 (전체의 30%)
    if rag_extract_size > total_count:
        rag_extract_size = total_count

    # random.sample을 사용하여 비복원 추출
    temp_rag_samples = random.sample(raw_data, rag_extract_size)

    # 3. RAG 데이터에 ID 부여 (ID가 golden에도 포함되도록 먼저 부여)
    for item in temp_rag_samples:
        # q, c, a 만 남기기 전에 모든 속성이 포함된 상태에서 id를 추가
        item['id'] = str(uuid.uuid4())
    print(f"RAG 초기 샘플 {len(temp_rag_samples)}개에 고유 ID(UUID)를 부여했습니다.")

    # 4. GOLDEN 샘플 추출 (ID가 부여된 temp_rag_samples 내에서 전체의 5%만큼 추출)
    if golden_size > rag_extract_size:
        print(f"경고: GOLDEN 목표 크기({golden_size})가 RAG 추출 크기({rag_extract_size})보다 큽니다. RAG 크기에 맞춥니다.")
        golden_size = rag_extract_size

    # ID가 부여된 temp_rag_samples 내에서 golden_size 만큼을 무작위로 추출
    golden_samples = random.sample(temp_rag_samples, golden_size)

    # 5. RAG 샘플 최종 구성 (GOLDEN과 겹치지 않도록 ID를 기준으로 중복 제거)
    # GOLDEN으로 추출된 항목의 ID를 저장
    golden_ids = set(item['id'] for item in golden_samples)

    # GOLDEN에 포함되지 않은 데이터만 최종 RAG 샘플로 구성
    rag_samples = [item for item in temp_rag_samples if item['id'] not in golden_ids]

    print(f"\n최종 RAG 샘플 수 (약 25%): {len(rag_samples)}개")
    print(f"최종 GOLDEN 샘플 수 (5%): {len(golden_samples)}개")

    print("\n--- RAG 샘플 오류 주입 시작 (Negative Sampling) ---")

    # 6. RAG 샘플에 오류 주입 (a 값 교환)
    # 오류 응답을 생성하기 위한 값 바꾸기
    swap_values(rag_samples, key='a', min_rate=0.03, max_rate=0.10)

    # 7. RAG 샘플에 오류 주입 (c 값 교환)
    # 오류 응답을 생성하기 위한 값 바꾸기
    swap_values(rag_samples, key='c', min_rate=0.00, max_rate=0.10)

    print("--- RAG 샘플 오류 주입 완료 ---\n")

    # 8. RAG 데이터 저장 (id, q, c, a 만 남김)
    rag_output_name = f"{PREPARE_OUT_PREFIX}_rag.jsonl"
    # keys_to_keep에 'id' 포함
    save_data(rag_samples, rag_output_name, keys_to_keep=['id', 'q', 'c', 'a'])

    # 9. GOLDEN 데이터 저장 (모든 속성 유지, ID 포함)
    golden_output_name = f"{PREPARE_OUT_PREFIX}_golden.jsonl"
    save_data(golden_samples, golden_output_name)


if __name__ == "__main__":
    generate_samples()