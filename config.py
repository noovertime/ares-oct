# config.py

import os

# 모델 관련 설정 (ARES 심사관 로딩용)
MODEL_NAME = "monologg/distilkobert"
MODEL_DIR = r"./model"

# 데이터 경로 설정 (상대 경로 기준)
DATA_ROOT = r"./data"

# 데이터 준비용
PREPARE_DIR = os.path.join(DATA_ROOT, "prepare")
#PREPARE_FILE_NAME = "qna_1_train.jsonl"
PREPARE_FILE_NAME = "qna_3_test.jsonl"
PREPARE_OUT_PREFIX = "random_sample"

# ARES 평가 관련 데이터
#DATA_DIR = DATA_ROOT + "/set1"
DATA_DIR = DATA_ROOT + "/set2"
# ARES 평가 프로세스 경로
DATA_IN_DIR = os.path.join(DATA_DIR, "in")        # 1. 배치 평가 입력 (QCA 트리플)
DATA_OUT_DIR = os.path.join(DATA_DIR, "out")      # 2. 배치 평가 출력 (PPI *.jsonl 파일)
DATA_REPORT_DIR = os.path.join(DATA_DIR, "report") # 3. 최종 통계 보고서 출력 (.md 파일)
DATA_GOLDEN_DIR = os.path.join(DATA_DIR, "golden") # 4. 골든셋

#
DATA_GOLDEN_FILE_NAME = "random_sample_golden.jsonl"



KEY_CR = 'contextrelevance'
KEY_AF = 'answerfaithfulness'
KEY_AR = 'answerrelevance'

# ARES 심사관 타입 리스트
JUDGE_TYPES = [KEY_CR, KEY_AF, KEY_AR]

# 예측 결과 필드 명칭 정의 (키와 값이 동일함)
JUDGE_PREDICTION_FIELDS = {
    KEY_CR: KEY_CR,
    KEY_AF: KEY_AF,
    KEY_AR: KEY_AR
}

# 골든 라벨 필드 명칭 정의
GOLD_LABEL_FIELDS = {
    KEY_CR: 'L_CR',
    KEY_AF: 'L_AF',
    KEY_AR: 'L_AR'
}
