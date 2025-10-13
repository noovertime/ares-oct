# config.py

import os

# 모델 관련 설정 (ARES 심사관 로딩용)
MODEL_NAME = "monologg/distilkobert"
MODEL_DIR = r"C:\dev\workspaces\pycharm\ares-oct\model"

# 데이터 경로 설정 (상대 경로 기준)
DATA_ROOT = r"./data"

# 데이터 준비용
PREPARE_DIR = os.path.join(DATA_ROOT, "prepare")


# ARES 평가 관련 데이터
DATA_DIR = DATA_ROOT + "/set1"
# ARES 평가 프로세스 경로
DATA_IN_DIR = os.path.join(DATA_DIR, "in")        # 1. 배치 평가 입력 (QCA 트리플)
DATA_OUT_DIR = os.path.join(DATA_DIR, "out")      # 2. 배치 평가 출력 (PPI *.jsonl 파일)
DATA_REPORT_DIR = os.path.join(DATA_DIR, "report") # 3. 최종 통계 보고서 출력 (.md 파일)
DATA_GOLDEN_DIR = os.path.join(DATA_DIR, "golden") # 4. 골든셋

#
DATA_SET1_GOLDEN_FILE_NAME = "qna_1_test.jsonl"