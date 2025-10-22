import time
from typing import Dict, Any, List
from config import KEY_CR, KEY_AF, KEY_AR


def generate_summary_report(golden_markdown_map: Dict[str, str], model_summaries: List[Dict[str, Any]]) -> str:
    """
    Markdown 형식 보고서 생성.
    다중 골든셋 통계 (golden_markdown_map)와 다중 보정 결과 (model_summaries)를 처리합니다.
    """
    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없습니다."

    # 'overall' 점수를 기준으로 내림차순 정렬
    model_summaries.sort(key=lambda x: float(x["overall"]), reverse=True)
    current_time: str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 🚨 주의: 'labeled_n_rep'은 CR 축의 labeled_n을 대표값으로 사용하므로, 이 섹션은 단순화합니다.
    model_list: str = "\n".join([f"   - {m['model_name']} (vs. {m['golden_set_name']})" for m in model_summaries])

    # -------------------------------------------------------------
    # 1. 보고서 기본 정보 섹션
    # -------------------------------------------------------------
    report_content: str = f"""
## 🧭 ARES 결과 보고서
평가 일자: {current_time}

--- 
### 1️⃣ 프로젝트 개요 
- 프로젝트명: ARES 심사관 로컬 배치 평가
- 평가 프레임워크: Stanford ARES (골든셋 기반 PPI 보정 로직 통합)
- 평가 대상 (조합) : 평가셋(QCA) * 골든셋(보정통계) 트리플 셋으로 구성 <br>
{model_list}

--- 
### 2️⃣ 평가 
- Context Relevance (CR, 문맥 적합성) : 검색된 문서가 질문과 얼마나 관련 있는가
- Answer Faithfulness (AF, 응답 충실도) : 생성된 답변이 검색 문서 내용에 충실한가 
- Answer Relevance (AR, 응답 적절성) : 답변이 질문에 직접적이고 구체적인가

--- 
### 3️⃣ PPI 추정 성능 점수
#### 🎯 성능점수 요약 

| 순번 | 평가대상 | 적용 골든셋 | 종합 점수 | CR(보정) | AF(보정) | AR(보정)|
|:--|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    # -------------------------------------------------------------
    # 2. 요약 테이블 (마크다운) - 올바른 키 사용
    # -------------------------------------------------------------
    for i, summary in enumerate(model_summaries):
        report_content += (
            f"| {i + 1} "
            f"| {summary['model_name']} "
            f"| {summary['golden_set_name']} "
            f"| {summary['overall']:.2f} "  # 종합 점수
            f"| {summary[KEY_CR]['corrected_mean']:.2f} "  # CR(보정)
            f"| {summary[KEY_AF]['corrected_mean']:.2f} "  # AF(보정)
            f"| {summary[KEY_AR]['corrected_mean']:.2f} |\n"  # AR(보정)
        )

    report_content += """
    📝 의미 요약 
    * 종합점수 : 평가 대상 모델의 전반적인 성능 추정치 
    * CR/AR/AF(보정) : 문맥 적합성(CR), 응답 충실도(AF), 응답 적절성(AR) 성능 추정치


<br>

#### 🎯 성능 점수 세부 항목 값\n
    """

    report_content += """
<table>
  <thead>
    <tr>
        <td rowspan="2">순번</td>
        <td rowspan="2">평가대상</td>
        <td rowspan="2">골든셋</td>
        <td colspan="3" align="center">CR</td>
        <td colspan="3" align="center">AF</td>
        <td colspan="3" align="center">AR</td>
        <td rowspan="2">심사관 예측 평균</td>
        <td rowspan="2">총 샘플 수</td>
    </tr>
    <tr>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
    </tr>
  </thead>
  <tbody>
"""
    # -------------------------------------------------------------
    # 3. 세부내용 테이블 (HTML) - 올바른 키 사용
    # -------------------------------------------------------------
    for i, summary in enumerate(model_summaries):
        # 🚨 수정: 축 통계를 올바른 키로 조회
        cr = summary[KEY_CR]  # 'contextrelevance'
        af = summary[KEY_AF]  # 'answerfaithfulness'
        ar = summary[KEY_AR]  # 'answerrelevance'

        # 모델의 모든 기계 예측 평균 (CR 축의 machine_mean 사용)
        model_machine_mean: float = cr['machine_mean']
        golden_set_name: str = summary['golden_set_name']

        report_content += "    <tr>\n"
        report_content += f"        <td>{i + 1}</td>\n"
        report_content += f"        <td>{summary['model_name']}</td>\n"
        report_content += f"        <td>{golden_set_name}</td>\n"

        # CR
        report_content += f"        <td>{cr['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{cr['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{cr['ci']:.2f}</td>\n"

        # AF
        report_content += f"        <td>{af['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{af['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{af['ci']:.2f}</td>\n"

        # AR
        report_content += f"        <td>{ar['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{ar['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{ar['ci']:.2f}</td>\n"

        # 모델 예측 및 샘플 수
        report_content += f"        <td>{model_machine_mean:.2f}</td>\n"
        report_content += f"        <td>{summary['n']}</td>\n"
        report_content += "    </tr>\n"

    report_content += """
  </tbody>
</table>


    📝 의미 요약 
    * CR/AR/AF(보정) : 문맥 적합성(CR), 응답 충실도(AF), 응답 적절성(AR) 성능 추정치 ( 0 ~ 1 : 최고점 ) 
    * 편향 : ARES 심사관의 예측과 골든라벨의 평균 차이. 부호에 상관없이 0에 가까울수록 심사관의 평가가 인간평가(골든라벨)과 같음을 의미.
    * CI : CR/AR/AF(보정) 값의 신뢰구간 ( 작을수록 좋음 )
    * 심사관 예측평균 : ARES 심사관이 부여한 평균 점수 
    * 총 샘플 수 : 평가에 사용된 Q-C-A 트리플의 전체 개수

---
### 4️⃣ PPI 보정 의미 
* PPI(Prediction-Powered Inference) 역할 
  * PPI는 ARES 심사관 모델($\hat{Y}$)의 예측 편향을 제거하여 평가 결과의 신뢰도와 통계적 효율성을 높이는 방법론 
  * 효율성 결합 
    * ARES 심사관이 수행한 대규모 예측 ($\hat{Y}$)의 정보력과 고비용으로 얻은 소규모 골든셋 ($Y$)의 정확성을 결합 
  * 편향계산 (Rectifier)
    * 소수의 골든셋에서 심사관 예측과 골든라벨의 평균 차이를 계산해 편향 수정자로 사용
* PPI 보정 값의 의미 
  * PPI 보정 값은 PPI 방법론을 통해 산출된 모델의 성능 추정치 
  * 참된 성능 추정 (true performance)
    * ARES 심사관의 예측 점수에서 골든셋 기반의 편향이 제거된 모델의 참된 성능을 신뢰할 수 있게 추정한 값 
  * 계산공식 
    * 보정값 = ARES 심사관 모델 예측평균 - 편향 
  * 보고서 표시
      * CR(보정), AF(보정), AR(보정) 값에 해당 
* 보고서 값의 의미 상세
  * $\text{종합 점수} = \frac{\text{CR}(\text{보정}) + \text{AF}(\text{보정}) + \text{AR}(\text{보정})}{3}$
  * 보정 : 예측값에서 편향을 제거한 성능 추정치 
  * 편향 : 골든셋과 심사관 예측 사이의 격차 ( -1 ~ 1 이상 )
  * CI (Confidence Interval, 신뢰구간)
    * PPI 보정 값(CR보정, AF보정, AR보정)의 신뢰도와 정밀도를 나타냄 
    * 값이 작을수록 추정된 보정값에 대한 오차 범위가 좁아져 신뢰도가 높다는 것을 의미 
  * 심사관예측점수 
    * CR/AF/AR 항목을 평가한 점수의 평균
    * PPI 보정의 입력값이며 편향이 포함되어 있어 이 값만으로는 모델 성능을 대표하지 않음
    * 보정을 거친 후에 CR/AF/AR 보정값으로 표현됨 


---
### 5️⃣ 참고자료 
"""

    # -------------------------------------------------------------
    # 5. 참고자료 섹션 (다중 골든셋 마크다운 출력)
    # -------------------------------------------------------------
    for name, markdown_string in golden_markdown_map.items():
        # 골든셋 이름 섹션을 추가합니다.
        report_content += f"\n### 🔸 골든셋 통계: {name}\n"

        # 이전 출력 오류와 KeyError를 모두 해결하는 수정 사항입니다.
        report_content += markdown_string

    report_content += "\n"

    return report_content