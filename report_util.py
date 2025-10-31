import time
from typing import Dict, Any, List, Set
from config import KEY_CR, KEY_AF, KEY_AR

# 축 순서를 위한 리스트 (편의상 여기서 정의)
JUDGE_TYPES_ORDERED = [KEY_CR, KEY_AF, KEY_AR]


# ------------------------------------------------------------------
# 헬퍼 함수 1: 골든셋 통계 데이터 마크다운 포맷팅
# ------------------------------------------------------------------

def _format_golden_stats_to_md(golden_report_data: Dict[str, Dict[str, float | str]]) -> str:
    """
    순수 데이터 딕셔너리(mean_pred, accuracy)를 받아 Markdown 테이블 (GFM 구문) 문자열로 변환합니다.
    """
    # 데이터 추출 및 포맷팅 (CR, AF, AR 순서 보장)
    cr = golden_report_data.get(KEY_CR, {})
    af = golden_report_data.get(KEY_AF, {})
    ar = golden_report_data.get(KEY_AR, {})

    # 숫자 값 포맷팅 헬퍼 (GFM에 맞게 정렬 포맷 유지)
    def format_val(val, is_acc=False):
        if val == 'N/A':
            return 'N/A'
        if is_acc:
            return f"{float(val):.3f}"
        return f"{float(val):.3f}"

    # 데이터 로우 값 준비
    cr_mean = format_val(cr.get('mean_pred', 'N/A'))
    af_mean = format_val(af.get('mean_pred', 'N/A'))
    ar_mean = format_val(ar.get('mean_pred', 'N/A'))

    cr_acc = format_val(cr.get('accuracy', 'N/A'), is_acc=True)
    af_acc = format_val(af.get('accuracy', 'N/A'), is_acc=True)
    ar_acc = format_val(ar.get('accuracy', 'N/A'), is_acc=True)

    # 1. 헤더 (Header)
    header = f"| 구분 | CR | AF | AR | 비고 |"

    # 2. 구분선 (Separator) - GFM 표준 중앙 정렬 사용
    separator = f"|:---:|:---:|:---:|:---:|:---|"

    # 3. 예측 평균 (Mean Row)
    mean_row = (
        f"| 예측평균 | {cr_mean} | {af_mean} | {ar_mean} |"
        f" 심사관이 1이라고 예측한 비율 (긍정 편향) |"
    )

    # 4. 정답 비율 (Accuracy Row)
    acc_row = (
        f"| 정답비율 | {cr_acc} | {af_acc} | {ar_acc} |"
        f" 심사관 예측의 정확도 |"
    )

    markdown_content = "\n".join([header, separator, mean_row, acc_row])

    return markdown_content


# ------------------------------------------------------------------
# 헬퍼 함수 2: 확신도 통계 테이블 생성 (기술 통계량)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# 헬퍼 함수 2: 확신도 통계 테이블 생성 (기술 통계량)
# ------------------------------------------------------------------

def _get_axis_label(axis_key: str) -> str:
    """긴 축 키(e.g., 'contextrelevance')를 짧은 약어(e.g., 'CR')로 변환합니다."""
    if axis_key == KEY_CR: return "CR"
    if axis_key == KEY_AF: return "AF"
    if axis_key == KEY_AR: return "AR"
    return axis_key


def _create_confidence_table(
        subject_list: List[Dict[str, Any]],
        source_type: str  # '평가셋' 또는 '골든셋'
) -> str:
    """
    고유한 평가 대상(subject) 리스트를 받아 Softmax 확신도 기술 통계량을
    CR, AF, AR 통합 테이블로 생성합니다.
    """

    # Softmax 통계 지표 명칭 축약
    header_mapping = {
        'prob_pos_avg': 'P+Avg',
        'prob_pos_median': 'P+Med',
        'conf_pos_avg': 'Conf+',
        'conf_neg_avg': 'Conf-',
        'mean_margin': 'Margin',
        'prob_pos_min': 'P+Min',
        'prob_pos_max': 'P+Max',
        'prob_neg_min': 'P-Min',
        'prob_neg_median': 'P-Med',
        'prob_neg_max': 'P-Max',
    }

    # 🚨 수정: 요청된 최종 컬럼 순서 (P-Avg 제외, 10개 컬럼)
    all_stats_cols = [
        'prob_pos_avg', 'conf_pos_avg', 'conf_neg_avg', 'mean_margin',
        'prob_pos_min', 'prob_pos_median', 'prob_pos_max',
        'prob_neg_min', 'prob_neg_median', 'prob_neg_max'
    ]

    table_content = ""

    for subject_data in subject_list:
        subject_name = subject_data['name']

        # 테이블 제목: '대상명 (구분) 확신도 통계' 형태
        table_content += f"\n#### 🔸 {subject_name} ({source_type}) 확신도 통계 \n"

        # 1. 헤더 (기준(Axis)을 행으로 가져옴)
        header_labels = [header_mapping[key] for key in all_stats_cols]
        header = "| 순번 | 기준 | " + " | ".join(header_labels) + " |\n"
        separator = "|:---:|:---:|" + ":---:|" * len(all_stats_cols)

        table_content += header
        table_content += separator + "\n"

        # 2. 데이터 로우 구성 (행: CR, AF, AR)
        for i, axis_key in enumerate(JUDGE_TYPES_ORDERED):
            # subject_data의 축 키에서 통계 딕셔너리를 추출
            stats = subject_data.get(axis_key, {})

            # 축 약어 결정 (CR, AF, AR)
            axis_label = _get_axis_label(axis_key)

            row = f"| {i + 1} | {axis_label} |"

            # 통계 값 출력 (요청된 10개 컬럼 순서대로)
            for key_short in all_stats_cols:
                value = stats.get(key_short, 0.0)
                row += f" {value:.4f} |"

            table_content += row + "\n"

    return table_content


# ------------------------------------------------------------------
# 헬퍼 함수 3: Softmax 확률 분포 시각화 (히스토그램)
# ------------------------------------------------------------------

def _create_bar(value: float, max_perc: float, length: int = 20) -> str:
    """확률 값(%)을 ASCII 막대로 변환합니다. max_perc에 따라 정규화됩니다."""
    if max_perc <= 0: return ' ' * length

    normalized_value = value / max_perc

    filled_count = int(round(normalized_value * length))
    empty_count = length - filled_count

    return '█' * filled_count + '░' * empty_count


def _create_distribution_chart(data_name: str, bins_data: List[Dict[str, Any]], axis_label: str) -> str:
    """
    Softmax Binning 데이터를 사용하여 분포 차트(막대 그래프)를 Markdown으로 생성합니다.
    (이 함수는 개별 축 차트 생성에 사용)
    """

    # 보고서 헤더
    title = f"{data_name} - {axis_label} 축"
    table_content = f"\n### 📊 Softmax 확률 분포 ({title})\n"

    # Markdown 테이블 헤더
    header = "| 확률 구간 | 샘플 수 | 비율(%) | 분포 막대 |\n"
    separator = "|:---:|:---:|:---:|:---:|\n"

    table_content += header
    table_content += separator

    # 데이터 로우 구성
    if not bins_data:
        table_content += "| N/A | 0 | 0.0 | ░░░░░░░░░░░░░░░░░░░░ |\n"
        return table_content

    # max_perc는 모든 빈에서 공유되므로 첫 번째 요소에서 추출
    max_perc = bins_data[0]['max_perc'] if bins_data else 0

    for item in bins_data:
        bar = _create_bar(item['percentage'], max_perc)
        row = (
            f"| {item['range']} | {item['count']} | {item['percentage']:.1f} | {bar} |"
        )
        table_content += row + "\n"

    return table_content


# ------------------------------------------------------------------
# 헬퍼 함수 4 (신규): CR, AF, AR Softmax 분포 통합 차트 생성
# ------------------------------------------------------------------

def _create_integrated_chart(model_summary: Dict[str, Any], chart_title: str) -> str:
    """
    CR, AF, AR 세 축의 Softmax 확률 분포를 단일 Markdown 테이블로 통합하여 생성합니다.
    """

    # 데이터 키 결정
    is_golden = "(골든셋)" in chart_title
    bins_key = 'golden_prob_bins' if is_golden else 'prob_bins'

    # 보고서 헤더: chart_title을 바로 사용
    # 🚨 수정: 히스토그램은 6️⃣ Softmax 기술 통계량의 하위 항목이므로 ### 대신 ####을 사용합니다.
    table_content = f"\n#### 🔸 {chart_title}\n"

    # 헤더 및 구분선 (CR, AF, AR 통합)
    header = "| 확률 구간 | CR (%) | CR Bar | AF (%) | AF Bar | AR (%) | AR Bar |\n"
    separator = "|:---:|" + ":---:|" * 6

    table_content += header
    table_content += separator + "\n"

    # 모든 축 데이터와 최대 비율 추출
    axis_keys = JUDGE_TYPES_ORDERED  # 수정: 순서를 보장
    # prob_bins 데이터가 비어있을 경우 대비하여 빈 리스트 사용
    all_bins_data = {ax: model_summary[ax].get(bins_key, []) for ax in axis_keys}

    # 2. 통합 최대 비율(max_perc) 계산: 세 축의 모든 빈도 중 가장 높은 비율을 찾습니다.
    all_percentages = [item['percentage'] for ax_data in all_bins_data.values() for item in ax_data if ax_data]
    integrated_max_perc = max(all_percentages) if all_percentages else 0.0

    # 3. 데이터 로우 구성 (확률 구간은 CR 축을 기준으로 순회)
    # CR 데이터가 없으면 다른 축 데이터도 없다고 가정하고 N/A 출력
    if all_bins_data[KEY_CR]:
        for i, cr_item in enumerate(all_bins_data[KEY_CR]):
            row = f"| {cr_item['range']} |"

            for axis_key in axis_keys:
                item = all_bins_data[axis_key][i]
                perc = item['percentage']
                bar = _create_bar(perc, integrated_max_perc)  # 통합된 max_perc 사용

                row += f" {perc:.1f} | {bar} |"

            table_content += row + "\n"
    else:
        # 데이터가 없을 경우 N/A 로우 추가 (안전성 확보)
        table_content += "| N/A | 0.0 | ░░░░░░░░░░░░░░░░░░░░ | 0.0 | ░░░░░░░░░░░░░░░░░░░░ | 0.0 | ░░░░░░░░░░░░░░░░░░░░ |\n"

    return table_content


# ------------------------------------------------------------------
# 메인 보고서 생성 함수
# ------------------------------------------------------------------

def generate_summary_report(golden_report_data: Dict[str, Dict], model_summaries: List[Dict[str, Any]]) -> str:
    """
    Markdown 형식 보고서 생성.
    다중 골든셋 통계 (golden_report_data)와 다중 보정 결과 (model_summaries)를 처리합니다.
    """
    if not model_summaries:
        return "[WARN] 분석할 모델 데이터가 없습니다."

    # 🚨 추가된 스타일 변수
    BORDER_STYLE = "border-right: 1px solid #999;"
    HEADER_BOTTOM_STYLE = "border-bottom: 2px solid #333;"

    # 'overall' 점수를 기준으로 내림차순 정렬
    model_summaries.sort(key=lambda x: float(x["overall"]), reverse=True)
    current_time: str = time.strftime("%Y-%m-%d %H:%M:%S")

    # 🚨 수정: model_list는 조합 형태를 유지합니다.
    model_list: str = "\n".join([f"   - {m['model_name']} (vs. {m['golden_set_name']})" for m in model_summaries])

    # -------------------------------------------------------------
    # 1, 2, 3 섹션 내용 (PPI 성능 테이블까지)
    # -------------------------------------------------------------
    report_content = f"""
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

    report_content += f"""
> 📝 의미 요약 
> * 종합점수 : 평가 대상 모델의 전반적인 성능 추정치 
> * CR/AR/AF(보정) : 문맥 적합성(CR), 응답 충실도(AF), 응답 적절성(AR) 성능 추정치

<br>

#### 🎯 성능 점수 세부 항목 값\n
    """

    # -------------------------------------------------------------
    # 3. 세부내용 테이블 (HTML) - 헤더 및 본문 스타일 수정
    # -------------------------------------------------------------
    report_content += f"""
<table>
  <thead style="{HEADER_BOTTOM_STYLE}">
    <tr>
        <td rowspan="2">순번</td>
        <td rowspan="2">평가대상</td>
        <td rowspan="2" style="{BORDER_STYLE}">골든셋</td>
        <td colspan="3" align="center" style="{BORDER_STYLE}">CR</td>
        <td colspan="3" align="center" style="{BORDER_STYLE}">AF</td>
        <td colspan="3" align="center">AR</td>
        <td rowspan="2">심사관 예측 평균</td>
        <td rowspan="2">총 샘플 수</td>
    </tr>
    <tr>
        <td>보정</td>
        <td>편향</td>
        <td style="{BORDER_STYLE}">CI</td>
        <td>보정</td>
        <td>편향</td>
        <td style="{BORDER_STYLE}">CI</td>
        <td>보정</td>
        <td>편향</td>
        <td>CI</td>
    </tr>
  </thead>
  <tbody>
"""
    for i, summary in enumerate(model_summaries):
        cr = summary[KEY_CR]
        af = summary[KEY_AF]
        ar = summary[KEY_AR]
        model_machine_mean: float = cr['machine_mean']
        golden_set_name: str = summary['golden_set_name']

        report_content += "    <tr>\n"
        report_content += f"        <td>{i + 1}</td>\n"
        report_content += f"        <td>{summary['model_name']}</td>\n"

        # 🚨 수정: 골든셋 우측 경계선 적용
        report_content += f"        <td style=\"{BORDER_STYLE}\">{golden_set_name}</td>\n"

        # CR
        report_content += f"        <td>{cr['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{cr['applied_rectifier']:.3f}</td>\n"
        # 🚨 수정: CR CI 우측 경계선 적용
        report_content += f"        <td style=\"{BORDER_STYLE}\">{cr['ci']:.2f}</td>\n"

        # AF
        report_content += f"        <td>{af['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{af['applied_rectifier']:.3f}</td>\n"
        # 🚨 수정: AF CI 우측 경계선 적용
        report_content += f"        <td style=\"{BORDER_STYLE}\">{af['ci']:.2f}</td>\n"

        # AR (경계선 없음)
        report_content += f"        <td>{ar['corrected_mean']:.2f}</td>\n"
        report_content += f"        <td>{ar['applied_rectifier']:.3f}</td>\n"
        report_content += f"        <td>{ar['ci']:.2f}</td>\n"

        report_content += f"        <td>{model_machine_mean:.2f}</td>\n"
        report_content += f"        <td>{summary['n']}</td>\n"
        report_content += "    </tr>\n"

    report_content += """
  </tbody>
</table>


>📝 의미 요약 
>* PPI 보정 값의 의미 
>   * PPI 보정 값은 PPI 방법론을 통해 산출된 모델의 성능 추정치 (true performance)
>   * ARES 심사관의 예측 점수에서 골든셋 기반의 편향이 제거된 모델의 참된 성능을 신뢰할 수 있게 추정한 값 
>   * 계산공식 
>     * 보정값 = ARES 심사관 모델 예측평균 - 편향 
> * 보고서 표시
>   * CR(보정), AF(보정), AR(보정) 값에 해당 
> * 보고서 값의 의미 상세
>   * $\\text{{종합 점수}} = \\frac{{\\text{{CR}}(\\text{{보정}}) + \\text{{AF}}(\\text{{보정}}) + \\text{{AR}}(\\text{{보정}})}}{{3}}$
>   * 보정 : 예측값에서 편향을 제거한 성능 추정치 
>   * 편향 : 골든셋과 심사관 예측 사이의 격차 ( -1 ~ 1 이상 )
>   * CI (Confidence Interval, 신뢰구간)
> * PPI 보정 값(CR보정, AF보정, AR보정)의 신뢰도와 정밀도를 나타냄 
>   * 값이 작을수록 추정된 보정값에 대한 오차 범위가 좁아져 신뢰도가 높다는 것을 의미 
> * 심사관예측점수 
>   * CR/AF/AR 항목을 평가한 점수의 평균
>   * PPI 보정의 입력값이며 편향이 포함되어 있어 이 값만으로는 모델 성능을 대표하지 않음
>   * 보정을 거친 후에 CR/AF/AR 보정값으로 표현됨 


---
### 5️⃣ 참고자료 
"""

    # -------------------------------------------------------------
    # 5. 참고자료 섹션 - 골든셋 통계 (GFM 포맷팅)
    # -------------------------------------------------------------

    # 중복 출력을 피하기 위해 이미 출력된 골든셋 이름을 추적합니다.
    processed_golden_sets: Set[str] = set()

    for name, report_data in golden_report_data.items():
        if name in processed_golden_sets:
            continue

        report_content += f"\n#### 🔸 골든셋 통계: {name}\n"

        markdown_table = _format_golden_stats_to_md(report_data)
        report_content += markdown_table
        report_content += "\n"

        processed_golden_sets.add(name)

    report_content += "\n"

    # -------------------------------------------------------------
    # 6. Softmax 기술 통계 및 히스토그램 (N+M 원천 평가 주체 기준)
    # -------------------------------------------------------------

    # 🚨 수정 1: Softmax 기술 통계 섹션을 6️⃣ 레벨로 승격
    report_content += "---\n\n"
    report_content += "### 6️⃣ Softmax 기술 통계량 및 확신도 지표\n"

    # -------------------------------------------------------------
    # Step 6.1: 고유한 평가 대상(N) 및 골든셋(M) 데이터 추출 및 테이블 생성
    # -------------------------------------------------------------

    unique_eval_subjects: Dict[str, Dict[str, Any]] = {}  # {model_name: {stats, ...}}
    unique_golden_subjects: Dict[str, Dict[str, Any]] = {}  # {golden_name: {stats, ...}}

    # model_summaries에서 고유한 평가 대상과 골든셋 통계를 추출
    for summary in model_summaries:
        model_name = summary['model_name']
        golden_name = summary['golden_set_name']

        # 1. 평가 대상셋 (Evaluation Model) 통계 추출 (N개) - 통계는 CR/AF/AR 키에 이미 저장되어 있음
        if model_name not in unique_eval_subjects:
            unique_eval_subjects[model_name] = {
                'name': model_name,
                'type': '평가셋',
                KEY_CR: summary[KEY_CR],
                KEY_AF: summary[KEY_AF],
                KEY_AR: summary[KEY_AR],
            }

        # 2. 골든셋 (Golden Set) 통계 추출 (M개) - 골든셋 통계는 현재 코드에 명시적으로 저장되지 않음.
        #    => 이 부분의 출력을 위해 골든셋은 N/A로 처리될 가능성이 높음.
        if golden_name not in unique_golden_subjects:
            # 🚨 주의: 골든셋의 Softmax 기술 통계가 model_summaries에 없으므로, 평가셋 데이터를 복사하지 않습니다.
            #         (Softmax 기술 통계는 평가 대상셋의 CR/AF/AR 키에 저장된 값과 구조가 동일해야 함.)
            #         여기서는 빈 딕셔너리를 사용하여 테이블에 0.0이나 N/A를 출력하도록 유도합니다.
            unique_golden_subjects[golden_name] = {
                'name': golden_name,
                'type': '골든셋',
                KEY_CR: {},
                KEY_AF: {},
                KEY_AR: {},
            }

    eval_list = list(unique_eval_subjects.values())
    golden_list = list(unique_golden_subjects.values())

    # -------------------------------------------------------------
    # Step 6.2: 확신도 테이블 생성 (평가셋 및 골든셋)
    # -------------------------------------------------------------

    # 평가 대상셋 테이블 출력 (N개)
    if eval_list:
        report_content += _create_confidence_table(eval_list, "평가셋")

    report_content += "\n"
    report_content += """
    > 📝 의미 요약
    > * P+Avg : 평균 긍정 확률 
    >   * 전체 샘플에 대해 모델이 긍정(1)을 부여한 확률 평균 값 (편향 경향 파악 목적) 
    > * Conf+ : 평균 긍정 확신도 
    >   * 모델이 긍정으로 예측한 샘플들만 대상으로, 해당 샘플들의 P_pos 값 평균 ( 긍정판단의 강도 )    
    > * Conf- : 평균 부정 확신도 
    >   * 모델이 부정으로 예측한 샘플들만 대상으로, 해당 샘플들의 P_neg 값 평균 ( 부정판단의 강도 )     
    > * Margin : 평균 확률 마진 
    >   * 긍정과 부정 확률의 차이로 값이 클수록 단호한 판단
    > * P+Min : 긍정 확률 최소값         
    > * P+Med : 긍정 확률 중간값 
    > * P+Max : 긍정 확률 최대값 
    > * P-Min : 부정 확률 최소값 
    > * P-Med : 부정 확률 중간값 
    > * P-Max : 부정 확률 최대값
    """


    # 2. Softmax 분포 시각화 (N+M 원천 평가 주체 기준)
    report_content += "\n\n---\n\n"
    report_content += "\n### 7️⃣ Softmax 확률 분포 히스토그램\n"

    # 평가 대상셋(N)과 골든셋(M)을 한 번씩만 출력하기 위해 추적
    processed_eval_models: Set[str] = set()
    processed_golden_names: Set[str] = set()

    for summary in model_summaries:
        model_name = summary['model_name']
        golden_name = summary['golden_set_name']

        # --- N개 평가 대상셋 히스토그램 출력 (중복 방지) ---
        if model_name not in processed_eval_models:
            chart_title_eval = f"{model_name} (평가셋) 분포"
            report_content += _create_integrated_chart(summary, chart_title_eval)
            processed_eval_models.add(model_name)

        # --- M개 골든셋 히스토그램 출력 (중복 방지) ---
        if golden_name not in processed_golden_names:
            # 통합 차트 헬퍼 함수에 골든셋 데이터를 전달하기 위해 summary 복사 및 데이터 이동
            temp_summary = summary.copy()
            for axis in [KEY_CR, KEY_AF, KEY_AR]:
                # 'prob_bins' 키에 'golden_prob_bins' 데이터를 덮어씌움 (통합 헬퍼 함수 재활용)
                temp_summary[axis]['prob_bins'] = temp_summary[axis]['golden_prob_bins']

            chart_title_golden = f"{golden_name} (골든셋) 분포"
            report_content += _create_integrated_chart(temp_summary, chart_title_golden)
            processed_golden_names.add(golden_name)

    report_content += "\n"

    return report_content