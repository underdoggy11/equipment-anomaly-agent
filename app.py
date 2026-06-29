import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Equipment Log Anomaly Detection Agent",
    layout="wide"
)

st.title("Equipment Log Anomaly Detection Agent")
st.caption("CSV 장비 로그를 업로드하면 분석 계획 수립, 자동 시각화, 이상치 탐지, 점검 액션 제안까지 수행합니다.")


def guess_time_column(df):
    for col in df.columns:
        lower = col.lower()
        if "time" in lower or "date" in lower or "timestamp" in lower:
            return col
    return df.columns[0]


def guess_equipment_column(df):
    keywords = ["eqp", "equipment", "tool", "machine", "chamber", "device"]
    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in keywords):
            return col
    return df.columns[0]


def detect_iqr_anomaly(df, target_col):
    result = df.copy()
    q1 = result[target_col].quantile(0.25)
    q3 = result[target_col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    result["anomaly_iqr"] = (result[target_col] < lower) | (result[target_col] > upper)
    return result, lower, upper


def detect_zscore_anomaly(df, target_col, threshold=3.0):
    result = df.copy()
    mean = result[target_col].mean()
    std = result[target_col].std()

    if std == 0 or pd.isna(std):
        result["z_score"] = 0
        result["anomaly_zscore"] = False
    else:
        result["z_score"] = (result[target_col] - mean) / std
        result["anomaly_zscore"] = result["z_score"].abs() > threshold

    return result


def make_analysis_plan(time_col, equipment_col, target_col, numeric_cols):
    return f"""
### Agent 분석 계획

1. `{target_col}`의 기본 통계량을 계산합니다.
2. `{time_col}` 기준으로 시간에 따른 drift, spike, trend를 확인합니다.
3. `{equipment_col}` 기준으로 장비별 평균 shift와 분산 차이를 확인합니다.
4. IQR 및 z-score 기준으로 1차 이상치 후보를 추출합니다.
5. 이상치가 특정 장비 또는 특정 시간 구간에 집중되는지 확인합니다.
6. 결과를 바탕으로 원인 후보와 후속 점검 액션을 제안합니다.

분석 가능한 수치형 컬럼: {", ".join(numeric_cols)}
"""


def make_report(df, target_col, equipment_col, anomaly_df, eqp_summary, stats):
    total_count = len(df)
    anomaly_count = len(anomaly_df)
    anomaly_rate = anomaly_count / total_count * 100 if total_count > 0 else 0

    if len(eqp_summary) > 0:
        top_eqp = eqp_summary.iloc[0][equipment_col]
        top_eqp_count = int(eqp_summary.iloc[0]["anomaly_count"])
        top_eqp_rate = float(eqp_summary.iloc[0]["anomaly_rate_percent"])
    else:
        top_eqp = "N/A"
        top_eqp_count = 0
        top_eqp_rate = 0.0

    return f"""
### AI Agent 분석 리포트

분석 대상은 `{target_col}`입니다.

전체 데이터 **{total_count:,}건** 중 이상치 후보는 **{anomaly_count:,}건**이며,  
이상치 비율은 **{anomaly_rate:.2f}%**입니다.

`{target_col}`의 평균은 **{stats["mean"]:.3f}**, 표준편차는 **{stats["std"]:.3f}**입니다.  
최소값은 **{stats["min"]:.3f}**, 최대값은 **{stats["max"]:.3f}**입니다.

장비 기준으로는 **`{top_eqp}`**에서 이상치 후보가 가장 많이 발생했습니다.  
해당 장비의 이상치 후보 수는 **{top_eqp_count:,}건**, 이상치 비율은 **{top_eqp_rate:.2f}%**입니다.

#### 원인 후보

1. 특정 장비의 sensor offset 또는 calibration drift 가능성
2. 특정 시간대의 공정 조건 변화 가능성
3. chamber, MFC, valve 등 장비 상태 변화 가능성
4. recipe 또는 lot 조건 차이에 따른 분포 변화 가능성

#### 권장 점검 액션

1. 이상치가 집중된 장비의 최근 PM 및 calibration 이력을 확인합니다.
2. 이상 발생 시간대의 recipe, lot, chamber 정보를 확인합니다.
3. 동일 시간대의 pressure, temperature 등 다른 센서값과 동시 이상 여부를 비교합니다.
4. 이상치 발생 전후 구간을 비교하여 spike인지 drift인지 구분합니다.

#### 주의사항

본 결과는 통계 기반 1차 이상 후보 탐지입니다. 실제 원인 확정에는 장비 이력, 공정 조건, 센서 상태 등 추가 확인이 필요합니다.
"""


uploaded_file = st.file_uploader("장비 로그 CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 분석을 시작할 수 있습니다.")

else:
    df = pd.read_csv(uploaded_file)

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(20), use_container_width=True)

    st.write(f"총 Row 수: **{len(df):,}**")
    st.write(f"총 Column 수: **{len(df.columns):,}**")

    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.error("분석 가능한 수치형 컬럼이 없습니다.")
        st.stop()

    default_time = guess_time_column(df)
    default_equipment = guess_equipment_column(df)

    st.subheader("Agent 컬럼 추정 결과")
    st.write(f"시간 컬럼 후보: **{default_time}**")
    st.write(f"장비 컬럼 후보: **{default_equipment}**")
    st.write(f"분석 가능한 수치형 컬럼: **{', '.join(numeric_cols)}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        time_col = st.selectbox(
            "시간 컬럼",
            columns,
            index=columns.index(default_time)
        )

    with col2:
        equipment_col = st.selectbox(
            "장비 컬럼",
            columns,
            index=columns.index(default_equipment)
        )

    with col3:
        target_col = st.selectbox(
            "분석 대상 수치 컬럼",
            numeric_cols
        )

    if st.button("Agent 분석 실행", type="primary"):
        working_df = df.copy()

        working_df[target_col] = pd.to_numeric(working_df[target_col], errors="coerce")
        working_df = working_df.dropna(subset=[target_col])

        st.subheader("Agent 분석 계획")
        st.markdown(make_analysis_plan(time_col, equipment_col, target_col, numeric_cols))

        stats = {
            "count": int(working_df[target_col].count()),
            "mean": float(working_df[target_col].mean()),
            "std": float(working_df[target_col].std()),
            "min": float(working_df[target_col].min()),
            "median": float(working_df[target_col].median()),
            "max": float(working_df[target_col].max()),
            "missing_count": int(df[target_col].isna().sum()),
            "missing_rate_percent": float(df[target_col].isna().mean() * 100),
        }

        st.subheader("기본 통계")
        st.json(stats)

        analyzed_df, lower, upper = detect_iqr_anomaly(working_df, target_col)
        analyzed_df = detect_zscore_anomaly(analyzed_df, target_col)

        analyzed_df["is_anomaly"] = analyzed_df["anomaly_iqr"] | analyzed_df["anomaly_zscore"]
        anomaly_df = analyzed_df[analyzed_df["is_anomaly"]]

        st.subheader("이상치 탐지 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("이상치 후보 수", f"{len(anomaly_df):,}")
        c2.metric("이상치 비율", f"{len(anomaly_df) / len(analyzed_df) * 100:.2f}%")
        c3.metric("IQR 정상 범위", f"{lower:.2f} ~ {upper:.2f}")

        st.subheader("자동 생성 Plot")

        try:
            analyzed_df[time_col] = pd.to_datetime(analyzed_df[time_col], errors="coerce")
            time_df = analyzed_df.dropna(subset=[time_col]).sort_values(time_col)

            fig_time = px.line(
                time_df,
                x=time_col,
                y=target_col,
                title=f"{target_col} Time Trend"
            )
            st.plotly_chart(fig_time, use_container_width=True)

            fig_anomaly = px.scatter(
                time_df,
                x=time_col,
                y=target_col,
                color="is_anomaly",
                title=f"{target_col} Anomaly Detection"
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)

        except Exception as e:
            st.warning(f"시간 기반 plot 생성 중 문제가 발생했습니다: {e}")

        fig_box = px.box(
            analyzed_df,
            x=equipment_col,
            y=target_col,
            title=f"{target_col} by {equipment_col}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        eqp_summary = (
            analyzed_df.groupby(equipment_col)
            .agg(
                total_count=("is_anomaly", "count"),
                anomaly_count=("is_anomaly", "sum")
            )
            .reset_index()
        )
        eqp_summary["anomaly_rate_percent"] = (
            eqp_summary["anomaly_count"] / eqp_summary["total_count"] * 100
        )
        eqp_summary = eqp_summary.sort_values("anomaly_count", ascending=False)

        st.subheader("장비별 이상치 순위")
        st.dataframe(eqp_summary, use_container_width=True)

        st.subheader("이상치 후보 테이블")
        st.dataframe(anomaly_df.head(200), use_container_width=True)

        st.subheader("AI Agent 분석 리포트")
        st.markdown(
            make_report(
                analyzed_df,
                target_col,
                equipment_col,
                anomaly_df,
                eqp_summary,
                stats
            )
        )