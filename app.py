import html
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="PECVD Process Anomaly Agent",
    layout="wide",
)

st.title("공정 이상 진단")
st.caption("Recipe 흐름과 주요 제어 신호를 한 화면에서 확인합니다.")


st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }
    .workflow-step {
        border-top: 1px solid #e5e7eb;
        padding-top: 1.2rem;
        margin-top: 1.6rem;
        margin-bottom: 0.8rem;
    }
    .workflow-step:first-of-type {
        border-top: 0;
        margin-top: 0.4rem;
    }
    .workflow-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.15rem;
    }
    .workflow-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.8rem;
        height: 1.8rem;
        border-radius: 999px;
        background: #0f172a;
        color: white;
        font-weight: 700;
        font-size: 0.92rem;
        flex: 0 0 auto;
    }
    .workflow-heading {
        font-size: 1.22rem;
        line-height: 1.25;
        font-weight: 750;
        color: #0f172a;
    }
    .workflow-caption {
        color: #64748b;
        margin-left: 2.55rem;
        font-size: 0.9rem;
    }
    .data-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border: 1px solid #dbeafe;
        background: #eff6ff;
        color: #1e3a8a;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        font-size: 0.88rem;
        font-weight: 650;
        margin: 0.35rem 0 0.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_step_header(number, title, caption=None):
    caption_html = f'<div class="workflow-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <section class="workflow-step">
            <div class="workflow-title">
                <span class="workflow-number">{number}</span>
                <span class="workflow-heading">{title}</span>
            </div>
            {caption_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def guess_time_column(df):
    for col in df.columns:
        lower = col.lower()
        if "time" in lower or "date" in lower or "timestamp" in lower:
            return col
    return df.columns[0]


def guess_step_column(df):
    for preferred in ["chA_VIR_Step_Name", "Step_Name", "step_name"]:
        if preferred in df.columns:
            return preferred

    for col in df.columns:
        lower = col.lower()
        if "step" in lower and "name" in lower:
            return col
    return None


def guess_status_column(df):
    for preferred in ["chA_Recipe_Curr_Chamber_Status", "ControlState", "Communication State"]:
        if preferred in df.columns:
            return preferred

    for col in df.columns:
        lower = col.lower()
        if "status" in lower or "state" in lower:
            return col
    return None


def guess_recipe_column(df):
    for preferred in ["chA_VIR_Recipe_Name", "CTC_chA_ProcRcpID", "CTC_chA_SeqRcpID"]:
        if preferred in df.columns:
            return preferred

    for col in df.columns:
        lower = col.lower()
        if "recipe" in lower or "rcp" in lower:
            return col
    return None


def parse_datetime_series(series):
    parsed = pd.to_datetime(series, format="%y/%m/%d %H:%M:%S", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def normalize_signal_name(col):
    name = col.lower()
    name = re.sub(r"^cha_", "", name)
    name = name.replace("_ao_", "_").replace("_ai_", "_").replace("_vir_", "_")
    name = name.replace("setpoint", "").replace("_set_", "_").replace("_set", "")
    name = name.replace("flow", "").replace("actual", "")
    name = name.replace("delivery", "").replace("position", "")
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name


def classify_pair(setpoint_col, actual_col):
    combined = f"{setpoint_col} {actual_col}".lower()
    if "mfc" in combined or "flow" in combined:
        return "Gas Flow"
    if "temp" in combined:
        return "Temperature"
    if "power" in combined:
        return "Power"
    if "apc" in combined:
        return "APC"
    return "Control"


def build_pair_label(setpoint_col, actual_col):
    tokens = []
    combined = f"{setpoint_col}_{actual_col}"

    mfc_match = re.search(r"mfc\d+", combined, flags=re.IGNORECASE)
    if mfc_match:
        tokens.append(mfc_match.group(0).upper())

    gas_match = re.search(
        r"_(SiH2Cl2|Si2H6|N2O|NF3|Ar|TN2|BN2)(?:_|$)",
        combined,
        flags=re.IGNORECASE,
    )
    if gas_match:
        tokens.append(gas_match.group(1))

    if not tokens:
        tokens.append(normalize_signal_name(setpoint_col).replace("_", " ").title())

    return " ".join(tokens)


def find_control_pairs(df):
    columns = df.columns.tolist()
    lower_map = {col: col.lower() for col in columns}
    pairs = []
    used_actuals = set()

    for set_col in columns:
        lower = lower_map[set_col]
        if "setpoint" not in lower and not lower.endswith("_set"):
            continue

        candidates = []

        if "mfc" in lower:
            set_mfc = re.search(r"mfc(\d+)", lower)
            if set_mfc:
                for actual_col in columns:
                    actual_lower = lower_map[actual_col]
                    actual_mfc = re.search(r"mfc(\d+)", actual_lower)
                    if (
                        actual_mfc
                        and actual_mfc.group(1) == set_mfc.group(1)
                        and "flow" in actual_lower
                    ):
                        candidates.append(actual_col)

        if "temp" in lower:
            set_base = normalize_signal_name(set_col).replace("temp", "")
            for actual_col in columns:
                actual_lower = lower_map[actual_col]
                if actual_col == set_col or "temp" not in actual_lower:
                    continue
                if "setpoint" in actual_lower or actual_lower.endswith("_set"):
                    continue
                actual_base = normalize_signal_name(actual_col).replace("temp", "")
                if set_base == actual_base:
                    candidates.append(actual_col)

        if "power" in lower:
            for actual_col in columns:
                actual_lower = lower_map[actual_col]
                if actual_col != set_col and "power" in actual_lower and "setpoint" not in actual_lower:
                    candidates.append(actual_col)

        for actual_col in candidates:
            if actual_col in used_actuals:
                continue
            pairs.append(
                {
                    "label": build_pair_label(set_col, actual_col),
                    "kind": classify_pair(set_col, actual_col),
                    "setpoint_col": set_col,
                    "actual_col": actual_col,
                }
            )
            used_actuals.add(actual_col)
            break

    return pairs


def tolerance_for_kind(kind):
    if kind == "Gas Flow":
        return 2.0, 5.0
    if kind == "Temperature":
        return 1.0, 1.0
    if kind == "Power":
        return 5.0, 5.0
    return 3.0, 1.0


def prepare_base_df(df, time_col):
    result = df.copy()
    result[time_col] = parse_datetime_series(result[time_col])
    return result.dropna(subset=[time_col]).sort_values(time_col)


def filter_process_rows(df, step_col, status_col):
    mask = pd.Series(False, index=df.index)

    if status_col and status_col in df.columns:
        status = df[status_col].astype(str).str.lower()
        mask |= status.str.contains("recipe running|running|depo|stable", regex=True, na=False)

    if step_col and step_col in df.columns:
        step = df[step_col].astype(str).str.strip()
        mask |= step.ne("") & step.ne("nan")

    process_df = df[mask].copy()
    if process_df.empty:
        return df.copy()
    return process_df


def add_tracking_columns(df, pair, time_col, step_col=None, status_col=None):
    set_col = pair["setpoint_col"]
    actual_col = pair["actual_col"]
    keep_cols = [time_col, set_col, actual_col]
    if step_col and step_col in df.columns:
        keep_cols.append(step_col)
    if status_col and status_col in df.columns:
        keep_cols.append(status_col)

    result = df[keep_cols].copy()
    result[set_col] = pd.to_numeric(result[set_col], errors="coerce")
    result[actual_col] = pd.to_numeric(result[actual_col], errors="coerce")
    result = result.dropna(subset=[set_col, actual_col]).sort_values(time_col)

    result["error"] = result[actual_col] - result[set_col]
    result["abs_error"] = result["error"].abs()
    result["active"] = result[set_col].abs() > 1e-9
    result["setpoint_changed"] = result[set_col].ne(result[set_col].shift())
    result["settled_points"] = result.groupby(result["setpoint_changed"].cumsum()).cumcount()
    result["settled"] = result["active"] & (result["settled_points"] >= 3)
    result["abs_pct_error"] = np.where(
        result["active"],
        result["abs_error"] / result[set_col].abs().clip(lower=1e-9) * 100,
        np.nan,
    )

    tol_pct, tol_abs_min = tolerance_for_kind(pair["kind"])
    result["tolerance_abs"] = np.maximum(result[set_col].abs() * tol_pct / 100, tol_abs_min)
    result["tracking_anomaly"] = result["settled"] & (result["abs_error"] > result["tolerance_abs"])
    return result


def calculate_pair_metrics(df, pairs, time_col, step_col=None, status_col=None):
    rows = []

    for pair in pairs:
        pair_df = add_tracking_columns(df, pair, time_col, step_col, status_col)
        settled = pair_df[pair_df["settled"]]
        if settled.empty:
            settled = pair_df[pair_df["active"]]
        if settled.empty:
            continue

        bad_count = int(settled["tracking_anomaly"].sum())
        checked_count = int(len(settled))
        bad_rate = bad_count / checked_count * 100 if checked_count else 0
        p95_pct = float(settled["abs_pct_error"].quantile(0.95))
        max_pct = float(settled["abs_pct_error"].max())
        mean_abs_error = float(settled["abs_error"].mean())
        max_abs_error = float(settled["abs_error"].max())
        tol_pct, _ = tolerance_for_kind(pair["kind"])
        score = bad_rate + max(0, p95_pct - tol_pct)

        rows.append(
            {
                "signal": pair["label"],
                "kind": pair["kind"],
                "status": judge_signal_status(bad_rate, p95_pct, tol_pct),
                "checked_count": checked_count,
                "bad_count": bad_count,
                "bad_rate_percent": bad_rate,
                "p95_abs_pct_error": p95_pct,
                "max_abs_pct_error": max_pct,
                "mean_abs_error": mean_abs_error,
                "max_abs_error": max_abs_error,
                "score": score,
                "setpoint_col": pair["setpoint_col"],
                "actual_col": pair["actual_col"],
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["score", "bad_count"], ascending=False)


def judge_signal_status(bad_rate, p95_pct, tol_pct):
    if bad_rate >= 10 or p95_pct >= tol_pct * 3:
        return "위험"
    if bad_rate > 0 or p95_pct >= tol_pct * 1.5:
        return "주의"
    return "정상"


def judge_overall(metrics):
    if metrics.empty:
        return "판단 불가", "분석 가능한 target/actual 신호가 없습니다."

    worst = metrics.iloc[0]
    if (metrics["status"] == "위험").any():
        return "위험", f"{worst['signal']} 신호의 추적 오차가 큽니다."
    if (metrics["status"] == "주의").any():
        return "주의", f"{worst['signal']} 신호를 우선 확인하세요."
    return "정상", "주요 target/actual 신호가 허용 범위 안에 있습니다."


def get_step_options(df, step_col):
    if not step_col or step_col not in df.columns:
        return ["전체 공정"]

    ordered_steps = (
        df[step_col]
        .astype(str)
        .str.strip()
        .replace("nan", "")
        .drop_duplicates()
    )
    steps = [step for step in ordered_steps.tolist() if step]
    return ["전체 공정"] + steps[:12]


def make_recipe_timeline_figure(df, time_col, recipe_col=None, step_col=None, status_col=None):
    if df.empty:
        return None

    timeline_df = df.sort_values(time_col).copy()
    timeline_df["_recipe"] = (
        timeline_df[recipe_col].astype(str).str.strip()
        if recipe_col and recipe_col in timeline_df.columns
        else "Recipe"
    )
    timeline_df["_step"] = (
        timeline_df[step_col].astype(str).str.strip()
        if step_col and step_col in timeline_df.columns
        else ""
    )
    timeline_df["_status"] = (
        timeline_df[status_col].astype(str).str.strip()
        if status_col and status_col in timeline_df.columns
        else ""
    )

    timeline_df["_recipe"] = timeline_df["_recipe"].replace({"": "Unknown", "nan": "Unknown"})
    timeline_df["_step"] = timeline_df["_step"].replace({"": "No step", "nan": "No step"})
    timeline_df["_status"] = timeline_df["_status"].replace({"": "-", "nan": "-"})

    timeline_df["_segment_key"] = (
        timeline_df["_recipe"] + "||" + timeline_df["_step"] + "||" + timeline_df["_status"]
    )
    segment_id = timeline_df["_segment_key"].ne(timeline_df["_segment_key"].shift()).cumsum()

    segments = (
        timeline_df.groupby(segment_id)
        .agg(
            start=(time_col, "first"),
            end=(time_col, "last"),
            recipe=("_recipe", "first"),
            step=("_step", "first"),
            status=("_status", "first"),
            rows=(time_col, "size"),
        )
        .reset_index(drop=True)
    )

    if len(timeline_df) > 1:
        deltas = timeline_df[time_col].diff().dropna()
        fallback_delta = deltas.median()
    else:
        fallback_delta = pd.Timedelta(seconds=1)
    if pd.isna(fallback_delta) or fallback_delta <= pd.Timedelta(0):
        fallback_delta = pd.Timedelta(seconds=1)

    if len(segments) > 1:
        segments.iloc[:-1, segments.columns.get_loc("end")] = segments["start"].shift(-1).iloc[:-1].to_numpy()
    segments.loc[segments.index[-1], "end"] = segments.loc[segments.index[-1], "end"] + fallback_delta
    segments["duration_ms"] = (segments["end"] - segments["start"]).dt.total_seconds() * 1000
    segments["label"] = segments["recipe"] + " / " + segments["step"]

    palette = [
        "#2563eb",
        "#059669",
        "#d97706",
        "#7c3aed",
        "#dc2626",
        "#0891b2",
        "#65a30d",
        "#9333ea",
    ]
    step_colors = {step: palette[idx % len(palette)] for idx, step in enumerate(segments["step"].unique())}

    fig = go.Figure()
    for row in segments.itertuples(index=False):
        fig.add_trace(
            go.Bar(
                x=[row.duration_ms],
                y=[row.recipe],
                base=[row.start],
                orientation="h",
                name=row.step,
                marker=dict(color=step_colors[row.step]),
                text=[row.step],
                textposition="inside",
                hovertemplate=(
                    "Recipe: %{y}<br>"
                    f"Step: {row.step}<br>"
                    f"Status: {row.status}<br>"
                    f"Rows: {row.rows}<br>"
                    "Start: %{base|%Y-%m-%d %H:%M:%S}<br>"
                    f"End: {row.end:%Y-%m-%d %H:%M:%S}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Recipe 타임라인",
        height=max(220, 90 + 48 * segments["recipe"].nunique()),
        barmode="stack",
        margin=dict(l=16, r=16, t=48, b=24),
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            title="시간",
            type="date",
            showgrid=True,
            gridcolor="#e5e7eb",
            griddash="dot",
            zeroline=False,
            showline=True,
            linecolor="#cbd5e1",
            mirror=True,
        ),
        yaxis=dict(
            title="",
            autorange="reversed",
            showgrid=True,
            gridcolor="#eef2f7",
            zeroline=False,
            showline=True,
            linecolor="#cbd5e1",
            mirror=True,
        ),
    )
    return fig


def make_tracking_figure(pair_df, pair, time_col):
    set_col = pair["setpoint_col"]
    actual_col = pair["actual_col"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pair_df[time_col],
            y=pair_df[set_col],
            mode="lines",
            name="Target",
            line=dict(color="#2563eb", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pair_df[time_col],
            y=pair_df[actual_col],
            mode="lines",
            name="Actual",
            line=dict(color="#059669", width=2),
        )
    )

    anomaly_df = pair_df[pair_df["tracking_anomaly"]]
    if not anomaly_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_df[time_col],
                y=anomaly_df[actual_col],
                mode="markers",
                name="이탈 구간",
                marker=dict(color="#dc2626", size=8, symbol="x"),
            )
        )

    fig.update_layout(
        title=f"{pair['label']} 추적 상태",
        height=430,
        margin=dict(l=16, r=16, t=56, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def find_pair_by_columns(pairs, setpoint_col, actual_col):
    for pair in pairs:
        if pair["setpoint_col"] == setpoint_col and pair["actual_col"] == actual_col:
            return pair
    return pairs[0]


render_step_header(
    1,
    "데이터 선택",
    "파일을 올리거나 예시 데이터를 선택하세요.",
)
uploaded_file = st.file_uploader("CSV 업로드", type=["csv"], label_visibility="collapsed")

sample_datasets = {
    "정상 예시 데이터 업로드하기": "sample_1_normal.csv",
    "주의 예시 데이터 업로드하기": "sample_2_short_gas_drift.csv",
    "위험 예시 데이터 업로드하기": "sample_3_severe_multi_signal.csv",
    "현실형 예시 데이터 업로드하기": "sample_4_realistic_long_run.csv",
}

if uploaded_file is not None:
    st.session_state["selected_sample_data"] = None

sample_cols = st.columns(4)
for idx, (sample_label, sample_path) in enumerate(sample_datasets.items()):
    if sample_cols[idx].button(sample_label, type="secondary", width="stretch"):
        st.session_state["selected_sample_data"] = sample_path

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_name = uploaded_file.name
elif st.session_state.get("selected_sample_data"):
    data_name = st.session_state["selected_sample_data"]
    df = pd.read_csv(data_name)
else:
    st.info("분석할 데이터를 선택하세요.")
    st.stop()

st.markdown(
    f'<div class="data-chip">분석 데이터 · {html.escape(data_name)}</div>',
    unsafe_allow_html=True,
)

time_col = guess_time_column(df)
step_col = guess_step_column(df)
status_col = guess_status_column(df)
recipe_col = guess_recipe_column(df)
base_df = prepare_base_df(df, time_col)
process_df = filter_process_rows(base_df, step_col, status_col)
control_pairs = find_control_pairs(process_df)

if not control_pairs:
    st.error("자동으로 찾은 target/actual 신호가 없습니다. setpoint-flow 또는 temp set-temp 컬럼명을 확인하세요.")
    st.stop()

timeline_fig = make_recipe_timeline_figure(base_df, time_col, recipe_col, step_col, status_col)
if timeline_fig is not None:
    render_step_header(
        2,
        "Recipe 타임라인",
        "공정 흐름을 먼저 확인합니다.",
    )
    with st.container(border=True):
        st.plotly_chart(timeline_fig, width="stretch")

render_step_header(
    3,
    "진단 결과",
    "분석할 구간을 선택하면 결과가 바로 갱신됩니다.",
)
step_options = get_step_options(process_df, step_col)
scope = st.selectbox("구간", step_options)

scoped_df = process_df.copy()
if scope != "전체 공정" and step_col:
    scoped_df = scoped_df[scoped_df[step_col].astype(str).str.strip() == scope]

metrics = calculate_pair_metrics(scoped_df, control_pairs, time_col, step_col, status_col)
overall_status, overall_reason = judge_overall(metrics)

summary_col, issue_col, signal_col = st.columns([1.4, 1, 1])
summary_col.metric("상태", overall_status)
summary_col.caption(overall_reason)
issue_count = int(metrics["bad_count"].sum()) if not metrics.empty else 0
issue_col.metric("이상 후보", f"{issue_count:,}")
signal_col.metric("감시 신호", f"{len(metrics):,}")

if overall_status == "정상":
    st.success("주요 신호가 안정적으로 추적 중입니다.")
elif overall_status == "주의":
    st.warning("일부 신호에 편차가 있습니다.")
else:
    st.error("이상 가능성이 높습니다.")

if metrics.empty:
    st.stop()

render_step_header(
    4,
    "핵심 신호",
    "가장 먼저 볼 신호를 확인합니다.",
)
signal_options = [
    f"{row.status} · {row.signal} · {row.kind}"
    for row in metrics.itertuples(index=False)
]
selected_signal = st.selectbox("추천 신호", signal_options)
selected_row = metrics.iloc[signal_options.index(selected_signal)]
selected_pair = find_pair_by_columns(
    control_pairs,
    selected_row["setpoint_col"],
    selected_row["actual_col"],
)
selected_pair_df = add_tracking_columns(scoped_df, selected_pair, time_col, step_col, status_col)

plot_col, table_col = st.columns([2.1, 1])

with plot_col:
    st.plotly_chart(make_tracking_figure(selected_pair_df, selected_pair, time_col), width="stretch")

with table_col:
    st.subheader("요약")
    st.metric("상태", selected_row["status"])
    st.metric("이상 후보", f"{int(selected_row['bad_count']):,}")

    issue_rows = selected_pair_df[selected_pair_df["tracking_anomaly"]]
    if issue_rows.empty:
        st.caption("눈에 띄는 이탈 구간이 없습니다.")
    else:
        display_cols = [time_col, selected_pair["setpoint_col"], selected_pair["actual_col"], "abs_error", "abs_pct_error"]
        if step_col and step_col in issue_rows.columns:
            display_cols.insert(1, step_col)
        with st.expander("이탈 구간 보기"):
            st.dataframe(issue_rows[display_cols].head(20), width="stretch", hide_index=True)

    with st.expander("상세 지표 보기"):
        st.metric("이상 비율", f"{selected_row['bad_rate_percent']:.2f}%")
        st.metric("P95 오차율", f"{selected_row['p95_abs_pct_error']:.2f}%")
        st.metric("최대 절대 오차", f"{selected_row['max_abs_error']:.3f}")

render_step_header(
    5,
    "점검 우선순위",
    "문제가 커 보이는 신호부터 정렬했습니다.",
)
priority_cols = [
    "status",
    "signal",
    "kind",
    "bad_count",
]
st.dataframe(
    metrics[priority_cols].rename(
        columns={
            "status": "상태",
            "signal": "신호",
            "kind": "종류",
            "bad_count": "이상 후보",
        }
    ),
    width="stretch",
    hide_index=True,
)

with st.expander("설정 정보"):
    st.write(f"데이터: **{data_name}**")
    st.write(f"시간 컬럼: `{time_col}`")
    st.write(f"Recipe 컬럼: `{recipe_col or '미탐지'}`")
    st.write(f"Step 컬럼: `{step_col or '미탐지'}`")
    st.write(f"Status 컬럼: `{status_col or '미탐지'}`")
    st.write(f"공정 Row: **{len(process_df):,} / {len(base_df):,}**")
    st.dataframe(pd.DataFrame(control_pairs), width="stretch", hide_index=True)

with st.expander("원본 데이터 보기"):
    st.dataframe(df.head(30), width="stretch")
