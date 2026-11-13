"""Interactive Streamlit dashboard for the bakery forecasting case."""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modeling import (
    load_and_validate_data,
    regression_metrics,
    run_full_analysis,
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "bakerysales.csv"

NAVY = "#18364A"
TEAL = "#1F7A75"
ORANGE = "#C56A45"
GOLD = "#D9A441"
INK = "#1D252B"
MUTED = "#65727A"
PAPER = "#F7F4EE"


st.set_page_config(
    page_title="Bakery Forecast Lab",
    page_icon="🥐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

        :root {
            --paper: #F7F4EE;
            --card: #FFFFFF;
            --ink: #1D252B;
            --muted: #65727A;
            --navy: #18364A;
            --teal: #1F7A75;
            --orange: #C56A45;
            --gold: #D9A441;
            --line: #E6E0D7;
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", Arial, sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: var(--paper);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            font-family: "Source Serif 4", Georgia, serif;
            color: var(--navy);
            letter-spacing: -0.02em;
        }

        .hero {
            background: var(--navy);
            color: white;
            padding: 2.4rem 2.7rem 2.2rem 2.7rem;
            border-radius: 20px;
            margin-bottom: 1.3rem;
            box-shadow: 0 14px 36px rgba(24, 54, 74, 0.14);
        }

        .hero .eyebrow {
            color: #F0C979;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }

        .hero h1 {
            color: white;
            font-size: clamp(2.2rem, 4vw, 4.2rem);
            line-height: 1.03;
            margin: 0 0 0.85rem 0;
            max-width: 900px;
        }

        .hero p {
            color: #DCE5E9;
            font-size: 1.08rem;
            line-height: 1.55;
            margin: 0;
            max-width: 900px;
        }

        .status-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 1.25rem;
        }

        .status-pill {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.20);
            color: #FFFFFF;
            border-radius: 999px;
            padding: 0.38rem 0.72rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,0.82);
            border: 1px solid var(--line) !important;
            border-radius: 16px;
            box-shadow: 0 7px 22px rgba(24,54,74,0.045);
        }

        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 15px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 6px 18px rgba(24,54,74,0.04);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: var(--navy);
            font-family: "Source Serif 4", Georgia, serif;
        }

        .section-kicker {
            color: var(--orange);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: -0.25rem;
        }

        .callout {
            background: #EDF6F4;
            border-left: 4px solid var(--teal);
            border-radius: 8px 14px 14px 8px;
            padding: 1rem 1.2rem;
            color: #284541;
            margin: 0.4rem 0 1rem 0;
        }

        .warning-callout {
            background: #FFF7E8;
            border-left-color: var(--gold);
            color: #594520;
        }

        .timeline-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
            margin: 0.5rem 0 1.5rem 0;
        }

        .timeline-card {
            background: white;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1rem;
            min-height: 128px;
        }

        .timeline-card strong {
            display: block;
            color: var(--orange);
            font-size: 1.05rem;
            margin-bottom: 0.45rem;
        }

        .timeline-card span {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.35;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.5;
        }

        .footer-note {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 2rem;
            padding-top: 1rem;
        }

        div[data-testid="stTabs"] button {
            font-weight: 650;
        }

        @media (max-width: 900px) {
            .block-container { padding: 1rem 0.8rem 3rem 0.8rem; }
            .hero { padding: 1.7rem 1.4rem; border-radius: 16px; }
            .timeline-grid { grid-template-columns: 1fr 1fr; }
        }

        @media (max-width: 560px) {
            .timeline-grid { grid-template-columns: 1fr; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Fitting candidate models and preparing the 2017 holdout…")
def load_analysis() -> tuple[pd.DataFrame, dict]:
    """Load the source data and cache the complete chronological analysis."""

    source_data = load_and_validate_data(DATA_PATH)
    analysis_results = run_full_analysis(source_data)
    return source_data, analysis_results


def metric_pair(actual: pd.Series, frame: pd.DataFrame) -> tuple[dict, dict]:
    """Calculate baseline and final metrics for the active filters."""

    baseline = regression_metrics(actual, frame["baseline_forecast"])
    final = regression_metrics(actual, frame["final_forecast"])
    return baseline, final


def improvement_pct(baseline_value: float, final_value: float) -> float:
    """Calculate the percent reduction in forecast error."""

    return 100.0 * (1.0 - final_value / baseline_value)


def forecast_chart(frame: pd.DataFrame, show_interval: bool) -> go.Figure:
    """Build the daily actual-versus-forecast chart."""

    figure = go.Figure()
    if show_interval:
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["interval_upper"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name="80% upper",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["interval_lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(31, 122, 117, 0.14)",
                hoverinfo="skip",
                name="Empirical 80% band",
            )
        )

    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["baseline_forecast"],
            mode="lines",
            name="Promotion-only baseline",
            line=dict(color=ORANGE, width=2, dash="dot"),
            hovertemplate="Baseline: %{y:.1f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["final_forecast"],
            mode="lines",
            name="Final forecast",
            line=dict(color=TEAL, width=2.8),
            hovertemplate="Final: %{y:.1f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["sales"],
            mode="lines",
            name="Actual sales",
            line=dict(color=NAVY, width=2.3),
            hovertemplate="Actual: %{y:.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=515,
        margin=dict(l=10, r=10, t=22, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            linecolor="#D9D3CA",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Daily units sold",
            gridcolor="#ECE7DF",
            zeroline=False,
        ),
        font=dict(family="DM Sans, Arial", color=INK),
    )
    return figure


def metric_comparison_chart(baseline: dict, final: dict) -> go.Figure:
    """Build a compact RMSE/MAE comparison chart for the active filters."""

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Baseline",
            x=["RMSE", "MAE"],
            y=[baseline["rmse"], baseline["mae"]],
            marker_color=ORANGE,
            text=[f'{baseline["rmse"]:.1f}', f'{baseline["mae"]:.1f}'],
            textposition="outside",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Final model",
            x=["RMSE", "MAE"],
            y=[final["rmse"], final["mae"]],
            marker_color=TEAL,
            text=[f'{final["rmse"]:.1f}', f'{final["mae"]:.1f}'],
            textposition="outside",
        )
    )
    figure.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        yaxis=dict(title="Forecast error (units)", gridcolor="#ECE7DF"),
        xaxis=dict(title=None),
        legend=dict(orientation="h", y=1.12, x=0),
        font=dict(family="DM Sans, Arial", color=INK),
    )
    return figure


def residual_histogram(diagnostics: pd.DataFrame) -> go.Figure:
    """Build the required residual histogram on the modeled log scale."""

    figure = go.Figure(
        go.Histogram(
            x=diagnostics["residual_log_sales"],
            nbinsx=55,
            marker_color=TEAL,
            opacity=0.88,
            hovertemplate="Residual: %{x:.3f}<br>Count: %{y}<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=2, line_dash="dash", line_color=NAVY)
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Residual in log(1 + sales)",
        yaxis_title="Days × stores",
        bargap=0.03,
        font=dict(family="DM Sans, Arial", color=INK),
    )
    return figure


def residual_scatter(diagnostics: pd.DataFrame) -> go.Figure:
    """Build the required residual-versus-fitted diagnostic chart."""

    figure = go.Figure(
        go.Scattergl(
            x=diagnostics["fitted_log_sales"],
            y=diagnostics["residual_log_sales"],
            mode="markers",
            marker=dict(color=ORANGE, size=4, opacity=0.25),
            hovertemplate="Fitted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=2, line_dash="dash", line_color=NAVY)
    figure.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis_title="Fitted log(1 + sales)",
        yaxis_title="Residual in log(1 + sales)",
        font=dict(family="DM Sans, Arial", color=INK),
    )
    return figure


def driver_chart(driver_changes: pd.DataFrame) -> go.Figure:
    """Show the direction and size of modeled operational effects."""

    colors = [TEAL if value >= 0 else ORANGE for value in driver_changes["modeled_change_pct"]]
    figure = go.Figure(
        go.Bar(
            x=driver_changes["modeled_change_pct"],
            y=driver_changes["driver"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.1f}%" for value in driver_changes["modeled_change_pct"]],
            textposition="outside",
            customdata=driver_changes["comparison"],
            hovertemplate="%{customdata}: %{x:+.1f}%<extra></extra>",
        )
    )
    figure.add_vline(x=0, line_width=1.5, line_color=NAVY)
    figure.update_layout(
        height=340,
        margin=dict(l=10, r=55, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(title="Modeled change in daily sales", ticksuffix="%", gridcolor="#ECE7DF"),
        yaxis=dict(title=None, autorange="reversed"),
        showlegend=False,
        font=dict(family="DM Sans, Arial", color=INK),
    )
    return figure


source_data, analysis = load_analysis()
forecasts = analysis["forecasts"]
summary = analysis["summary"]

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Bakery demand lab · one croissant · ten stores</div>
        <h1>What should tomorrow's croissant demand look like?</h1>
        <p>
            Explore the untouched 2017 test year and compare management's
            promotion-only baseline with a fuller daily demand model.
        </p>
        <div class="status-row">
            <span class="status-pill">2017 holdout only</span>
            <span class="status-pill">Daily forecasts</span>
            <span class="status-pill">RMSE + MAE</span>
            <span class="status-pill">Empirical uncertainty band</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_forecast, tab_evidence, tab_method = st.tabs(
    ["Forecast explorer", "Model evidence", "Method & limitations"]
)


with tab_forecast:
    st.markdown('<div class="section-kicker">Interactive holdout</div>', unsafe_allow_html=True)
    st.subheader("Choose a store and a continuous 2017 period")

    store_types = forecasts.groupby("store_id")["store_type"].first().to_dict()
    with st.container(border=True):
        control_1, control_2, control_3 = st.columns([1.0, 2.1, 1.0])
        with control_1:
            selected_store = st.selectbox(
                "Store",
                options=sorted(forecasts["store_id"].unique()),
                format_func=lambda store: f"Store {store} · {store_types[store].title()}",
            )
        with control_2:
            selected_months = st.select_slider(
                "Continuous month range",
                options=list(range(1, 13)),
                value=(1, 12),
                format_func=lambda month: calendar.month_abbr[month],
            )
        with control_3:
            show_interval = st.toggle("Show 80% forecast band", value=True)

    start_month, end_month = selected_months
    selected = forecasts[
        (forecasts["store_id"] == selected_store)
        & forecasts["month"].between(start_month, end_month)
    ].copy()
    baseline_metrics, final_metrics = metric_pair(selected["sales"], selected)
    rmse_improvement = improvement_pct(
        baseline_metrics["rmse"], final_metrics["rmse"]
    )
    mae_improvement = improvement_pct(
        baseline_metrics["mae"], final_metrics["mae"]
    )
    coverage = 100.0 * selected["inside_interval"].mean()

    period_label = (
        calendar.month_name[start_month]
        if start_month == end_month
        else f"{calendar.month_name[start_month]}–{calendar.month_name[end_month]}"
    )
    st.caption(
        f"Store {selected_store} · {period_label} 2017 · {len(selected):,} daily observations"
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Baseline RMSE", f'{baseline_metrics["rmse"]:.1f} units')
    metric_2.metric(
        "Final RMSE",
        f'{final_metrics["rmse"]:.1f} units',
        f"{rmse_improvement:.1f}% lower",
    )
    metric_3.metric("Baseline MAE", f'{baseline_metrics["mae"]:.1f} units')
    metric_4.metric(
        "Final MAE",
        f'{final_metrics["mae"]:.1f} units',
        f"{mae_improvement:.1f}% lower",
    )

    st.plotly_chart(
        forecast_chart(selected, show_interval),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    chart_col, note_col = st.columns([1.45, 1.0], gap="large")
    with chart_col:
        st.markdown('<div class="section-kicker">Error comparison</div>', unsafe_allow_html=True)
        st.subheader("Lower is better")
        st.plotly_chart(
            metric_comparison_chart(baseline_metrics, final_metrics),
            width="stretch",
            config={"displayModeBar": False},
        )
    with note_col:
        st.markdown('<div class="section-kicker">Read the forecast</div>', unsafe_allow_html=True)
        st.subheader("What changed?")
        st.markdown(
            f"""
            <div class="callout">
                The final model reduces RMSE by <strong>{rmse_improvement:.1f}%</strong>
                and MAE by <strong>{mae_improvement:.1f}%</strong> for this selection.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            "The baseline moves only with promotion. The final forecast also accounts "
            "for the store, weekday, bad weather, and promotion."
        )
        st.metric("Observed 80% band coverage", f"{coverage:.1f}%")
        st.caption(
            "The band uses 2016 validation errors and was fixed before the 2017 test "
            "year was evaluated. It is an empirical planning range, not a formal "
            "conditional prediction interval."
        )

    with st.expander("View and download the selected daily forecasts"):
        export = selected[
            [
                "date",
                "store_id",
                "store_type",
                "sales",
                "baseline_forecast",
                "final_forecast",
                "interval_lower",
                "interval_upper",
                "promotion",
                "bad_weather",
                "day",
            ]
        ].copy()
        export = export.rename(
            columns={
                "sales": "actual_sales",
                "interval_lower": "final_80pct_lower",
                "interval_upper": "final_80pct_upper",
            }
        )
        numeric_columns = export.select_dtypes(include=np.number).columns
        export[numeric_columns] = export[numeric_columns].round(2)
        st.dataframe(export, width="stretch", hide_index=True)
        st.download_button(
            "Download selected forecasts as CSV",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name=f"store_{selected_store}_{start_month:02d}_{end_month:02d}_2017_forecasts.csv",
            mime="text/csv",
        )


with tab_evidence:
    st.markdown('<div class="section-kicker">Model selection</div>', unsafe_allow_html=True)
    st.subheader("The decision was made on 2016, not 2017")
    best_validation_rmse = analysis["validation_comparison"]["rmse"].min()
    selected_validation_rmse = analysis["validation_comparison"].loc[
        analysis["validation_comparison"]["selected"], "rmse"
    ].iloc[0]
    practical_gap_pct = 100.0 * (
        selected_validation_rmse / best_validation_rmse - 1.0
    )
    st.markdown(
        f"""
        <div class="callout">
            Three log models finished within 0.1% of the best 2016 RMSE. The
            18-parameter operational log model was only
            <strong>{practical_gap_pct:.2f}%</strong> above the absolute minimum.
            Applying the documented practical-tie rule, the fewest-parameter model
            was retained to reduce unnecessary complexity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    validation_display = analysis["validation_comparison"].copy()
    validation_display["selected"] = validation_display["selected"].map(
        {True: "✓ Final", False: ""}
    )
    validation_display = validation_display.rename(
        columns={
            "model": "Candidate model",
            "rmse": "2016 RMSE",
            "mae": "2016 MAE",
            "bias": "2016 bias",
            "parameters": "Parameters",
            "selected": "Decision",
        }
    )
    for column in ["2016 RMSE", "2016 MAE", "2016 bias"]:
        validation_display[column] = validation_display[column].round(2)
    st.dataframe(
        validation_display,
        width="stretch",
        hide_index=True,
        column_config={
            "2016 RMSE": st.column_config.NumberColumn(format="%.2f"),
            "2016 MAE": st.column_config.NumberColumn(format="%.2f"),
            "2016 bias": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.markdown('<div class="section-kicker">One final test</div>', unsafe_allow_html=True)
    st.subheader("Overall performance on the untouched 2017 year")
    test_display = analysis["test_comparison"].rename(
        columns={"model": "Model", "rmse": "RMSE", "mae": "MAE", "bias": "Bias"}
    ).copy()
    for column in ["RMSE", "MAE", "Bias"]:
        test_display[column] = test_display[column].round(2)
    test_col, score_col = st.columns([1.45, 1.0], gap="large")
    with test_col:
        st.dataframe(test_display, width="stretch", hide_index=True)
    with score_col:
        score_1, score_2 = st.columns(2)
        score_1.metric(
            "RMSE reduction",
            f'{summary["test_rmse_improvement_pct"]:.1f}%',
        )
        score_2.metric(
            "MAE reduction",
            f'{summary["test_mae_improvement_pct"]:.1f}%',
        )
        st.caption(
            "These test scores describe generalization. They were not used to choose "
            "features, transformations, or interactions."
        )

    st.markdown('<div class="section-kicker">Operational effects</div>', unsafe_allow_html=True)
    st.subheader("What the final model learned")
    st.plotly_chart(
        driver_chart(analysis["driver_changes"]),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Effects hold the reference store and other listed conditions constant. They "
        "describe model associations, not randomized causal effects."
    )

    st.markdown('<div class="section-kicker">Required diagnostics</div>', unsafe_allow_html=True)
    st.subheader("Residual behavior after the log transformation")
    diag_1, diag_2 = st.columns(2, gap="large")
    with diag_1:
        st.plotly_chart(
            residual_histogram(analysis["diagnostics"]),
            width="stretch",
            config={"displayModeBar": False},
        )
    with diag_2:
        st.plotly_chart(
            residual_scatter(analysis["diagnostics"]),
            width="stretch",
            config={"displayModeBar": False},
        )
    st.markdown(
        f"""
        <div class="callout">
            Log-scale residual skewness is <strong>{summary['residual_skew_log_scale']:.2f}</strong>.
            The transformation makes the residual distribution more symmetric and
            stabilizes spread, although high-demand days still produce some large errors.
        </div>
        """,
        unsafe_allow_html=True,
    )
    assumption_checks = pd.DataFrame(
        [
            {
                "Assumption": "Functional form",
                "Evidence": "Log response and categorical store/weekday effects; no strong residual pattern",
                "Remaining concern": "Holidays and other omitted nonlinear effects may remain",
            },
            {
                "Assumption": "Residual normality",
                "Evidence": f"Log-residual skewness = {summary['residual_skew_log_scale']:.3f}",
                "Remaining concern": "Individual high-demand days can still create large errors",
            },
            {
                "Assumption": "Constant variance",
                "Evidence": (
                    "Correlation of |residual| with fitted value = "
                    f"{summary['absolute_residual_fitted_correlation']:.3f}"
                ),
                "Remaining concern": "A visual diagnostic is more informative than one correlation",
            },
            {
                "Assumption": "Independent errors",
                "Evidence": (
                    "Mean within-store lag-1 residual correlation = "
                    f"{summary['mean_store_lag1_residual_correlation']:.3f}"
                ),
                "Remaining concern": "Repeated store-day observations can still share unmodeled dependence",
            },
            {
                "Assumption": "Exogenous predictors",
                "Evidence": "Not established by observational data",
                "Remaining concern": "Promotion timing may respond to expected demand; effects are not causal",
            },
        ]
    )
    st.subheader("Assumption check")
    st.dataframe(assumption_checks, width="stretch", hide_index=True)


with tab_method:
    st.markdown('<div class="section-kicker">Leakage-safe workflow</div>', unsafe_allow_html=True)
    st.subheader("Every year has one job")
    st.markdown(
        f"""
        <div class="timeline-grid">
            <div class="timeline-card">
                <strong>2013–2015</strong>
                <span>Fit candidate models<br>{summary['development_rows']:,} rows</span>
            </div>
            <div class="timeline-card">
                <strong>2016</strong>
                <span>Compare RMSE and MAE<br>{summary['validation_rows']:,} rows</span>
            </div>
            <div class="timeline-card">
                <strong>2013–2016</strong>
                <span>Refit the locked models<br>{summary['final_training_rows']:,} rows</span>
            </div>
            <div class="timeline-card">
                <strong>2017</strong>
                <span>Evaluate once and explore<br>{summary['test_rows']:,} rows</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_col, metric_col = st.columns(2, gap="large")
    with model_col:
        st.subheader("Two forecasts")
        st.markdown("**Management baseline**")
        st.latex(r"Sales = \beta_0 + \beta_1(Promotion) + \varepsilon")
        st.markdown("**Selected final model**")
        st.latex(
            r"\log(1+Sales) = \beta_0 + \beta_1Promotion + "
            r"\beta_2Weather + Store + Weekday + \varepsilon"
        )
        st.write(
            "Predictions are returned to the sales scale with Duan's smearing "
            f"correction (factor {summary['smearing_factor']:.3f}) and clipped at zero."
        )
    with metric_col:
        st.subheader("Two error measures")
        st.markdown(
            "**RMSE** penalizes a few large misses more heavily, which matters when a "
            "bakery wants to avoid extreme under- or over-production."
        )
        st.markdown(
            "**MAE** is the typical absolute miss in croissants and is easier to read "
            "in operational units."
        )
        st.markdown(
            "The 80% band adds the 10th and 90th percentiles of 2016 validation errors "
            "to each final forecast."
        )

    st.markdown('<div class="section-kicker">Interpret with care</div>', unsafe_allow_html=True)
    st.subheader("Assumptions and limitations")
    limitation_1, limitation_2 = st.columns(2, gap="large")
    with limitation_1:
        st.markdown(
            """
            - The data cover one croissant item and may not generalize to other products
            - Promotion and weather effects are observational associations, not causal estimates
            - Store fixed effects assume each store's underlying demand level is stable
            - Store fixed effects do not support forecasts for a brand-new, unseen store
            - Forecasts require the future promotion plan and weather indicator to be known
            """
        )
    with limitation_2:
        st.markdown(
            """
            - The model does not include price, holidays, stockouts, local events, or competitor activity
            - Consecutive daily errors may be correlated even though OLS treats rows as independent
            - Model choice relies on one validation year rather than repeated rolling-origin validation
            - The empirical band is global and may be too wide or narrow for a particular store or day
            - Accuracy does not directly optimize waste, shortage, or profit costs
            """
        )

    st.markdown(
        """
        <div class="callout warning-callout">
            A practical next step would attach the cost of one leftover croissant and
            one missed sale, then convert demand forecasts into store-level bake quantities.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <div class="footer-note">
        Source: bakerysales.csv supplied with the MSBX 5415 forecasting case ·
        Models and dashboard prepared for the 2013–2017 case analysis
    </div>
    """,
    unsafe_allow_html=True,
)

