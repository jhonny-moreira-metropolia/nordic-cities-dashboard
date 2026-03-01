"""
app.py - Nordic Climate Dashboard
Streamlit app for visualizing monthly temperatures of Nordic capitals
with AI forecasting using Facebook Prophet.

Expected repository structure:
    app.py
    data_processor.py
    forecaster.py
    requirements.txt
    data/
        Helsinki.csv
        Copenhagen.csv
        Stockholm.csv
        Reykjavik.csv
        Oslo.csv

Local usage:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_processor import (
    load_all_cities,
    CITY_FILES,
    DATA_DIR,
    MONTH_SHORT,
    MIN_DAYS_PER_MONTH,
)
from forecaster import forecast_all_cities, forecast_to_dataframe

# ------------------------------------------------------------------------------
CITY_COLORS = {
    "Helsinki":   "#4DA8DA",
    "Stockholm":  "#F4A942",
    "Copenhagen": "#6BCB77",
    "Oslo":       "#C77DFF",
    "Reykjavik":  "#FF6B6B",
}

PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="'DM Sans', sans-serif", color="#c9d1d9", size=13),
    hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_size=13),
)

# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Nordic Climate Dashboard",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; letter-spacing:-0.02em; color: #e6edf3 !important; }
.main { background: #0d1117 !important; }
.block-container { padding: 1.5rem 2.5rem 3rem; background: #0d1117 !important; }
[data-testid="stAppViewContainer"] { background: #0d1117 !important; }
[data-testid="stMain"] { background: #0d1117 !important; }
[data-testid="stHeader"] { background: #0d1117 !important; }
header[data-testid="stHeader"] { background: #0d1117 !important; }
p, span, div, label { color: #c9d1d9; }

.kpi-card {
    background: linear-gradient(135deg,#161b22,#1c2128);
    border: 1px solid #30363d; border-radius: 14px;
    padding: 1.1rem 1.3rem 1rem; text-align: center;
}
.kpi-label { font-size: 0.68rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.kpi-value { font-size: 1.9rem; font-weight: 500; color: #e6edf3; line-height: 1.1; }
.kpi-sub   { font-size: 0.75rem; color: #8b949e; margin-top: 0.25rem; }

.sec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #c9d1d9 !important;
    margin: 2rem 0 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #4DA8DA;
}

[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] label { font-size: 0.78rem !important; color: #8b949e; text-transform: uppercase; letter-spacing: 0.07em; }

.warn-box {
    background: #1c1a10; border: 1px solid #4a3d00;
    border-left: 3px solid #d4a017; border-radius: 8px;
    padding: 0.7rem 1rem; font-size: 0.85rem; color: #c9a227; margin: 0.5rem 0;
}
.info-box {
    background: #0d1e2e; border: 1px solid #1f4068;
    border-left: 3px solid #4DA8DA; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.85rem; color: #8b949e; margin-bottom: 1rem;
}
.ai-badge {
    display: inline-block;
    background: linear-gradient(90deg,#1a1a2e,#16213e);
    border: 1px solid #4DA8DA; border-radius: 20px;
    padding: 0.2rem 0.8rem; font-size: 0.75rem;
    color: #4DA8DA; letter-spacing: 0.05em; margin-bottom: 0.5rem;
}

[data-testid="stDownloadButton"] button {
    background: #161b22 !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #21262d !important; border-color: #4DA8DA !important;
}

[data-testid="stSelectbox"] > div > div {
    background: #161b22 !important; color: #c9d1d9 !important;
    border-color: #30363d !important;
}
div[data-baseweb="select"] > div {
    background: #161b22 !important; color: #c9d1d9 !important;
    border-color: #30363d !important;
}
div[data-baseweb="popover"] { background: #161b22 !important; }
div[data-baseweb="menu"]    { background: #161b22 !important; }
div[data-baseweb="option"]  { background: #161b22 !important; color: #c9d1d9 !important; }
div[data-baseweb="option"]:hover { background: #21262d !important; }
div[data-baseweb="popover"] ul { background: #161b22 !important; }
div[data-baseweb="popover"] li { background: #161b22 !important; color: #c9d1d9 !important; }
div[data-baseweb="popover"] li:hover { background: #21262d !important; }

[data-testid="stElementToolbar"] {
    background: #161b22 !important; border: 1px solid #30363d !important;
}
[data-testid="stElementToolbar"] button { color: #c9d1d9 !important; }
div[class*="Toolbar"] { background: #161b22 !important; }
div[class*="toolbar"] { background: #161b22 !important; }
div[data-testid="stTooltipContent"], div[role="tooltip"] {
    background: #161b22 !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
def kpi(label, value, sub=""):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def sec(title):
    st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

def apply_layout(fig, height=380, show_legend=True, rangeslider=False):
    fig.update_layout(
        **PLOTLY_BASE, height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            gridcolor="#21262d", showgrid=False,
            rangeslider=dict(visible=rangeslider, thickness=0.05) if rangeslider else {},
        ),
        yaxis=dict(gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1) if show_legend else dict(visible=False),
    )
    return fig


# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading climate data...")
def get_data():
    return load_all_cities(DATA_DIR)

try:
    df_daily, df_monthly = get_data()
except FileNotFoundError as e:
    st.error(f"**Missing data files.**\n\n{e}")
    st.markdown("""
    Make sure the `data/` folder exists in the repo with these files:
    ```
    data/
      Helsinki.csv
      Copenhagen.csv
      Stockholm.csv
      Reykjavik.csv
      Oslo.csv
    ```
    """)
    st.stop()
except Exception as e:
    st.error(f"**Error loading data:** {e}")
    st.stop()


# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Nordic Climate")
    st.markdown("---")
    st.markdown("#### Filters")

    all_cities = sorted(df_monthly["CITY"].unique())
    sel_cities = st.multiselect(
        "Cities", all_cities,
        default=["Helsinki"] if "Helsinki" in all_cities else all_cities[:1],
    )

    if sel_cities:
        # Use the common overlapping range: latest start, earliest end
        city_starts = [int(df_monthly[df_monthly["CITY"]==c]["YEAR"].min()) for c in sel_cities]
        city_ends   = [int(df_monthly[df_monthly["CITY"]==c]["YEAR"].max()) for c in sel_cities]
        ymin = max(city_starts)
        ymax = min(city_ends)
    else:
        ymin = int(df_monthly["YEAR"].min())
        ymax = int(df_monthly["YEAR"].max())
    if ymin >= ymax:
        ymin = ymax - 1
    sel_years = st.slider("Year range", ymin, ymax, (ymin, ymax))                 if ymin < ymax else (ymin, ymax)

    temp_var = st.selectbox(
        "Temperature variable",
        ["TAVG_mean", "TMAX_mean", "TMIN_mean"],
        format_func=lambda x: {
            "TAVG_mean": "Mean temperature (TAVG)",
            "TMAX_mean": "Max temperature (TMAX)",
            "TMIN_mean": "Min temperature (TMIN)",
        }[x]
    )

    st.markdown("---")
    st.markdown("#### AI Forecast (Prophet)")
    forecast_periods = st.slider("Months to forecast", 6, 60, 36, step=6)
    run_forecast = st.button("Run forecast", type="primary", use_container_width=True)

    if run_forecast:
        with st.spinner("Training Prophet models..."):
            results = forecast_all_cities(
                df_monthly, sel_cities, temp_var, forecast_periods,
            )
        st.session_state["forecast_results"] = results
        st.session_state["forecast_var"]     = temp_var

        ok   = sum(1 for r in results.values() if r.ok)
        fail = [c for c, r in results.items() if not r.ok]
        if ok:
            st.sidebar.success(f"{ok} model(s) trained")
        for city in fail:
            st.sidebar.warning(f"Warning - {city}: {results[city].error}")

    st.markdown("---")
    st.markdown(
        "<small style='color:#484f58'>ECA&D - European Climate Assessment & Dataset<br>"
        "ecad.eu</small>", unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
mdf = df_monthly[
    df_monthly["CITY"].isin(sel_cities) &
    df_monthly["YEAR"].between(sel_years[0], sel_years[1])
].copy()

TEMP_LABEL = {
    "TAVG_mean": "Mean Temp (°C)",
    "TMAX_mean": "Max Temp (°C)",
    "TMIN_mean": "Min Temp (°C)",
}.get(temp_var, "°C")

if mdf.empty or temp_var not in mdf.columns:
    st.warning("No data available for the current selection.")
    st.stop()

fc_periods = forecast_periods


# ------------------------------------------------------------------------------
st.markdown("# Nordic Climate Dashboard")
cities_str = " - ".join(sorted(df_monthly["CITY"].unique()))
st.markdown(
    f"<span style='color:#8b949e;font-size:0.9rem'>"
    f"Historical climate data - ECA&D - {cities_str}"
    f"</span>",
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION 1 - HISTORICAL ANALYSIS
# ==============================================================================

# Key metrics - one row per selected city
for city in sel_cities:
    cdf = mdf[mdf["CITY"] == city]
    if cdf.empty or temp_var not in cdf.columns:
        continue
    color = CITY_COLORS.get(city, "#4DA8DA")
    st.markdown(
        f"<div style='font-size:0.8rem;font-weight:600;color:{color};"
        f"text-transform:uppercase;letter-spacing:0.08em;"
        f"margin:1.2rem 0 0.4rem'>{city}</div>",
        unsafe_allow_html=True,
    )
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi("Mean temperature", f"{cdf[temp_var].mean():.1f} °C",
            f"{int(cdf['YEAR'].min())}-{int(cdf['YEAR'].max())}")
    with k2:
        if "TMAX_mean" in cdf.columns and cdf["TMAX_mean"].notna().any():
            r = cdf.loc[cdf["TMAX_mean"].idxmax()]
            kpi("Warmest month", f"{r['TMAX_mean']:.1f} °C",
                f"{MONTH_SHORT[int(r['MONTH'])]} {int(r['YEAR'])}")
    with k3:
        if "TMIN_mean" in cdf.columns and cdf["TMIN_mean"].notna().any():
            r = cdf.loc[cdf["TMIN_mean"].idxmin()]
            kpi("Coldest month", f"{r['TMIN_mean']:.1f} °C",
                f"{MONTH_SHORT[int(r['MONTH'])]} {int(r['YEAR'])}")

# Time series
sec("Monthly temperature evolution")

show_trend = st.toggle(
    "Show 12-month trend line", value=False,
    help="Smooths seasonal variation to reveal long-term warming or cooling trends.",
)

fig_line = go.Figure()
for city in sel_cities:
    cdf   = mdf[mdf["CITY"] == city].sort_values(["YEAR","MONTH"])
    if cdf.empty or temp_var not in cdf.columns: continue
    color = CITY_COLORS.get(city, "#aaa")
    good  = cdf[~cdf["LOW_COVERAGE"]]
    low   = cdf[cdf["LOW_COVERAGE"]]

    fig_line.add_trace(go.Scatter(
        x=good["PERIOD"], y=good[temp_var],
        name=city, mode="lines",
        line=dict(color=color, width=1.2),
        opacity=0.3 if show_trend else 1.0,
        customdata=good["COVERAGE"],
        hovertemplate=(
            f"<b>{city}</b><br>%{{x}}<br>{TEMP_LABEL}: %{{y:.1f}} °C"
            "<br>Coverage: %{customdata:.0f}%<extra></extra>"
        ),
        legendgroup=city,
    ))
    if not low.empty:
        fig_line.add_trace(go.Scatter(
            x=low["PERIOD"], y=low[temp_var],
            name=f"{city} (low cov.)", mode="markers",
            marker=dict(color=color, size=5, symbol="x", opacity=0.5),
            customdata=low["COVERAGE"],
            hovertemplate=(
                f"<b>{city}</b><br>%{{x}}<br>{TEMP_LABEL}: %{{y:.1f}} °C"
                "<br>Warning: Coverage: %{customdata:.0f}%<extra></extra>"
            ),
            legendgroup=city, showlegend=False,
        ))

    # 12-month rolling average
    if show_trend and len(good) >= 12:
        trend = good[temp_var].rolling(window=12, center=True, min_periods=6).mean()
        r_c = int(color[1:3], 16)
        g_c = int(color[3:5], 16)
        b_c = int(color[5:7], 16)
        fig_line.add_trace(go.Scatter(
            x=good["PERIOD"], y=trend,
            name=f"{city} (trend)",
            mode="lines",
            line=dict(color=f"rgba({r_c},{g_c},{b_c},1)", width=2.8),
            hovertemplate=(
                f"<b>{city} — 12m trend</b><br>%{{x}}<br>"
                f"{TEMP_LABEL}: %{{y:.2f}} °C<extra></extra>"
            ),
            legendgroup=city,
        ))

apply_layout(fig_line, height=420, rangeslider=True)
fig_line.update_layout(
    yaxis=dict(gridcolor="#21262d", title=TEMP_LABEL),
    legend=dict(font=dict(color="#c9d1d9")),
)
st.plotly_chart(fig_line, use_container_width=True)

# City comparison
sec("City comparison")

col_a, col_b = st.columns(2)

with col_a:
    fig_box = go.Figure()
    for city in sel_cities:
        cdf = mdf[(mdf["CITY"] == city) & (~mdf["LOW_COVERAGE"])]
        if cdf.empty: continue
        fig_box.add_trace(go.Box(
            y=cdf[temp_var], name=city,
            marker_color=CITY_COLORS.get(city, "#aaa"), boxmean="sd",
            hovertemplate=f"<b>{city}</b><br>%{{y:.1f}} °C<extra></extra>",
        ))
    apply_layout(fig_box, height=360, show_legend=False)
    fig_box.update_layout(
        title=dict(text="Monthly distribution", font=dict(size=14, color="#8b949e")),
        yaxis=dict(gridcolor="#21262d", title=TEMP_LABEL),
        legend=dict(font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_b:
    seasonal = (
        mdf[~mdf["LOW_COVERAGE"]]
        .groupby(["CITY","MONTH"])[temp_var].mean().reset_index()
    )
    fig_sea = go.Figure()
    for city in sel_cities:
        cdf = seasonal[seasonal["CITY"] == city]
        if cdf.empty: continue
        color = CITY_COLORS.get(city, "#aaa")
        fig_sea.add_trace(go.Scatter(
            x=cdf["MONTH"], y=cdf[temp_var],
            name=city, mode="lines+markers",
            line=dict(color=color, width=2.2), marker=dict(size=7),
            hovertemplate=(
                f"<b>{city}</b><br>%{{x}}<br>{TEMP_LABEL}: %{{y:.1f}} °C<extra></extra>"
            ),
        ))
    apply_layout(fig_sea, height=360)
    fig_sea.update_layout(
        title=dict(text="Average seasonal cycle", font=dict(size=14, color="#8b949e")),
        xaxis=dict(tickvals=list(range(1,13)),
                   ticktext=list(MONTH_SHORT.values()), showgrid=False),
        yaxis=dict(gridcolor="#21262d", title=TEMP_LABEL),
        legend=dict(font=dict(color="#c9d1d9")),
    )
    st.plotly_chart(fig_sea, use_container_width=True)

# Heatmap
sec("Monthly heatmap")

valid_hm = [
    c for c in sel_cities
    if temp_var in mdf[mdf["CITY"]==c].columns
    and not mdf[(mdf["CITY"]==c) & (~mdf["LOW_COVERAGE"])].empty
]
if valid_hm:
    hm_city = st.selectbox("City", valid_hm, key="hm_sel")
    current_year = pd.Timestamp.now().year
    hm_data = mdf[
        (mdf["CITY"] == hm_city) &
        (~mdf["LOW_COVERAGE"]) &
        (mdf["YEAR"] < current_year)
    ]
    if not hm_data.empty:
        pivot = hm_data.pivot_table(
            index="YEAR", columns="MONTH", values=temp_var, aggfunc="mean"
        )
        pivot.columns = [MONTH_SHORT[m] for m in pivot.columns]
        fig_hm = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
            colorscale=[
                [0.00,"#072459"],[0.20,"#1a5fa3"],[0.45,"#4da6d4"],
                [0.60,"#f7f7f7"],[0.75,"#f4a942"],[0.90,"#e85c2a"],[1.00,"#b01a0e"],
            ],
            text=np.round(pivot.values, 1),
            texttemplate="%{text}", textfont=dict(size=10),
            hovertemplate=(
                "Year: %{y}<br>Month: %{x}<br>"+TEMP_LABEL+": %{z:.1f} °C<extra></extra>"
            ),
            colorbar=dict(title="°C", tickfont=dict(color="#c9d1d9")),
        ))
        fig_hm.update_layout(
            **PLOTLY_BASE,
            height=max(350, len(pivot) * 22 + 80),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(side="top", showgrid=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# ==============================================================================
# SECTION 2 - AI FORECAST
# ==============================================================================
st.markdown("---")
st.markdown('<div class="ai-badge">* PROPHET - FACEBOOK AI RESEARCH</div>', unsafe_allow_html=True)
sec("AI Forecast - Monthly temperature prediction")
st.markdown(
    f"<span style='color:#8b949e;font-size:0.88rem'>"
    f"Prophet model trained on historical data - "
    f"<b style='color:#c9d1d9'>{fc_periods}-month</b> projection - "
    f"95% confidence interval - "
    f"Variable: <b style='color:#c9d1d9'>{TEMP_LABEL}</b>"
    f"</span>",
    unsafe_allow_html=True,
)

forecast_results = st.session_state.get("forecast_results", {})
forecast_var     = st.session_state.get("forecast_var", temp_var)

if not forecast_results:
    st.markdown("""
    <div class="info-box">
    Set the number of months in the sidebar and press
    <strong>Run forecast</strong> to train the models.<br><br>
    Prophet automatically captures <strong>annual seasonality</strong>
    (warm summers, cold winters) and <strong>long-term trends</strong>.
    </div>
    """, unsafe_allow_html=True)

else:
    if forecast_var != temp_var:
        st.markdown(
            f'<div class="warn-box">Warning: Forecast was calculated for <b>{forecast_var}</b> '
            f'but you are now viewing <b>{temp_var}</b>. '
            f'Re-run the forecast to update.</div>',
            unsafe_allow_html=True,
        )

    for city in sel_cities:
        if city not in forecast_results:
            continue
        res   = forecast_results[city]
        color = CITY_COLORS.get(city, "#aaa")

        if not res.ok:
            st.markdown(
                f'<div class="warn-box">Warning: <b>{city}</b>: {res.error}</div>',
                unsafe_allow_html=True,
            )
            continue

        sec(city)
        hist = res.historical
        pred = res.future_only

        fig_fc = go.Figure()
        r_c, g_c, b_c = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([pred["DATE"], pred["DATE"][::-1]]),
            y=pd.concat([pred["UPPER"], pred["LOWER"][::-1]]),
            fill="toself",
            fillcolor=f"rgba({r_c},{g_c},{b_c},0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=True, name="95% interval",
        ))
        fig_fc.add_trace(go.Scatter(
            x=pred["DATE"], y=pred["FORECAST"],
            mode="lines", name="AI Forecast",
            line=dict(color=color, width=2.5, dash="dash"),
            hovertemplate=(
                "<b>Forecast</b><br>%{x|%b %Y}<br>"
                f"{TEMP_LABEL}: %{{y:.1f}} °C<extra></extra>"
            ),
        ))
        fig_fc.add_trace(go.Scatter(
            x=hist["ds"], y=hist["y"],
            mode="lines", name="Historical data",
            line=dict(color="#6e7681", width=1.2),
            hovertemplate=(
                "<b>Historical</b><br>%{x|%b %Y}<br>"
                f"{TEMP_LABEL}: %{{y:.1f}} °C<extra></extra>"
            ),
        ))
        last_date = hist["ds"].max()
        fig_fc.add_trace(go.Scatter(
            x=[last_date, last_date],
            y=[pred["LOWER"].min(), pred["UPPER"].max()],
            mode="lines",
            line=dict(color="#484f58", width=1, dash="dot"),
            name="Forecast start", showlegend=False,
        ))

        apply_layout(fig_fc, height=380)
        fig_fc.update_layout(
            yaxis=dict(gridcolor="#21262d", title=TEMP_LABEL),
            xaxis=dict(showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(color="#c9d1d9")),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Predicted mean temp", f"{pred['FORECAST'].mean():.1f} °C",
                f"next {fc_periods} months")
        with k2:
            hot = pred.loc[pred["FORECAST"].idxmax()]
            kpi("Warmest predicted month", f"{hot['FORECAST']:.1f} °C",
                hot["DATE"].strftime("%b %Y"))
        with k3:
            cold = pred.loc[pred["FORECAST"].idxmin()]
            kpi("Coldest predicted month", f"{cold['FORECAST']:.1f} °C",
                cold["DATE"].strftime("%b %Y"))
        with k4:
            kpi("Mean uncertainty",
                f"+/- {((pred['UPPER']-pred['LOWER'])/2).mean():.1f} °C",
                "95% interval")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Comparison across all cities
    valid = {c: r for c, r in forecast_results.items() if c in sel_cities and r.ok}
    if len(valid) > 1:
        sec("Forecast comparison - all cities")

        first_dates  = [r.future_only["DATE"].min() for r in valid.values()]
        latest_start = max(first_dates)
        comparable   = {
            c: r for c, r in valid.items()
            if r.future_only["DATE"].max() >= latest_start
        }

        if len(comparable) < 2:
            st.markdown(
                '<div class="warn-box">Not enough cities share a common forecast '
                'period for comparison.</div>',
                unsafe_allow_html=True,
            )
        else:
            fig_comp = go.Figure()
            for city, res in comparable.items():
                color = CITY_COLORS.get(city, "#aaa")
                pred  = res.future_only[res.future_only["DATE"] >= latest_start]
                fig_comp.add_trace(go.Scatter(
                    x=pred["DATE"], y=pred["FORECAST"],
                    name=city, mode="lines",
                    line=dict(color=color, width=2.2),
                    hovertemplate=(
                        f"<b>{city}</b><br>%{{x|%b %Y}}<br>"
                        f"{TEMP_LABEL}: %{{y:.1f}} °C<extra></extra>"
                    ),
                ))
            apply_layout(fig_comp, height=360)
            fig_comp.update_layout(
                yaxis=dict(gridcolor="#21262d", title=TEMP_LABEL),
                xaxis=dict(showgrid=False),
                legend=dict(font=dict(color="#c9d1d9")),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    # Table
    sec("Predicted values by month")
    all_pred = forecast_to_dataframe(
        {c: r for c, r in forecast_results.items() if c in sel_cities}
    )
    if not all_pred.empty:
        all_pred["Month"] = all_pred["DATE"].dt.strftime("%b %Y")
        table_fc = (
            all_pred[["CITY","Month","DATE","FORECAST","LOWER","UPPER"]]
            .sort_values(["DATE","CITY"])
            .drop(columns=["DATE"])
            .rename(columns={
                "CITY": "City",
                "FORECAST": f"{TEMP_LABEL} (forecast)",
                "LOWER": "Lower 95%",
                "UPPER": "Upper 95%",
            })
            .round(2)
            .reset_index(drop=True)
        )
        st.dataframe(table_fc, use_container_width=True, height=320)
        st.download_button(
            "Download forecast CSV",
            table_fc.to_csv(index=False).encode("utf-8"),
            "nordic_climate_forecast.csv", "text/csv",
        )

    with st.expander("About the Prophet model"):
        st.markdown(f"""
**Facebook Prophet** is an additive time series model designed for data with
strong seasonality and trend.

**Configuration:**
- Annual seasonality: enabled - captures the summer/winter cycle
- Mode: additive - appropriate for temperature data
- Confidence interval: 95%
- Trend regularization: moderate (0.05) - avoids overfitting

**Strengths:** seasonal cycles, gradual long-term trends (e.g. warming).

**Limitations:** does not anticipate extreme one-off events;
uncertainty grows with forecast horizon;
uses only `{TEMP_LABEL}` without external variables (CO2, NAO index, etc.)
        """)


# ==============================================================================
# SECTION 3 - DATA
# ==============================================================================
st.markdown("---")
sec("Monthly aggregated data")

RENAME = {
    "CITY":"City","PERIOD":"Period","YEAR":"Year","MONTH":"Month",
    "TAVG_mean":"TAVG mean (°C)","TMAX_mean":"TMAX mean (°C)","TMIN_mean":"TMIN mean (°C)",
    "TMAX_abs":"TMAX abs (°C)","TMIN_abs":"TMIN abs (°C)",
    "DATA_DAYS":"Data days","COVERAGE":"Coverage %","LOW_COVERAGE":"Low cov.",
}
show_cols = [c for c in RENAME if c in mdf.columns]
table = mdf[show_cols].rename(columns=RENAME).copy()
table["Month"] = table["Month"].map(MONTH_SHORT)

st.dataframe(
    table.sort_values(["Period","City"]).reset_index(drop=True),
    use_container_width=True, height=450,
)

dl1, dl2, _ = st.columns([1, 1, 4])
with dl1:
    st.download_button(
        "Monthly CSV",
        mdf[show_cols].to_csv(index=False).encode("utf-8"),
        "nordic_climate_monthly.csv", "text/csv",
    )
with dl2:
    daily_filt = df_daily[
        df_daily["CITY"].isin(sel_cities) &
        df_daily["DATE"].dt.year.between(sel_years[0], sel_years[1])
    ]
    st.download_button(
        "Daily CSV",
        daily_filt.to_csv(index=False).encode("utf-8"),
        "nordic_climate_daily.csv", "text/csv",
    )

st.markdown("---")
st.markdown(
    "<small style='color:#484f58'>Data - ECA&D European Climate Assessment & Dataset - ecad.eu</small>",
    unsafe_allow_html=True,
)
