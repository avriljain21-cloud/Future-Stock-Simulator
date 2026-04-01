import streamlit as st
import numpy as np
import plotly.graph_objects as go

from data import get_stock_data
from monte_carlo import (
    ewma_volatility,
    fit_t_distribution,
    simulate_gbm,
    compute_statistics,
)

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Monte Carlo Stock Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================
# CSS STYLING
# =======================
st.markdown(
    """
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #f3f4f6;
}
.main-title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    background: linear-gradient(90deg, #3b82f6, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom:5px;
}
.subtitle {
    text-align:center;
    font-size:16px;
    color:#9ca3af;
    margin-bottom:25px;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding:18px;
    border-radius:16px;
    text-align:center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin-bottom:8px;
}
.metric-label {font-size:13px;color:#9ca3af;}
.metric-value {font-size:28px;font-weight:600;}
.metric-delta {font-size:14px;}
.section-title {
    font-size:20px;
    font-weight:600;
    margin-top:25px;
    margin-bottom:15px;
}
.tooltip {font-size:11px;color:#6b7280;margin-top:4px;}
</style>
""",
    unsafe_allow_html=True,
)

# =======================
# HEADER
# =======================
st.markdown(
    '<div class="main-title">Monte Carlo Stock Simulator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">'
    "Probabilistic Forecasting with Fat-Tail Support & EWMA Volatility"
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

# =======================
# SIDEBAR CONTROLS
# =======================
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Stock Ticker", "NVDA").upper()
lookback_days = st.sidebar.slider("Historical Window (trading days)", 30, 504, 120)
forecast_days = st.sidebar.slider("Forecast Horizon (trading days)", 5, 252, 30)
simulations = st.sidebar.slider(
    "Number of Simulations", 1000, 20000, 5000, step=500
)

st.sidebar.markdown("---")
st.sidebar.subheader("Volatility")

vol_method = st.sidebar.radio(
    "Volatility Estimation",
    ["EWMA (Recommended)", "Simple Std Dev"],
    help=(
        "EWMA gives more weight to recent price action, "
        "adapting faster to changing market conditions."
    ),
)

vol_shock = st.sidebar.slider(
    "Volatility Multiplier",
    0.5,
    3.0,
    1.0,
    0.1,
    help=(
        "Scale the estimated volatility. "
        "1.0 = baseline. >1 = stress scenario. <1 = calm markets."
    ),
)

if vol_shock < 1.0:
    st.sidebar.caption(
        f"Multiplier {vol_shock:.1f}x — Lower volatility, smoother movements."
    )
elif vol_shock == 1.0:
    st.sidebar.caption(
        f"Multiplier {vol_shock:.1f}x — Baseline volatility."
    )
else:
    st.sidebar.caption(
        f"Multiplier {vol_shock:.1f}x — Elevated volatility, larger swings."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Distribution")

use_fat_tails = st.sidebar.checkbox(
    "Enable Fat-Tail Distribution (Student-t)",
    value=True,
    help=(
        "Real stock returns have fatter tails than a normal distribution. "
        "Student-t captures extreme moves more realistically."
    ),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Geometric Brownian Motion with configurable innovations."
)


# =======================
# DATA LOADING
# =======================
@st.cache_data(ttl=300)
def load_data(tick, lookback):
    return get_stock_data(tick, lookback)


try:
    current_price, prices, returns = load_data(ticker, lookback_days)
except Exception as e:
    st.error(f"Data load failed: {e}")
    st.stop()

# =======================
# MODEL PARAMETERS
# =======================
mu = float(np.mean(returns))

if vol_method == "EWMA (Recommended)":
    sigma_base = ewma_volatility(returns, span=min(30, len(returns)))
else:
    sigma_base = float(np.std(returns))

sigma = sigma_base * vol_shock

# Fit t-distribution if enabled
t_df = 5.0
if use_fat_tails:
    t_df, _, _ = fit_t_distribution(returns)

# =======================
# RUN SIMULATION
# =======================
paths = simulate_gbm(
    S0=current_price,
    mu=mu,
    sigma=sigma,
    days=forecast_days,
    sims=simulations,
    use_t_dist=use_fat_tails,
    t_df=t_df,
)

# Compute all statistics (NO artificial clipping)
stats = compute_statistics(paths, current_price, forecast_days)


# =======================
# CARD HELPERS
# =======================
def arrow_indicator(value, reference):
    delta = value - reference
    if delta > 0:
        return "▲", "#22c55e", delta
    elif delta < 0:
        return "▼", "#ef4444", delta
    return "→", "#9ca3af", 0.0


def render_card(col, label, value, tooltip=None, show_delta=False, reference=0.0):
    tooltip_html = f'<div class="tooltip">{tooltip}</div>' if tooltip else ""
    delta_html = ""
    if show_delta:
        sym, color, delta = arrow_indicator(
            float(value.replace("$", "").replace("%", "")), reference
        )
        delta_html = (
            f'<div class="metric-delta" style="color:{color}">'
            f"{sym} {abs(delta):.2f}</div>"
        )
    col.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
            {tooltip_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =======================
# PRICE METRICS
# =======================
st.markdown(
    '<div class="section-title">Price Estimates</div>', unsafe_allow_html=True
)
cols = st.columns(5)
render_card(cols[0], "Current Price", f"${current_price:.2f}", "Latest closing price")
render_card(
    cols[1],
    "Expected Mean",
    f"${stats['mean_price']:.2f}",
    "Average across all simulations",
    show_delta=True,
    reference=current_price,
)
render_card(
    cols[2],
    "Expected Median",
    f"${stats['median_price']:.2f}",
    "Middle value of simulations",
    show_delta=True,
    reference=current_price,
)
render_card(cols[3], "5th Percentile", f"${stats['p5']:.2f}", "Worst 5% scenario")
render_card(cols[4], "95th Percentile", f"${stats['p95']:.2f}", "Best 5% scenario")

# =======================
# RISK METRICS
# =======================
st.markdown(
    '<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True
)
rcols = st.columns(4)
render_card(
    rcols[0],
    "Expected Return",
    f"{stats['expected_return']:.2f}%",
    f"Over {forecast_days} trading days",
)
render_card(
    rcols[1],
    "Sharpe Ratio",
    f"{stats['sharpe_ratio']:.2f}",
    "Annualized return per unit risk",
)
render_card(
    rcols[2],
    "Value at Risk (5%)",
    f"{stats['var_5']:.2f}%",
    "Max loss in 95% of scenarios",
)
render_card(
    rcols[3],
    "Conditional VaR",
    f"{stats['cvar_5']:.2f}%",
    "Avg loss in worst 5% of scenarios",
)

# Probability & drawdown row
pcols = st.columns(3)
render_card(
    pcols[0],
    "Probability of Profit",
    f"{stats['prob_profit']:.1f}%",
    "Chance price ends above current",
)
render_card(
    pcols[1],
    "Probability of Loss",
    f"{stats['prob_loss']:.1f}%",
    "Chance price ends below current",
)
render_card(
    pcols[2],
    "Max Drawdown (Median)",
    f"{stats['max_drawdown']:.2f}%",
    "Largest peak-to-trough drop on median path",
)

# =======================
# MONTE CARLO CHART
# =======================
st.markdown(
    '<div class="section-title">Monte Carlo Price Paths</div>',
    unsafe_allow_html=True,
)

median_path = np.median(paths, axis=1)
p5_path = np.percentile(paths, 5, axis=1)
p10_path = np.percentile(paths, 10, axis=1)
p25_path = np.percentile(paths, 25, axis=1)
p75_path = np.percentile(paths, 75, axis=1)
p90_path = np.percentile(paths, 90, axis=1)
p95_path = np.percentile(paths, 95, axis=1)

future_days = list(range(len(median_path)))
hist_days = list(range(-len(prices), 0))

fig = go.Figure()

# Historical prices
fig.add_trace(
    go.Scatter(
        x=hist_days,
        y=prices.values,
        name="Historical",
        line=dict(color="#636efa", width=2),
    )
)

# Confidence bands (widest first for proper layering)
fig.add_trace(
    go.Scatter(
        x=future_days,
        y=p95_path,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=future_days,
        y=p5_path,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(99,102,241,0.1)",
        line=dict(width=0),
        name="5th-95th Percentile",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=future_days,
        y=p75_path,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=future_days,
        y=p25_path,
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(width=0),
        name="25th-75th Percentile",
        hoverinfo="skip",
    )
)

# Median forecast
fig.add_trace(
    go.Scatter(
        x=future_days,
        y=median_path,
        name="Median Forecast",
        line=dict(color="#8b5cf6", width=3),
    )
)

# Current price reference
fig.add_hline(
    y=current_price,
    line_dash="dash",
    line_color="#9ca3af",
    annotation_text="Current Price",
    annotation_font_color="#9ca3af",
)

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Days (negative = historical, positive = forecast)",
    yaxis_title="Price ($)",
    height=500,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)

# =======================
# DISTRIBUTION CHART
# =======================
st.markdown(
    '<div class="section-title">Final Price Distribution</div>',
    unsafe_allow_html=True,
)

final_prices = stats["final_prices"]
hist_fig = go.Figure()
hist_fig.add_trace(
    go.Histogram(
        x=final_prices,
        nbinsx=80,
        marker_color="#8b5cf6",
        opacity=0.85,
        name="Simulated Prices",
    )
)
hist_fig.add_vline(
    x=current_price,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Current Price",
    annotation_font_color="#ef4444",
)
hist_fig.add_vline(
    x=stats["median_price"],
    line_dash="dot",
    line_color="#22c55e",
    annotation_text="Median",
    annotation_font_color="#22c55e",
)
hist_fig.update_layout(
    template="plotly_dark",
    xaxis_title="Price ($)",
    yaxis_title="Frequency",
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)
st.plotly_chart(hist_fig, use_container_width=True)

# =======================
# MODEL INFO
# =======================
with st.expander("Model Details"):
    dist_label = (
        f"Student-t (df={t_df:.1f})" if use_fat_tails else "Normal (Gaussian)"
    )
    st.markdown(
        f"""
    | Parameter | Value |
    |-----------|-------|
    | Ticker | {ticker} |
    | Historical window | {lookback_days} trading days |
    | Forecast horizon | {forecast_days} trading days |
    | Simulations | {simulations:,} |
    | Daily drift (mu) | {mu:.6f} |
    | Base volatility | {sigma_base:.6f} |
    | Adjusted volatility | {sigma:.6f} (x{vol_shock:.1f}) |
    | Vol estimation method | {vol_method.split(" (")[0]} |
    | Innovation distribution | {dist_label} |
    """
    )

st.divider()
st.markdown(
    "**Disclaimer:** This application is for educational and research purposes only. "
    "Simulations are based on historical data and theoretical assumptions that may not "
    "reflect future market conditions. Nothing displayed is financial advice."
)
