import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from scipy import stats as scipy_stats

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
# DATA FUNCTIONS
# =======================
def get_stock_data(ticker, lookback_days):
    """
    Downloads historical stock data with robust error handling.

    Parameters:
    - ticker (str): Stock symbol, e.g., "AAPL"
    - lookback_days (int): Number of past trading days to retrieve

    Returns:
    - current_price (float): Latest closing price
    - prices (pd.Series): Historical closing prices
    - log_returns (np.ndarray): Daily log returns
    """
    # Use calendar days with buffer to ensure enough trading days
    calendar_days = int(lookback_days * 1.6) + 10
    period = f"{calendar_days}d"

    data = yf.download(ticker, period=period, progress=False)

    if data.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Please check the symbol and try again."
        )

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise ValueError(f"No 'Close' column found in data for '{ticker}'.")

    prices = data["Close"].dropna()

    if len(prices) < 10:
        raise ValueError(
            f"Insufficient data for '{ticker}': only {len(prices)} data points. "
            "Need at least 10 trading days."
        )

    # Trim to requested lookback (trading days)
    prices = prices.tail(lookback_days)

    current_price = float(prices.iloc[-1])
    log_returns = np.log(prices / prices.shift(1)).dropna().to_numpy()

    return current_price, prices, log_returns


# =======================
# MONTE CARLO FUNCTIONS
# =======================
def ewma_volatility(returns, span=30):
    """
    Compute Exponentially Weighted Moving Average volatility.
    Gives more weight to recent observations, capturing volatility clustering.

    Parameters:
    - returns (np.ndarray): Array of log returns
    - span (int): EWMA span (half-life ~ span/2)

    Returns:
    - float: EWMA volatility estimate
    """
    weights = np.array(
        [(1 - 2 / (span + 1)) ** i for i in range(len(returns) - 1, -1, -1)]
    )
    weights /= weights.sum()
    weighted_mean = np.dot(weights, returns)
    weighted_var = np.dot(weights, (returns - weighted_mean) ** 2)
    return float(np.sqrt(weighted_var))


def fit_t_distribution(returns):
    """
    Fit a Student-t distribution to log returns to capture fat tails.

    Returns:
    - df (float): Degrees of freedom (lower = fatter tails)
    - loc (float): Location parameter (mean)
    - scale (float): Scale parameter (volatility)
    """
    df, loc, scale = scipy_stats.t.fit(returns)
    # Clamp degrees of freedom to a reasonable range
    df = max(df, 2.1)  # must be > 2 for finite variance
    df = min(df, 100.0)  # beyond 100, essentially normal
    return df, loc, scale


def simulate_gbm(S0, mu, sigma, days, sims, use_t_dist=False, t_df=5.0):
    """
    Simulate future stock prices using Geometric Brownian Motion.
    Fully vectorized for performance.

    Parameters:
    - S0 (float): Current stock price
    - mu (float): Drift (mean daily log return)
    - sigma (float): Daily volatility
    - days (int): Forecast horizon in trading days
    - sims (int): Number of simulation paths
    - use_t_dist (bool): If True, use Student-t innovations for fat tails
    - t_df (float): Degrees of freedom for t-distribution

    Returns:
    - paths (np.ndarray): Shape (days+1, sims), price paths starting at S0
    """
    dt = 1.0
    drift = (mu - 0.5 * sigma ** 2) * dt

    if use_t_dist:
        # Student-t innovations, scaled to have unit variance
        # Var of t(df) = df/(df-2), so scale down
        raw = scipy_stats.t.rvs(df=t_df, size=(days, sims))
        scale_factor = np.sqrt(t_df / (t_df - 2)) if t_df > 2 else 1.0
        innovations = raw / scale_factor
    else:
        innovations = np.random.standard_normal((days, sims))

    daily_log_returns = drift + sigma * np.sqrt(dt) * innovations

    # Prepend zeros for starting price
    cumulative = np.vstack([np.zeros(sims), daily_log_returns])
    paths = S0 * np.exp(np.cumsum(cumulative, axis=0))

    return paths


def compute_statistics(paths, current_price, forecast_days):
    """
    Compute comprehensive statistics from simulation paths.

    Parameters:
    - paths (np.ndarray): Shape (days+1, sims)
    - current_price (float): Starting price
    - forecast_days (int): Number of forecast days

    Returns:
    - dict: Statistics including prices, returns, and risk metrics
    """
    final_prices = paths[-1]
    sim_returns = (final_prices - current_price) / current_price

    # Price statistics
    mean_price = float(np.mean(final_prices))
    median_price = float(np.median(final_prices))
    p5 = float(np.percentile(final_prices, 5))
    p10 = float(np.percentile(final_prices, 10))
    p25 = float(np.percentile(final_prices, 25))
    p75 = float(np.percentile(final_prices, 75))
    p90 = float(np.percentile(final_prices, 90))
    p95 = float(np.percentile(final_prices, 95))

    # Return statistics
    expected_return = float(np.mean(sim_returns) * 100)
    return_std = float(np.std(sim_returns))

    # Annualized Sharpe ratio (assuming 0% risk-free rate for simplicity)
    annualization = np.sqrt(252 / max(forecast_days, 1))
    sharpe = (
        float(np.mean(sim_returns) / np.std(sim_returns) * annualization)
        if np.std(sim_returns) > 1e-9
        else 0.0
    )

    # Value at Risk (5%) - how much you could lose in the worst 5% of cases
    var_5 = float(np.percentile(sim_returns, 5) * 100)

    # Conditional VaR (Expected Shortfall) - average loss in worst 5%
    tail_returns = sim_returns[sim_returns <= np.percentile(sim_returns, 5)]
    cvar_5 = float(np.mean(tail_returns) * 100) if len(tail_returns) > 0 else var_5

    # Probability of profit vs loss
    prob_profit = float(np.mean(final_prices > current_price) * 100)
    prob_loss = 100.0 - prob_profit

    # Max drawdown across median path
    median_path = np.median(paths, axis=1)
    running_max = np.maximum.accumulate(median_path)
    drawdowns = (median_path - running_max) / running_max
    max_drawdown = float(np.min(drawdowns) * 100)

    return {
        "mean_price": mean_price,
        "median_price": median_price,
        "p5": p5,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "expected_return": expected_return,
        "return_std": return_std,
        "sharpe_ratio": sharpe,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "prob_profit": prob_profit,
        "prob_loss": prob_loss,
        "max_drawdown": max_drawdown,
        "final_prices": final_prices,
    }


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
