import numpy as np
from scipy import stats as scipy_stats


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
