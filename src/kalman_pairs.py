from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import BacktestReport
from src.features import FeatureCalculationError, compute_z_score

logger = logging.getLogger(__name__)


class KalmanPairError(ValueError):
    """Raised when the dynamic hedge filter or benchmark cannot be run.

    Parameters
    ----------
    message : str
        Description of the invalid input or numerical failure.

    Returns
    -------
    None
        Exception constructors do not return values.

    Raises
    ------
    None
        Construction of this exception does not raise an error.

    Notes
    -----
    Domain-specific errors distinguish invalid Kalman/backtest inputs from
    failures in unrelated pipeline phases.
    """


@dataclass(frozen=True)
class KalmanFilterResult:
    """Filtered hedge states and one-step-ahead measurement innovations.

    Parameters
    ----------
    states : pandas.DataFrame
        Posterior state estimates indexed by date, with ``alpha`` and ``beta``
        columns.
    innovations : pandas.Series
        One-step-ahead residuals ``y_t - H_t theta_(t|t-1)``.
    innovation_variance : pandas.Series
        One-step-ahead innovation variances.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are returned by :func:`estimate_dynamic_hedge_ratio`.

    Notes
    -----
    Innovations are formed before each observation updates the state, making
    them valid as-of-date residuals. Posterior states include the current
    observation and may be used only for subsequent decisions.
    """

    states: pd.DataFrame
    innovations: pd.Series
    innovation_variance: pd.Series


@dataclass(frozen=True)
class KalmanPerformance:
    """Out-of-sample performance of the causally hedged pair strategy.

    Parameters
    ----------
    pnl : float
        Cumulative spread PnL in price units.
    trade_count : int
        Number of trades opened during the evaluation period.
    win_rate : float
        Fraction of closed trades with positive PnL; zero if none closed.
    max_drawdown : float
        Minimum peak-to-trough drawdown of cumulative daily PnL, including
        zero as the initial equity high-water mark.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are constructed by :func:`run_kalman_pair_backtest`.

    Notes
    -----
    Metrics use the same uncapitalized price-spread PnL convention as the
    static backtest and exclude transaction costs, financing, and execution
    constraints.
    """

    pnl: float
    trade_count: int
    win_rate: float
    max_drawdown: float


@dataclass(frozen=True)
class KalmanBacktestReport:
    """Dynamic-hedge OOS result with its causal filter diagnostics.

    Parameters
    ----------
    performance : KalmanPerformance
        OOS PnL, trade count, win rate, and maximum drawdown.
    daily_pnl : pandas.Series
        OOS daily PnL indexed by its earning date.
    filter_result : KalmanFilterResult
        Full-history one-step innovations and posterior coefficient estimates.
    process_noise : float
        Scalar multiplier for the identity process covariance matrix.
    observation_variance : float
        Fixed scalar observation-noise variance.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are returned by :func:`run_kalman_pair_backtest`.

    Notes
    -----
    The report keeps the dynamic filter series together with its OOS summary
    so users can inspect the estimated hedge path instead of relying only on
    aggregate performance.
    """

    performance: KalmanPerformance
    daily_pnl: pd.Series
    filter_result: KalmanFilterResult
    process_noise: float
    observation_variance: float


def estimate_dynamic_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    process_noise: float = 1e-5,
    observation_variance: float = 1.0,
    initial_state: tuple[float, float] = (0.0, 1.0),
    initial_covariance: float = 1e6,
) -> KalmanFilterResult:
    """Estimate time-varying intercept and hedge ratio with a NumPy Kalman filter.

    Parameters
    ----------
    price_a : pandas.Series
        Dependent asset prices, indexed by observation date.
    price_b : pandas.Series
        Explanatory asset prices with an identical date index.
    process_noise : float, default=1e-5
        Non-negative scalar ``q`` defining ``Q = q * I``.
    observation_variance : float, default=1.0
        Positive, fixed measurement-noise variance ``R`` in price units
        squared. It is not estimated from future or OOS observations.
    initial_state : tuple of float, default=(0.0, 1.0)
        Prior state ``(alpha, beta)`` before the first observation.
    initial_covariance : float, default=1e6
        Positive scale for the initial covariance ``P_0 = scale * I``.

    Returns
    -------
    KalmanFilterResult
        Posterior state path, one-step-ahead innovations, and innovation
        variances, all indexed by the input dates.

    Raises
    ------
    KalmanPairError
        If series are empty, misaligned, non-finite, unsorted, or if filter
        parameters are invalid or a numerical invariant fails.

    Notes
    -----
    The local-level state follows ``theta_t = theta_(t-1) + w_t`` with
    ``Q = process_noise * I``. The observation is
    ``y_t = [1, x_t] theta_t + v_t``. For each date, the function first forms
    the prior prediction and innovation, then applies the scalar measurement
    update using only that date's observation. The Joseph covariance update
    improves positive-semidefinite numerical stability. ``R`` and ``Q`` are
    fixed inputs; there is no parameter selection on the OOS sample.
    """
    if price_a.empty or price_b.empty:
        raise KalmanPairError("price series must not be empty")
    if not price_a.index.equals(price_b.index):
        raise KalmanPairError("price series must have identical date indexes")
    if not price_a.index.is_unique or not price_a.index.is_monotonic_increasing:
        raise KalmanPairError("price indexes must be unique and chronologically increasing")
    if not np.isfinite(process_noise) or process_noise < 0.0:
        raise KalmanPairError("process_noise must be finite and non-negative")
    if not np.isfinite(observation_variance) or observation_variance <= 0.0:
        raise KalmanPairError("observation_variance must be finite and positive")
    if not np.isfinite(initial_covariance) or initial_covariance <= 0.0:
        raise KalmanPairError("initial_covariance must be finite and positive")
    try:
        initial_state_array = np.asarray(initial_state, dtype=float)
    except (TypeError, ValueError) as error:
        raise KalmanPairError("initial_state must contain two finite values") from error
    if initial_state_array.shape != (2,) or not np.isfinite(initial_state_array).all():
        raise KalmanPairError("initial_state must contain two finite values")

    try:
        dates = pd.Index(
            [pd.Timestamp(value).date() for value in price_a.index],
            name=price_a.index.name or "Date",
        )
        observed_a = price_a.to_numpy(dtype=float)
        observed_b = price_b.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise KalmanPairError("prices must have valid dates and numeric values") from error
    if not np.isfinite(observed_a).all() or not np.isfinite(observed_b).all():
        raise KalmanPairError("prices must contain only finite values")
    if (observed_a <= 0.0).any() or (observed_b <= 0.0).any():
        raise KalmanPairError("prices must be strictly positive")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise KalmanPairError("normalized price dates must be unique and increasing")

    process_covariance = float(process_noise) * np.eye(2, dtype=float)
    identity = np.eye(2, dtype=float)
    state = initial_state_array.copy()
    covariance = float(initial_covariance) * identity
    state_path = np.empty((len(dates), 2), dtype=float)
    innovations = np.empty(len(dates), dtype=float)
    innovation_variances = np.empty(len(dates), dtype=float)

    for position, (observed_y, observed_x) in enumerate(zip(observed_a, observed_b)):
        predicted_state = state.copy()
        predicted_covariance = covariance + process_covariance
        observation_vector = np.array([1.0, observed_x], dtype=float)
        innovation = float(observed_y - observation_vector @ predicted_state)
        innovation_variance = float(
            observation_vector @ predicted_covariance @ observation_vector
            + observation_variance
        )
        if not np.isfinite(innovation) or not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise KalmanPairError(f"invalid Kalman innovation at {dates[position]}")

        gain = predicted_covariance @ observation_vector / innovation_variance
        state = predicted_state + gain * innovation
        residual_operator = identity - np.outer(gain, observation_vector)
        covariance = (
            residual_operator @ predicted_covariance @ residual_operator.T
            + observation_variance * np.outer(gain, gain)
        )
        covariance = 0.5 * (covariance + covariance.T)
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise KalmanPairError(f"non-finite Kalman state at {dates[position]}")
        if np.linalg.eigvalsh(covariance).min() < -1e-8:
            raise KalmanPairError(f"Kalman covariance lost positive semidefiniteness at {dates[position]}")

        innovations[position] = innovation
        innovation_variances[position] = innovation_variance
        state_path[position] = state

    states = pd.DataFrame(state_path, index=dates, columns=["alpha", "beta"])
    innovation_series = pd.Series(innovations, index=dates, name="innovation")
    variance_series = pd.Series(
        innovation_variances,
        index=dates,
        name="innovation_variance",
    )
    assert len(states) == len(price_a)
    assert states.index.equals(innovation_series.index)
    assert states.index.equals(variance_series.index)
    assert np.isfinite(states.to_numpy()).all()
    assert np.isfinite(innovation_series.to_numpy()).all()
    logger.info(
        "Kalman hedge filter complete: observations=%d process_noise=%.3g observation_variance=%.6g",
        len(states),
        process_noise,
        observation_variance,
    )
    return KalmanFilterResult(
        states=states,
        innovations=innovation_series,
        innovation_variance=variance_series,
    )


def run_kalman_pair_backtest(
    prices: pd.DataFrame,
    static_report: BacktestReport,
    tickers: tuple[str, str],
    process_noise: float = 1e-5,
    observation_variance: float = 1.0,
) -> KalmanBacktestReport:
    """Benchmark a causal dynamic hedge against the static OLS strategy OOS.

    Parameters
    ----------
    prices : pandas.DataFrame
        Aligned pair prices through ``static_report.as_of``.
    static_report : BacktestReport
        Static benchmark report. Its In-Sample-selected entry/exit/window
        settings are reused unchanged for the dynamic strategy.
    tickers : tuple of str
        Ordered dependent and explanatory asset column names.
    process_noise : float, default=1e-5
        Scalar process-noise covariance multiplier.
    observation_variance : float, default=1.0
        Fixed measurement-noise variance.

    Returns
    -------
    KalmanBacktestReport
        OOS dynamic-hedge metrics and full-history filter diagnostics.

    Raises
    ------
    KalmanPairError
        If dates, columns, prices, or split boundaries are invalid.
    FeatureCalculationError
        If causal rolling z-scores cannot be calculated.

    Notes
    -----
    Filter states are updated sequentially across the full observed history,
    so the state on an OOS date depends only on observations available through
    that date. The strategy signal uses the previous date's innovation-based
    z-score, and each PnL increment hedges the realized price change with the
    beta estimate available before that increment. Static OLS-selected
    strategy thresholds and window are reused for a fair, non-optimized
    benchmark; neither Kalman hyperparameters nor thresholds are tuned on OOS.
    """
    if len(tickers) != 2 or tickers[0] == tickers[1]:
        raise KalmanPairError("tickers must name two distinct assets")
    if any(ticker not in prices.columns for ticker in tickers):
        raise KalmanPairError("prices must contain both requested ticker columns")
    if prices.empty:
        raise KalmanPairError("prices must not be empty")

    pair = prices.loc[:, list(tickers)].copy()
    try:
        normalized_dates = pd.Index(
            [pd.Timestamp(value).date() for value in pair.index],
            name=pair.index.name or "Date",
        )
        pair.index = normalized_dates
        pair = pair.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise KalmanPairError("prices must have valid dates and numeric values") from error
    if normalized_dates.has_duplicates or not normalized_dates.is_monotonic_increasing:
        raise KalmanPairError("price dates must be unique and increasing")
    if any(observation_date > static_report.as_of for observation_date in normalized_dates):
        raise KalmanPairError("prices contain observations after the static report as_of")
    if not any(observation_date < static_report.split_date for observation_date in normalized_dates):
        raise KalmanPairError("prices must include in-sample observations before the split")
    if not any(observation_date >= static_report.split_date for observation_date in normalized_dates):
        raise KalmanPairError("prices must include out-of-sample observations")
    if not np.isfinite(pair.to_numpy()).all() or (pair <= 0.0).any().any():
        raise KalmanPairError("prices must be finite and strictly positive")

    filtered = estimate_dynamic_hedge_ratio(
        pair[tickers[0]],
        pair[tickers[1]],
        process_noise=process_noise,
        observation_variance=observation_variance,
    )
    z_score = compute_z_score(
        filtered.innovations,
        static_report.selected_parameters.rolling_window,
    )

    oos_mask = normalized_dates >= static_report.split_date
    oos_positions = np.flatnonzero(oos_mask)
    first_oos_position = int(oos_positions[0])
    if first_oos_position == 0:
        raise KalmanPairError("OOS period requires a prior observation for causal PnL")

    parameters = static_report.selected_parameters
    position = 0
    trade_count = 0
    closed_trade_pnls: list[float] = []
    current_trade_pnl = 0.0
    daily_pnl_values: list[float] = []

    for current_index in oos_positions:
        previous_index = int(current_index) - 1
        previous_z = float(z_score.iloc[previous_index])
        next_position = position
        if np.isfinite(previous_z):
            if position == 0:
                if previous_z >= parameters.entry_z:
                    next_position = -1
                elif previous_z <= -parameters.entry_z:
                    next_position = 1
            elif abs(previous_z) <= parameters.exit_z:
                closed_trade_pnls.append(current_trade_pnl)
                current_trade_pnl = 0.0
                next_position = 0

        if position == 0 and next_position != 0:
            trade_count += 1
        position = next_position
        beta_before_price_move = float(filtered.states["beta"].iloc[previous_index])
        pnl = position * (
            float(pair[tickers[0]].iloc[current_index] - pair[tickers[0]].iloc[previous_index])
            - beta_before_price_move
            * float(pair[tickers[1]].iloc[current_index] - pair[tickers[1]].iloc[previous_index])
        )
        daily_pnl_values.append(float(pnl))
        if position != 0:
            current_trade_pnl += float(pnl)

    daily_pnl = pd.Series(
        daily_pnl_values,
        index=normalized_dates[oos_positions],
        name="kalman_daily_pnl",
        dtype=float,
    )
    equity = daily_pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    win_rate = (
        sum(trade_pnl > 0.0 for trade_pnl in closed_trade_pnls) / len(closed_trade_pnls)
        if closed_trade_pnls
        else 0.0
    )
    performance = KalmanPerformance(
        pnl=float(daily_pnl.sum()),
        trade_count=trade_count,
        win_rate=float(win_rate),
        max_drawdown=float(drawdown.min()),
    )
    assert daily_pnl.index.equals(pair.index[oos_mask])
    assert all(observation_date >= static_report.split_date for observation_date in daily_pnl.index)
    assert all(observation_date <= static_report.as_of for observation_date in daily_pnl.index)
    assert np.isfinite(daily_pnl.to_numpy()).all()
    assert 0.0 <= performance.win_rate <= 1.0
    logger.info(
        "Dynamic Kalman OOS benchmark: q=%.3g R=%.6g entry_z=%.2f window=%d "
        "PnL=%.4f trades=%d win_rate=%.2f%% max_drawdown=%.4f",
        process_noise,
        observation_variance,
        parameters.entry_z,
        parameters.rolling_window,
        performance.pnl,
        performance.trade_count,
        performance.win_rate * 100.0,
        performance.max_drawdown,
    )
    return KalmanBacktestReport(
        performance=performance,
        daily_pnl=daily_pnl,
        filter_result=filtered,
        process_noise=float(process_noise),
        observation_variance=float(observation_variance),
    )
