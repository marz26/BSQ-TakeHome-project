from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller

from src.stability import (
    StabilityAnalysisError,
    compare_stability_with_performance,
    compute_rolling_adf,
)


def _sample_spread(seed: int = 11, observations: int = 140) -> pd.Series:
    """Create a deterministic mean-reverting spread for rolling ADF tests.

    Parameters
    ----------
    seed : int, default=11
        Seed for the fixed pseudo-random innovation sequence.
    observations : int, default=140
        Number of calendar-daily spread observations.

    Returns
    -------
    pandas.Series
        Mean-reverting spread indexed by Python calendar dates.

    Raises
    ------
    None
        Fixed-size numeric generation is well-defined.

    Notes
    -----
    The autoregressive coefficient is below one, creating a stationary
    synthetic process without relying on a network or external market data.
    """
    random = np.random.default_rng(seed)
    innovations = random.normal(size=observations)
    values = np.zeros(observations, dtype=float)
    for position in range(1, observations):
        values[position] = 0.35 * values[position - 1] + innovations[position]
    index = pd.date_range("2024-01-01", periods=observations, freq="D").date
    return pd.Series(values, index=pd.Index(index, name="Date"), name="spread")


def test_compute_rolling_adf_returns_trailing_pvalues_without_lookahead() -> None:
    """Verify rolling window alignment and prefix invariance to future changes.

    Parameters
    ----------
    None
        This test uses deterministic synthetic spread data.

    Returns
    -------
    None
        Assertions verify date alignment, output range, and causality.

    Raises
    ------
    AssertionError
        If rolling output is misaligned or uses observations after its label.

    Notes
    -----
    ADF p-values are written at the final observation of each window.
    Modifying only later observations must not change any earlier p-value.
    """
    spread = _sample_spread()
    window = 40
    cutoff_position = 100

    p_values = compute_rolling_adf(spread, window=window, as_of=spread.index[-1])
    altered = spread.copy()
    altered.iloc[cutoff_position:] += np.linspace(10.0, 20.0, len(altered) - cutoff_position)
    altered_p_values = compute_rolling_adf(
        altered,
        window=window,
        as_of=altered.index[-1],
    )

    assert p_values.index.equals(spread.index)
    assert p_values.iloc[: window - 1].isna().all()
    assert p_values.iloc[window - 1 :].notna().any()
    assert p_values.dropna().between(0.0, 1.0).all()
    pd.testing.assert_series_equal(
        p_values.iloc[:cutoff_position],
        altered_p_values.iloc[:cutoff_position],
    )
    expected = adfuller(
        spread.iloc[-window:].to_numpy(),
        regression="c",
        autolag="AIC",
    )[1]
    assert p_values.iloc[-1] == pytest.approx(expected)
    assert all(observation_date <= spread.index[-1] for observation_date in p_values.index)


def test_compute_rolling_adf_rejects_invalid_and_out_of_cutoff_inputs() -> None:
    """Reject invalid rolling windows and dates beyond the explicit cutoff.

    Parameters
    ----------
    None
        This test uses a deterministic synthetic spread.

    Returns
    -------
    None
        Assertions verify explicit validation failures.

    Raises
    ------
    AssertionError
        If invalid inputs are accepted instead of raising the domain error.

    Notes
    -----
    A minimum practical window size and explicit date bound make the ADF test
    reproducible and prevent accidental inclusion of future observations.
    """
    spread = _sample_spread(observations=100)
    with pytest.raises(StabilityAnalysisError, match="greater than or equal to 20"):
        compute_rolling_adf(spread, window=10)
    with pytest.raises(StabilityAnalysisError, match="after as_of"):
        compute_rolling_adf(
            spread,
            window=40,
            as_of=date.fromordinal(spread.index[-1].toordinal() - 1),
        )
    with pytest.raises(StabilityAnalysisError, match="must not exceed"):
        compute_rolling_adf(spread, window=101)


def test_compare_stability_with_performance_matches_dates_and_drawdowns() -> None:
    """Compare matched stable and unstable PnL with timeline drawdown values.

    Parameters
    ----------
    None
        This test uses fixed p-values and daily strategy PnL.

    Returns
    -------
    None
        Assertions validate classification counts, PnL, and drawdowns.

    Raises
    ------
    AssertionError
        If date matching or the stable/unstable group metrics are incorrect.

    Notes
    -----
    The leading missing p-value is excluded because its rolling window is
    incomplete. Drawdowns are measured on cumulative PnL in date order and
    associated with the stability classification at the drawdown date.
    """
    index = pd.date_range("2024-01-01", periods=5, freq="D").date
    p_values = pd.Series(
        [np.nan, 0.01, 0.10, 0.20, 0.03],
        index=pd.Index(index, name="Date"),
        name="adf_pvalue",
    )
    daily_pnl = pd.Series(
        [100.0, -2.0, -3.0, 4.0, 1.0],
        index=pd.Index(index, name="Date"),
        name="daily_pnl",
    )

    comparison = compare_stability_with_performance(p_values, daily_pnl)

    assert comparison.stable_observations == 2
    assert comparison.unstable_observations == 2
    assert comparison.stable_pnl == pytest.approx(-1.0)
    assert comparison.unstable_pnl == pytest.approx(1.0)
    assert comparison.stable_mean_daily_pnl == pytest.approx(-0.5)
    assert comparison.unstable_mean_daily_pnl == pytest.approx(0.5)
    assert comparison.stable_max_drawdown == pytest.approx(-2.0)
    assert comparison.unstable_max_drawdown == pytest.approx(-5.0)


def test_compare_stability_rejects_misaligned_or_invalid_inputs() -> None:
    """Reject comparisons that cannot be associated with valid dates.

    Parameters
    ----------
    None
        This test uses short deterministic series.

    Returns
    -------
    None
        Assertions verify invalid alignment and significance-level checks.

    Raises
    ------
    AssertionError
        If invalid comparison data is accepted.

    Notes
    -----
    A non-overlapping pair provides no evidence for a stability-performance
    comparison, so the function reports an explicit error rather than empty
    metrics.
    """
    adf_index = pd.Index([date(2024, 1, 1)], name="Date")
    pnl_index = pd.Index([date(2024, 1, 2)], name="Date")
    p_values = pd.Series([0.01], index=adf_index)
    daily_pnl = pd.Series([1.0], index=pnl_index)
    with pytest.raises(StabilityAnalysisError, match="no valid overlapping"):
        compare_stability_with_performance(p_values, daily_pnl)
    with pytest.raises(StabilityAnalysisError, match="strictly between"):
        compare_stability_with_performance(p_values, daily_pnl, significance_level=1.0)
