from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestReport, run_backtest
from src.config import ProjectConfig
from src.kalman_pairs import (
    KalmanPairError,
    estimate_dynamic_hedge_ratio,
    run_kalman_pair_backtest,
)


def _sample_prices(observations: int = 220) -> pd.DataFrame:
    """Build deterministic pair prices with a slowly varying hedge ratio.

    Parameters
    ----------
    observations : int, default=220
        Number of daily price observations.

    Returns
    -------
    pandas.DataFrame
        Positive, aligned dependent and explanatory asset price history.

    Raises
    ------
    None
        The fixture uses fixed deterministic NumPy calculations.

    Notes
    -----
    A slowly changing beta gives the online filter a dynamic relation to
    estimate while bounded sinusoidal residuals generate finite innovations.
    """
    dates = pd.date_range("2020-01-01", periods=observations, freq="D").date
    time = np.arange(observations, dtype=float)
    price_b = 35.0 + 0.12 * time + 2.0 * np.sin(time / 13.0)
    beta = 1.0 + 0.0004 * time
    alpha = 3.0 + 0.002 * time
    residual = 0.8 * np.sin(time / 3.7) + 0.15 * np.cos(time / 1.9)
    price_a = alpha + beta * price_b + residual
    return pd.DataFrame(
        {"INTC": price_a, "AMD": price_b},
        index=pd.Index(dates, name="Date"),
    )


def _static_report(prices: pd.DataFrame) -> BacktestReport:
    """Fit the existing static strategy on the deterministic fixture.

    Parameters
    ----------
    prices : pandas.DataFrame
        Pair history to split and backtest.

    Returns
    -------
    src.backtest.BacktestReport
        Static report with coefficients and parameters selected in-sample.

    Raises
    ------
    src.backtest.BacktestError
        If the history cannot support the chosen split or grid.

    Notes
    -----
    The split leaves more than 90 training observations for the default grid;
    the dynamic benchmark then reuses its selected strategy parameters rather
    than tuning them on OOS data.
    """
    split = date.fromisoformat(str(prices.index[130]))
    config = replace(ProjectConfig(), train_test_split_date=split.isoformat())
    return run_backtest(prices, config, date.fromisoformat(str(prices.index[-1])))


def test_filter_shapes_are_finite_and_variance_is_positive() -> None:
    """Validate output dimensions and finite stable Kalman diagnostics.

    Parameters
    ----------
    None
        Uses fixed deterministic pair prices.

    Returns
    -------
    None
        Assertions validate state, innovation, and variance dimensions.

    Raises
    ------
    AssertionError
        If filter outputs have invalid shapes or numerical values.

    Notes
    -----
    The state has two components (intercept and hedge ratio); innovations and
    their scalar variances must align one-for-one with price observations.
    """
    prices = _sample_prices()
    result = estimate_dynamic_hedge_ratio(prices["INTC"], prices["AMD"])

    assert result.states.shape == (len(prices), 2)
    assert list(result.states.columns) == ["alpha", "beta"]
    assert result.innovations.shape == (len(prices),)
    assert result.innovation_variance.shape == (len(prices),)
    assert result.states.index.equals(result.innovations.index)
    assert np.isfinite(result.states.to_numpy()).all()
    assert np.isfinite(result.innovations.to_numpy()).all()
    assert (result.innovation_variance > 0.0).all()


def test_filter_is_causal_for_every_prefix() -> None:
    """Ensure future observations cannot alter earlier filter outputs.

    Parameters
    ----------
    None
        Compares full-sample output with a separately filtered prefix.

    Returns
    -------
    None
        Assertions verify exact causal prefix equivalence.

    Raises
    ------
    AssertionError
        If future observations affect prior states or innovations.

    Notes
    -----
    The filter is deterministic with fixed parameters. Filtering a prefix
    independently must therefore produce the same states and innovations as
    the corresponding prefix of the longer run.
    """
    prices = _sample_prices(120)
    prefix_length = 73
    full = estimate_dynamic_hedge_ratio(prices["INTC"], prices["AMD"])
    prefix = estimate_dynamic_hedge_ratio(
        prices["INTC"].iloc[:prefix_length],
        prices["AMD"].iloc[:prefix_length],
    )

    pd.testing.assert_frame_equal(
        full.states.iloc[:prefix_length],
        prefix.states,
    )
    pd.testing.assert_series_equal(
        full.innovations.iloc[:prefix_length],
        prefix.innovations,
    )
    pd.testing.assert_series_equal(
        full.innovation_variance.iloc[:prefix_length],
        prefix.innovation_variance,
    )


def test_filter_rejects_misaligned_and_invalid_inputs() -> None:
    """Reject misaligned dates and invalid measurement variance.

    Parameters
    ----------
    None
        Uses deterministic prices and an intentionally shifted index.

    Returns
    -------
    None
        Expected domain errors are asserted.

    Raises
    ------
    AssertionError
        If invalid filter inputs are accepted.

    Notes
    -----
    Exact index alignment is necessary because each measurement row must
    represent the same time for both assets.
    """
    prices = _sample_prices(40)
    shifted = prices["AMD"].copy()
    shifted.index = pd.Index(
        pd.date_range("2020-01-02", periods=len(shifted), freq="D").date,
        name="Date",
    )

    with pytest.raises(KalmanPairError, match="identical date indexes"):
        estimate_dynamic_hedge_ratio(prices["INTC"], shifted)
    with pytest.raises(KalmanPairError, match="observation_variance"):
        estimate_dynamic_hedge_ratio(
            prices["INTC"],
            prices["AMD"],
            observation_variance=0.0,
        )


def test_dynamic_backtest_reports_causal_oos_metrics() -> None:
    """Benchmark dynamic hedge performance only on dates in the OOS period.

    Parameters
    ----------
    None
        Uses deterministic prices and a static IS-selected backtest report.

    Returns
    -------
    None
        Assertions verify OOS date bounds and finite benchmark metrics.

    Raises
    ------
    AssertionError
        If the report includes pre-split PnL dates or invalid metrics.

    Notes
    -----
    The dynamic benchmark receives the static report's already-selected
    threshold and rolling window. The filter updates online, but no Kalman
    setting or trading threshold is selected from the OOS result.
    """
    prices = _sample_prices()
    static = _static_report(prices)

    dynamic = run_kalman_pair_backtest(
        prices,
        static_report=static,
        tickers=("INTC", "AMD"),
    )

    assert len(dynamic.daily_pnl) == len(static.out_of_sample_daily_pnl)
    assert all(observation_date >= static.split_date for observation_date in dynamic.daily_pnl.index)
    assert all(observation_date <= static.as_of for observation_date in dynamic.daily_pnl.index)
    assert np.isfinite(dynamic.daily_pnl.to_numpy()).all()
    assert np.isfinite(dynamic.performance.pnl)
    assert np.isfinite(dynamic.performance.max_drawdown)
    assert dynamic.performance.trade_count >= 0
    assert 0.0 <= dynamic.performance.win_rate <= 1.0
