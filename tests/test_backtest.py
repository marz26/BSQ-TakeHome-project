from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestError, analyze_parameter_sensitivity, run_backtest
from src.config import ProjectConfig


def _sample_prices() -> pd.DataFrame:
    """Create deterministic pair prices with a mean-reverting residual.

    Parameters
    ----------
    None
        This test fixture has no configurable inputs.

    Returns
    -------
    pandas.DataFrame
        Positive INTC and AMD prices over 150 daily observations.

    Raises
    ------
    None
        Construction uses fixed finite numeric values.

    Notes
    -----
    The independent price varies enough for OLS, while a sinusoidal residual
    provides deterministic excursions around a stable linear relationship.
    """
    observations = 150
    index = pd.date_range("2020-01-01", periods=observations, freq="D").date
    time = np.arange(observations, dtype=float)
    amd = 40.0 + 0.08 * time + 1.5 * np.sin(time / 7.0)
    residual = 1.2 * np.sin(time / 3.5) + 0.3 * np.sin(time / 1.8)
    intc = 5.0 + 1.1 * amd + residual
    return pd.DataFrame(
        {"INTC": intc, "AMD": amd},
        index=pd.Index(index, name="Date"),
    )


def test_backtest_records_attempts_and_splits_performance() -> None:
    """Verify grid audit count and separate pre/post-split performance.

    Parameters
    ----------
    None
        This test uses fixed deterministic prices and parameter grids.

    Returns
    -------
    None
        Assertions validate report contents and the split boundary.

    Raises
    ------
    AssertionError
        If trials are not fully recorded or the periods overlap.

    Notes
    -----
    The Cartesian product contains four attempts. The configured split day is
    included in OOS and excluded from the training regression and IS metrics.
    """
    split_date = date(2020, 3, 20)
    config = replace(ProjectConfig(), train_test_split_date=split_date.isoformat())

    report = run_backtest(
        _sample_prices(),
        config=config,
        as_of=date(2020, 5, 29),
        entry_thresholds=(1.5, 2.0),
        rolling_windows=(10, 20),
    )

    assert report.attempt_count == 4
    assert len(report.trials) == 4
    assert sum(trial.selected for trial in report.trials) == 1
    assert report.split_date == split_date
    assert report.as_of == date(2020, 5, 29)
    assert report.in_sample.trade_count >= 0
    assert report.out_of_sample.trade_count >= 0
    assert 0.0 <= report.in_sample.win_rate <= 1.0
    assert 0.0 <= report.out_of_sample.win_rate <= 1.0
    assert report.selected_parameters == next(
        trial.parameters for trial in report.trials if trial.selected
    )
    assert report.out_of_sample_spread.index.equals(report.out_of_sample_daily_pnl.index)
    assert report.out_of_sample_spread.index[0] == split_date


def test_oos_prices_cannot_change_fit_or_parameter_selection() -> None:
    """Prove OOS values do not influence training estimates or grid selection.

    Parameters
    ----------
    None
        This test runs two backtests with identical training data.

    Returns
    -------
    None
        Assertions verify identical in-sample choices and fit coefficients.

    Raises
    ------
    AssertionError
        If changed out-of-sample prices affect training state.

    Notes
    -----
    The second dataset changes only post-split INTC prices. Since hedge
    coefficients and the trial objective are strictly in-sample, both must
    remain identical; OOS PnL is evaluated afterward using those frozen values.
    """
    split_date = date(2020, 3, 20)
    config = replace(ProjectConfig(), train_test_split_date=split_date.isoformat())
    prices = _sample_prices()
    altered = prices.copy()
    altered.loc[altered.index >= split_date, "INTC"] += 3.0
    common_arguments = {
        "config": config,
        "as_of": date(2020, 5, 29),
        "entry_thresholds": (1.5, 2.0),
        "rolling_windows": (10, 20),
    }

    original_report = run_backtest(prices, **common_arguments)
    altered_report = run_backtest(altered, **common_arguments)

    assert original_report.alpha == pytest.approx(altered_report.alpha)
    assert original_report.beta == pytest.approx(altered_report.beta)
    assert original_report.selected_parameters == altered_report.selected_parameters
    assert [trial.in_sample for trial in original_report.trials] == [
        trial.in_sample for trial in altered_report.trials
    ]


def test_backtest_rejects_prices_beyond_as_of() -> None:
    """Reject input with a date beyond the explicit information cutoff.

    Parameters
    ----------
    None
        This test uses fixed synthetic prices.

    Returns
    -------
    None
        The assertion verifies look-ahead protection.

    Raises
    ------
    AssertionError
        If post-cutoff prices are accepted.

    Notes
    -----
    The price validator rejects future rows before fitting or selecting
    parameters, and an explicit assertion in the validation path enforces the
    D-1 bound.
    """
    split_date = date(2020, 3, 20)
    config = replace(ProjectConfig(), train_test_split_date=split_date.isoformat())
    with pytest.raises(AssertionError):
        run_backtest(
            _sample_prices(),
            config=config,
            as_of=date(2020, 5, 28),
            entry_thresholds=(1.5,),
            rolling_windows=(10,),
        )


def test_backtest_rejects_invalid_grid_and_missing_period() -> None:
    """Reject ambiguous parameter grids and missing OOS observations.

    Parameters
    ----------
    None
        This test uses deterministic synthetic inputs.

    Returns
    -------
    None
        Assertions verify invalid protocol inputs raise domain errors.

    Raises
    ------
    AssertionError
        If an invalid grid or absent OOS period is accepted.

    Notes
    -----
    An explicit grid is necessary for an auditable attempt count, and both
    periods must exist for a valid split report.
    """
    prices = _sample_prices()
    split_date = date(2020, 3, 20)
    config = replace(ProjectConfig(), train_test_split_date=split_date.isoformat())

    with pytest.raises(BacktestError, match="must not contain duplicates"):
        run_backtest(
            prices,
            config=config,
            as_of=date(2020, 5, 29),
            entry_thresholds=(1.5, 1.5),
            rolling_windows=(10,),
        )

    config_without_oos = replace(
        config,
        train_test_split_date=date(2020, 5, 30).isoformat(),
    )
    with pytest.raises(BacktestError, match="before and on/after"):
        run_backtest(
            prices,
            config=config_without_oos,
            as_of=date(2020, 5, 29),
            entry_thresholds=(1.5,),
            rolling_windows=(10,),
        )


def test_parameter_sensitivity_detects_neighbor_sign_reversal() -> None:
    """Classify the selected configuration as fragile when an adjacent trial loses.

    Parameters
    ----------
    None
        This test uses a synthetic ledger produced by the backtest.

    Returns
    -------
    None
        Assertions verify immediate-neighbor comparisons and fragility reasons.

    Raises
    ------
    AssertionError
        If neighboring trials or the sign-reversal rule are not reported.

    Notes
    -----
    Only the closest adjacent value on each parameter axis is included. Which
    axis controls this synthetic example depends on its fitted training data,
    so the invariant is the detected adjacent sign reversal itself.
    """
    prices = _sample_prices()
    config = replace(
        ProjectConfig(),
        train_test_split_date=date(2020, 4, 15).isoformat(),
    )
    report = run_backtest(
        prices,
        config=config,
        as_of=date(2020, 5, 29),
        entry_thresholds=(1.5, 2.0, 2.5),
        rolling_windows=(40, 60, 90),
    )

    sensitivity = analyze_parameter_sensitivity(report)

    assert sensitivity.baseline.selected
    assert sensitivity.neighbors
    assert sensitivity.is_fragile
    assert any(neighbor.pnl <= 0 for neighbor in sensitivity.neighbors)
    assert sensitivity.fragility_reasons


def test_nonprofitable_sensitivity_baseline_is_inconclusive_not_fragile() -> None:
    """Keep an unprofitable baseline distinct from neighbor-driven fragility.

    Parameters
    ----------
    None
        Uses the deterministic test price history and a fitted trial ledger.

    Returns
    -------
    None
        Assertions validate the inconclusive classification.

    Raises
    ------
    AssertionError
        If an unprofitable baseline is labeled fragile or robust.

    Notes
    -----
    The ledger PnLs are replaced with negative values to isolate the
    classification rule from the fixture's particular market outcome.
    """
    prices = _sample_prices()
    config = replace(
        ProjectConfig(),
        train_test_split_date=date(2020, 4, 15).isoformat(),
    )
    report = run_backtest(
        prices,
        config=config,
        as_of=date(2020, 5, 29),
        entry_thresholds=(1.5, 2.0, 2.5),
        rolling_windows=(40, 60, 90),
    )
    unprofitable_trials = tuple(
        replace(
            trial,
            in_sample=replace(trial.in_sample, pnl=-1.0),
        )
        for trial in report.trials
    )
    unprofitable_report = replace(report, trials=unprofitable_trials)

    sensitivity = analyze_parameter_sensitivity(unprofitable_report)

    assert sensitivity.baseline.in_sample.pnl == -1.0
    assert not sensitivity.is_fragile
    assert any("inconclusive" in reason for reason in sensitivity.fragility_reasons)
