from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestReport, run_backtest
from src.config import ProjectConfig
from src.simulation import (
    SimulationError,
    generate_joint_block_bootstrap_paths,
    run_joint_block_bootstrap,
)


def _prices(observations: int = 150) -> pd.DataFrame:
    """Create deterministic positive pair prices for bootstrap tests.

    Parameters
    ----------
    observations : int, default=150
        Number of daily price observations to generate.

    Returns
    -------
    pandas.DataFrame
        Positive INTC and AMD prices with a daily date index.

    Raises
    ------
    None
        Inputs are used only to size deterministic NumPy arrays.

    Notes
    -----
    AMD follows a smooth trend and INTC combines a linear relation with a
    bounded residual so the OLS hedge ratio and rolling signals are defined.
    """
    dates = pd.date_range("2020-01-01", periods=observations, freq="D").date
    time = np.arange(observations, dtype=float)
    amd = 40.0 + 0.08 * time + np.sin(time / 9.0)
    intc = 5.0 + 1.1 * amd + 1.5 * np.sin(time / 3.5)
    return pd.DataFrame(
        {"INTC": intc, "AMD": amd},
        index=pd.Index(dates, name="Date"),
    )


def _backtest_report(prices: pd.DataFrame) -> BacktestReport:
    """Fit a small-grid report whose training window supports simulation.

    Parameters
    ----------
    prices : pandas.DataFrame
        Deterministic INTC/AMD history.

    Returns
    -------
    src.backtest.BacktestReport
        Report fitted before the requested split date.

    Raises
    ------
    src.backtest.BacktestError
        If the fixture cannot support the backtest split.

    Notes
    -----
    The selected window is deliberately shorter than the available training
    history to keep this test focused and fast.
    """
    split_date = date(2020, 4, 30)
    config = replace(
        ProjectConfig(),
        train_test_split_date=split_date.isoformat(),
        z_score_exit=0.5,
    )
    return run_backtest(
        prices,
        config=config,
        as_of=date.fromisoformat(str(prices.index[-1])),
        entry_thresholds=(1.5,),
        rolling_windows=(20,),
    )


def test_joint_bootstrap_paths_are_reproducible_and_keep_joint_rows() -> None:
    """Verify seeded generation and contemporaneous paired returns.

    Parameters
    ----------
    None
        Fixed returns and simulation settings are used.

    Returns
    -------
    None
        Assertions validate path dimensions, compounding, and reproducibility.

    Raises
    ------
    AssertionError
        If output shape, seed determinism, or row pairing is violated.

    Notes
    -----
    Asset B returns are an exact multiple of asset A returns; every synthetic
    step should preserve that row-wise joint relationship because sampling is
    performed on two-column return blocks together.
    """
    returns_a = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
    returns = pd.DataFrame(
        {"INTC": returns_a, "AMD": 2.0 * returns_a},
        index=pd.date_range("2022-01-01", periods=len(returns_a), freq="D"),
    )
    initial_prices = pd.Series({"INTC": 100.0, "AMD": 50.0})

    paths = generate_joint_block_bootstrap_paths(
        returns,
        initial_prices,
        horizon=9,
        n_paths=5,
        block_size=3,
        seed=123,
    )
    repeated = generate_joint_block_bootstrap_paths(
        returns,
        initial_prices,
        horizon=9,
        n_paths=5,
        block_size=3,
        seed=123,
    )
    different_seed = generate_joint_block_bootstrap_paths(
        returns,
        initial_prices,
        horizon=9,
        n_paths=5,
        block_size=3,
        seed=456,
    )

    assert paths.shape == (5 * 10, 2)
    pd.testing.assert_frame_equal(paths, repeated)
    assert not paths.equals(different_seed)
    assert (paths > 0.0).all().all()
    for _, path in paths.groupby(level="path"):
        sampled_returns = path.pct_change(fill_method=None).iloc[1:]
        np.testing.assert_allclose(
            sampled_returns["AMD"].to_numpy(),
            2.0 * sampled_returns["INTC"].to_numpy(),
        )


def test_run_joint_bootstrap_uses_only_pre_split_prices() -> None:
    """Run a small simulation and verify its report and temporal boundary.

    Parameters
    ----------
    None
        Fixed synthetic prices and a small path count are used.

    Returns
    -------
    None
        Assertions validate summary dimensions and a successful simulation.

    Raises
    ------
    AssertionError
        If the simulation result is malformed.
    src.simulation.SimulationError
        If the routine rejects the in-sample fixture.

    Notes
    -----
    The backtest report supplies frozen training coefficients and parameters;
    the simulator is given only dates strictly before its split.
    """
    prices = _prices()
    report = _backtest_report(prices)
    in_sample = prices.loc[
        [observation_date < report.split_date for observation_date in prices.index]
    ]
    assert all(observation_date < report.split_date for observation_date in in_sample.index)

    result = run_joint_block_bootstrap(
        in_sample_prices=in_sample,
        report=report,
        tickers=("INTC", "AMD"),
        horizon=12,
        n_paths=8,
        block_size=5,
        seed=77,
    )

    assert len(result.path_pnl) == 8
    assert result.horizon == 12
    assert result.block_size == 5
    assert result.seed == 77
    assert np.isfinite(result.path_pnl.to_numpy()).all()
    assert 0.0 <= result.profitable_fraction <= 1.0


def test_bootstrap_rejects_oos_prices() -> None:
    """Reject any bootstrap calibration data at or beyond the split.

    Parameters
    ----------
    None
        Uses deterministic prices and a fitted backtest report.

    Returns
    -------
    None
        The expected domain exception is asserted.

    Raises
    ------
    AssertionError
        If OOS observations are accepted as bootstrap calibration data.

    Notes
    -----
    Enforcing the date boundary inside the simulation API protects callers
    beyond the main pipeline from accidentally leaking OOS returns.
    """
    prices = _prices()
    report = _backtest_report(prices)

    with pytest.raises(SimulationError, match="strictly before the split"):
        run_joint_block_bootstrap(
            in_sample_prices=prices,
            report=report,
            tickers=("INTC", "AMD"),
            horizon=5,
            n_paths=2,
            block_size=5,
            seed=1,
        )


def test_path_generator_rejects_block_larger_than_history() -> None:
    """Reject bootstrap blocks that cannot be sampled from available data.

    Parameters
    ----------
    None
        Fixed return data uses only two observations.

    Returns
    -------
    None
        The expected domain exception is asserted.

    Raises
    ------
    AssertionError
        If invalid block sizing is accepted.

    Notes
    -----
    Restricting block size to the observed return history guarantees every
    sampled block is a valid contiguous slice rather than a wrapped or padded
    synthetic segment.
    """
    returns = pd.DataFrame(
        {"INTC": [0.01, -0.01], "AMD": [0.02, -0.02]},
        index=pd.date_range("2022-01-01", periods=2, freq="D"),
    )
    initial_prices = pd.Series({"INTC": 100.0, "AMD": 50.0})

    with pytest.raises(SimulationError, match="must not exceed"):
        generate_joint_block_bootstrap_paths(
            returns,
            initial_prices,
            horizon=5,
            n_paths=2,
            block_size=3,
        )
