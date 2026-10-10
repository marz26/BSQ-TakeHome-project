from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main
from src.backtest import BacktestReport
from src.config import ProjectConfig
from src.simulation import SimulationReport


def test_full_pipeline_shares_one_price_snapshot_with_both_phases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Phase 1 and Phase 2 use one as-of-bounded price snapshot.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace data loading and plot output.
    tmp_path : pathlib.Path
        Isolated temporary cache and plot directories.

    Returns
    -------
    None
        Assertions verify the pipeline report and its phase inputs.

    Raises
    ------
    AssertionError
        If the phases do not share input data or all nine attempts are omitted.

    Notes
    -----
    The synthetic history contains enough pre-split observations for the
    largest default rolling window and post-split observations for OOS scoring.
    Backtesting itself remains real; only remote data access and image writes
    are replaced to keep this integration test deterministic and offline.
    """
    observations = 220
    dates = pd.date_range("2020-01-01", periods=observations, freq="D").date
    time = np.arange(observations, dtype=float)
    amd = 40.0 + 0.1 * time + np.sin(time / 8.0)
    intc = 4.0 + 1.2 * amd + np.sin(time / 3.0)
    prices = pd.DataFrame(
        {"INTC": intc, "AMD": amd},
        index=pd.Index(dates, name="Date"),
    )
    loads: list[tuple[tuple[str, ...], date, Path]] = []
    plots: list[tuple[pd.DataFrame, Path, date]] = []
    simulations: list[tuple[pd.DataFrame, int, int, int]] = []

    def fake_load(
        tickers: tuple[str, ...],
        start_date: date,
        as_of: date,
        cache_dir: Path,
    ) -> pd.DataFrame:
        """Return the deterministic fixture and record the loader invocation.

        Parameters
        ----------
        tickers : tuple of str
            Requested pair symbols.
        start_date : datetime.date
            Requested first history date.
        as_of : datetime.date
            Requested latest history date.
        cache_dir : pathlib.Path
            Requested cache directory.

        Returns
        -------
        pandas.DataFrame
            The shared synthetic price snapshot.

        Raises
        ------
        None
            The fixture loader always succeeds.

        Notes
        -----
        This fake makes the number and exact cutoff of fetches observable.
        """
        assert start_date <= as_of
        loads.append((tickers, as_of, cache_dir))
        return prices

    def fake_plot(
        plot_prices: pd.DataFrame,
        output_dir: Path,
        as_of: date,
    ) -> tuple[Path, ...]:
        """Record Phase 1's price snapshot without producing image files.

        Parameters
        ----------
        plot_prices : pandas.DataFrame
            Prices passed to the diagnostics.
        output_dir : pathlib.Path
            Requested diagnostic output directory.
        as_of : datetime.date
            Inclusive date cutoff.

        Returns
        -------
        tuple of pathlib.Path
            Empty tuple because plot creation is replaced for this test.

        Raises
        ------
        None
            The plotting fake always succeeds.

        Notes
        -----
        Object identity is recorded so the test can assert Phase 1 and Phase 2
        share the same loaded DataFrame rather than independently fetching it.
        """
        plots.append((plot_prices, output_dir, as_of))
        return ()

    def fake_simulation(
        in_sample_prices: pd.DataFrame,
        report: BacktestReport,
        tickers: tuple[str, str],
        horizon: int,
        n_paths: int = 500,
        block_size: int = 10,
        seed: int = 42,
    ) -> SimulationReport:
        """Record bootstrap input and return a deterministic summary.

        Parameters
        ----------
        in_sample_prices : pandas.DataFrame
            Historical prices supplied to Phase 5.
        report : object
            Frozen backtest result; accepted broadly in this test fake.
        tickers : tuple of str
            Ordered asset names.
        horizon : int
            Number of simulated future observations.
        n_paths : int, default=500
            Requested path count.
        block_size : int, default=10
            Requested block size.
        seed : int, default=42
            Requested generator seed.

        Returns
        -------
        SimulationReport
            Small deterministic report used to avoid running Monte Carlo paths.

        Raises
        ------
        AssertionError
            If Phase 5 receives unexpected pair configuration.

        Notes
        -----
        The test records the input snapshot to prove observations on or after
        the split are excluded before bootstrap sampling begins.
        """
        assert tickers == ("INTC", "AMD")
        simulations.append((in_sample_prices.copy(), horizon, block_size, seed))
        return SimulationReport(
            path_pnl=pd.Series([0.0] * n_paths),
            mean_pnl=0.0,
            median_pnl=0.0,
            standard_deviation=0.0,
            minimum_pnl=0.0,
            maximum_pnl=0.0,
            percentile_05=0.0,
            percentile_95=0.0,
            profitable_fraction=0.0,
            n_paths=n_paths,
            horizon=horizon,
            block_size=block_size,
            seed=seed,
        )

    monkeypatch.setattr(main, "load_price_data", fake_load)
    monkeypatch.setattr(main, "_plot_warmup", fake_plot)
    monkeypatch.setattr(main, "run_joint_block_bootstrap", fake_simulation)
    as_of = dates[-1]
    start_date = dates[0]
    short_interest = pd.DataFrame(
        {"short_interest": [100_000.0, 120_000.0, 90_000.0]},
        index=pd.Index([dates[115], dates[150], dates[185]], name="settlement_date"),
    )
    monkeypatch.setattr(
        main,
        "fetch_nasdaq_short_interest",
        lambda ticker: short_interest,
    )
    config = ProjectConfig(
        start_date=start_date.isoformat(),
        end_date=as_of.isoformat(),
        train_test_split_date=dates[110].isoformat(),
    )

    report = main.run_full_pipeline(
        config=config,
        as_of=as_of,
        start_date=start_date,
        cache_dir=tmp_path / "data",
        output_dir=tmp_path / "plots",
    )

    assert len(loads) == 1
    assert loads[0] == (("INTC", "AMD"), as_of, tmp_path / "data")
    assert len(plots) == 1
    assert plots[0][0] is prices
    assert plots[0][1:] == (tmp_path / "plots", as_of)
    assert report.attempt_count == 9
    assert report.as_of == as_of
    assert len(simulations) == 1
    simulation_prices, horizon, block_size, seed = simulations[0]
    split_date = date.fromisoformat(config.train_test_split_date)
    assert all(observation_date < split_date for observation_date in simulation_prices.index)
    assert len(simulation_prices) == sum(observation_date < split_date for observation_date in dates)
    assert horizon == len(report.out_of_sample_daily_pnl)
    assert (block_size, seed) == (10, 42)
