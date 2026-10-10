from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import (
    BacktestError,
    BacktestReport,
    analyze_parameter_sensitivity,
    run_backtest,
)
from src.config import ProjectConfig
from src.data_loader import DataFetchError, load_price_data
from src.features import FeatureCalculationError
from src.stability import (
    StabilityAnalysisError,
    compare_stability_with_performance,
    compute_rolling_adf,
)
from src.short_interest import (
    ShortInterestError,
    analyze_short_interest_impact,
    fetch_nasdaq_short_interest,
)
from src.simulation import SimulationError, run_joint_block_bootstrap

logger = logging.getLogger(__name__)


def _parse_date(value: str) -> date:
    """Parse an ISO-8601 calendar date for the command-line cutoff.

    Parameters
    ----------
    value : str
        Date in ``YYYY-MM-DD`` format.

    Returns
    -------
    datetime.date
        Parsed calendar date.

    Raises
    ------
    ValueError
        If ``value`` is not a valid ISO-8601 calendar date.

    Notes
    -----
    Keeping the CLI representation as ``date`` ensures the same explicit cutoff
    type is passed through to the data loader.
    """
    return date.fromisoformat(value)


def _plot_warmup(prices: pd.DataFrame, output_dir: Path, as_of: date) -> tuple[Path, ...]:
    """Save diagnostic warm-up plots from already-loaded pair prices.

    Parameters
    ----------
    prices : pandas.DataFrame
        Date-indexed INTC and AMD adjusted prices.
    output_dir : pathlib.Path, default=Path("outputs/warmup")
        Directory where the diagnostic PNG files are saved.
    as_of : datetime.date
        Inclusive latest permitted observation.

    Returns
    -------
    tuple of pathlib.Path
        Paths to the saved price, ratio, difference, and rolling-correlation plots.

    Raises
    ------
    ValueError
        If prices are empty or do not contain both configured pair columns.
    AssertionError
        If the input has observations beyond ``as_of``.
    OSError
        If the output directory or plot files cannot be written.

    Notes
    -----
    Price data is passed in by the pipeline and reused by the backtest, avoiding
    a duplicate data request and ensuring both phases use the same snapshot.
    """
    if prices.empty:
        raise ValueError("prices must not be empty")
    if "INTC" not in prices.columns or "AMD" not in prices.columns:
        raise ValueError("prices must contain INTC and AMD columns")

    assert all(pd.Timestamp(observation_date).date() <= as_of for observation_date in prices.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dates = pd.to_datetime(prices.index)
    output_paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_dates, prices["INTC"], label="INTC")
    axis.plot(plot_dates, prices["AMD"], label="AMD")
    axis.set(title="INTC and AMD adjusted close prices", xlabel="Date", ylabel="Price (USD)")
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    price_plot = output_dir / "prices.png"
    figure.savefig(price_plot, dpi=150)
    plt.close(figure)
    output_paths.append(price_plot)

    ratio = prices["INTC"] / prices["AMD"]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_dates, ratio)
    axis.set(title="INTC / AMD price ratio", xlabel="Date", ylabel="Price ratio")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    ratio_plot = output_dir / "price_ratio.png"
    figure.savefig(ratio_plot, dpi=150)
    plt.close(figure)
    output_paths.append(ratio_plot)

    difference = prices["INTC"] - prices["AMD"]
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_dates, difference)
    axis.set(title="INTC - AMD price difference", xlabel="Date", ylabel="Price difference (USD)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    difference_plot = output_dir / "price_difference.png"
    figure.savefig(difference_plot, dpi=150)
    plt.close(figure)
    output_paths.append(difference_plot)

    returns = prices.pct_change(fill_method=None)
    rolling_correlation = returns["INTC"].rolling(window=60, min_periods=60).corr(
        returns["AMD"]
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_dates, rolling_correlation)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    axis.set(
        title="60-session rolling correlation of daily returns",
        xlabel="Date",
        ylabel="Correlation",
        ylim=(-1, 1),
    )
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    correlation_plot = output_dir / "rolling_correlation_60d.png"
    figure.savefig(correlation_plot, dpi=150)
    plt.close(figure)
    output_paths.append(correlation_plot)

    logger.info(
        "Phase 1 complete: %d aligned sessions from %s through %s; plots saved to %s",
        len(prices),
        prices.index.min(),
        prices.index.max(),
        output_dir,
    )
    return tuple(output_paths)


def run_warmup(
    as_of: date,
    start_date: date = date(2010, 1, 1),
    cache_dir: Path = Path("data"),
    output_dir: Path = Path("outputs/warmup"),
) -> tuple[Path, ...]:
    """Load the INTC/AMD history and save diagnostic warm-up plots.

    Parameters
    ----------
    as_of : datetime.date
        Inclusive latest permitted observation, normally the prior calendar day.
    start_date : datetime.date, default=2010-01-01
        Inclusive beginning of the historical period.
    cache_dir : pathlib.Path, default=Path("data")
        Directory used by the Yahoo Finance pickle cache.
    output_dir : pathlib.Path, default=Path("outputs/warmup")
        Directory where the diagnostic PNG files are saved.

    Returns
    -------
    tuple of pathlib.Path
        Paths to the saved price, ratio, difference, and rolling-correlation plots.

    Raises
    ------
    ValueError
        If the requested date range or returned prices are invalid.
    src.data_loader.DataFetchError
        If historical prices cannot be loaded.
    OSError
        If the output directory or plot files cannot be written.

    Notes
    -----
    This public helper fetches prices for a standalone Phase 1 run. The full
    pipeline instead loads once and passes the shared snapshot to
    :func:`_plot_warmup` and the backtest.
    """
    if start_date > as_of:
        raise ValueError("start_date must be on or before as_of")
    prices = load_price_data(
        ("INTC", "AMD"),
        start_date=start_date,
        as_of=as_of,
        cache_dir=cache_dir,
    )
    return _plot_warmup(prices, output_dir, as_of)


def run_full_pipeline(
    config: ProjectConfig,
    as_of: date,
    start_date: date,
    cache_dir: Path = Path("data"),
    output_dir: Path = Path("outputs/warmup"),
) -> BacktestReport:
    """Execute the data warm-up and blind in-sample/out-of-sample backtest.

    Parameters
    ----------
    config : ProjectConfig
        Pair tickers, split date, and strategy exit threshold.
    as_of : datetime.date
        Inclusive latest permitted price date.
    start_date : datetime.date
        Inclusive start of downloaded price history.
    cache_dir : pathlib.Path, default=Path("data")
        Directory used for local pickle checkpoints.
    output_dir : pathlib.Path, default=Path("outputs/warmup")
        Directory used for Phase 1 diagnostic plots.

    Returns
    -------
    BacktestReport
        Separate in-sample/out-of-sample metrics and the complete nine-attempt
        default parameter search ledger.

    Raises
    ------
    ValueError
        If the date range is invalid.
    DataFetchError
        If the pair history cannot be loaded.
    BacktestError
        If the split or backtest protocol is invalid.
    FeatureCalculationError
        If the requested features cannot be computed.
    StabilityAnalysisError
        If the OOS history is too short or invalid for rolling stability tests.
    SimulationError
        If the pre-split history cannot support Phase 5 bootstrap simulation.
    OSError
        If cache or diagnostic plot files cannot be written.

    Notes
    -----
    Prices are loaded once and the identical as-of-bounded frame is used for
    Phase 1 plots and Phase 2 backtesting. The fixed default parameter grid
    contains three entry thresholds and three rolling windows, for nine logged
    in-sample attempts before one blind OOS evaluation. Phase 3 runs a
    90-observation rolling ADF test on the fixed-hedge OOS spread and logs
    retrospective PnL/drawdown comparisons for stable and unstable dates.
    Sensitivity is reported from the IS trial ledger; OOS results are not used
    for robustness classification or parameter selection.
    Phase 4 retrieves short-interest data best-effort. Endpoint coverage or
    availability failures are reported as warnings without invalidating prior
    phases. Phase 5 uses only prices strictly before the split to sample joint
    return blocks, then evaluates the frozen strategy over synthetic paths.
    Its horizon matches the observed OOS session count; the seeded run logs
    summary statistics but does not use OOS outcomes for model selection.
    """
    if start_date > as_of:
        raise ValueError("start_date must be on or before as_of")
    split_date = date.fromisoformat(config.train_test_split_date)
    if split_date > as_of:
        raise ValueError("train_test_split_date must not be after as_of")
    if config.ticker_a == config.ticker_b:
        raise ValueError("ProjectConfig must define two distinct tickers")

    prices = load_price_data(
        (config.ticker_a, config.ticker_b),
        start_date=start_date,
        as_of=as_of,
        cache_dir=cache_dir,
    )
    assert not prices.empty
    assert all(pd.Timestamp(observation_date).date() <= as_of for observation_date in prices.index)

    _plot_warmup(prices, output_dir, as_of)

    entry_thresholds = (1.5, 2.0, 2.5)
    rolling_windows = (40, 60, 90)
    expected_attempts = len(entry_thresholds) * len(rolling_windows)
    logger.info(
        "Phase 2 starting: split=%s; parameter grid=%d entry thresholds x %d rolling windows (%d attempts)",
        split_date,
        len(entry_thresholds),
        len(rolling_windows),
        expected_attempts,
    )
    report = run_backtest(
        prices,
        config=config,
        as_of=as_of,
        entry_thresholds=entry_thresholds,
        rolling_windows=rolling_windows,
    )
    assert report.attempt_count == expected_attempts
    assert report.split_date == split_date
    assert report.as_of == as_of
    logger.info("Phase 2 complete: %d parameter attempts recorded.", report.attempt_count)
    sensitivity = analyze_parameter_sensitivity(report)
    if sensitivity.baseline.in_sample.pnl <= 0.0:
        sensitivity_verdict = "INCONCLUSIVE"
    elif sensitivity.is_fragile:
        sensitivity_verdict = "FRAGILE"
    else:
        sensitivity_verdict = "no fragility trigger (not proof of robustness)"
    logger.info(
        "Parameter sensitivity verdict: %s; baseline IS PnL=%.4f; "
        "OOS PnL (diagnostic only)=%.4f",
        sensitivity_verdict,
        sensitivity.baseline.in_sample.pnl,
        report.out_of_sample.pnl,
    )

    adf_p_values = compute_rolling_adf(
        report.out_of_sample_spread,
        window=90,
        as_of=as_of,
    )
    stability_comparison = compare_stability_with_performance(
        adf_p_values,
        report.out_of_sample_daily_pnl,
        significance_level=0.05,
        as_of=as_of,
    )
    assert (
        stability_comparison.stable_observations
        + stability_comparison.unstable_observations
        == int(adf_p_values.notna().sum())
    )
    logger.info(
        "Phase 3 complete: rolling ADF and performance comparison over %d valid OOS dates.",
        stability_comparison.stable_observations + stability_comparison.unstable_observations,
    )

    try:
        intc_short_interest = fetch_nasdaq_short_interest(config.ticker_a)
        amd_short_interest = fetch_nasdaq_short_interest(config.ticker_b)
        short_interest_impact = analyze_short_interest_impact(
            intc=intc_short_interest,
            amd=amd_short_interest,
            spread=report.out_of_sample_spread,
            daily_pnl=report.out_of_sample_daily_pnl,
            adf_p_values=adf_p_values,
            as_of=as_of,
        )
        logger.info(
            "Phase 4 complete: matched observations=%d; report updates=%d; "
            "INTC correlation=%s; AMD correlation=%s; unstable mean PnL=%s; "
            "stable mean PnL=%s",
            short_interest_impact.observations,
            short_interest_impact.report_updates,
            short_interest_impact.intc_pct_change_spread_change_correlation,
            short_interest_impact.amd_pct_change_spread_change_correlation,
            short_interest_impact.unstable_mean_daily_pnl,
            short_interest_impact.stable_mean_daily_pnl,
        )
    except ShortInterestError as error:
        logger.warning(
            "Phase 4 unavailable; Nasdaq short-interest data could not be fetched "
            "or aligned. Price backtest and stability analysis remain valid: %s",
            error,
        )

    in_sample_prices = prices.loc[
        [pd.Timestamp(observation_date).date() < split_date for observation_date in prices.index]
    ]
    simulation_report = run_joint_block_bootstrap(
        in_sample_prices=in_sample_prices,
        report=report,
        tickers=(config.ticker_a, config.ticker_b),
        horizon=len(report.out_of_sample_daily_pnl),
        n_paths=500,
        block_size=10,
        seed=42,
    )
    assert simulation_report.horizon == len(report.out_of_sample_daily_pnl)
    logger.info(
        "Phase 5 complete: %d joint-bootstrap paths (horizon=%d, block_size=%d, seed=%d); "
        "PnL mean=%.4f median=%.4f p05=%.4f p95=%.4f profitable_fraction=%.2f%%",
        simulation_report.n_paths,
        simulation_report.horizon,
        simulation_report.block_size,
        simulation_report.seed,
        simulation_report.mean_pnl,
        simulation_report.median_pnl,
        simulation_report.percentile_05,
        simulation_report.percentile_95,
        simulation_report.profitable_fraction * 100.0,
    )
    return report


def main() -> None:
    """Run both historical warm-up and the blind pair backtest from the CLI.

    Parameters
    ----------
    None
        Command-line arguments are read from ``sys.argv``.

    Returns
    -------
    None
        Logs completion after saving diagnostics and reporting Phases 2–5.

    Raises
    ------
    SystemExit
        If command-line arguments are invalid or pipeline execution fails.

    Notes
    -----
    Without ``--as-of``, the prior calendar day is selected as the explicit
    D-1 information cutoff. Configuration defaults provide the pair, start
    date, and split; command-line options allow reproducible date overrides.
    Pipeline failures are logged and surfaced as ``SystemExit`` with the
    original data, backtest, feature, stability, simulation, or I/O error
    preserved as the exception cause.
    """
    config = ProjectConfig()
    parser = argparse.ArgumentParser(description="Warm up and backtest the INTC/AMD pair.")
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=date.today() - timedelta(days=1),
        help="Inclusive last permitted date in YYYY-MM-DD format (default: yesterday).",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=date.fromisoformat(config.start_date),
        help=f"Inclusive first date in YYYY-MM-DD format (default: {config.start_date}).",
    )
    parser.add_argument(
        "--split-date",
        type=_parse_date,
        default=date.fromisoformat(config.train_test_split_date),
        help=(
            "First out-of-sample date in YYYY-MM-DD format "
            f"(default: {config.train_test_split_date})."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data"),
        help="Directory for local Yahoo Finance pickle checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/warmup"),
        help="Directory for generated diagnostic plots.",
    )
    arguments = parser.parse_args()

    if arguments.start_date > arguments.as_of:
        parser.error("--start-date must be on or before --as-of")
    if arguments.split_date > arguments.as_of:
        parser.error("--split-date must be on or before --as-of")

    config = ProjectConfig(
        ticker_a=config.ticker_a,
        ticker_b=config.ticker_b,
        start_date=arguments.start_date.isoformat(),
        end_date=arguments.as_of.isoformat(),
        train_test_split_date=arguments.split_date.isoformat(),
        z_score_entry=config.z_score_entry,
        z_score_exit=config.z_score_exit,
        rolling_window=config.rolling_window,
    )
    try:
        run_full_pipeline(
            config=config,
            as_of=arguments.as_of,
            start_date=arguments.start_date,
            cache_dir=arguments.cache_dir,
            output_dir=arguments.output_dir,
        )
    except (
        DataFetchError,
        BacktestError,
        FeatureCalculationError,
        StabilityAnalysisError,
        SimulationError,
        OSError,
        ValueError,
    ) as error:
        logger.error("Pipeline failed: %s", error, exc_info=True)
        raise SystemExit(1) from error


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()