from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.backtest import BacktestReport
from src.features import FeatureCalculationError, compute_z_score

logger = logging.getLogger(__name__)


class SimulationError(ValueError):
    """Raised when a joint block-bootstrap simulation cannot be performed.

    Parameters
    ----------
    message : str
        Description of the invalid simulation input or failed invariant.

    Returns
    -------
    None
        Exception constructors do not return a value.

    Raises
    ------
    None
        This exception class does not raise errors during construction.

    Notes
    -----
    Expected simulation input and numerical failures use a domain-specific
    exception, making an unsuccessful experiment explicit to pipeline callers.
    """


@dataclass(frozen=True)
class SimulationReport:
    """Summary of strategy PnL across joint-bootstrap synthetic price paths.

    Parameters
    ----------
    path_pnl : pandas.Series
        Simulated spread-price PnL, indexed by synthetic path number.
    mean_pnl : float
        Arithmetic mean path PnL.
    median_pnl : float
        Median path PnL.
    standard_deviation : float
        Sample standard deviation of path PnL; zero for a single path.
    minimum_pnl : float
        Worst observed simulated path PnL.
    maximum_pnl : float
        Best observed simulated path PnL.
    percentile_05 : float
        Fifth percentile of simulated path PnL.
    percentile_95 : float
        Ninety-fifth percentile of simulated path PnL.
    profitable_fraction : float
        Fraction of paths with strictly positive PnL.
    n_paths : int
        Number of simulated paths.
    horizon : int
        Number of simulated daily return observations per path.
    block_size : int
        Number of consecutive joint return observations per sampled block.
    seed : int
        Random generator seed used for reproducibility.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are created by :func:`run_joint_block_bootstrap`.

    Notes
    -----
    PnL is the same uncapitalized fixed-hedge spread-price PnL used by the
    project backtest. Percentiles and profitable fraction describe conditional
    simulations from historical in-sample blocks; they are not probabilities
    of future outcomes and exclude transaction costs.
    """

    path_pnl: pd.Series
    mean_pnl: float
    median_pnl: float
    standard_deviation: float
    minimum_pnl: float
    maximum_pnl: float
    percentile_05: float
    percentile_95: float
    profitable_fraction: float
    n_paths: int
    horizon: int
    block_size: int
    seed: int


def generate_joint_block_bootstrap_paths(
    daily_returns: pd.DataFrame,
    initial_prices: pd.Series,
    horizon: int,
    n_paths: int = 500,
    block_size: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible two-asset price paths from joint return blocks.

    Parameters
    ----------
    daily_returns : pandas.DataFrame
        Historical daily simple returns for exactly two assets, with aligned
        observations in chronological order.
    initial_prices : pandas.Series
        Positive starting prices; its index must match the two return columns.
    horizon : int
        Number of future daily return observations in every path.
    n_paths : int, default=500
        Number of synthetic paths to draw.
    block_size : int, default=10
        Number of consecutive paired daily return observations in each block.
    seed : int, default=42
        Seed for NumPy's random generator.

    Returns
    -------
    pandas.DataFrame
        Synthetic prices indexed by a ``(path, step)`` MultiIndex. Each path
        includes step 0 at ``initial_prices`` and steps 1 through ``horizon``.

    Raises
    ------
    SimulationError
        If the input shape, dates, returns, prices, or simulation dimensions
        are invalid, or if generated price paths are not finite and positive.

    Notes
    -----
    Both asset returns are sampled from the same contiguous row blocks, which
    preserves contemporaneous cross-asset dependence and short-run serial
    structure within each block. Blocks are sampled with replacement from the
    historical moving-block set, without wrapping across the sample boundary.
    Prices are reconstructed by compounding simple returns. A local seeded
    generator avoids mutating global random state.
    """
    if daily_returns.shape[1] != 2:
        raise SimulationError("daily_returns must contain exactly two asset columns")
    if daily_returns.empty:
        raise SimulationError("daily_returns must not be empty")
    if not daily_returns.columns.is_unique or set(daily_returns.columns) != set(initial_prices.index):
        raise SimulationError("initial_prices index must match the two return columns")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise SimulationError("horizon must be a positive integer")
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise SimulationError("n_paths must be a positive integer")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise SimulationError("block_size must be a positive integer")
    if block_size > len(daily_returns):
        raise SimulationError("block_size must not exceed available return observations")
    if not np.isfinite(daily_returns.to_numpy(dtype=float)).all():
        raise SimulationError("daily_returns must contain finite values")
    if (daily_returns <= -1.0).any().any():
        raise SimulationError("simple returns must be greater than -1")
    try:
        start_prices = initial_prices.loc[daily_returns.columns].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise SimulationError("initial_prices must provide one numeric value per asset") from error
    if not np.isfinite(start_prices).all() or (start_prices <= 0.0).any():
        raise SimulationError("initial_prices must be finite and strictly positive")
    if not daily_returns.index.is_unique or not daily_returns.index.is_monotonic_increasing:
        raise SimulationError("daily_returns index must be unique and chronologically increasing")

    return_values = daily_returns.to_numpy(dtype=float)
    block_starts = np.arange(len(return_values) - block_size + 1)
    random_generator = np.random.default_rng(seed)
    sampled_paths = np.empty((n_paths, horizon + 1, 2), dtype=float)
    sampled_paths[:, 0, :] = start_prices

    for path_number in range(n_paths):
        path_returns: list[np.ndarray] = []
        remaining = horizon
        while remaining > 0:
            block_start = int(random_generator.choice(block_starts))
            block = return_values[block_start : block_start + block_size]
            used_block = block[:remaining]
            path_returns.append(used_block)
            remaining -= len(used_block)
        concatenated_returns = np.concatenate(path_returns, axis=0)
        assert concatenated_returns.shape == (horizon, 2)
        sampled_paths[path_number, 1:, :] = start_prices * np.cumprod(
            1.0 + concatenated_returns,
            axis=0,
        )

    assert sampled_paths.shape == (n_paths, horizon + 1, 2)
    assert np.isfinite(sampled_paths).all()
    assert (sampled_paths > 0.0).all()
    path_index = np.repeat(np.arange(n_paths), horizon + 1)
    step_index = np.tile(np.arange(horizon + 1), n_paths)
    result = pd.DataFrame(
        sampled_paths.reshape(n_paths * (horizon + 1), 2),
        index=pd.MultiIndex.from_arrays(
            [path_index, step_index],
            names=["path", "step"],
        ),
        columns=daily_returns.columns.copy(),
    )
    logger.info(
        "Generated %d joint-bootstrap paths: horizon=%d, block_size=%d, seed=%d",
        n_paths,
        horizon,
        block_size,
        seed,
    )
    return result


def _evaluate_path_pnl(
    synthetic_prices: pd.DataFrame,
    calibration_spread: pd.Series,
    report: BacktestReport,
) -> float:
    """Apply frozen strategy parameters to one synthetic future price path.

    Parameters
    ----------
    synthetic_prices : pandas.DataFrame
        One path including the final in-sample starting price at step zero.
    calibration_spread : pandas.Series
        Historical in-sample residual spread calculated with frozen report
        coefficients.
    report : BacktestReport
        Frozen alpha, beta, and selected entry/exit/window parameters.

    Returns
    -------
    float
        Cumulative synthetic spread-price PnL over the future path.

    Raises
    ------
    SimulationError
        If path columns, observations, or values are inconsistent.
    FeatureCalculationError
        If the causal rolling z-score cannot be calculated.

    Notes
    -----
    The trailing historical residuals warm up the rolling z-score but earn no
    simulated PnL. At each synthetic step the position is set using the prior
    step's z-score before earning the current spread change, matching the
    backtest's one-observation signal lag and avoiding close-price look-ahead.
    """
    parameters = report.selected_parameters
    if synthetic_prices.shape[1] != 2 or synthetic_prices.empty:
        raise SimulationError("synthetic_prices must contain two assets and a starting row")
    if not synthetic_prices.index.is_unique or not synthetic_prices.index.is_monotonic_increasing:
        raise SimulationError("synthetic path steps must be unique and increasing")
    if len(calibration_spread) < parameters.rolling_window:
        raise SimulationError("in-sample spread is too short to warm up the selected z-score window")
    if not np.isfinite(synthetic_prices.to_numpy(dtype=float)).all() or (
        synthetic_prices <= 0.0
    ).any().any():
        raise SimulationError("synthetic path prices must be finite and strictly positive")

    synthetic_spread = (
        synthetic_prices.iloc[:, 0]
        - (
            report.alpha
            + report.beta * synthetic_prices.iloc[:, 1]
        )
    )
    historical_tail = calibration_spread.iloc[-parameters.rolling_window :]
    combined_spread = pd.concat([historical_tail, synthetic_spread.iloc[1:]])
    try:
        z_score = compute_z_score(combined_spread, parameters.rolling_window)
    except FeatureCalculationError as error:
        raise SimulationError("could not calculate z-score for synthetic path") from error

    first_simulated_position = len(historical_tail)
    position = 0
    total_pnl = 0.0
    for current_position in range(first_simulated_position, len(combined_spread)):
        previous_z = z_score.iloc[current_position - 1]
        next_position = position
        if np.isfinite(previous_z):
            if position == 0:
                if previous_z >= parameters.entry_z:
                    next_position = -1
                elif previous_z <= -parameters.entry_z:
                    next_position = 1
            elif abs(previous_z) <= parameters.exit_z:
                next_position = 0
        position = next_position
        total_pnl += position * (
            float(combined_spread.iloc[current_position])
            - float(combined_spread.iloc[current_position - 1])
        )

    assert np.isfinite(total_pnl)
    assert len(combined_spread) == len(historical_tail) + len(synthetic_prices) - 1
    return float(total_pnl)


def run_joint_block_bootstrap(
    in_sample_prices: pd.DataFrame,
    report: BacktestReport,
    tickers: tuple[str, str],
    horizon: int,
    n_paths: int = 500,
    block_size: int = 10,
    seed: int = 42,
) -> SimulationReport:
    """Simulate the frozen pair strategy over joint-bootstrap return paths.

    Parameters
    ----------
    in_sample_prices : pandas.DataFrame
        Positive daily asset prices strictly before ``report.split_date``.
    report : BacktestReport
        Backtest report providing the IS-fitted hedge and selected parameters.
    tickers : tuple of str
        Pair ticker column names in the order expected by the fitted report.
    horizon : int
        Number of synthetic sessions to simulate, normally the observed OOS
        session count.
    n_paths : int, default=500
        Number of synthetic return paths.
    block_size : int, default=10
        Number of adjacent paired return rows per bootstrap block.
    seed : int, default=42
        Seed used for reproducible path generation.

    Returns
    -------
    SimulationReport
        Distribution summary and path-level PnL from the frozen strategy.

    Raises
    ------
    SimulationError
        If prices are not strictly in-sample, contain invalid observations, or
        do not provide enough history/returns for the chosen simulation.

    Notes
    -----
    Only pre-split prices calibrate the joint return blocks. Hedge coefficients
    and strategy parameters are taken unchanged from ``report``; no synthetic
    outcome is used for selection. The start point is the final IS price and
    the horizon is supplied by the caller, commonly the OOS duration. This
    conditional bootstrap assumes historical IS blocks are informative for
    scenarios; it cannot represent regimes absent from that finite sample and
    does not prove the spread stationary.
    """
    if len(tickers) != 2 or tickers[0] == tickers[1]:
        raise SimulationError("tickers must name two distinct assets")
    if any(ticker not in in_sample_prices.columns for ticker in tickers):
        raise SimulationError("in_sample_prices must contain both requested tickers")
    if in_sample_prices.empty:
        raise SimulationError("in_sample_prices must not be empty")

    prices = in_sample_prices.loc[:, list(tickers)].copy()
    try:
        normalized_dates = pd.Index(
            [pd.Timestamp(index_value).date() for index_value in prices.index],
            name=prices.index.name or "Date",
        )
        prices.index = normalized_dates
        prices = prices.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise SimulationError("in_sample_prices must have valid dates and numeric values") from error
    if normalized_dates.has_duplicates or not normalized_dates.is_monotonic_increasing:
        raise SimulationError("in_sample_prices dates must be unique and increasing")
    if any(observation_date >= report.split_date for observation_date in normalized_dates):
        raise SimulationError("in_sample_prices must contain observations strictly before the split")
    if any(observation_date > report.as_of for observation_date in normalized_dates):
        raise SimulationError("in_sample_prices contains observations beyond report.as_of")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0.0).any().any():
        raise SimulationError("in_sample_prices must be finite and strictly positive")

    parameters = report.selected_parameters
    if len(prices) < parameters.rolling_window + 2:
        raise SimulationError("in-sample history is too short to initialize the selected strategy")
    daily_returns = prices.pct_change(fill_method=None).iloc[1:]
    if daily_returns.empty:
        raise SimulationError("at least two in-sample price observations are required")
    if block_size > len(daily_returns):
        raise SimulationError("block_size must not exceed available in-sample returns")

    initial_prices = prices.iloc[-1].rename_axis("asset")
    synthetic_paths = generate_joint_block_bootstrap_paths(
        daily_returns=daily_returns,
        initial_prices=initial_prices,
        horizon=horizon,
        n_paths=n_paths,
        block_size=block_size,
        seed=seed,
    )
    calibration_spread = (
        prices[tickers[0]]
        - (report.alpha + report.beta * prices[tickers[1]])
    ).rename("spread")
    path_results: list[float] = []
    for path_number in range(n_paths):
        path_prices = synthetic_paths.xs(path_number, level="path")
        path_results.append(
            _evaluate_path_pnl(
                path_prices,
                calibration_spread,
                report,
            )
        )

    path_pnl = pd.Series(path_results, index=pd.RangeIndex(n_paths, name="path"), name="pnl")
    if not np.isfinite(path_pnl.to_numpy()).all():
        raise SimulationError("simulated PnL distribution contains non-finite values")
    pnl_values = path_pnl.to_numpy()
    standard_deviation = float(np.std(pnl_values, ddof=1)) if n_paths > 1 else 0.0
    simulation_report = SimulationReport(
        path_pnl=path_pnl,
        mean_pnl=float(np.mean(pnl_values)),
        median_pnl=float(np.median(pnl_values)),
        standard_deviation=standard_deviation,
        minimum_pnl=float(np.min(pnl_values)),
        maximum_pnl=float(np.max(pnl_values)),
        percentile_05=float(np.quantile(pnl_values, 0.05)),
        percentile_95=float(np.quantile(pnl_values, 0.95)),
        profitable_fraction=float(np.mean(pnl_values > 0.0)),
        n_paths=n_paths,
        horizon=horizon,
        block_size=block_size,
        seed=seed,
    )
    assert len(simulation_report.path_pnl) == n_paths
    assert 0.0 <= simulation_report.profitable_fraction <= 1.0
    logger.info(
        "Joint block-bootstrap summary: paths=%d horizon=%d block=%d seed=%d "
        "PnL[mean=%.4f median=%.4f std=%.4f p05=%.4f p95=%.4f "
        "min=%.4f max=%.4f profitable=%.2f%%]",
        n_paths,
        horizon,
        block_size,
        seed,
        simulation_report.mean_pnl,
        simulation_report.median_pnl,
        simulation_report.standard_deviation,
        simulation_report.percentile_05,
        simulation_report.percentile_95,
        simulation_report.minimum_pnl,
        simulation_report.maximum_pnl,
        simulation_report.profitable_fraction * 100.0,
    )
    return simulation_report
