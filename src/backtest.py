from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from itertools import product

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.features import FeatureCalculationError, compute_hedge_ratio, compute_z_score

logger = logging.getLogger(__name__)


class BacktestError(ValueError):
    """Raised when a disciplined in-sample/out-of-sample test cannot run.

    Parameters
    ----------
    message : str
        Description of the backtest input or execution failure.

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
    Invalid protocol inputs are separated from feature-calculation failures so
    the caller can identify whether the backtest itself or its features failed.
    """


@dataclass(frozen=True)
class StrategyParameters:
    """Immutable strategy parameters for one recorded grid-search attempt.

    Parameters
    ----------
    entry_z : float
        Absolute z-score threshold that opens a position.
    rolling_window : int
        Number of trailing spread observations used by the z-score.
    exit_z : float
        Absolute z-score threshold at or below which a position is closed.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Validation is performed by :func:`run_backtest`.

    Notes
    -----
    Each instance represents one explicit attempted configuration. The grid
    ledger stores every instance rather than reporting only the selected one.
    """

    entry_z: float
    rolling_window: int
    exit_z: float


@dataclass(frozen=True)
class PerformanceMetrics:
    """Period-specific, unlevered spread backtest summary.

    Parameters
    ----------
    pnl : float
        Cumulative mark-to-market spread PnL in price units.
    trade_count : int
        Number of completed or still-open trades entered during the period.
    win_rate : float
        Fraction of closed trades with strictly positive cumulative PnL;
        zero when there are no closed trades.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Values are created internally by the backtest.

    Notes
    -----
    PnL is position times the change in the fixed-coefficient residual spread.
    It is not a capital-normalized return and excludes commissions, borrow,
    slippage, financing, and market impact. Win rate covers closed trades only;
    any open trade remains marked-to-market in total PnL but is excluded from
    win-rate calculation.
    """

    pnl: float
    trade_count: int
    win_rate: float


@dataclass(frozen=True)
class ParameterTrial:
    """Audit record for one in-sample parameter configuration.

    Parameters
    ----------
    parameters : StrategyParameters
        Exact strategy parameters evaluated.
    in_sample : PerformanceMetrics
        Result used for selection; contains no out-of-sample information.
    selected : bool
        Whether this attempt was chosen for the final blind evaluation.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Values are created internally by the backtest.

    Notes
    -----
    The trial tuple in :class:`BacktestReport` is the formal data-snooping
    ledger: its length records how many configurations were actually tried.
    """

    parameters: StrategyParameters
    in_sample: PerformanceMetrics
    selected: bool


@dataclass(frozen=True)
class BacktestReport:
    """Complete audit and separate performance report for one pair backtest.

    Parameters
    ----------
    alpha : float
        Intercept fitted only on in-sample prices.
    beta : float
        Hedge ratio fitted only on in-sample prices.
    split_date : datetime.date
        First date assigned to the out-of-sample period.
    as_of : datetime.date
        Latest permitted input observation.
    selected_parameters : StrategyParameters
        Configuration selected using in-sample PnL only.
    in_sample : PerformanceMetrics
        Performance of the selected configuration before the split.
    out_of_sample : PerformanceMetrics
        Blind performance at and after the split.
    out_of_sample_spread : pandas.Series
        Fixed in-sample hedge residuals during the OOS period.
    out_of_sample_daily_pnl : pandas.Series
        Daily OOS spread PnL aligned to the OOS date index.
    trials : tuple of ParameterTrial
        Full parameter-search ledger.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Values are created internally by the backtest.

    Notes
    -----
    The hedge coefficients are estimated once from in-sample observations.
    Out-of-sample observations may update the causal rolling z-score and
    produce signals, but they never select or alter a parameter.
    The aligned OOS spread and PnL series support retrospective stability
    diagnostics without reconstructing trades from aggregate metrics.
    """

    alpha: float
    beta: float
    split_date: date
    as_of: date
    selected_parameters: StrategyParameters
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics
    out_of_sample_spread: pd.Series
    out_of_sample_daily_pnl: pd.Series
    trials: tuple[ParameterTrial, ...]

    @property
    def attempt_count(self) -> int:
        """Return the number of parameter configurations evaluated.

        Parameters
        ----------
        None
            The count is derived from the audit ledger.

        Returns
        -------
        int
            Number of recorded parameter trials.

        Raises
        ------
        None
            The property reads an immutable tuple.

        Notes
        -----
        Deriving the count from the ledger prevents a separately maintained
        counter from drifting away from the recorded attempts.
        """
        return len(self.trials)


@dataclass(frozen=True)
class SensitivityNeighbor:
    """In-sample outcome for a one-step neighboring parameter configuration.

    Parameters
    ----------
    parameters : StrategyParameters
        Neighboring configuration evaluated in the original ledger.
    changed_parameter : str
        Parameter axis varied from the selected configuration.
    pnl : float
        Neighbor's in-sample PnL in spread-price units.
    pnl_change : float
        Difference from selected configuration's in-sample PnL.
    relative_change : float or None
        Fractional PnL change relative to baseline, or ``None`` if baseline
        PnL is zero.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are created by :func:`analyze_parameter_sensitivity`.

    Notes
    -----
    Neighbors differ on exactly one parameter and use the immediately adjacent
    value present in the evaluated grid. Outcomes are read from the existing
    attempt ledger; no additional strategy trials are performed.
    """

    parameters: StrategyParameters
    changed_parameter: str
    pnl: float
    pnl_change: float
    relative_change: float | None


@dataclass(frozen=True)
class ParameterSensitivityReport:
    """Sensitivity summary around the in-sample selected configuration.

    Parameters
    ----------
    baseline : ParameterTrial
        Selected ledger entry used as the sensitivity reference.
    neighbors : tuple of SensitivityNeighbor
        Evaluated immediate one-axis neighboring configurations.
    minimum_neighbor_pnl : float or None
        Lowest neighboring in-sample PnL, or ``None`` if there are no neighbors.
    maximum_neighbor_pnl : float or None
        Highest neighboring in-sample PnL, or ``None`` if there are no neighbors.
    is_fragile : bool
        Whether a neighbor reverses positive PnL or reduces a positive baseline
        by at least 50 percent.
    fragility_reasons : tuple of str
        Explicit reasons for the fragile classification.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are created by :func:`analyze_parameter_sensitivity`.

    Notes
    -----
    Fragility is a transparent diagnostic rule, not a statistical test:
    immediate grid-neighbor results are fragile if any neighbor changes a
    positive baseline to zero/negative or reduces it by at least half.
    """

    baseline: ParameterTrial
    neighbors: tuple[SensitivityNeighbor, ...]
    minimum_neighbor_pnl: float | None
    maximum_neighbor_pnl: float | None
    is_fragile: bool
    fragility_reasons: tuple[str, ...]


def analyze_parameter_sensitivity(report: BacktestReport) -> ParameterSensitivityReport:
    """Summarize ledger outcomes at immediate neighbors of the selected parameters.

    Parameters
    ----------
    report : BacktestReport
        Completed backtest report containing the complete trial ledger.

    Returns
    -------
    ParameterSensitivityReport
        Neighbor PnLs, changes from the selected IS result, and an explicit
        robustness/fragility classification.

    Raises
    ------
    BacktestError
        If the trial ledger is empty, has multiple selected entries, or does
        not include the selected attempt.

    Notes
    -----
    For each parameter axis, the analysis includes only the closest lower and
    higher values actually present in the ledger, holding the other parameter
    fixed. It marks a positive baseline fragile when an immediate neighbor
    flips the PnL sign or reduces PnL by at least 50 percent. A non-positive
    baseline is labeled inconclusive rather than robust. This diagnostic uses
    existing trials only and never examines OOS PnL to select or tune a model.
    """
    if not report.trials:
        raise BacktestError("parameter sensitivity requires a non-empty trial ledger")
    selected_trials = tuple(trial for trial in report.trials if trial.selected)
    if len(selected_trials) != 1:
        raise BacktestError("parameter sensitivity requires exactly one selected trial")

    baseline = selected_trials[0]
    baseline_parameters = baseline.parameters
    entry_values = sorted({trial.parameters.entry_z for trial in report.trials})
    window_values = sorted({trial.parameters.rolling_window for trial in report.trials})
    neighboring_entries = {
        value
        for value in (
            max((item for item in entry_values if item < baseline_parameters.entry_z), default=None),
            min((item for item in entry_values if item > baseline_parameters.entry_z), default=None),
        )
        if value is not None
    }
    neighboring_windows = {
        value
        for value in (
            max((item for item in window_values if item < baseline_parameters.rolling_window), default=None),
            min((item for item in window_values if item > baseline_parameters.rolling_window), default=None),
        )
        if value is not None
    }

    neighbor_specs: list[tuple[ParameterTrial, str]] = []
    for trial in report.trials:
        parameters = trial.parameters
        if (
            parameters.rolling_window == baseline_parameters.rolling_window
            and parameters.entry_z in neighboring_entries
        ):
            neighbor_specs.append((trial, "entry_z"))
        if (
            parameters.entry_z == baseline_parameters.entry_z
            and parameters.rolling_window in neighboring_windows
        ):
            neighbor_specs.append((trial, "rolling_window"))

    unique_neighbor_specs = {
        (trial.parameters, changed_parameter): (trial, changed_parameter)
        for trial, changed_parameter in neighbor_specs
    }
    neighbors = tuple(
        SensitivityNeighbor(
            parameters=trial.parameters,
            changed_parameter=changed_parameter,
            pnl=trial.in_sample.pnl,
            pnl_change=trial.in_sample.pnl - baseline.in_sample.pnl,
            relative_change=(
                (trial.in_sample.pnl - baseline.in_sample.pnl) / abs(baseline.in_sample.pnl)
                if baseline.in_sample.pnl != 0.0
                else None
            ),
        )
        for trial, changed_parameter in unique_neighbor_specs.values()
    )

    reasons: list[str] = []
    baseline_pnl = baseline.in_sample.pnl
    if baseline_pnl <= 0:
        reasons.append("selected in-sample baseline is not profitable; robustness is inconclusive")
    else:
        for neighbor in neighbors:
            if neighbor.pnl <= 0:
                reasons.append(
                    f"{neighbor.changed_parameter} neighbor "
                    f"{neighbor.parameters} reverses positive PnL"
                )
            elif neighbor.pnl <= baseline_pnl * 0.5:
                reasons.append(
                    f"{neighbor.changed_parameter} neighbor "
                    f"{neighbor.parameters} reduces baseline PnL by at least 50%"
                )

    neighbor_pnls = [neighbor.pnl for neighbor in neighbors]
    is_fragile = baseline_pnl > 0.0 and any(
        neighbor.pnl <= 0.0 or neighbor.pnl <= baseline_pnl * 0.5
        for neighbor in neighbors
    )
    sensitivity = ParameterSensitivityReport(
        baseline=baseline,
        neighbors=neighbors,
        minimum_neighbor_pnl=min(neighbor_pnls) if neighbor_pnls else None,
        maximum_neighbor_pnl=max(neighbor_pnls) if neighbor_pnls else None,
        is_fragile=is_fragile,
        fragility_reasons=tuple(reasons),
    )
    if baseline_pnl <= 0.0:
        classification = "inconclusive"
    elif sensitivity.is_fragile:
        classification = "FRAGILE"
    else:
        classification = "no fragility trigger"
    logger.info(
        "Parameter sensitivity around entry_z=%.2f window=%d: baseline IS PnL=%.4f, "
        "%d immediate neighbors, neighbor PnL range=%s..%s, classification=%s",
        baseline_parameters.entry_z,
        baseline_parameters.rolling_window,
        baseline_pnl,
        len(neighbors),
        sensitivity.minimum_neighbor_pnl,
        sensitivity.maximum_neighbor_pnl,
        classification,
    )
    for neighbor in sensitivity.neighbors:
        logger.info(
            "Sensitivity neighbor axis=%s entry_z=%.2f window=%d IS PnL=%.4f "
            "change=%+.4f relative=%s",
            neighbor.changed_parameter,
            neighbor.parameters.entry_z,
            neighbor.parameters.rolling_window,
            neighbor.pnl,
            neighbor.pnl_change,
            neighbor.relative_change,
        )
    for reason in sensitivity.fragility_reasons:
        logger.warning("Parameter sensitivity: %s", reason)
    return sensitivity


def _normalize_prices(prices: pd.DataFrame, as_of: date, config: ProjectConfig) -> pd.DataFrame:
    """Validate and normalize paired daily prices before the split.

    Parameters
    ----------
    prices : pandas.DataFrame
        Pair price history with one column for each configured ticker.
    as_of : datetime.date
        Inclusive latest allowed price date.
    config : ProjectConfig
        Configuration naming the two ticker columns.

    Returns
    -------
    pandas.DataFrame
        Sorted float prices indexed by Python calendar dates.

    Raises
    ------
    BacktestError
        If required columns are absent, prices are empty/invalid, or dates
        cannot be ordered unambiguously.

    Notes
    -----
    Rows are not silently dropped or reindexed: any missing or non-finite pair
    observation is rejected to preserve transparent alignment. The explicit
    cutoff is asserted after normalization to prevent look-ahead leakage.
    """
    required_columns = (config.ticker_a, config.ticker_b)
    if prices.empty:
        raise BacktestError("prices must not be empty")
    if any(ticker not in prices.columns for ticker in required_columns):
        raise BacktestError(f"prices must contain columns {required_columns}")

    normalized = prices.loc[:, list(required_columns)].copy()
    try:
        normalized.index = pd.Index(
            [pd.Timestamp(index_value).date() for index_value in normalized.index],
            name="Date",
        )
        normalized = normalized.astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise BacktestError("prices must have valid dates and numeric values") from error

    if normalized.index.has_duplicates:
        raise BacktestError("prices must not contain duplicate dates")
    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()
    if not np.isfinite(normalized.to_numpy()).all():
        raise BacktestError("prices must not contain missing or non-finite values")
    if (normalized <= 0).any().any():
        raise BacktestError("prices must be strictly positive")

    assert len(normalized) == len(prices)
    assert all(observation_date <= as_of for observation_date in normalized.index)
    return normalized


def _simulate_period(
    spread: pd.Series,
    z_score: pd.Series,
    start_position: int,
    end_position: int,
    parameters: StrategyParameters,
) -> tuple[PerformanceMetrics, pd.Series]:
    """Simulate a flat-start period using lagged z-score signals.

    Parameters
    ----------
    spread : pandas.Series
        Full fixed-hedge-ratio residual history.
    z_score : pandas.Series
        Causal rolling standardized spread aligned to ``spread``.
    start_position : int
        Inclusive row position at which period performance begins.
    end_position : int
        Exclusive row position ending the period.
    parameters : StrategyParameters
        Entry, exit, and rolling-window configuration.

    Returns
    -------
    tuple of PerformanceMetrics and pandas.Series
        Cumulative spread PnL, entries, closed-trade win rate, and daily
        mark-to-market PnL aligned to the simulated period dates.

    Raises
    ------
    BacktestError
        If series are misaligned or period boundaries are invalid.

    Notes
    -----
    At observation t, the position is updated from z-score t-1, then earns
    that position times the residual change from t-1 to t. This one-observation
    lag prevents trading on a signal that uses the same close whose PnL is
    being measured. The period begins flat; a carried trade at a reporting
    boundary is not credited as an entry in the next period. The initial
    history session has zero PnL because no prior close is available.
    """
    if not spread.index.equals(z_score.index):
        raise BacktestError("spread and z_score must have identical indexes")
    if start_position < 0 or end_position > len(spread) or start_position >= end_position:
        raise BacktestError("period boundaries must describe a non-empty valid range")

    assert len(spread) == len(z_score)
    position = 0
    trade_count = 0
    cumulative_trade_pnl = 0.0
    closed_trade_pnls: list[float] = []
    total_pnl = 0.0
    daily_pnls: list[float] = []

    for current_position in range(start_position, end_position):
        if current_position == 0:
            daily_pnls.append(0.0)
            continue

        previous_z = z_score.iloc[current_position - 1]
        previous_spread = float(spread.iloc[current_position - 1])
        current_spread = float(spread.iloc[current_position])
        next_position = position

        if np.isfinite(previous_z):
            if position == 0:
                if previous_z >= parameters.entry_z:
                    next_position = -1
                elif previous_z <= -parameters.entry_z:
                    next_position = 1
            elif abs(previous_z) <= parameters.exit_z:
                closed_trade_pnls.append(cumulative_trade_pnl)
                cumulative_trade_pnl = 0.0
                next_position = 0

        if position == 0 and next_position != 0:
            trade_count += 1
        position = next_position
        daily_pnl = position * (current_spread - previous_spread)
        daily_pnls.append(float(daily_pnl))
        total_pnl += daily_pnl
        if position != 0:
            cumulative_trade_pnl += daily_pnl

    win_rate = (
        sum(trade_pnl > 0 for trade_pnl in closed_trade_pnls) / len(closed_trade_pnls)
        if closed_trade_pnls
        else 0.0
    )
    assert 0.0 <= win_rate <= 1.0
    metrics = PerformanceMetrics(
        pnl=float(total_pnl),
        trade_count=trade_count,
        win_rate=float(win_rate),
    )
    daily_pnl_series = pd.Series(
        daily_pnls,
        index=spread.index[start_position:end_position],
        name="daily_pnl",
        dtype=float,
    )
    assert len(daily_pnl_series) == end_position - start_position
    assert daily_pnl_series.index.equals(spread.index[start_position:end_position])
    return metrics, daily_pnl_series


def run_backtest(
    prices: pd.DataFrame,
    config: ProjectConfig,
    as_of: date,
    entry_thresholds: tuple[float, ...] = (1.5, 2.0, 2.5),
    rolling_windows: tuple[int, ...] = (40, 60, 90),
) -> BacktestReport:
    """Select strategy parameters in-sample and evaluate them blindly out-of-sample.

    Parameters
    ----------
    prices : pandas.DataFrame
        Aligned daily prices with columns named by ``config.ticker_a`` and
        ``config.ticker_b``.
    config : ProjectConfig
        Pair and ISO ``YYYY-MM-DD`` in-sample/out-of-sample split configuration.
    as_of : datetime.date
        Inclusive latest allowed observation; observations after this date are
        rejected.
    entry_thresholds : tuple of float, default=(1.5, 2.0, 2.5)
        Explicit absolute z-score entry thresholds included in the trial grid.
    rolling_windows : tuple of int, default=(40, 60, 90)
        Explicit trailing window sizes included in the trial grid.

    Returns
    -------
    BacktestReport
        Fitted in-sample hedge coefficients, selected parameters, separate
        period performance metrics, and the full attempted-parameter ledger.

    Raises
    ------
    BacktestError
        If the input dates, split, grid, or period sizes are invalid.
    FeatureCalculationError
        If in-sample hedge estimation or z-score features cannot be calculated.

    Notes
    -----
    The split date is the first out-of-sample date: training observations are
    strictly earlier, and test observations are on/after it. OLS is fitted only
    on training prices. Each explicit threshold/window combination is scored
    only by training PnL, ties preserve grid order, and OOS is evaluated once
    with the selected frozen coefficients and thresholds. Rolling z-scores use
    only trailing observations; the daily position uses the prior session's
    z-score. Reported PnL is spread-price PnL without transaction costs.
    """
    if not entry_thresholds or not rolling_windows:
        raise BacktestError("parameter grids must not be empty")
    if any(not np.isfinite(threshold) or threshold <= 0 for threshold in entry_thresholds):
        raise BacktestError("entry thresholds must be finite and positive")
    if any(isinstance(window, bool) or not isinstance(window, int) or window < 2 for window in rolling_windows):
        raise BacktestError("rolling windows must be integers greater than or equal to 2")
    if len(set(entry_thresholds)) != len(entry_thresholds):
        raise BacktestError("entry thresholds must not contain duplicates")
    if len(set(rolling_windows)) != len(rolling_windows):
        raise BacktestError("rolling windows must not contain duplicates")
    if not np.isfinite(config.z_score_exit) or config.z_score_exit < 0:
        raise BacktestError("config.z_score_exit must be finite and non-negative")
    if any(threshold <= config.z_score_exit for threshold in entry_thresholds):
        raise BacktestError("each entry threshold must exceed the configured exit threshold")

    try:
        split_date = date.fromisoformat(config.train_test_split_date)
    except ValueError as error:
        raise BacktestError("train_test_split_date must use YYYY-MM-DD format") from error

    normalized = _normalize_prices(prices, as_of, config)
    in_sample_mask = normalized.index < split_date
    out_of_sample_mask = normalized.index >= split_date
    in_sample_prices = normalized.loc[in_sample_mask]
    out_of_sample_prices = normalized.loc[out_of_sample_mask]
    if in_sample_prices.empty or out_of_sample_prices.empty:
        raise BacktestError("prices must include observations before and on/after the split date")
    if len(in_sample_prices) < max(rolling_windows) + 2:
        raise BacktestError("in-sample period is too short for the largest rolling window")
    if len(normalized) < 2:
        raise BacktestError("at least two observations are required")

    assert all(observation_date < split_date for observation_date in in_sample_prices.index)
    assert all(observation_date >= split_date for observation_date in out_of_sample_prices.index)
    assert all(observation_date <= as_of for observation_date in normalized.index)

    alpha, beta, _ = compute_hedge_ratio(
        in_sample_prices[config.ticker_a],
        in_sample_prices[config.ticker_b],
    )
    full_spread = (
        normalized[config.ticker_a]
        - (alpha + beta * normalized[config.ticker_b])
    ).rename("spread")

    candidates = tuple(
        StrategyParameters(
            entry_z=float(entry_threshold),
            rolling_window=window,
            exit_z=float(config.z_score_exit),
        )
        for entry_threshold, window in product(entry_thresholds, rolling_windows)
    )
    if not candidates:
        raise BacktestError("parameter grid did not produce any attempts")

    trial_results: list[ParameterTrial] = []
    for parameters in candidates:
        z_score = compute_z_score(full_spread, parameters.rolling_window)
        in_sample_end = len(in_sample_prices)
        in_sample_metrics, _ = _simulate_period(
            full_spread,
            z_score,
            start_position=0,
            end_position=in_sample_end,
            parameters=parameters,
        )
        trial_results.append(
            ParameterTrial(
                parameters=parameters,
                in_sample=in_sample_metrics,
                selected=False,
            )
        )

    selected_index = max(
        range(len(trial_results)),
        key=lambda index: trial_results[index].in_sample.pnl,
    )
    selected_parameters = trial_results[selected_index].parameters
    selected_trial = trial_results[selected_index]
    trial_results[selected_index] = ParameterTrial(
        parameters=selected_trial.parameters,
        in_sample=selected_trial.in_sample,
        selected=True,
    )
    for trial in trial_results:
        logger.info(
            "Parameter attempt%s entry_z=%.2f window=%d IS[pnl=%.4f trades=%d win_rate=%.2f%%]",
            " (selected)" if trial.selected else "",
            trial.parameters.entry_z,
            trial.parameters.rolling_window,
            trial.in_sample.pnl,
            trial.in_sample.trade_count,
            trial.in_sample.win_rate * 100,
        )

    selected_z_score = compute_z_score(full_spread, selected_parameters.rolling_window)
    split_position = len(in_sample_prices)
    in_sample_metrics, _ = _simulate_period(
        full_spread,
        selected_z_score,
        start_position=0,
        end_position=split_position,
        parameters=selected_parameters,
    )
    out_of_sample_metrics, out_of_sample_daily_pnl = _simulate_period(
        full_spread,
        selected_z_score,
        start_position=split_position,
        end_position=len(normalized),
        parameters=selected_parameters,
    )

    assert split_position > 0
    assert in_sample_metrics.pnl == selected_trial.in_sample.pnl
    out_of_sample_spread = full_spread.iloc[split_position:].copy()
    assert out_of_sample_spread.index.equals(out_of_sample_daily_pnl.index)
    assert all(observation_date >= split_date for observation_date in out_of_sample_daily_pnl.index)
    assert all(observation_date <= as_of for observation_date in out_of_sample_daily_pnl.index)
    logger.info(
        "Backtest split=%s as_of=%s attempts=%d selected(entry_z=%.2f, window=%d) "
        "IS[pnl=%.4f trades=%d win_rate=%.2f%%] OOS[pnl=%.4f trades=%d win_rate=%.2f%%]",
        split_date,
        as_of,
        len(trial_results),
        selected_parameters.entry_z,
        selected_parameters.rolling_window,
        in_sample_metrics.pnl,
        in_sample_metrics.trade_count,
        in_sample_metrics.win_rate * 100,
        out_of_sample_metrics.pnl,
        out_of_sample_metrics.trade_count,
        out_of_sample_metrics.win_rate * 100,
    )

    return BacktestReport(
        alpha=alpha,
        beta=beta,
        split_date=split_date,
        as_of=as_of,
        selected_parameters=selected_parameters,
        in_sample=in_sample_metrics,
        out_of_sample=out_of_sample_metrics,
        out_of_sample_spread=out_of_sample_spread,
        out_of_sample_daily_pnl=out_of_sample_daily_pnl,
        trials=tuple(trial_results),
    )
