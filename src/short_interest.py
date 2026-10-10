from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Mapping

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

_NASDAQ_URL = "https://api.nasdaq.com/api/quote/{ticker}/short-interest"
_NASDAQ_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
}


class ShortInterestError(RuntimeError):
    """Raised when Nasdaq short-interest data cannot be fetched or parsed.

    Parameters
    ----------
    message : str
        Description of the request or response failure.

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
    Explicit errors let the pipeline distinguish missing vendor coverage from
    a valid analysis that found no relationship.
    """


@dataclass(frozen=True)
class ShortInterestImpact:
    """Descriptive comparison of short-interest changes and pair outcomes.

    Parameters
    ----------
    observations : int
        Daily price/PnL observations with at least one published short-interest
        report available.
    report_updates : int
        Dates when a newly published report became available for either ticker.
    intc_updates : int
        Number of available INTC short-interest report changes.
    amd_updates : int
        Number of available AMD short-interest report changes.
    intc_pct_change_spread_change_correlation : float or None
        Correlation of INTC short-interest percentage changes with absolute
        daily spread changes, measured only on report update dates.
    amd_pct_change_spread_change_correlation : float or None
        Corresponding correlation for AMD.
    unstable_observations : int or None
        Matched dates labeled unstable by the provided rolling ADF p-values.
    unstable_mean_daily_pnl : float or None
        Mean daily PnL on matched unstable dates.
    stable_mean_daily_pnl : float or None
        Mean daily PnL on matched stable dates.
    aligned_data : pandas.DataFrame
        As-of-aligned daily prices/outcomes and last-published short-interest
        values, with publication availability enforced.

    Returns
    -------
    None
        Dataclass construction does not return a value.

    Raises
    ------
    None
        Instances are created by :func:`analyze_short_interest_impact`.

    Notes
    -----
    Correlations are descriptive and based only on report update dates; they
    do not imply causality. Short-interest share counts are not directly
    comparable between firms with different share counts or floats.
    """

    observations: int
    report_updates: int
    intc_updates: int
    amd_updates: int
    intc_pct_change_spread_change_correlation: float | None
    amd_pct_change_spread_change_correlation: float | None
    unstable_observations: int | None
    unstable_mean_daily_pnl: float | None
    stable_mean_daily_pnl: float | None
    aligned_data: pd.DataFrame


def _normalize_key(key: object) -> str:
    """Normalize a Nasdaq field label for tolerant schema matching.

    Parameters
    ----------
    key : object
        Candidate key from the JSON payload.

    Returns
    -------
    str
        Lowercase alphanumeric key.

    Raises
    ------
    None
        Every key is converted through its string representation.

    Notes
    -----
    Nasdaq has returned both camelCase field identifiers and display labels;
    removing separators lets the parser recognize both forms.
    """
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _find_short_interest_rows(payload: object) -> list[Mapping[object, object]]:
    """Recursively locate a Nasdaq table containing settlement and SI fields.

    Parameters
    ----------
    payload : object
        Decoded Nasdaq JSON payload.

    Returns
    -------
    list of Mapping
        Rows from the first matching historical table, or an empty list.

    Raises
    ------
    None
        Unsupported payload shapes are treated as no matching table.

    Notes
    -----
    The endpoint has nested through response.data tables, so discovery is
    recursive instead of depending on a single brittle JSON path.
    """
    date_keys = {"settlementdate", "settlement", "settledate"}
    interest_keys = {
        "interest",
        "shortinterest",
        "shortinterestshares",
        "shortinterestvolume",
        "shortshares",
    }
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list) and rows:
            first_row = rows[0]
            if isinstance(first_row, dict):
                keys = {_normalize_key(key) for key in first_row}
                if keys.intersection(date_keys) and keys.intersection(interest_keys):
                    return [
                        row for row in rows if isinstance(row, dict)
                    ]
        for value in payload.values():
            found_rows = _find_short_interest_rows(value)
            if found_rows:
                return found_rows
    elif isinstance(payload, list):
        for value in payload:
            found_rows = _find_short_interest_rows(value)
            if found_rows:
                return found_rows
    return []


def _row_value(row: Mapping[object, object], aliases: set[str]) -> object | None:
    """Return the value associated with the first matching normalized key.

    Parameters
    ----------
    row : Mapping
        Nasdaq historical data row.
    aliases : set of str
        Normalized candidate keys.

    Returns
    -------
    object or None
        Matched value, or ``None`` when no alias is present.

    Raises
    ------
    None
        Missing keys return ``None``.

    Notes
    -----
    Matching is case- and punctuation-insensitive to tolerate display names
    such as ``"Short Interest"`` and machine keys such as ``shortInterest``.
    """
    for key, value in row.items():
        if _normalize_key(key) in aliases:
            return value
    return None


def _parse_number(value: object, field_name: str, ticker: str) -> float | None:
    """Parse a formatted Nasdaq numeric cell.

    Parameters
    ----------
    value : object
        API value that may contain commas, units, or a not-available marker.
    field_name : str
        Field name used in error descriptions.
    ticker : str
        Ticker symbol associated with the value.

    Returns
    -------
    float or None
        Parsed finite number, or ``None`` for a documented missing marker.

    Raises
    ------
    ShortInterestError
        If a non-empty value cannot be parsed as a finite number.

    Notes
    -----
    Commas and common share-count suffixes are removed; placeholders such as
    ``"--"`` and ``"N/A"`` remain missing rather than being converted to zero.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"--", "n/a", "na", "null", "none"}:
        return None
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    cleaned = re.sub(r"\s*(shares?|days?)\s*$", "", cleaned, flags=re.IGNORECASE)
    try:
        result = float(cleaned)
    except ValueError as error:
        raise ShortInterestError(
            f"Could not parse {field_name} value {value!r} for {ticker}"
        ) from error
    if not np.isfinite(result):
        raise ShortInterestError(f"Non-finite {field_name} value for {ticker}")
    return result


def fetch_nasdaq_short_interest(
    ticker: str,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch historical Nasdaq short interest and parse its dated observations.

    Parameters
    ----------
    ticker : str
        Nasdaq ticker symbol, for example ``"INTC"``.
    timeout : float, default=20.0
        HTTP request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Historical short interest indexed by settlement date, with numeric
        ``short_interest``, ``average_daily_volume``, and ``days_to_cover``
        columns where provided by Nasdaq.

    Raises
    ------
    ShortInterestError
        If the ticker/timeout is invalid, the request fails, JSON is invalid,
        or the response has no parseable short-interest rows.

    Notes
    -----
    A browser-like User-Agent plus Nasdaq's expected origin/referer headers are
    sent to avoid routine 403 responses. Returned settlement dates describe
    measurement dates, not publication dates; callers must use a conservative
    availability lag before joining the values to daily historical outcomes.
    """
    normalized_ticker = ticker.strip().upper()
    if not normalized_ticker or not re.fullmatch(r"[A-Z0-9.-]+", normalized_ticker):
        raise ShortInterestError("ticker must contain only letters, digits, dots, or hyphens")
    if timeout <= 0 or not np.isfinite(timeout):
        raise ShortInterestError("timeout must be finite and greater than zero")

    url = _NASDAQ_URL.format(ticker=normalized_ticker)
    try:
        response = requests.get(url, headers=_NASDAQ_HEADERS, timeout=timeout)
        response.raise_for_status()
        payload: object = response.json()
    except requests.RequestException as error:
        raise ShortInterestError(
            f"Nasdaq short-interest request failed for {normalized_ticker}"
        ) from error
    except ValueError as error:
        raise ShortInterestError(
            f"Nasdaq returned invalid JSON for {normalized_ticker}"
        ) from error

    rows = _find_short_interest_rows(payload)
    if not rows:
        raise ShortInterestError(
            f"Nasdaq response contains no historical short-interest table for {normalized_ticker}"
        )

    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        settlement_value = _row_value(
            row,
            {"settlementdate", "settlement", "settledate"},
        )
        interest_value = _row_value(
            row,
            {
                "interest",
                "shortinterest",
                "shortinterestshares",
                "shortinterestvolume",
                "shortshares",
            },
        )
        if settlement_value is None or interest_value is None:
            continue
        try:
            settlement_text = str(settlement_value).strip()
            try:
                settlement_date = datetime.strptime(settlement_text, "%m/%d/%Y").date()
            except ValueError:
                settlement_date = date.fromisoformat(settlement_text)
        except (TypeError, ValueError) as error:
            raise ShortInterestError(
                f"Invalid settlement date {settlement_value!r} for {normalized_ticker}"
            ) from error

        short_interest = _parse_number(interest_value, "short_interest", normalized_ticker)
        if short_interest is None:
            continue
        average_volume = _parse_number(
            _row_value(
                row,
                {
                    "averagedailyvolume",
                    "averagedailysharevolume",
                    "avgdailyvolume",
                    "avgdailysharevolume",
                },
            ),
            "average_daily_volume",
            normalized_ticker,
        )
        days_to_cover = _parse_number(
            _row_value(row, {"daystocover", "daystocoverratio"}),
            "days_to_cover",
            normalized_ticker,
        )
        parsed_rows.append(
            {
                "settlement_date": settlement_date,
                "short_interest": short_interest,
                "average_daily_volume": average_volume,
                "days_to_cover": days_to_cover,
            }
        )

    if not parsed_rows:
        raise ShortInterestError(
            f"Nasdaq returned no parseable short-interest records for {normalized_ticker}"
        )

    result = pd.DataFrame(parsed_rows).set_index("settlement_date").sort_index()
    result.index.name = "settlement_date"
    if result.index.has_duplicates:
        result = result.groupby(level=0).last()
    assert result.index.is_monotonic_increasing
    assert result["short_interest"].notna().all()
    logger.info(
        "Fetched %d Nasdaq short-interest records for %s (%s through %s)",
        len(result),
        normalized_ticker,
        result.index[0],
        result.index[-1],
    )
    return result


def _availability_date(settlement_date: date) -> date:
    """Estimate the conservative public availability date for a report.

    Parameters
    ----------
    settlement_date : datetime.date
        Nasdaq/FINRA short-interest measurement date.

    Returns
    -------
    datetime.date
        Fourteenth calendar day after settlement, used as the earliest
        eligible daily price/PnL observation in this conservative model.

    Raises
    ------
    None
        Date arithmetic is deterministic.

    Notes
    -----
    The endpoint provides settlement dates but does not expose a historical
    publication timestamp. A fixed 14-calendar-day lag conservatively covers
    the commonly stated dissemination window without assuming exact historical
    market holidays. It is an operational proxy, not a vendor-verified release
    timestamp.
    """
    return settlement_date + timedelta(days=14)


def _prepare_short_interest(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate and add publication-availability dates to a Nasdaq history.

    Parameters
    ----------
    frame : pandas.DataFrame
        Historical Nasdaq data with a settlement-date index.
    ticker : str
        Ticker used to name standardized output columns.

    Returns
    -------
    pandas.DataFrame
        Sorted records indexed by estimated publication availability date.

    Raises
    ------
    ShortInterestError
        If required data is missing, malformed, non-finite, or duplicated in a
        way that cannot be resolved.

    Notes
    -----
    Settlement observations become available only after the conservative
    14-calendar-day reporting lag, preventing a historical backtest from
    seeing short interest before the market could reasonably have received
    the report.
    """
    if frame.empty or "short_interest" not in frame.columns:
        raise ShortInterestError(f"{ticker} history must include short_interest values")
    try:
        settlement_dates = pd.Index(
            [pd.Timestamp(item).date() for item in frame.index],
            name="settlement_date",
        )
        interest = pd.to_numeric(frame["short_interest"], errors="raise").astype(float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ShortInterestError(f"{ticker} short-interest history is invalid") from error
    if settlement_dates.has_duplicates:
        raise ShortInterestError(f"{ticker} history must have unique settlement dates")
    if not np.isfinite(interest.to_numpy()).all() or (interest < 0).any():
        raise ShortInterestError(f"{ticker} short interest must be finite and non-negative")

    prepared = pd.DataFrame(
        {
            f"{ticker}_short_interest": interest.to_numpy(),
            f"{ticker}_settlement_date": settlement_dates,
            f"{ticker}_availability_date": [_availability_date(item) for item in settlement_dates],
        },
        index=pd.Index(
            [_availability_date(item) for item in settlement_dates],
            name="date",
        ),
    ).sort_index()
    if prepared.index.has_duplicates:
        raise ShortInterestError(f"{ticker} reports have duplicate availability dates")
    assert len(prepared) == len(frame)
    assert all(
        availability_date >= settlement_date
        for availability_date, settlement_date in zip(
            prepared.index,
            prepared[f"{ticker}_settlement_date"],
        )
    )
    return prepared


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """Return a finite Pearson correlation when enough varying pairs exist.

    Parameters
    ----------
    left : pandas.Series
        First numeric vector.
    right : pandas.Series
        Second numeric vector with identical index.

    Returns
    -------
    float or None
        Correlation or ``None`` when fewer than two valid observations exist,
        either input is constant, or the statistic is undefined.

    Raises
    ------
    ShortInterestError
        If the two series do not have identical indexes.

    Notes
    -----
    Returning ``None`` for underdetermined correlations avoids presenting a
    meaningless numeric estimate for the short historical vendor series.
    """
    if not left.index.equals(right.index):
        raise ShortInterestError("correlation inputs must have identical indexes")
    valid = pd.concat([left, right], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None
    correlation = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
    return correlation if np.isfinite(correlation) else None


def analyze_short_interest_impact(
    intc: pd.DataFrame,
    amd: pd.DataFrame,
    spread: pd.Series,
    daily_pnl: pd.Series,
    adf_p_values: pd.Series | None = None,
    as_of: date | None = None,
    significance_level: float = 0.05,
) -> ShortInterestImpact:
    """Align Nasdaq short interest to daily outcomes and summarize associations.

    Parameters
    ----------
    intc : pandas.DataFrame
        INTC short-interest history indexed by settlement date.
    amd : pandas.DataFrame
        AMD short-interest history indexed by settlement date.
    spread : pandas.Series
        Daily residual spread indexed by date.
    daily_pnl : pandas.Series
        Daily strategy PnL indexed by its earning date.
    adf_p_values : pandas.Series or None, default=None
        Optional rolling ADF p-values indexed by test-window end date.
    as_of : datetime.date or None, default=None
        Inclusive information cutoff; future daily outcomes or settlements
        beyond the cutoff are rejected.
    significance_level : float, default=0.05
        ADF p-value threshold used to classify unstable dates.

    Returns
    -------
    ShortInterestImpact
        Date-aligned vendor values, report-update correlations with absolute
        spread changes, and optional stable/unstable daily PnL comparisons.

    Raises
    ------
    ShortInterestError
        If histories, dates, values, cutoff, or time-series alignment are
        invalid or no short-interest report was available during the analysis.

    Notes
    -----
    Short interest is carried forward only from its estimated publication date
    (settlement plus 14 calendar days), never from the settlement date itself.
    This deliberately conservative availability proxy is used because the API
    does not return historical publication timestamps. Pair share counts are
    not divided or compared as if they represented equivalent floats.
    Associations are descriptive, using newly published report changes rather
    than repeated forward-filled daily values.
    """
    if not np.isfinite(significance_level) or not 0.0 < significance_level < 1.0:
        raise ShortInterestError("significance_level must be strictly between 0 and 1")
    if not spread.index.equals(daily_pnl.index):
        raise ShortInterestError("spread and daily_pnl must have identical date indexes")
    if spread.empty or daily_pnl.empty:
        raise ShortInterestError("spread and daily_pnl must not be empty")

    try:
        dates = pd.Index([pd.Timestamp(item).date() for item in spread.index], name="date")
        normalized_spread = pd.Series(
            pd.to_numeric(spread, errors="raise").to_numpy(dtype=float),
            index=dates,
            name="spread",
        )
        normalized_pnl = pd.Series(
            pd.to_numeric(daily_pnl, errors="raise").to_numpy(dtype=float),
            index=dates,
            name="daily_pnl",
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ShortInterestError("spread and daily_pnl must contain valid dates and numbers") from error
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ShortInterestError("daily outcome dates must be unique and increasing")
    if not np.isfinite(normalized_spread.to_numpy()).all() or not np.isfinite(normalized_pnl.to_numpy()).all():
        raise ShortInterestError("daily outcomes must be finite")
    cutoff = as_of if as_of is not None else dates[-1]
    if any(observation_date > cutoff for observation_date in dates):
        raise ShortInterestError("daily outcomes contain observations after as_of")

    intc_prepared = _prepare_short_interest(intc, "INTC")
    amd_prepared = _prepare_short_interest(amd, "AMD")
    for prepared, ticker in ((intc_prepared, "INTC"), (amd_prepared, "AMD")):
        if any(
            settlement_date > cutoff
            for settlement_date in prepared[f"{ticker}_settlement_date"]
        ):
            raise ShortInterestError(f"{ticker} short-interest history contains settlement dates after as_of")

    daily_frame = pd.concat(
        [normalized_spread, normalized_pnl],
        axis=1,
    )
    daily_frame.index.name = "date"
    intc_prepared = intc_prepared.loc[intc_prepared.index <= cutoff]
    amd_prepared = amd_prepared.loc[amd_prepared.index <= cutoff]
    intc_daily =     intc_prepared[["INTC_short_interest"]].reindex(
        daily_frame.index,
        method="ffill",
    )
    amd_daily =     amd_prepared[["AMD_short_interest"]].reindex(
        daily_frame.index,
        method="ffill",
    )
    aligned = pd.concat([daily_frame, intc_daily, amd_daily], axis=1)
    aligned = aligned.dropna(subset=["INTC_short_interest", "AMD_short_interest"], how="all")
    if aligned.empty:
        raise ShortInterestError("no short-interest reports were available in the outcome period")

    intc_report_dates = set(intc_prepared.index)
    amd_report_dates = set(amd_prepared.index)
    intc_updates = aligned.index.isin(intc_report_dates)
    amd_updates = aligned.index.isin(amd_report_dates)
    aligned["INTC_short_interest_pct_change"] = aligned["INTC_short_interest"].pct_change(
        fill_method=None
    )
    aligned["AMD_short_interest_pct_change"] = aligned["AMD_short_interest"].pct_change(
        fill_method=None
    )
    aligned["absolute_spread_change"] = aligned["spread"].diff().abs()
    aligned["INTC_report_update"] = intc_updates
    aligned["AMD_report_update"] = amd_updates
    if adf_p_values is not None:
        if adf_p_values.index.has_duplicates:
            raise ShortInterestError("adf_p_values must not contain duplicate dates")
        try:
            adf_dates = pd.Index(
                [pd.Timestamp(item).date() for item in adf_p_values.index],
                name="date",
            )
            adf_values = pd.Series(
                pd.to_numeric(adf_p_values, errors="raise").to_numpy(dtype=float),
                index=adf_dates,
                name="adf_pvalue",
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ShortInterestError("adf_p_values must contain valid dates and numeric values") from error
        if adf_dates.has_duplicates or not adf_dates.is_monotonic_increasing:
            raise ShortInterestError("ADF dates must be unique and increasing")
        if any(observation_date > cutoff for observation_date in adf_dates):
            raise ShortInterestError("adf_p_values contains observations after as_of")
        aligned = aligned.join(adf_values, how="left")
        if (aligned["adf_pvalue"].dropna() > 1.0).any() or (
            aligned["adf_pvalue"].dropna() < 0.0
        ).any():
            raise ShortInterestError("ADF p-values must be between 0 and 1")
        aligned["unstable"] = aligned["adf_pvalue"] > significance_level

    intc_update_rows = aligned.loc[
        aligned["INTC_report_update"],
        ["INTC_short_interest_pct_change", "absolute_spread_change"],
    ]
    amd_update_rows = aligned.loc[
        aligned["AMD_report_update"],
        ["AMD_short_interest_pct_change", "absolute_spread_change"],
    ]
    intc_correlation = _safe_correlation(
        intc_update_rows["INTC_short_interest_pct_change"],
        intc_update_rows["absolute_spread_change"],
    )
    amd_correlation = _safe_correlation(
        amd_update_rows["AMD_short_interest_pct_change"],
        amd_update_rows["absolute_spread_change"],
    )

    unstable_observations: int | None = None
    unstable_mean_daily_pnl: float | None = None
    stable_mean_daily_pnl: float | None = None
    if adf_p_values is not None:
        classified = aligned.dropna(subset=["adf_pvalue"])
        unstable_mask = classified["unstable"]
        stable_mask = ~unstable_mask
        unstable_observations = int(unstable_mask.sum())
        if unstable_observations:
            unstable_mean_daily_pnl = float(classified.loc[unstable_mask, "daily_pnl"].mean())
        if stable_mask.any():
            stable_mean_daily_pnl = float(classified.loc[stable_mask, "daily_pnl"].mean())

    assert len(aligned) <= len(daily_frame)
    assert aligned.index.is_monotonic_increasing
    assert all(observation_date <= cutoff for observation_date in aligned.index)
    logger.info(
        "Short-interest impact through %s: matched_days=%d report_updates=%d "
        "INTC_delta/spread_change_corr=%s AMD_delta/spread_change_corr=%s "
        "unstable_days=%s unstable_mean_pnl=%s stable_mean_pnl=%s",
        cutoff,
        len(aligned),
        int(intc_updates.sum()) + int(amd_updates.sum()),
        intc_correlation,
        amd_correlation,
        unstable_observations,
        unstable_mean_daily_pnl,
        stable_mean_daily_pnl,
    )
    return ShortInterestImpact(
        observations=len(aligned),
        report_updates=int(intc_updates.sum()) + int(amd_updates.sum()),
        intc_updates=int(intc_updates.sum()),
        amd_updates=int(amd_updates.sum()),
        intc_pct_change_spread_change_correlation=intc_correlation,
        amd_pct_change_spread_change_correlation=amd_correlation,
        unstable_observations=unstable_observations,
        unstable_mean_daily_pnl=unstable_mean_daily_pnl,
        stable_mean_daily_pnl=stable_mean_daily_pnl,
        aligned_data=aligned,
    )
