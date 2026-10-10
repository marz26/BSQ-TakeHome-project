from __future__ import annotations

import logging
import pickle
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetchError(RuntimeError):
    """Raised when requested historical market data cannot be obtained.

    Parameters
    ----------
    message : str
        Description of the data retrieval failure.

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
    API and empty-response failures are translated to this exception so callers
    can distinguish unavailable market data from downstream analysis errors.
    """


def _cache_path(ticker: str, start_date: date, as_of: date, cache_dir: Path) -> Path:
    """Build the deterministic path for a cached ticker history.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start_date : date
        Inclusive beginning of the requested history.
    as_of : date
        Inclusive end of the requested history.
    cache_dir : Path
        Directory where pickle snapshots are stored.

    Returns
    -------
    Path
        Path for the ticker and requested date range.

    Raises
    ------
    ValueError
        If ``ticker`` contains no usable filename characters.

    Notes
    -----
    Including both boundaries makes each cache file an unambiguous checkpoint
    for an exact request and prevents data fetched for a later date leaking into
    an earlier request.
    """
    safe_ticker = re.sub(r"[^A-Za-z0-9_-]", "_", ticker)
    if not safe_ticker:
        raise ValueError("ticker must contain at least one filename-safe character")
    return cache_dir / f"{safe_ticker}_{start_date.isoformat()}_{as_of.isoformat()}.pkl"


def _read_cache(cache_file: Path, ticker: str, start_date: date, as_of: date) -> pd.DataFrame | None:
    """Read and validate a cached ticker history.

    Parameters
    ----------
    cache_file : Path
        Pickle file to read.
    ticker : str
        Ticker whose close-price column is expected.
    start_date : date
        Inclusive beginning of the requested history.
    as_of : date
        Inclusive end of the requested history.

    Returns
    -------
    pandas.DataFrame or None
        One-column history when the cache is valid, otherwise ``None``.

    Raises
    ------
    None
        Unreadable cache files are logged and treated as cache misses.

    Notes
    -----
    A cache is accepted only when it contains the expected column and a
    date-indexed DataFrame. Corrupt or stale cache data is discarded so it is
    never silently used as market data.
    """
    if not cache_file.exists():
        return None

    try:
        with cache_file.open("rb") as cache_handle:
            cached = pickle.load(cache_handle)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as error:
        logger.warning("Ignoring unreadable cache %s: %s", cache_file, error)
        return None

    if not isinstance(cached, pd.DataFrame) or ticker not in cached.columns:
        logger.warning("Ignoring invalid cache structure in %s", cache_file)
        return None

    prices = cached.loc[:, [ticker]].copy()
    prices.index = pd.Index(
        [pd.Timestamp(index_value).date() for index_value in prices.index],
        name="Date",
    )
    prices = prices.loc[
        (prices.index >= start_date) & (prices.index <= as_of)
    ].sort_index()
    if prices.empty:
        logger.warning("Ignoring empty or out-of-range cache %s", cache_file)
        return None
    return prices


def _fetch_ticker_history(
    ticker: str,
    start_date: date,
    as_of: date,
) -> pd.DataFrame:
    """Download one ticker's adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    start_date : date
        Inclusive beginning of the requested history.
    as_of : date
        Inclusive latest permitted observation.

    Returns
    -------
    pandas.DataFrame
        A one-column DataFrame named for ``ticker``.

    Raises
    ------
    DataFetchError
        If Yahoo Finance fails or returns no usable closing prices.

    Notes
    -----
    Yahoo Finance treats ``end`` as exclusive, so the request ends on the day
    after ``as_of``. Adjusted prices are requested to account for splits and
    distributions; results are then clipped and asserted against the explicit
    information cutoff.
    """
    exclusive_end = as_of + timedelta(days=1)
    try:
        history = yf.Ticker(ticker).history(
            start=start_date.isoformat(),
            end=exclusive_end.isoformat(),
            auto_adjust=True,
            actions=False,
        )
    except Exception as error:
        raise DataFetchError(f"Yahoo Finance request failed for {ticker}") from error

    if history.empty or "Close" not in history.columns:
        raise DataFetchError(f"Yahoo Finance returned no closing prices for {ticker}")

    prices = history.loc[:, ["Close"]].rename(columns={"Close": ticker}).copy()
    prices.index = pd.Index(
        [pd.Timestamp(index_value).date() for index_value in prices.index],
        name="Date",
    )
    prices[ticker] = pd.to_numeric(prices[ticker], errors="coerce")
    prices = prices.dropna(subset=[ticker])
    prices = prices.loc[
        (prices.index >= start_date) & (prices.index <= as_of)
    ].sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    if prices.empty:
        raise DataFetchError(f"Yahoo Finance returned no usable prices for {ticker}")

    assert all(d <= as_of for d in prices.index)
    return prices


def _write_cache(cache_file: Path, prices: pd.DataFrame) -> None:
    """Persist a ticker history as an atomic local pickle checkpoint.

    Parameters
    ----------
    cache_file : Path
        Destination pickle file.
    prices : pandas.DataFrame
        Ticker history to persist.

    Returns
    -------
    None
        The cache is written in place.

    Raises
    ------
    OSError
        If the cache directory or destination cannot be written.
    pickle.PickleError
        If the DataFrame cannot be serialized.

    Notes
    -----
    A temporary sibling file is replaced into place only after serialization
    completes, preventing interrupted writes from leaving a partial checkpoint.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = cache_file.with_suffix(f"{cache_file.suffix}.tmp")
    try:
        with temporary_file.open("wb") as cache_handle:
            pickle.dump(prices, cache_handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_file.replace(cache_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


def load_price_data(
    tickers: tuple[str, ...],
    start_date: date,
    as_of: date,
    cache_dir: Path = Path("data"),
) -> pd.DataFrame:
    """Load adjusted daily closing prices, using local checkpoints when available.

    Parameters
    ----------
    tickers : tuple of str
        Ticker symbols to load, for example ``("INTC", "AMD")``.
    start_date : date
        Inclusive first date of the historical period.
    as_of : date
        Inclusive information cutoff; no returned observation may exceed it.
    cache_dir : pathlib.Path, default=Path("data")
        Directory used for per-ticker pickle checkpoints.

    Returns
    -------
    pandas.DataFrame
        Date-indexed adjusted close prices with one column per requested ticker.

    Raises
    ------
    ValueError
        If the date range or ticker list is invalid.
    DataFetchError
        If any required ticker history cannot be fetched.
    OSError
        If checkpoint data cannot be written.
    pickle.PickleError
        If checkpoint data cannot be serialized.

    Notes
    -----
    Each ticker is checkpointed independently to avoid repeating successful
    API calls after a partial failure. The explicit ``as_of`` bound is applied
    to both cached and downloaded observations before the aligned pair is
    returned, enforcing the D-1/no-look-ahead contract.
    """
    if start_date > as_of:
        raise ValueError("start_date must be on or before as_of")
    if not tickers or any(not ticker.strip() for ticker in tickers):
        raise ValueError("tickers must contain non-empty ticker symbols")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must not contain duplicates")

    ticker_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        cache_file = _cache_path(ticker, start_date, as_of, cache_dir)
        prices = _read_cache(cache_file, ticker, start_date, as_of)
        if prices is None:
            prices = _fetch_ticker_history(ticker, start_date, as_of)
            _write_cache(cache_file, prices)
            logger.info("Fetched and cached %s prices through %s", ticker, as_of)
        else:
            logger.info("Loaded cached %s prices through %s", ticker, as_of)
        ticker_frames.append(prices)

    df = pd.concat(ticker_frames, axis=1, join="inner").dropna().sort_index()
    if df.empty:
        raise DataFetchError("Requested tickers have no overlapping price observations")

    assert all(d <= as_of for d in df.index)
    return df
