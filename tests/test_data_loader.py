from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import main as warmup
from src import data_loader
from src.data_loader import DataFetchError, load_price_data


class FakeTicker:
    """Provide deterministic Yahoo Finance history for loader tests."""

    calls: list[tuple[str, dict[str, object]]]
    response: pd.DataFrame

    def __init__(self, ticker: str) -> None:
        """Record a requested ticker.

        Parameters
        ----------
        ticker : str
            Requested symbol.

        Returns
        -------
        None
            Initializes the fake ticker instance.

        Raises
        ------
        None
            Initialization does not raise errors.

        Notes
        -----
        Requests are collected in a class attribute so tests can verify cache
        behavior without contacting Yahoo Finance.
        """
        self.ticker = ticker

    def history(self, **kwargs: object) -> pd.DataFrame:
        """Return configured price data and record the API arguments.

        Parameters
        ----------
        kwargs : object
            Yahoo Finance history options supplied by the loader.

        Returns
        -------
        pandas.DataFrame
            Configured deterministic history.

        Raises
        ------
        None
            This fake always returns its configured history.

        Notes
        -----
        Recording options lets the tests verify that ``end`` is exclusive and
        set to the day after the requested inclusive cutoff.
        """
        self.calls.append((self.ticker, kwargs))
        return self.response


def _history() -> pd.DataFrame:
    """Create sample observations including one date after the cutoff.

    Parameters
    ----------
    None
        The sample has no parameters.

    Returns
    -------
    pandas.DataFrame
        Three daily close observations spanning the requested cutoff.

    Raises
    ------
    None
        DataFrame construction does not raise for these fixed values.

    Notes
    -----
    The post-cutoff observation provides a direct regression check for
    look-ahead prevention.
    """
    return pd.DataFrame(
        {"Close": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_load_price_data_caps_date_and_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify cutoff filtering and that a second load avoids API calls.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the Yahoo Finance client.
    tmp_path : pathlib.Path
        Temporary directory for isolated pickle checkpoints.

    Returns
    -------
    None
        Assertions validate both downloaded and cached behavior.

    Raises
    ------
    AssertionError
        If data extends beyond the cutoff or caching repeats a request.

    Notes
    -----
    The same fake history is used for both symbols; this isolates loader
    behavior from market data and network availability.
    """
    FakeTicker.calls = []
    FakeTicker.response = _history()
    monkeypatch.setattr(data_loader.yf, "Ticker", FakeTicker)
    as_of = date(2024, 1, 3)

    first = load_price_data(
        ("INTC", "AMD"),
        start_date=date(2024, 1, 2),
        as_of=as_of,
        cache_dir=tmp_path,
    )
    assert list(first.columns) == ["INTC", "AMD"]
    assert list(first.index) == [date(2024, 1, 2), as_of]
    assert all(observation <= as_of for observation in first.index)
    assert len(FakeTicker.calls) == 2
    assert all(call[1]["end"] == "2024-01-04" for call in FakeTicker.calls)

    second = load_price_data(
        ("INTC", "AMD"),
        start_date=date(2024, 1, 2),
        as_of=as_of,
        cache_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(FakeTicker.calls) == 2


def test_load_price_data_wraps_yahoo_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose remote request failures as the loader's domain exception.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the Yahoo Finance client.
    tmp_path : pathlib.Path
        Temporary directory used to isolate cache state.

    Returns
    -------
    None
        The assertion validates the public exception contract.

    Raises
    ------
    AssertionError
        If a Yahoo Finance failure is not translated to ``DataFetchError``.

    Notes
    -----
    This test fails before writing any cache file, demonstrating that failed
    requests do not create success-shaped checkpoints.
    """
    class FailingTicker:
        """Simulate a Yahoo Finance request failure."""

        def __init__(self, ticker: str) -> None:
            """Accept the requested symbol.

            Parameters
            ----------
            ticker : str
                Requested symbol.

            Returns
            -------
            None
                Initializes the fake object.

            Raises
            ------
            None
                Initialization cannot fail.

            Notes
            -----
            The symbol is unused because the failure is deterministic.
            """

        def history(self, **kwargs: object) -> pd.DataFrame:
            """Raise a simulated upstream failure.

            Parameters
            ----------
            kwargs : object
                Yahoo Finance history options.

            Returns
            -------
            pandas.DataFrame
                This method never returns normally.

            Raises
            ------
            RuntimeError
                Always, to simulate an upstream request failure.

            Notes
            -----
            The loader should preserve this as the cause of ``DataFetchError``.
            """
            raise RuntimeError("network unavailable")

    monkeypatch.setattr(data_loader.yf, "Ticker", FailingTicker)
    with pytest.raises(DataFetchError) as error:
        load_price_data(
            ("INTC",),
            start_date=date(2024, 1, 2),
            as_of=date(2024, 1, 3),
            cache_dir=tmp_path,
        )
    assert isinstance(error.value.__cause__, RuntimeError)


def test_load_price_data_rejects_invalid_date_range(tmp_path: Path) -> None:
    """Reject a request whose start date is after its information cutoff.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory passed as the cache destination.

    Returns
    -------
    None
        The assertion validates input checking.

    Raises
    ------
    AssertionError
        If an invalid date range is accepted.

    Notes
    -----
    Validation occurs before any network or cache operation.
    """
    with pytest.raises(ValueError, match="start_date must be on or before as_of"):
        load_price_data(
            ("INTC",),
            start_date=date(2024, 1, 4),
            as_of=date(2024, 1, 3),
            cache_dir=tmp_path,
        )


def test_run_warmup_writes_all_diagnostic_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Generate all required plots from local deterministic price data.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the external data loader.
    tmp_path : pathlib.Path
        Temporary directory for generated PNG files.

    Returns
    -------
    None
        Assertions verify each expected plot was written.

    Raises
    ------
    AssertionError
        If a requested chart is missing or has an unexpected filename.

    Notes
    -----
    Seventy daily rows exercise the 60-session rolling-correlation calculation
    while keeping the test independent of Yahoo Finance and network access.
    """
    index = pd.date_range("2024-01-01", periods=70, freq="D").date
    prices = pd.DataFrame(
        {
            "INTC": [20.0 + day for day in range(70)],
            "AMD": [15.0 + day * 0.8 for day in range(70)],
        },
        index=pd.Index(index, name="Date"),
    )
    monkeypatch.setattr(warmup, "load_price_data", lambda *args, **kwargs: prices)

    plots = warmup.run_warmup(
        as_of=date(2024, 3, 10),
        start_date=date(2024, 1, 1),
        cache_dir=tmp_path / "data",
        output_dir=tmp_path / "plots",
    )

    assert {plot.name for plot in plots} == {
        "prices.png",
        "price_ratio.png",
        "price_difference.png",
        "rolling_correlation_60d.png",
    }
    assert all(plot.is_file() and plot.stat().st_size > 0 for plot in plots)
