from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
import requests

from src.short_interest import (
    ShortInterestError,
    analyze_short_interest_impact,
    fetch_nasdaq_short_interest,
)


class FakeResponse:
    """Provide a deterministic response compatible with requests.Response."""

    def __init__(self, payload: object, status_error: requests.HTTPError | None = None) -> None:
        """Store JSON payload and optional HTTP status failure.

        Parameters
        ----------
        payload : object
            Value returned by :meth:`json`.
        status_error : requests.HTTPError or None, default=None
            Optional exception raised by :meth:`raise_for_status`.

        Returns
        -------
        None
            Initializes this mock response.

        Raises
        ------
        None
            Construction does not raise errors.

        Notes
        -----
        This minimal response fake exercises request parsing without network
        access or dependence on a live Nasdaq response.
        """
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        """Raise the configured HTTP error, if present.

        Parameters
        ----------
        None
            The response has no call-time parameters.

        Returns
        -------
        None
            Returns normally for a successful status.

        Raises
        ------
        requests.HTTPError
            If configured to simulate a non-success HTTP status.

        Notes
        -----
        Mirrors the response status check used by the production fetcher.
        """
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> object:
        """Return the configured JSON-compatible payload.

        Parameters
        ----------
        None
            The response has no call-time parameters.

        Returns
        -------
        object
            Configured API payload.

        Raises
        ------
        None
            The fake payload is returned directly.

        Notes
        -----
        Returning object keeps the fixture permissive enough to test malformed
        or alternate nested response structures.
        """
        return self.payload


def _payload() -> dict[str, object]:
    """Build a nested Nasdaq-like historical short-interest payload.

    Parameters
    ----------
    None
        The fixture has fixed values.

    Returns
    -------
    dict of str to object
        Nested data table with formatted values and two settlement dates.

    Raises
    ------
    None
        The literal payload is deterministic.

    Notes
    -----
    Values intentionally use commas and string fields to cover common Nasdaq
    presentation formatting.
    """
    return {
        "data": {
            "shortInterestTable": {
                "headers": {"settlementDate": "Settlement Date"},
                "rows": [
                    {
                        "settlementDate": "06/13/2025",
                        "shortInterest": "1,234,567",
                        "avgDailyShareVolume": "50,000",
                        "daysToCover": "24.69",
                    },
                    {
                        "settlementDate": "05/30/2025",
                        "shortInterest": "1,100,000",
                        "avgDailyShareVolume": "44,000",
                        "daysToCover": "25.00",
                    },
                ],
            }
        }
    }


def test_fetch_nasdaq_short_interest_parses_payload_and_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse Nasdaq history and send browser-compatible HTTP headers.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the requests HTTP call.

    Returns
    -------
    None
        Assertions verify parsed data and request metadata.

    Raises
    ------
    AssertionError
        If the response is parsed incorrectly or required request headers are
        absent.

    Notes
    -----
    This test uses an offline fake and does not validate the live endpoint's
    current availability or response schema.
    """
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        """Record the request and return a fixed response.

        Parameters
        ----------
        url : str
            Requested endpoint URL.
        kwargs : object
            Request parameters and headers.

        Returns
        -------
        FakeResponse
            Successful Nasdaq-like response.

        Raises
        ------
        None
            The fake always succeeds.

        Notes
        -----
        Captured headers allow a deterministic User-Agent assertion.
        """
        calls.append((url, kwargs))
        return FakeResponse(_payload())

    monkeypatch.setattr("src.short_interest.requests.get", fake_get)

    result = fetch_nasdaq_short_interest(" intc ")

    assert calls[0][0] == "https://api.nasdaq.com/api/quote/INTC/short-interest"
    request_headers = calls[0][1]["headers"]
    assert isinstance(request_headers, dict)
    user_agent = request_headers["User-Agent"]
    assert isinstance(user_agent, str)
    assert "Mozilla" in user_agent
    assert result.index.name == "settlement_date"
    assert list(result.index) == [date(2025, 5, 30), date(2025, 6, 13)]
    assert result.loc[date(2025, 6, 13), "short_interest"] == pytest.approx(1_234_567.0)
    assert result.loc[date(2025, 6, 13), "average_daily_volume"] == pytest.approx(50_000.0)
    assert result.loc[date(2025, 6, 13), "days_to_cover"] == pytest.approx(24.69)


def test_fetch_nasdaq_actual_table_schema_uses_interest_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parse the Nasdaq endpoint's current shortInterestTable row schema.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the requests HTTP call.

    Returns
    -------
    None
        Assertions verify the production field names and date format.

    Raises
    ------
    AssertionError
        If the ``interest`` API field is not recognized as short interest.

    Notes
    -----
    Nasdaq's observed payload calls the value column ``interest`` even though
    the surrounding object is ``shortInterestTable``; this fixture protects
    the parser against regressing to the assumed ``shortInterest`` name.
    """
    payload: dict[str, object] = {
        "data": {
            "symbol": "intc",
            "shortInterestTable": {
                "headers": {
                    "settlementDate": "Settlement Date",
                    "interest": "Short Interest",
                    "avgDailyShareVolume": "Avg. Daily Share Volume",
                    "daysToCover": "Days to Cover",
                },
                "rows": [
                    {
                        "settlementDate": "09/30/2026",
                        "interest": "167,199,908",
                        "avgDailyShareVolume": "120,610,361",
                        "daysToCover": 1.386281,
                    }
                ],
            },
        },
        "message": None,
        "status": {"rCode": 200},
    }
    monkeypatch.setattr(
        "src.short_interest.requests.get",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    result = fetch_nasdaq_short_interest("INTC")

    assert list(result.index) == [date(2026, 9, 30)]
    assert result.loc[date(2026, 9, 30), "short_interest"] == pytest.approx(167_199_908.0)
    assert result.loc[date(2026, 9, 30), "average_daily_volume"] == pytest.approx(120_610_361.0)
    assert result.loc[date(2026, 9, 30), "days_to_cover"] == pytest.approx(1.386281)


def test_fetch_nasdaq_short_interest_surfaces_api_and_schema_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise the domain error for failed HTTP requests and unknown payload shape.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the requests HTTP call.

    Returns
    -------
    None
        Assertions verify error translation.

    Raises
    ------
    AssertionError
        If an upstream or schema failure is silently accepted.

    Notes
    -----
    Network/API limitations are explicit so the main pipeline can log a
    transparent warning and continue prior validated phases.
    """
    monkeypatch.setattr(
        "src.short_interest.requests.get",
        lambda *args, **kwargs: FakeResponse({}, requests.HTTPError("403")),
    )
    with pytest.raises(ShortInterestError, match="request failed"):
        fetch_nasdaq_short_interest("INTC")

    monkeypatch.setattr(
        "src.short_interest.requests.get",
        lambda *args, **kwargs: FakeResponse({"data": {"unexpected": []}}),
    )
    with pytest.raises(ShortInterestError, match="no historical"):
        fetch_nasdaq_short_interest("AMD")


def test_short_interest_is_not_forward_filled_before_availability() -> None:
    """Compare SI with daily outcomes only after conservative publication lag.

    Parameters
    ----------
    None
        Fixed short-interest reports and daily outcomes are used.

    Returns
    -------
    None
        Assertions validate availability, forward-fill, and update matching.

    Raises
    ------
    AssertionError
        If settlement values appear before their estimated availability dates.

    Notes
    -----
    Each observation becomes available after a conservative 14-calendar-day
    settlement lag. The first report must not leak into preceding daily dates.
    """
    dates = pd.date_range("2025-01-01", periods=50, freq="D").date
    settlements = [date(2025, 1, 2), date(2025, 1, 20)]
    intc = pd.DataFrame(
        {"short_interest": [100.0, 120.0]},
        index=pd.Index(settlements, name="settlement_date"),
    )
    amd = pd.DataFrame(
        {"short_interest": [200.0, 180.0]},
        index=pd.Index(settlements, name="settlement_date"),
    )
    spread = pd.Series(
        [float(position) / 10.0 for position in range(len(dates))],
        index=pd.Index(dates, name="date"),
    )
    daily_pnl = spread.diff().fillna(0.0).rename("daily_pnl")
    result = analyze_short_interest_impact(
        intc,
        amd,
        spread,
        daily_pnl,
        as_of=dates[-1],
    )
    first_available = settlements[0] + timedelta(days=14)

    assert result.aligned_data.index.min() == first_available
    assert result.aligned_data.loc[first_available, "INTC_short_interest"] == pytest.approx(100.0)
    assert result.intc_updates == 2
    assert result.amd_updates == 2
    assert result.report_updates == 4
    assert all(observation_date <= dates[-1] for observation_date in result.aligned_data.index)


def test_short_interest_analysis_aligns_optional_adf_labels() -> None:
    """Report unstable/stable PnL summaries when ADF observations are supplied.

    Parameters
    ----------
    None
        Fixed short-interest, spread, PnL, and ADF data are used.

    Returns
    -------
    None
        Assertions verify matched stability groups.

    Raises
    ------
    AssertionError
        If optional ADF labels are not aligned with the daily PnL.

    Notes
    -----
    Only dates with both published short interest and a valid ADF p-value are
    counted in stable/unstable daily PnL summaries.
    """
    dates = pd.date_range("2025-01-01", periods=40, freq="D").date
    settlements = [date(2024, 12, 1), date(2025, 1, 10)]
    intc = pd.DataFrame(
        {"short_interest": [1_000.0, 1_500.0]},
        index=pd.Index(settlements, name="settlement_date"),
    )
    amd = pd.DataFrame(
        {"short_interest": [2_000.0, 2_500.0]},
        index=pd.Index(settlements, name="settlement_date"),
    )
    index = pd.Index(dates, name="date")
    spread = pd.Series(range(40), index=index, dtype=float)
    daily_pnl = pd.Series([1.0] * 40, index=index, dtype=float)
    adf_p_values = pd.Series([0.01] * 20 + [0.10] * 20, index=index, dtype=float)

    result = analyze_short_interest_impact(
        intc,
        amd,
        spread,
        daily_pnl,
        adf_p_values=adf_p_values,
        as_of=dates[-1],
    )

    assert result.unstable_observations is not None
    assert result.unstable_observations > 0
    assert result.unstable_mean_daily_pnl == pytest.approx(1.0)
    assert result.stable_mean_daily_pnl == pytest.approx(1.0)
    assert "unstable" in result.aligned_data.columns
