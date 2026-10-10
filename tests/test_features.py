from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureCalculationError,
    compute_hedge_ratio,
    compute_z_score,
)


def test_compute_hedge_ratio_recovers_linear_relationship() -> None:
    """Recover known OLS coefficients and the associated zero-residual spread.

    Parameters
    ----------
    None
        This test uses fixed deterministic input data.

    Returns
    -------
    None
        Assertions verify the fitted coefficients and residuals.

    Raises
    ------
    AssertionError
        If the estimates or spread differ from the known linear relationship.

    Notes
    -----
    The dependent prices are constructed exactly as alpha plus beta times the
    independent prices, so the fitted residuals should be numerically zero.
    """
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    price_b = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0], index=index)
    price_a = pd.Series(3.0 + 1.5 * price_b.to_numpy(), index=index)

    alpha, beta, spread = compute_hedge_ratio(price_a, price_b)

    assert alpha == pytest.approx(3.0)
    assert beta == pytest.approx(1.5)
    assert spread.index.equals(index)
    np.testing.assert_allclose(spread.to_numpy(), 0.0, atol=1e-12)


def test_compute_hedge_ratio_rejects_empty_and_misaligned_inputs() -> None:
    """Reject empty or differently indexed asset price histories.

    Parameters
    ----------
    None
        This test uses fixed deterministic input data.

    Returns
    -------
    None
        Assertions verify input validation behavior.

    Raises
    ------
    AssertionError
        If invalid input is not reported as ``FeatureCalculationError``.

    Notes
    -----
    Explicit failures prevent accidental pairwise alignment from producing a
    hedge ratio on an unintended set of observations.
    """
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    price_a = pd.Series([11.0, 12.0, 13.0], index=index)
    price_b = pd.Series([5.0, 6.0, 7.0], index=index)

    with pytest.raises(FeatureCalculationError, match="must not be empty"):
        compute_hedge_ratio(price_a.iloc[:0], price_b.iloc[:0])
    with pytest.raises(FeatureCalculationError, match="identical indexes"):
        compute_hedge_ratio(price_a, price_b.set_axis(index[::-1]))


def test_compute_hedge_ratio_rejects_nonfinite_data_and_constant_predictor() -> None:
    """Reject invalid observations and a predictor with no variation.

    Parameters
    ----------
    None
        This test uses fixed deterministic input data.

    Returns
    -------
    None
        Assertions verify both invalid-data checks.

    Raises
    ------
    AssertionError
        If invalid regression data is accepted.

    Notes
    -----
    A non-finite observation corrupts least-squares estimation, while a
    constant predictor makes the hedge-ratio slope unidentifiable.
    """
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    price_a = pd.Series([11.0, 12.0, 13.0], index=index)
    with pytest.raises(FeatureCalculationError, match="non-finite"):
        compute_hedge_ratio(price_a, pd.Series([5.0, np.inf, 7.0], index=index))
    with pytest.raises(FeatureCalculationError, match="distinct values"):
        compute_hedge_ratio(price_a, pd.Series([5.0, 5.0, 5.0], index=index))


def test_compute_z_score_uses_trailing_sample_statistics() -> None:
    """Calculate rolling standardized values with a complete trailing window.

    Parameters
    ----------
    None
        This test uses a fixed four-observation spread.

    Returns
    -------
    None
        Assertions verify the first available z-score and retained index.

    Raises
    ------
    AssertionError
        If the rolling z-score or output shape is incorrect.

    Notes
    -----
    For the first window [1, 2, 3], the rolling sample mean is 2 and its
    sample standard deviation is 1, giving a final-window z-score of 1.
    """
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    spread = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

    z_score = compute_z_score(spread, window=3)

    assert z_score.index.equals(index)
    assert z_score.iloc[:2].isna().all()
    assert z_score.iloc[2] == pytest.approx(1.0)
    assert z_score.iloc[3] == pytest.approx(1.0)


def test_compute_z_score_rejects_invalid_input_and_handles_zero_deviation() -> None:
    """Validate rolling inputs and preserve undefined zero-variance scores.

    Parameters
    ----------
    None
        This test uses fixed deterministic input data.

    Returns
    -------
    None
        Assertions verify validation and undefined-score behavior.

    Raises
    ------
    AssertionError
        If invalid window/input values are accepted or zero variance is hidden.

    Notes
    -----
    A constant spread has a zero rolling deviation; its standardized scores are
    mathematically undefined and therefore expected to remain NaN.
    """
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    spread = pd.Series([2.0, 2.0, 2.0, 2.0], index=index)

    with pytest.raises(FeatureCalculationError, match="must not be empty"):
        compute_z_score(spread.iloc[:0], window=2)
    with pytest.raises(FeatureCalculationError, match="integer"):
        compute_z_score(spread, window=1)
    with pytest.raises(FeatureCalculationError, match="must not exceed"):
        compute_z_score(spread, window=5)

    z_score = compute_z_score(spread, window=3)
    assert z_score.iloc[:2].isna().all()
    assert z_score.iloc[2:].isna().all()