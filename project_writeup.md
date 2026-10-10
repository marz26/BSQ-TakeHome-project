# Executive Writeup: INTC / AMD Pairs Trading Pipeline

## Overview & Phases Completed
This project implements a research-oriented quantitative pairs-trading pipeline for Intel (INTC) and Advanced Micro Devices (AMD), using price history from January 2010 through the stated `as_of` cutoff. The five phases below are implemented and covered by automated tests where applicable. Results that depend on external vendor availability are best-effort and must be interpreted for the specific run and data snapshot:
1. **Phase 1 (Data & Warm-up):** Automated retrieval of adjusted daily close prices with local pickle checkpointing and strict `as_of` temporal cutoff enforcement to prevent leakage.
2. **Phase 2 (In-Sample / Out-of-Sample Backtest):** Static OLS hedge ratio estimation, a disciplined grid search across 9 parameter combinations strictly on In-Sample data, and a one-time Out-of-Sample (OOS) evaluation. It includes deterministic parameter sensitivity diagnostics (flagging fragile vs. inconclusive baselines).
3. **Phase 3 (Cointegration Stability):** Rolling Augmented Dickey-Fuller (ADF) tests (90-day window) across the OOS timeline, coupled with a retrospective descriptive comparison mapping structural stability against daily PnL and drawdown paths.
4. **Phase 4 (Short Interest):** Best-effort retrieval and parsing of public Nasdaq short-interest reports. A 14-calendar-day delay after settlement is used as a conservative availability proxy when aligning reports to daily outcomes; it is not a vendor-verified publication timestamp. Short interest is not a direct measure of institutional sentiment or net institutional positioning.
5. **Phase 5 (Synthetic Path Simulation):** Joint Moving-Block Bootstrap (500 synthetic paths, block size 10, seed 42) producing a conditional distribution based on historical In-Sample paired returns. This is a scenario analysis, not a forecast or a calibrated probability model.

---

## Methodological Choices & Rationale
Where choices were required, decisions were guided strictly by statistical honesty and risk management over curve-fitting:
* **In-Sample Parameter Selection:** Hedge coefficients and strategy parameters are fitted/selected without using OOS performance; the selected values (`entry_z=2.0`, `window=90`) are then evaluated on the OOS period beginning `2024-01-01`. In the reproducible run with cached prices through `2026-10-09`, the OOS result was `-27.1274` spread-price units. This is a result for that run, not a general estimate of future performance.
* **ADF Interpretation and Bootstrap Choice:** The rolling ADF test uses a 90-observation window on the OOS residual spread with the In-Sample hedge coefficients held fixed. In the cached run, 602 of 607 valid rolling windows had p-values above 0.05. This means the unit-root null was not rejected at the 5% level in those windows; it does not prove that cointegration broke, that the spread is definitively non-stationary, or that any particular stochastic model is invalid. The Joint Moving-Block Bootstrap was chosen to resample observed paired asset returns without fitting an OU process to the spread. It preserves contemporaneous paired returns and within-block serial structure, but not serial continuity at block boundaries, and it cannot represent regimes absent from the In-Sample sample.
* **Conservative Alignment of Short-Interest Data:** A 14-calendar-day lag after settlement is used as a proxy for report availability because the endpoint does not provide historical publication timestamps. It is intended to reduce look-ahead risk, but cannot guarantee the exact date on which each report became public.

## Observed Run and Limitations

For the cached price snapshot through `2026-10-09`, split `2024-01-01`, and the default nine-attempt grid, the selected In-Sample PnL was `+8.0897` spread-price units and the blind OOS PnL was `-27.1274`. An immediate neighboring rolling window of 60 sessions changed the In-Sample PnL to `-5.9064`, so the selected result is sensitive to this parameter on this sample. These uncapitalized spread-price PnLs exclude transaction costs, slippage, borrow costs, market impact, and portfolio constraints.

With 500 bootstrap paths, block size 10, seed 42, and a 696-session horizon, the simulated PnL mean was `+13.9969`, median `+1.9003`, standard deviation `81.6046`, 5th–95th percentiles `-80.2708` to `+162.2771`, and 51.60% of paths had positive PnL. These are conditional Monte Carlo summaries of this bootstrap setup. They are not confidence intervals for future outcomes, calibrated probabilities of profit, a test against a 50% benchmark, or evidence that the realized OOS loss was or was not an anomaly.

Phase 4 depends on a live third-party endpoint and may fail or return limited/changed coverage. No Nasdaq empirical result is asserted here unless the exact response snapshot and its alignment outputs are preserved alongside the result. All statistical comparisons in this project are descriptive and do not establish causation or out-of-sample robustness.

---

## What We Would Do With More Time
If given an extended timeline to evolve this research framework, we would focus on:
1. **Dynamic Hedge Ratio Estimation:** Implementing a Kalman Filter (state-space model) or rolling OLS regression to capture time-varying cointegration parameters rather than relying on a static historical beta.
2. **Capital Normalization & Transaction Costs:** Converting spread price units into proper portfolio percentage returns ($PnL / Capital$) and integrating realistic execution friction models (bid-ask spread, borrowing fees for short legs, and market impact).
3. **Multivariate Expansion:** Expanding the cointegration universe beyond a simple binary pair (INTC/AMD) to a broader semiconductor basket using PCA or Johansen cointegration tests to isolate idiosyncratic shocks.