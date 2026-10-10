# AI Usage Disclosure

## Overview of AI Assistance

In compliance with the BlackSwan Quants project submission standards, this document provides a transparent account of AI usage throughout the development of the INTC and AMD quantitative pairs trading pipeline.

GitHub Copilot, acting as the AI programming assistant in VS Code, contributed substantially to implementation of the pipeline, unit tests, and project documentation in response to requirements and iterative feedback from the human author. The author supplied the project goals, methodological constraints, and review direction; generated code and analysis were iteratively inspected and revised. This disclosure does not imply that every line was authored by one party or that generated output was accepted without review.

## Division of Responsibilities

1. **AI assistance (GitHub Copilot):**
   - Drafted and iteratively revised implementation across the core modules (`src/data_loader.py`, `src/features.py`, `src/backtest.py`, `src/stability.py`, `src/short_interest.py`, `src/simulation.py`, and `main.py`).
   - Drafted and revised automated tests for data loading, features, backtesting, stability, short interest, simulation, and pipeline integration.
   - Assisted with project documentation, including the README, executive write-up, and this disclosure.

2. **Human author direction and review:**
   - Set the research objectives, pair, phase requirements, and methodological constraints, including the explicit `as_of` cutoff, In-Sample/Out-of-Sample separation, trial transparency, and the choice of a Joint Moving-Block Bootstrap rather than an OU simulation.
   - Reviewed outputs and requested corrections where data coverage, interpretation, or conclusions needed further investigation or more cautious wording.
   - Test commands and numerical checks were run with AI-assistant tooling during this development workflow. The author should separately disclose any additional independent testing or analysis performed outside this workflow.

## Human Corrections, Overrides, and Critical Fixes

The implementation was revised through iterative human direction and AI-assisted investigation:

- **Nasdaq API investigation (Phase 4):** An initial attempt did not find/parse the historical short-interest table for INTC. The human author pointed out that Phase 4 data should be retried and asked for another investigation. In response, the AI assistant revisited the endpoint payload and parsing logic, updated the field/date handling and request headers, and added offline parser/request tests. The later Nasdaq integration remains best-effort and subject to vendor availability; the 14-day lag is an analytical proxy, not a verified publication timestamp.
- **Sensitivity classification:** During review, an inconsistency was identified between the documented “inconclusive” interpretation of a non-profitable In-Sample baseline and the code flag. The AI assistant changed the implementation so that such a baseline is not marked as parameter-fragile and added a regression test.
- **Conservative interpretation:** The human author requested rigorous, transparent interpretation. The AI assistant revised the write-up to describe ADF and bootstrap outputs as conditional/descriptive results, not proof of a structural break, calibrated predictive probabilities, or evidence of a robust trading edge.