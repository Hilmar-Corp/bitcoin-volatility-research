# Bitcoin Volatility Research

HilmarCorp Research — HC-BTCVOL-001

Research question:
Can statistical models improve one-day-ahead
Bitcoin volatility forecasts out of sample?

Models:
- Rolling variance
- EWMA
- ARCH(1)
- GARCH(1,1)
- Rolling realized variance
- EWMA-RV
- HAR-RV

Data: Binance Spot BTCUSDT, 5-minute candles.
Period: 2020-01-01 to 2026-09-30 UTC.
Primary loss: QLIKE.
Evaluation: chronological walk-forward.

Raw data and full reconstructed datasets are excluded
from GitHub. Final forecast artifacts are in `results/`.

Code license: Apache License 2.0.

## Verified research dataset

Data quality policy: v1.1

- Binance BTCUSDT spot, 5-minute candles
- Study period: 2020-01-01 to 2026-09-30 UTC
- 82 SHA-256-verified source archives
- 709,920 expected 5-minute timestamps
- 709,458 observed candles
- 462 missing 5-minute candles
- 7 shortened candles
- 1 candle with invalid negative duration
- 2,465 daily observations
- 2,449 admissible realized-variance observations
- 16 documented invalid realized-variance days
- 2,465 admissible daily close-to-close returns

The final out-of-sample period, 2025-01-01 through
2026-09-30, contains 638 valid daily RV targets.

Missing observations are never interpolated.
The full calendar is preserved.

The original protocol is preserved under
`protocol-v1.0.0`.

The data-quality amendment is versioned separately.

See:
- `docs/data_quality_policy_v1.1.md`
- `docs/data_manifest_v1_1.json`
- `configs/source_quality_reference_v1.json`

Final holdout results: `docs/research_findings_v1.md`.

## Post-holdout scientific robustness review

The independent numerical replay, exploratory
robustness diagnostics, figures and academic discussion
are archived separately from the first-look results.

- `docs/postholdout_scientific_review_v1.md`
- `docs/postholdout_diagnostics_v1.json`
- `figures/`

The preregistered results remain unchanged under
`final-results-v1.0.0`.

Alternative HAC lags, bootstrap blocks and outlier
exclusions are post-holdout exploratory analyses.
They do not constitute new confirmatory tests.

This is an internally audited research artifact,
not an external peer-reviewed publication or
institutional model validation.

## Independent second-solver QMLE audit

All 1,276 ARCH/GARCH holdout estimations were
recomputed using an independently implemented
SLSQP optimizer.

Convergence: 1,276 / 1,276. Alerts: zero.

See `docs/qmle_second_solver_review_v1.md`
and `results/second_solver_qmle_v1/`.

The original confirmatory results remain unchanged.


## Reproducibility, citation and governance

The first-look results, post-holdout analyses and numerical audit are
**immutable historical releases**, verified with SHA-256 manifests.

The repository is **private**; third parties need access to reproduce
the released artifact-level results. Full data reconstruction also requires
the separately obtained Binance source archives.

Reproduce the available forecast results:

    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-lock.txt
    python scripts/verify_release_integrity.py
    python -W error::DeprecationWarning -m pytest tests/ -q
    python scripts/verify_public_results.py

- [Reproducibility and limitations](REPRODUCIBILITY.md)
- [Citation metadata](CITATION.cff)
- [Contributor and change-control policy](CONTRIBUTING.md)
- [Apache 2.0 license](LICENSE) and [copyright NOTICE](NOTICE)
- [Automated reproducibility checks](.github/workflows/reproducibility.yml)

CI checks released artifact bytes and numerical inference, **not** missing
third-party market-data archives or external peer-reviewed validation.
