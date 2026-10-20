# HC-BTCVOL-001 — Second-Solver QMLE Audit

**Classification:** Post-holdout numerical audit.

## Objective

Verify the numerical stability of the original
ARCH(1) and GARCH(1,1) Gaussian QMLE estimates.

Original optimizer: L-BFGS-B.
Independent implementation: SLSQP.
Independent starting points: three per window.

## Results

| Model | Windows | Maximum absolute normalized objective gap | Maximum forecast difference | Alerts |
|---|---:|---:|---:|---:|
| ARCH | 638 | 1.12354514581e-10 | 0.005625% | 0 |
| GARCH | 638 | 2.41823239211e-09 | 0.021160% | 0 |

All 1,276 second-solver estimations converged.

No estimation exceeded the numerical alert
thresholds: 0.0001 for the absolute objective
difference and 5% for the relative forecast
difference.

## Interpretation

The close agreement of fitted likelihood values
and conditional-variance forecasts supports
the numerical stability of the original QMLE
estimates.

This does not establish global optimality or
prove that ARCH/GARCH is the best specification.
Both implementations use SciPy.

## Research integrity

This audit occurred after the first-look holdout.

Original forecasts, frozen hypotheses and
confirmatory test results remain unchanged.

This is an internal second-implementation
verification, not external peer review.

## Reproduction

- scripts/audit_second_solver_qmle.py
- results/second_solver_qmle_v1/audit_report.json
- results/second_solver_qmle_v1/independent_qmle_refits.parquet
- results/second_solver_qmle_v1/provenance_manifest.json

The full source-data reconstruction requires
the local quality-controlled research dataset.
