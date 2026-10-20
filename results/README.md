# HC-BTCVOL-001 — Final holdout artifacts

First-look holdout: 2025-01-01 to 2026-09-30 UTC.

These files are byte-for-byte copies of the
original locally generated research outputs.

- final_predictions.parquet: 638 rows,
  seven model forecasts and realized-variance targets
- final_parameter_trace.parquet: 1914 parameter records
- final_report.json: original confirmatory results
- first_holdout_opened.json: first-look execution marker

The original data, model and execution freezes
preceded this final holdout evaluation.

Reproduce the statistical results with:

    python scripts/verify_public_results.py

The underlying Binance archives and complete
five-minute dataset are not redistributed here.

See docs/final_results_audit_v1.json
and docs/research_findings_v1.md.

No post-holdout parameter optimization was performed.
