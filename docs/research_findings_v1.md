# HC-BTCVOL-001 — Final Research Findings

## Research design

Asset: BTCUSDT, Binance Spot.

Daily realized variance from five-minute log returns.

Final out-of-sample period:
2025-01-01 through 2026-09-30, 638 UTC days.

Seven models were frozen before the holdout was
evaluated.

Primary loss: QLIKE, lower is better.

## Descriptive model ranking

| Rank | Model | Mean QLIKE |
|---|---|---:|
| 1 | HAR_RV | -6.63846313 |
| 2 | GARCH | -6.59243329 |
| 3 | D1_ewma_094 | -6.55394906 |
| 4 | I1_ewma_rv_094 | -6.55018691 |
| 5 | ARCH | -6.53958202 |
| 6 | I0_rolling_rv_30 | -6.50303889 |
| 7 | D0_rolling_30 | -6.49302190 |

All seven scores use the same 638 dates.

## Preregistered hypothesis tests

Two-sided normal-approximation Diebold-Mariano
tests, Newey-West HAC with seven lags,
Holm familywise correction at five percent.

Positive gain favors the candidate model.

| Comparison | Mean QLIKE gain | Holm p-value | Bootstrap 95% CI | Reject equal predictive accuracy |
|---|---:|---:|---|---|
| HAR-RV vs EWMA-RV | +0.08827622 | 0.00514178 | [+0.04149298, +0.15364046] | Yes |
| GARCH vs EWMA | +0.03848423 | 0.16063413 | [-0.00923804, +0.10013922] | No |

The moving-block bootstrap (14-day blocks,
10,000 replications) is supplementary,
not the preregistered decision rule.

## Primary finding

Log-HAR-RV(1,7,30) achieved a statistically
significant improvement in mean QLIKE against
EWMA-RV under the frozen confirmatory procedure.

The GARCH(1,1) improvement over daily-return
EWMA was not statistically significant.

The descriptive ranking does not establish
significance for unregistered pairwise contrasts.

## Scope and limitations

Results concern one asset, one venue,
one period, one-day-ahead forecasts,
and a realized-variance proxy.

QLIKE superiority is not evidence of
investment profitability, execution performance,
or generalization to all future market regimes.

The evaluation is historical, not a prospective
live forecasting experiment.

These are statistical research findings,
not investment recommendations.

## Reproducibility

- Data: data-policy-v1.1.0
- Models and inference: models-inference-v1.0.0
- First-look runner: holdout-runner-v1.0.0
- Post-holdout audit: docs/final_results_audit_v1.json
- Original forecast files: results/
- Verification: scripts/verify_public_results.py

No model was reoptimized using holdout results.
