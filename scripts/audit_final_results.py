"""HC-BTCVOL-001: post-holdout results integrity audit.

Audits the original frozen first-look outputs.
Does not refit models or generate new forecasts.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hc_vol.inference import (
    qlike_losses,
    hac_mean_test,
    holm_adjust,
    moving_block_ci,
)

OUT = ROOT / "data/processed/final_holdout_v1"
DOCS = ROOT / "docs"

MODELS = [
    "D0_rolling_30",
    "D1_ewma_094",
    "I0_rolling_rv_30",
    "I1_ewma_rv_094",
    "ARCH",
    "GARCH",
    "HAR_RV",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


report_path = OUT / "final_report.json"
pred_path = OUT / "final_predictions.parquet"
trace_path = OUT / "final_parameter_trace.parquet"
marker_path = OUT / "first_holdout_opened.json"

spec = json.loads(
    (ROOT / "configs/inference_v1.json").read_text()
)
report = json.loads(report_path.read_text())
marker = json.loads(marker_path.read_text())

assert marker["status"] == "first_holdout_opened"
assert report["stage"] == "final_holdout_first_look"
assert report["holdout_evaluated"] is True
assert report["model_reoptimization"] is False
assert report["n"] == 638

assert spec["models"] == MODELS

for name, expected in report["output_sha256"].items():
    assert digest(OUT / name) == expected, name

pred = pd.read_parquet(pred_path)
trace = pd.read_parquet(trace_path)

dates = pd.date_range(
    "2025-01-01",
    "2026-10-01",
    freq="D",
    inclusive="left",
    tz="UTC",
)

assert len(pred) == 638
assert pd.DatetimeIndex(pred.date_utc).equals(dates)
assert pred.date_utc.is_unique

assert len(trace) == 638 * 3
assert set(trace.model) == {
    "ARCH", "GARCH", "HAR_RV"
}
assert not trace.duplicated(
    ["date_utc", "model"]
).any()

assert np.isfinite(
    pred[["rv_target"] + MODELS].to_numpy()
).all()

assert (pred.rv_target >= 0).all()
assert (pred[MODELS] > 0).all().all()

print("=== 1. FROZEN OUTPUT INTEGRITY ===")
print("PREDICTION_ROWS=638")
print("PARAMETER_TRACE_ROWS=1914")
print("SOURCE_OUTPUT_SHA256=PASS")

y = pred.rv_target.to_numpy()

loss = {
    model: qlike_losses(
        y, pred[model].to_numpy()
    )
    for model in MODELS
}

print("=== 2. INDEPENDENT QLIKE REPLAY ===")

scores = {}

for model in MODELS:
    score = float(loss[model].mean())
    recorded = report["scores"][model]

    assert recorded["n"] == 638

    np.testing.assert_allclose(
        score,
        recorded["mean_qlike"],
        rtol=0,
        atol=1e-12,
    )

    scores[model] = score
    print(f"{model}={score:.8f}")

print("=== 3. CONFIRMATORY INFERENCE REPLAY ===")

comparisons = {}
p_values = {}

bootstrap = spec["supplementary_bootstrap"]

for name, contrast in spec[
    "primary_contrasts"
].items():

    difference = (
        loss[contrast["benchmark"]]
        - loss[contrast["candidate"]]
    )

    dm = hac_mean_test(
        difference,
        lags=spec["hac_lags"],
    )

    stored = report[
        "primary_contrasts"
    ][name]["dm_hac"]

    for key in (
        "mean_gain",
        "hac_standard_error",
        "z",
        "p_value_two_sided",
    ):
        np.testing.assert_allclose(
            dm[key],
            stored[key],
            rtol=0,
            atol=1e-12,
        )

    ci = moving_block_ci(
        difference,
        block=bootstrap["block_length_days"],
        replications=bootstrap["replications"],
        seed=bootstrap["random_seed"],
    )

    original_ci = report[
        "primary_contrasts"
    ][name]["bootstrap"]["ci_percentile_95"]

    np.testing.assert_allclose(
        ci["ci_percentile_95"],
        original_ci,
        rtol=0,
        atol=1e-12,
    )

    comparisons[name] = {
        "dm_hac": dm,
        "bootstrap": ci,
    }

    p_values[name] = dm["p_value_two_sided"]

corrected = holm_adjust(
    p_values,
    alpha=spec["familywise_alpha"],
)

for name, result in comparisons.items():
    decision = corrected[name]

    original = report[
        "primary_contrasts"
    ][name]

    np.testing.assert_allclose(
        decision["p_holm"],
        original["holm"]["p_holm"],
        rtol=0,
        atol=1e-12,
    )

    assert (
        decision["reject_at_0_05"]
        == original["holm"]["reject_at_0_05"]
    )

    ci = result[
        "bootstrap"
    ]["ci_percentile_95"]

    result["holm"] = decision

    print(f"\nCONTRAST={name}")
    print(
        "QLIKE_GAIN="
        f"{result['dm_hac']['mean_gain']:.8f}"
    )
    print(
        "P_HOLM="
        f"{decision['p_holm']:.8f}"
    )
    print(
        "BOOTSTRAP_95_CI="
        f"[{ci[0]:.8f}, {ci[1]:.8f}]"
    )
    print(
        "REJECT_AT_0_05="
        f"{decision['reject_at_0_05']}"
    )

print("=== 4. DESCRIPTIVE YEAR SPLIT ===")

years = {}

for year, expected_n in ((2025, 365), (2026, 273)):
    mask = (
        pred.date_utc.dt.year.to_numpy() == year
    )

    assert mask.sum() == expected_n

    year_scores = {
        model: float(loss[model][mask].mean())
        for model in MODELS
    }

    years[str(year)] = {
        "n": expected_n,
        "mean_qlike": year_scores,
    }

    print(
        f"YEAR={year} N={expected_n} "
        f"HAR={year_scores['HAR_RV']:.8f} "
        f"GARCH={year_scores['GARCH']:.8f}"
    )

public = {
    "protocol": "HC-BTCVOL-001",
    "stage": "post_holdout_verified_results",
    "holdout_start": "2025-01-01",
    "holdout_end": "2026-09-30",
    "common_sample": 638,
    "frozen_models_commit": report[
        "frozen_models_commit"
    ],
    "source_data_sha256": report[
        "data_source_sha256"
    ],
    "original_final_report_sha256": digest(
        report_path
    ),
    "original_predictions_sha256": digest(
        pred_path
    ),
    "original_parameter_trace_sha256": digest(
        trace_path
    ),
    "models": scores,
    "primary_contrasts": comparisons,
    "descriptive_year_split": years,
    "holdout_evaluated": True,
    "model_reoptimization": False,
    "bootstrap_role": "supplementary_only",
    "post_holdout_audit": True,
}

DOCS.mkdir(exist_ok=True)

dest = DOCS / "final_results_audit_v1.json"
dest.write_text(
    json.dumps(public, indent=2) + "\n"
)

print()
print("=== FINAL RESEARCH AUDIT ===")
print("ORIGINAL_OUTPUTS_UNMODIFIED=YES")
print("QLIKE_REPRODUCTION=PASS")
print("DM_HAC_REPRODUCTION=PASS")
print("HOLM_REPRODUCTION=PASS")
print("BOOTSTRAP_REPRODUCTION=PASS")
print("COMMON_SAMPLE=638")
print("FINAL_RESULTS_AUDIT=PASS")
print(f"PUBLIC_AUDIT={dest}")
