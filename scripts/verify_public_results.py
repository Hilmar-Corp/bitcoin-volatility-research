"""Reproduce published HC-BTCVOL-001 inference.

Requires published forecast artifacts, but does not
require the raw five-minute dataset or refitting.
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

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

base = ROOT / "results"

audit = json.loads(
    (ROOT / "docs/final_results_audit_v1.json").read_text()
)

report = json.loads(
    (base / "final_report.json").read_text()
)

spec = json.loads(
    (ROOT / "configs/inference_v1.json").read_text()
)

assert digest(base / "final_report.json") == (
    audit["original_final_report_sha256"]
)
assert digest(base / "final_predictions.parquet") == (
    audit["original_predictions_sha256"]
)
assert digest(base / "final_parameter_trace.parquet") == (
    audit["original_parameter_trace_sha256"]
)

pred = pd.read_parquet(
    base / "final_predictions.parquet"
)

trace = pd.read_parquet(
    base / "final_parameter_trace.parquet"
)

dates = pd.date_range(
    "2025-01-01",
    "2026-10-01",
    freq="D",
    inclusive="left",
    tz="UTC",
)

assert len(pred) == 638
assert len(trace) == 1914
assert pd.DatetimeIndex(pred.date_utc).equals(dates)
assert not trace.duplicated(
    ["date_utc", "model"]
).any()

models = spec["models"]
y = pred.rv_target.to_numpy()

losses = {}

for model in models:
    loss = qlike_losses(
        y,
        pred[model].to_numpy()
    )

    loss_mean = float(loss.mean())
    expected = audit["models"][model]

    np.testing.assert_allclose(
        loss_mean, expected,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        loss_mean,
        report["scores"][model]["mean_qlike"],
        rtol=0, atol=1e-12
    )

    losses[model] = loss

p_values = {}

for name, contrast in spec[
    "primary_contrasts"
].items():

    differences = (
        losses[contrast["benchmark"]]
        - losses[contrast["candidate"]]
    )

    actual = hac_mean_test(
        differences,
        lags=spec["hac_lags"]
    )

    recorded = audit[
        "primary_contrasts"
    ][name]["dm_hac"]

    for key in (
        "mean_gain",
        "hac_standard_error",
        "z",
        "p_value_two_sided",
    ):
        np.testing.assert_allclose(
            actual[key],
            recorded[key],
            atol=1e-12,
            rtol=0,
        )

    bootstrap = spec[
        "supplementary_bootstrap"
    ]

    interval = moving_block_ci(
        differences,
        block=bootstrap["block_length_days"],
        replications=bootstrap["replications"],
        seed=bootstrap["random_seed"],
    )

    np.testing.assert_allclose(
        interval["ci_percentile_95"],
        audit["primary_contrasts"][
            name
        ]["bootstrap"]["ci_percentile_95"],
        atol=1e-12,
        rtol=0,
    )

    p_values[name] = actual[
        "p_value_two_sided"
    ]

adjusted = holm_adjust(
    p_values,
    alpha=spec["familywise_alpha"],
)

for name, result in adjusted.items():
    expected = audit[
        "primary_contrasts"
    ][name]["holm"]

    np.testing.assert_allclose(
        result["p_holm"],
        expected["p_holm"],
        rtol=0, atol=1e-12
    )

    assert result["reject_at_0_05"] == (
        expected["reject_at_0_05"]
    )

print("=== PUBLIC REPRODUCIBILITY ===")
print("ORIGINAL_ARTIFACT_HASHES=PASS")
print("QLIKE_REPRODUCTION=PASS")
print("DM_HAC_REPRODUCTION=PASS")
print("HOLM_REPRODUCTION=PASS")
print("BOOTSTRAP_REPRODUCTION=PASS")
print("COMMON_SAMPLE=638")
print("PUBLIC_RESULT_REPLAY=PASS")
