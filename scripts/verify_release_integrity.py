"""Read-only, standard-library audit of HC-BTCVOL-001 frozen artifacts.

This validates release integrity and provenance. It does not refit models
and cannot reproduce the excluded 5-minute raw/source dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEX_SHA = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise SystemExit(f"INTEGRITY_FAIL: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def digest(relative: str) -> str:
    path = ROOT / relative
    require(path.is_file(), f"missing file: {relative}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def check(relative: str, expected: str) -> None:
    require(isinstance(expected, str) and HEX_SHA.fullmatch(expected) is not None,
            f"invalid expected SHA-256: {relative}")
    actual = digest(relative)
    require(actual == expected, f"SHA-256 mismatch: {relative}")


def read_json(relative: str) -> dict:
    path = ROOT / relative
    require(path.is_file(), f"missing JSON: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {relative}")
    return value


def main() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (ROOT / "NOTICE").read_text(encoding="utf-8")
    require("Apache License" in license_text and
            "Version 2.0" in license_text, "Apache 2.0 license missing")
    require("Copyright 2026 HilmarCorp" in notice_text,
            "copyright attribution missing")
    print("APACHE_LICENSE_AND_NOTICE=PASS")

    frozen = read_json("docs/model_freeze_manifest_v1.json")
    require(frozen["release"] == "models-inference-v1.0.0",
            "unexpected frozen model release")
    require(frozen["holdout_evaluated"] is False,
            "model freeze chronology altered")
    locked = frozen["locked_file_sha256"]
    require(len(locked) >= 10, "incomplete model freeze manifest")
    for relative, expected in locked.items():
        check(relative, expected)
    print(f"FROZEN_MODEL_AND_INFERENCE_HASHES=PASS ({len(locked)} files)")

    runner = read_json("docs/holdout_runner_manifest_v1.json")
    require(runner["release"] == "holdout-runner-v1.0.0",
            "unexpected first-look runner release")
    check("scripts/run_final_holdout.py", runner["runner_sha256"])
    print("FIRST_LOOK_RUNNER_SHA256=PASS")

    data = read_json("docs/data_manifest_v1_1.json")
    require(data["protocol_id"] == "HC-BTCVOL-001", "protocol mismatch")
    require(data["data_policy"] == "v1.1", "data policy mismatch")
    check("configs/source_quality_reference_v1.json",
          data["source_reference_sha256"])
    archives = data["source_archives"]
    require(len(archives) == 82, "expected 82 source archives")
    months = [item["month"] for item in archives]
    require(len(months) == len(set(months)), "duplicate archive month")
    for item in archives:
        require(item["url"].startswith("https://data.binance.vision/"),
                "unexpected source archive origin")
        require(HEX_SHA.fullmatch(item["sha256"]) is not None,
                "invalid source archive SHA-256")
    print("DATA_MANIFEST_PROVENANCE=PASS (82 declared archives)")
    print("SOURCE_ARCHIVE_BYTES_VERIFIED=NO (source files excluded)")

    audit = read_json("docs/final_results_audit_v1.json")
    report = read_json("results/final_report.json")
    require(audit["common_sample"] == 638, "wrong holdout length")
    require(report["n"] == 638 and
            report["holdout_evaluated"] is True and
            report["model_reoptimization"] is False,
            "first-look report state mismatch")
    check("results/final_predictions.parquet",
          audit["original_predictions_sha256"])
    check("results/final_parameter_trace.parquet",
          audit["original_parameter_trace_sha256"])
    check("results/final_report.json",
          audit["original_final_report_sha256"])
    require(audit["original_final_report_sha256"] ==
            digest("results/final_report.json"),
            "final report changed")
    print("FIRST_LOOK_RESULTS_SHA256=PASS")

    post = read_json("docs/postholdout_diagnostics_v1.json")
    require(post["status"] == "POST_HOLDOUT_EXPLORATORY",
            "post-holdout classification missing")
    require(post["original_report_sha256"] ==
            audit["original_final_report_sha256"] and
            post["original_predictions_sha256"] ==
            audit["original_predictions_sha256"],
            "post-holdout study points to changed outputs")
    print("POST_HOLDOUT_DIAGNOSTICS_PROVENANCE=PASS")

    provenance = read_json(
        "results/second_solver_qmle_v1/provenance_manifest.json"
    )
    second = read_json("results/second_solver_qmle_v1/audit_report.json")
    require(provenance["release"] == "qmle-audit-v1.0.0",
            "unexpected QMLE audit version")
    require(second["attempts_total"] == 1276 and
            second["converged"] == 1276 and
            second["review_required"] == 0 and
            second["status"] == "PASS", "QMLE release metadata mismatch")
    require(second["original_forecasts_sha256"] ==
            audit["original_predictions_sha256"],
            "QMLE audit refers to different first-look forecasts")
    check("results/second_solver_qmle_v1/audit_report.json",
          provenance["report_sha256"])
    check("results/second_solver_qmle_v1/independent_qmle_refits.parquet",
          provenance["refits_sha256"])
    check("scripts/audit_second_solver_qmle.py",
          provenance["audit_script_sha256"])
    print("SECOND_SOLVER_QMLE_ARTIFACTS_SHA256=PASS")
    print("RESULTS_REFITTED=NO")
    print("RELEASE_INTEGRITY=PASS")


if __name__ == "__main__":
    main()
