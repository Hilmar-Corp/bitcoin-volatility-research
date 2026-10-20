# Reproducibility and research integrity

**Study:** HC-BTCVOL-001 · Binance BTCUSDT spot volatility · 2020-01-01 to
2026-09-30 UTC. **Status:** internally validated research, not peer reviewed.
The repository is currently **private**; sharing this link is not the same
as granting external access or publishing an openly reproducible dataset.

## Scope of what another researcher can reproduce

| Layer | Repository-only reproduction? | Method |
| --- | --- | --- |
| Original model and inference code SHA-256 | Yes | verify_release_integrity.py |
| Seven first-look model QLIKE scores (638 UTC days) | Yes | verify_public_results.py |
| Two preregistered DM-HAC tests and Holm p-values | Yes | verify_public_results.py |
| Original 10,000-replication, 14-day-block bootstrap | Yes | verify_public_results.py |
| Original first-look output-byte integrity | Yes | verify_release_integrity.py |
| 1,276 second-solver QMLE output-byte integrity | Yes | verify_release_integrity.py |
| Reconstruct 5-minute BTC prices / 82 Binance ZIPs | **No** | Obtain archives listed in data manifest; verify each SHA-256 |
| Refit all seven models from original observations | **No** | Needs excluded processed source datasets |
| Independently rerun all QMLE refits | **No** | Needs excluded quality-controlled daily-return Parquet |
| External peer review / institutional certification | **No** | Requires independent review |

The original **first-look evaluation must not be re-run in a way that
rewrites its marker or published outputs**. The scripts below only check
existing results. The baseline result is permanently identified by tag
final-results-v1.0.0; follow-on work is clearly labeled post-holdout.

## Reproduce published results from a clean clone

Use an x86_64 or arm64 environment capable of running Python 3.12.
CI uses ubuntu-24.04 / Python 3.12. For an exact published-code snapshot:

    git clone https://github.com/Hilmar-Corp/bitcoin-volatility-research.git
    cd bitcoin-volatility-research
    git checkout qmle-audit-v1.0.0
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements-lock.txt
    python -m pip check
    python -W error::DeprecationWarning -m pytest tests/ -q
    python scripts/verify_public_results.py

For the repository with the additional reproducibility safeguards and
CI verification, check out the reviewed merge commit or its future tag,
and also run:

    python scripts/verify_release_integrity.py

The latter validates SHA-256 hashes of frozen source files and available
released artifacts, including the second-solver audit. All verification
commands are read-only. Running tests does not evaluate a fresh holdout.

Expected original confirmatory findings: HAR-RV vs EWMA-RV mean QLIKE
gain +0.08827622, Holm-adjusted p=0.00514178; GARCH vs daily EWMA mean
gain +0.03848423, Holm-adjusted p=0.16063413. The first is significant
under the **frozen** two-test protocol; the second is not.

## Environment and precision boundaries

- requirements-lock.txt fixes the exact *Python package versions* from
  the frozen original release. It does **not** pin distribution artifact
  hashes (wheels/sdists), operating-system libraries, native compiler
  versions, BLAS, CPU architecture, or a container image digest.
- GitHub Actions pins each action to an immutable full commit SHA and
  uses a stated runner image. Dependency resolution still relies on a
  reachable package index. The run is neither hermetic nor guaranteed
  bit-for-bit identical across different hardware.
- The public replay script checks the published numerical outputs with
  explicit precision tolerances; CI success is evidence for the tested
  environment only.
- If an upstream wheel disappears, record the environment failure
  separately; do not change the frozen requirements-lock.txt to make
  an old result appear to reproduce.

## Data provenance and licensing

The original 82 monthly Binance five-minute archives are *not* included.
See docs/data_manifest_v1_1.json (archive URLs and source SHA-256),
docs/data_quality_policy_v1.1.md, and docs/research_findings_v1.md.
The manifest confirms declared provenance, **not** that the missing
archive bytes have been independently rechecked in CI.

The repository contains Apache-2.0-licensed research code and a NOTICE.
This license does **not** license third-party Binance data, the Binance
trademarks, or any data that the research organization does not own.
The numerical findings are research outputs, not audited investment
performance or investment advice.

## Frozen history and changes

| Milestone | Git tag |
| --- | --- |
| Original preregistration | protocol-v1.0.0 |
| Source-data quality amendment | data-policy-v1.1.0 |
| Model and inference freeze | models-inference-v1.0.0 |
| Locked first-look execution | holdout-runner-v1.0.0 |
| First-look final results | final-results-v1.0.0 |
| Post-holdout sensitivity | postholdout-review-v1.0.0 |
| Second-solver numerical audit | qmle-audit-v1.0.0 |

Do not rewrite or retag the historical releases. Scientific extensions,
different estimators, new metrics or data corrections require separately
versioned analyses labeled *post-holdout* where appropriate. The original
confirmation cannot be strengthened by trying alternatives on the same
already-opened test sample.

See CONTRIBUTING.md for review and change-control rules.
