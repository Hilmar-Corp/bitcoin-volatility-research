# Contributions and release controls

This repository records **one finished, preregistered research experiment**.
The frozen historical releases are research evidence, not a development
sandbox.

## Review requirements

1. Work on a topic branch and open a pull request. Do not force-push
   frozen tags or rewrite historical commits.
2. Run: python scripts/verify_release_integrity.py
3. Run: python -W error::DeprecationWarning -m pytest tests/ -q
4. Run: python scripts/verify_public_results.py
5. Document exactly which files and conclusions are affected. Any new
   robustness exercise, exclusion, estimator, hypothesis or re-analysis
   following the first-look holdout must be identified as **post-holdout**
   and must not replace a preregistered inferential decision.

Historical first-look outputs under results/, source/model freeze manifests,
and requirements-lock.txt must not be edited. Fixes to the reproduction
packaging or CI belong in new commits without mutating those artifacts.

Never upload the excluded raw or processed market-data files, secrets, API
keys, production service configuration, client NAVs or private credentials.

The automated workflow provides reproducibility checks, not external peer
review or an assurance opinion. Access to the private repository is managed
separately; a passing check does not make it public.
