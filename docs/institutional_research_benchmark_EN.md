# HC-BTCVOL-001 — Academic and Institutional Research Benchmark (EN)

**Status:** documentary research and publication specification, October 20, 2026.  
**Non-endorsement:** MIT, Cambridge, ENSAE, École polytechnique, BlackRock, Amundi, NYU, and regulatory bodies are independent *reference sources*, not sponsors, certifiers, reviewers or partners in this study. This document makes no change to the frozen first-look experiment.

## 1. Publication strategy

Deliver **two distinct editorial layers**, each in **English and French**:

- A transparent **academic working paper**, including equations, data-quality methodology, the precise econometric implementations, preregistered inference, uncertainties, literature and reproducible appendices.
- A short **institutional research note** translating *forecast precision* into narrowly justified model-risk and monitoring implications, without making portfolio-performance or live-deployment claims.

French and English editions must contain the same numerical scores, sample size, statistical decisions, source citations, figures and qualifications. Neither should be labeled as peer-reviewed or institutionally certified.

## 2. Verified institutional sourcebook

| Reference source | Official link | Editorial/technical lesson |
|---|---|---|
| MIT Sloan Finance research | https://mitsloan.mit.edu/faculty/academic-groups/finance/research | Explicit finance question, falsifiable empirical contribution and critical limitations |
| Cambridge Judge working papers | https://www.jbs.cam.ac.uk/faculty-research/publications/working-papers/ | A working paper is preliminary, separate from peer-reviewed publication |
| ENSAE Paris / CREST / École polytechnique | https://www.ensae.fr/recherche/centre-de-recherche-en-economie-et-statistique-crest | Formal statistical methods and empirical confrontation |
| École polytechnique / CMAP MathsFi | https://portail.polytechnique.edu/cmap/en/research/probability/mathsfi | Financial risk modeling, numerical optimization and market microstructure |
| NYU Stern V-Lab | https://vlab.stern.nyu.edu/about | Volatility forecasting and transparent risk measurement, distinguishing historical replay from live output |
| BlackRock Investment Institute | https://www.blackrock.com/institutions/en-global/institutional-insights/thought-leadership/capital-market-assumptions | Assumptions, provenance, uncertainty and fit-for-purpose interpretation |
| Amundi Investment Institute papers | https://research-center.amundi.com/publications/working-paper | Clear abstract, quantitative method, finance application and readable implications |
| Amundi sequential IV forecasting (2026) | https://research-center.amundi.com/article/modeling-and-forecasting-large-unbalanced-option-implied-volatility-surfaces | Example of benchmark-based *sequential* empirical research; option IV is **not** our BTC realized variance |
| Amundi / CREST DCC-GARCH (2016) | https://research-center.amundi.com/article/stationarity-dynamic-conditional-correlation-models | Research/asset-management interface; different multivariate model |
| US Federal Reserve SR 26-2 (April 17, 2026) | https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm | **Current** risk-proportionate development, validation, monitoring and governance benchmark; **not** a regulation certifying this paper |
| BIS BCBS stress testing principles (2018) | https://www.bis.org/publications/201810-guidelines-stress-testing-principles | Data completeness, methodology, governance, documented limitations |

**Regulatory version control:** SR 26-2 *replaced* SR 11-7. Avoid labeling the superseded 2011 guidance as the current supervisory benchmark. These US bank supervisory publications are informative comparisons, not proof of an Amundi/BlackRock acceptance requirement.

## 3. Primary academic references

1. Corsi (2009), *A Simple Approximate Long-Memory Model of Realized Volatility*, *Journal of Financial Econometrics*. https://doi.org/10.1093/jjfinec/nbp001 — Our **log-HAR** with *mean log-RV* regressors differs from arithmetic weekly/monthly aggregation in the original formulation.
2. Patton (2011), *Volatility forecast comparison using imperfect volatility proxies*, *Journal of Econometrics*. https://doi.org/10.1016/j.jeconom.2010.03.034 — Latent variance, imperfect RV proxies and QLIKE evaluation.
3. Diebold and Mariano (1995), *Comparing Predictive Accuracy*, *JBES*. https://doi.org/10.1080/07350015.1995.10524599 — Paired forecast-loss tests.
4. Newey and West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, *Econometrica*. https://doi.org/10.2307/1913610 — HAC.
5. Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*, *Scandinavian Journal of Statistics*. https://www.jstor.org/stable/4615733 — Two-hypothesis multiplicity adjustment.
6. Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*, *Journal of Econometrics*. https://doi.org/10.1016/0304-4076(86)90063-1 — GARCH.
7. Andersen, Bollerslev, Diebold and Labys (2003), *Modeling and Forecasting Realized Volatility*, *Econometrica*. https://doi.org/10.1111/1468-0262.00418 — Realized-variance research.
8. Duan (1983), *Smearing Estimate: A Nonparametric Retransformation Method*, *JASA*. https://doi.org/10.1080/01621459.1983.10478017 — Log-to-level retransformation.
9. *Volatility prediction comparison via robust volatility proxies: An empirical deviation perspective* (2024), *Journal of Econometrics*, 239(2), 105633. https://doi.org/10.1016/j.jeconom.2023.105633 — Directly relevant **Bitcoin** forecast comparison and sensitivity to proxy noise.

Journal studies are intellectual benchmarks, **not** directly comparable numeric baselines without reproduction on exactly the same market, time period, loss and forecast horizon.

## 4. Frozen, verifiable study facts

- Binance Spot BTCUSDT, 5-minute closes; January 1, 2020 through September 30, 2026, UTC.
- 82 source archives, 709,920 expected timestamp slots, 709,458 observations, 462 missing 5-minute timestamps, 16 RV-invalid calendar dates, no interpolation.
- Seven forecasting models: rolling daily variance, daily EWMA, ARCH(1), GARCH(1,1), rolling RV, EWMA-RV, and the implemented mean-log-RV **log-HAR-RV(1,7,30)**.
- 638 identical first-look final dates, Jan 1 2025–Sep 30 2026, with all model/hypothesis specifications frozen before evaluation.
- QLIKE loss: \`log(h_t) + RV_t / h_t\` (lower is better). Negative mean scores are admissible due to variance units; gain is **not** an investment return or a percentage.
- Preregistered HAR-RV vs EWMA-RV: mean QLIKE gain **+0.08827622**, Holm-adjusted **p=0.00514178**, supplemental original moving-block interval **[0.04149298, 0.15364046]**.
- Preregistered GARCH vs daily EWMA: mean gain **+0.03848423**, Holm-adjusted **p=0.16063413** (**not statistically significant**).
- Post-holdout extreme sensitivity: HAR advantage decreases to **+0.01913281** after trimming the largest 5% absolute paired differences; **no confirmatory significance claim** for trimmed data.
- 1,276 of 1,276 ARCH/GARCH second-solver SLSQP refits converged, zero numerical alerts; no external validation.

## 5. Scientific and institutional challenge checklist

| Question an institutional reviewer should ask | Evidence today | Reporting requirement |
|---|---|---|
| What was frozen before first look? | Versioned protocol, model and runner tags | Time-stamped release lineage |
| Could future targets leak into each prediction? | Walk-forward model code and unit tests | State the information set and strict lag semantics |
| How were incomplete source days handled? | Manifest and data-quality rules | Full missingness and candle boundary table |
| Are the forecasts directly comparable? | Identical 638 UTC dates | Seven-model score table + exactly two primary contrasts |
| Are confidence claims multiplicity-adjusted? | Predeclared DM-HAC lag=7; Holm over 2 tests | Full raw and adjusted p-values |
| Are post-holdout exclusions confirmatory? | **No** | Clearly isolated sensitivity section |
| Are model optimizer outputs stable? | Second SLSQP audit | State numerical tolerance and limits on global optimality |
| Can a clean clone reproduce published inference? | CI and released Parquets | Command and artifact SHA-256, machine/environment boundary |
| Can a clone regenerate the entire raw dataset? | **No**, ZIPs not shipped | Provenance manifest, access/redistribution caveats |
| Is the strategy profitable or traded live? | **Not tested here** | No P&L, alpha, portfolio Sharpe or execution claims |
| Has it passed external institutional review? | **No** | Do not claim certification or endorsement |

## 6. Matched paper architecture (English/French)

1. Cover, immutable release identifiers, authorship/ownership statement.
2. Abstract / Résumé — research question, method, matched sample, paired outcomes, central limitations.
3. Introduction — purpose and testable contribution.
4. Literature review — RV, HAR, GARCH, noise, Diebold–Mariano, robust proxies, Bitcoin.
5. Data and target construction — sources, 5-minute-to-daily RV, missingness policy, time-zone boundaries.
6. Seven models and estimation details — exact log-HAR feature formulas, QMLE starts, Duan correction.
7. Strict chronological out-of-sample evaluation — development, validation, first-look holdout, 730-day prior training.
8. Statistical inference — QLIKE, DM-HAC(7), Holm, supportive 10,000-draw block bootstrap.
9. Main findings — descriptive rankings and preregistered contrasts, including nonsignificant GARCH.
10. **Post-holdout exploratory** sensitivity and numerical audit.
11. Measurement error, regime changes, external validity, operational uses and non-uses.
12. Conclusion; references; appendices with math, parameter diagnostics, SHA-256 and instructions.

**Four separate deliverables:** full FR working paper, full EN working paper, short FR institutional research note, short EN institutional research note. All four share immutable numeric results and exactly the same methodological distinctions.

## 7. Publication acceptance gates

Release only if equations agree with the frozen implementation; every figure and table states units, source and time horizon; original and Holm-adjusted p-values are shown; post-holdout tests are distinctly marked; French and English numeric values match; raw external data are not mistakenly redistributed under Apache-2.0; and absence of external review is explicit.

**Never claim:** approved by BlackRock/Amundi/MIT, externally certified, HAR significantly beats all seven models, investment return generated, or full raw-market-data reproducibility from repository-only artifacts.

**This is a research benchmark/sourcebook, not the finished paper.**
