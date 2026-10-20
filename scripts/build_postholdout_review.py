"""Post-holdout vector figures and academic review note."""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = ROOT / "figures"
RESULTS = ROOT / "results"

report = json.loads(
    (RESULTS / "final_report.json").read_text()
)
review = json.loads(
    (DOCS / "postholdout_diagnostics_v1.json").read_text()
)
pred = pd.read_parquet(
    RESULTS / "final_predictions.parquet"
)

assert review["status"] == "POST_HOLDOUT_EXPLORATORY"
assert len(pred) == 638

FIG.mkdir(exist_ok=True)


def text(x, y, value, size=13, color="#263547"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" '
        f'fill="{color}" font-size="{size}" '
        f'font-family="Arial,sans-serif">'
        f'{escape(str(value))}</text>'
    )


def line(x1, y1, x2, y2, color="#d7dee6", width=1):
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" '
        f'x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"/>'
    )


def save(name, body, width=950, height=530):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        + body + "</svg>\n"
    )
    (FIG / name).write_text(svg)


# Figure 1 — All models on the same 638 dates.

scores = {
    name: item["mean_qlike"]
    for name, item in report["scores"].items()
}

ranking = sorted(
    scores.items(), key=lambda item: item[1]
)

best = ranking[0][1]

body = text(
    35, 44, "Final holdout: excess QLIKE over best model", 21
)
body += text(
    35, 72, "BTCUSDT | 638 matched UTC days | lower is better"
)

for i, (name, score) in enumerate(ranking):
    y = 116 + i*48
    gap = score - best
    width = max(2, 550 * gap / 0.16)
    color = "#087e71" if i == 0 else "#43679c"

    body += text(35, y+17, name, 14)
    body += (
        f'<rect x="250" y="{y}" width="{width:.2f}" '
        f'height="23" rx="3" fill="{color}"/>'
    )
    body += text(815, y+17, f"{gap:+.5f}")

body += text(
    250, 490,
    "Bar length = additional mean loss relative to HAR-RV."
)

save("01_qlike_comparison.svg", body)


# Figure 2 — Cumulative daily loss differences.

target = pred.rv_target.to_numpy()

comparisons = [
    (
        "HAR-RV vs EWMA-RV",
        "I1_ewma_rv_094",
        "HAR_RV",
    ),
    (
        "GARCH vs EWMA",
        "D1_ewma_094",
        "GARCH",
    ),
]

curves = []

for title, benchmark, candidate in comparisons:
    b = pred[benchmark].to_numpy()
    c = pred[candidate].to_numpy()

    difference = (
        np.log(b) + target/b
        - np.log(c) - target/c
    )

    curves.append((
        title,
        np.r_[0, np.cumsum(difference)]
    ))

low = min(
    0, min(float(curve.min()) for _, curve in curves)
)
high = max(
    0, max(float(curve.max()) for _, curve in curves)
)

padding = max(1, (high-low)*0.1)
low -= padding
high += padding

left, right = 90, 870
top, bottom = 105, 440

def px(i):
    return left + (right-left)*i/638

def py(value):
    return bottom - (bottom-top)*(value-low)/(high-low)

body = text(
    35, 42, "Cumulative QLIKE gains — final holdout", 21
)
body += text(
    35, 69, "Positive values favour the candidate"
)

for k in range(6):
    value = low + (high-low)*k/5
    y = py(value)
    body += line(left, y, right, y)
    body += text(
        24, y+4, f"{value:.1f}", 12
    )

body += line(
    left, py(0), right, py(0), "#ae614f", 2
)

for i, (name, values) in enumerate(curves):
    color = ["#087e71", "#43679c"][i]

    points = " ".join(
        f"{px(j):.1f},{py(float(v)):.1f}"
        for j, v in enumerate(values)
    )

    body += (
        f'<polyline points="{points}" fill="none" '
        f'stroke="{color}" stroke-width="2.5"/>'
    )

    body += text(470, 84+i*20, name, 12, color)

body += text(
    90, 478, "Day 0 = 2025-01-01; day 638 = 2026-09-30"
)

save("02_cumulative_qlike_gains.svg", body)


# Figure 3 — Exploratory HAC bandwidth sensitivity.

grid = review["exploratory_hac_lag_grid"]
lags = [0, 1, 3, 7, 14, 21, 30]
names = list(report["primary_contrasts"])

maximum = max(
    0.2,
    max(
        grid[str(lag)][name]["p_holm"]
        for lag in lags
        for name in names
    ) * 1.1
)

left, right = 90, 860
top, bottom = 105, 425

def x_lag(i):
    return left + (right-left)*i/6

def y_p(p):
    return bottom - (bottom-top)*p/maximum

body = text(
    35, 43,
    "HAC lag sensitivity — exploratory post-holdout",
    20,
)
body += text(
    35, 70,
    "Holm-adjusted p-values; lag 7 was preregistered"
)

for k in range(6):
    p = maximum*k/5
    yp = y_p(p)
    body += line(left, yp, right, yp)
    body += text(25, yp+4, f"{p:.2f}", 12)

body += line(
    left, y_p(0.05), right, y_p(0.05),
    "#b15b48", 2
)

body += line(
    x_lag(3), top, x_lag(3), bottom,
    "#a5b0bc", 2
)

for i, lag in enumerate(lags):
    body += text(
        x_lag(i)-8, bottom+24, str(lag)
    )

for i, name in enumerate(names):
    color = ["#087e71", "#43679c"][i]

    values = [
        grid[str(lag)][name]["p_holm"]
        for lag in lags
    ]

    points = " ".join(
        f"{x_lag(j):.1f},{y_p(v):.1f}"
        for j, v in enumerate(values)
    )

    body += (
        f'<polyline points="{points}" fill="none" '
        f'stroke="{color}" stroke-width="2.5"/>'
    )

    body += text(
        380, 86+i*20, name.replace("_", " "),
        11, color,
    )

body += text(
    90, 480,
    "Horizontal reference: p = 0.05; vertical marker: lag 7"
)

save("03_hac_sensitivity.svg", body)


# Academic discussion generated from actual audit values.

contrast = report["primary_contrasts"]
har = contrast["HAR_RV_vs_EWMA_RV"]
garch = contrast["GARCH_vs_EWMA"]

def conclusion(name, value):
    dm = value["dm_hac"]
    return (
        f"| {name} | {dm['mean_gain']:+.8f} | "
        f"{value['holm']['p_holm']:.8f} | "
        f"{value['bootstrap']['ci_percentile_95']} |"
    )

ranking_rows = "\n".join(
    f"| {name} | {score:.8f} |"
    for name, score in ranking
)

primary_rows = "\n".join((
    conclusion("HAR-RV vs EWMA-RV", har),
    conclusion("GARCH vs EWMA", garch),
))

lag_rows = "\n".join(
    f"| {lag} | "
    f"{grid[str(lag)]['HAR_RV_vs_EWMA_RV']['p_holm']:.6f} | "
    f"{grid[str(lag)]['GARCH_vs_EWMA']['p_holm']:.6f} |"
    for lag in lags
)

block_rows = "\n".join(
    f"| {block} | "
    f"{review['exploratory_bootstrap_block_grid'][str(block)]['HAR_RV_vs_EWMA_RV']['ci_95']} | "
    f"{review['exploratory_bootstrap_block_grid'][str(block)]['GARCH_vs_EWMA']['ci_95']} |"
    for block in (7, 14, 21, 28)
)

year_rows = "\n".join(
    f"| {year} | {v['n']} | "
    f"{v['gains']['HAR_RV_vs_EWMA_RV']:+.6f} | "
    f"{v['gains']['GARCH_vs_EWMA']:+.6f} |"
    for year, v in review[
        "exploratory_year_split"
    ].items()
)

extreme_rows = "\n".join(
    f"| {name} | {criterion} | "
    f"{value['n']} | {value['gain']:+.6f} |"
    for name, entries in review[
        "exploratory_outlier_sensitivity"
    ].items()
    for criterion, value in entries.items()
)

document = f"""# HC-BTCVOL-001 — Independent Numerical Review

**Status:** post-holdout exploratory audit, not peer-reviewed.

**Instrument:** Binance Spot BTCUSDT.

**Study period:** 2020-01-01 to 2026-09-30 UTC.

**First-look holdout:** 638 daily observations, 2025–2026.

## 1. Confirmatory results

The original model and inference specifications were
frozen before the holdout was opened.

The loss is QLIKE: log(forecast) + RV/forecast.

Two-sided Diebold–Mariano-style comparisons use
Newey–West HAC with seven lags and Holm correction
across exactly two registered hypotheses.

| Primary comparison | Mean gain | Holm p | Bootstrap 95% CI |
|---|---:|---:|---|
{primary_rows}

HAR-RV improves QLIKE relative to EWMA-RV under the
preregistered confirmatory procedure.

GARCH's numerical improvement over daily EWMA is
not statistically significant at the familywise 5% level.

The supplementary bootstrap intervals are not the
confirmatory decision rule.

## 2. Model ranking

| Model | Mean QLIKE |
|---|---:|
{ranking_rows}

![Model comparison](../figures/01_qlike_comparison.svg)

The ranking is descriptive: no registered pairwise
significance test between HAR-RV and GARCH was performed.

## 3. Methodology and numerical verification

The dataset contains 82 source archives, 462 missing
five-minute observations, seven shortened candles,
and one candle with negative duration.

The 16 incomplete daily RV observations are retained
as missing, without price interpolation.

Daily RV equals the sum of 288 squared five-minute
log returns, with the preceding daily close as anchor.

The first-look holdout contains 638 valid RV targets.

An independently written implementation reproduced:

- all 638 daily RV targets from five-minute source closes;
- rolling variance and the two EWMA recursions;
- ARCH and GARCH forecasts from stored fitted parameters;
- all HAR OLS forecasts and training-only smearing;
- all seven QLIKE means and the two DM–HAC/Holm results.

**Independent-verification boundary:** this replay
does not separately solve and validate every GARCH QMLE
optimization problem with a second solver.
External peer review has not yet occurred.

## 4. Important model specification distinction

Our log-HAR specification uses yesterday's log RV,
the mean of the previous seven log RV observations,
and the mean of the previous thirty log RV observations.

It is not mathematically identical to taking the
logarithm of weekly and monthly arithmetic-average RV.

Consequently, it should be called the
**implemented log-HAR-RV variant**, not an exact
replication of the canonical Corsi specification.

The model uses 730 calendar days for the training
window, removes inadmissible training observations,
and requires at least 500 training rows.

The retransformation applies Duan smearing computed
from training residuals only.

## 5. Post-holdout statistical sensitivity

All alternative HAC bandwidths, bootstrap lengths,
year subdivisions and exclusions below are exploratory.

| HAC lag | HAR Holm p | GARCH Holm p |
|---:|---:|---:|
{lag_rows}

![HAC sensitivity](../figures/03_hac_sensitivity.svg)

| Bootstrap block days | HAR 95% CI | GARCH 95% CI |
|---:|---|---|
{block_rows}

These alternative confidence intervals use 2,000
bootstrap repetitions each. The original frozen
bootstrap used 14-day blocks and 10,000 draws.

| Year | Days | HAR mean gain | GARCH mean gain |
|---:|---:|---:|---:|
{year_rows}

![Cumulative gain](../figures/02_cumulative_qlike_gains.svg)

### Extreme-observation influence

The next means exclude the highest 1% or 5%
by RV, or by the absolute paired loss differential.

These subsamples were chosen post-holdout, and no
additional confirmatory p-values are assigned.

| Contrast | Exclusion | N | Mean gain |
|---|---|---:|---:|
{extreme_rows}

## 6. Scientific limitations

**Measurement risk.** Five-minute realized variance
is a proxy for latent conditional variance. Noise,
jumps and exchange microstructure can matter.

**External validity.** One venue, one instrument,
one holdout period. Results do not establish
performance in future or cross-venue market regimes.

**Model assumptions.** ARCH/GARCH use zero-mean
Gaussian QMLE. Heavy tails, leverage effects,
alternative forecast horizons and model families
have not been prospectively adjudicated.

**Inference.** HAC standard errors are asymptotic.
Alternative bandwidths and bootstrap lengths were
examined post-holdout and cannot alter the original
confirmatory interpretation.

**Methodological independence.** Internal numerical
replication is not an external academic review or
institutional model-risk certification.

**Economic relevance.** Statistical forecast
accuracy does not establish investment profitability.

## 7. Selected academic references

1. Andersen, Bollerslev, Diebold & Labys (2003),
   *Modeling and Forecasting Realized Volatility*,
   Econometrica.
   https://doi.org/10.1111/1468-0262.00418

2. Bollerslev (1986), *Generalized Autoregressive
   Conditional Heteroskedasticity*, Journal of
   Econometrics.
   https://doi.org/10.1016/0304-4076(86)90063-1

3. Corsi (2009), *A Simple Approximate Long-Memory
   Model of Realized Volatility*, Journal of
   Financial Econometrics.
   https://doi.org/10.1093/jjfinec/nbp001

4. Diebold & Mariano (1995), *Comparing Predictive
   Accuracy*, Journal of Business & Economic Statistics.
   https://doi.org/10.1080/07350015.1995.10524599

5. Duan (1983), *Smearing Estimate: A Nonparametric
   Retransformation Method*, JASA.
   https://doi.org/10.1080/01621459.1983.10478017

6. Holm (1979), *A Simple Sequentially Rejective
   Multiple Test Procedure*, Scandinavian Journal
   of Statistics.
   https://www.jstor.org/stable/4615733

7. Newey & West (1987), *A Simple, Positive
   Semi-Definite, Heteroskedasticity and
   Autocorrelation Consistent Covariance Matrix*,
   Econometrica.
   https://doi.org/10.2307/1913610

8. Patton (2011), *Volatility Forecast Comparison
   Using Imperfect Volatility Proxies*,
   Journal of Econometrics.
   https://doi.org/10.1016/j.jeconom.2010.03.034

## 8. Research integrity

This review is explicitly post-holdout.

The original frozen model specifications and
confirmatory results remain unchanged.

The analyses above are not an additional
preregistered experiment.

External replication and critical review remain
necessary before claiming institutional validation.
"""

(DOCS / "postholdout_scientific_review_v1.md").write_text(
    document
)

assert len(list(FIG.glob("*.svg"))) == 3

print()
print("=== SCIENTIFIC REVIEW PACKAGE ===")
print("FIGURES=3")
print("ACADEMIC_REFERENCES=8")
print("INDEPENDENT_METHOD_REVIEW=DOCUMENTED")
print("LIMITATIONS=DOCUMENTED")
print("POST_HOLDOUT_ANALYSES=LABELED")
print("ORIGINAL_MODELS_MODIFIED=NO")
print("SCIENTIFIC_REVIEW_PACKAGE=PASS")
