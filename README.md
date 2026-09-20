# Frequency-Severity Insurance Claim Risk Model

A pricing-style risk model for French motor third-party liability (MTPL) insurance,
built the way an actuarial/pricing team would actually decompose the problem:
separately model *how often* a policy claims and *how much* a claim costs, then
combine the two into an expected loss per policy. This is the standard structure
behind real insurance pricing, as opposed to training a single model to predict
total loss directly (which gets dominated by the zero-inflation of "most policies
never claim" and loses the ability to reason about frequency and severity as
distinct risk drivers).

## The problem, in plain English

An insurer needs to price a policy before it's written, which means estimating
`E[loss] = P(claim occurs) × E[claim size | a claim occurs]` for a policy it hasn't
seen claims from yet. Getting `P(claim)` wrong misprices risk selection (who to
insure and at what base rate); getting `E[severity]` wrong misprices the size of the
bet once a claim happens. The two questions have almost nothing in common
statistically — whether *any* claim happens is dominated by exposure and rare-event
class imbalance, while *how expensive* it is once it happens is dominated by a
long-tailed cost distribution — so they're modeled as two separate problems and
only multiplied together at the end.

## Data

[freMTPL2freq](https://www.openml.org/d/41214) (678,013 policy-years: exposure,
vehicle/driver characteristics, claim count) joined with
[freMTPL2sev](https://www.openml.org/d/41215) (26,639 individual claim records) on
`IDpol`, both fetched live from OpenML via `sklearn.datasets.fetch_openml`.

### Cleaning decisions (see `src/data/merge_clean.py` for the code-level reasoning)

- **Severity aggregated by summing, not taking max/first, per IDpol.** A policy can
  have multiple claim rows (up to 5 observed); the sum is the correct total-loss
  counterpart to `ClaimNb`, which already counts claims per policy.
- **Exposure capped at 1.0, floored at ~1 day.** A policy-year can't exceed one year
  of exposure by definition — 1,224 rows exceeded 1.0 in the raw data (up to 2.01),
  a data-entry artifact, not a real quantity. 3,105 rows had near-zero exposure,
  floored to avoid divide-by-zero in exposure-normalized features.
- **ClaimNb capped at 4.** Only 9 rows exceeded this (up to 16), implausible for a
  single policy-year. Matches the cap used in sklearn's own reference treatment of
  this dataset, keeping results comparable.
- **A real data quirk, documented rather than silently patched:** 9,116 of 34,060
  policies with `ClaimNb > 0` (~27%) have *no matching severity record at all* —
  verified by direct IDpol-overlap inspection, not a join bug. This is a known
  artifact of how freMTPL2freq and freMTPL2sev were extracted from the source
  system. It matters downstream: the severity model can only be trained on the 73%
  of claiming policies with a real payout, and the combination step has to
  explicitly correct for this mismatch (see "Combining into expected loss" below) or
  it silently overstates expected loss by ~50%.

## EDA — what the plots show

(Figures in `reports/figures/`.)

- **Claim frequency** (`claim_frequency_distribution.png`): ~95% of policy-years
  have zero claims; overall claim rate is 5.02%. This single fact drives most of the
  frequency-model design choices below (`scale_pos_weight`, PR-AUC over accuracy).
- **Severity** (`severity_distribution.png`): raw severity has skew ≈ 106 — a
  handful of claims run into the millions. `log1p` brings skew to about -0.5, close
  enough to symmetric to justify squared-error loss on the log scale.
- **Exposure** (`exposure_distribution.png`): bimodal — a spike at 1.0 (full-year
  policies) plus a roughly uniform spread below it (new policies, mid-year
  cancellations).
- **Correlations** (`correlation_heatmap.png`): every linear correlation with
  `HasClaim`/`ClaimAmountTotal` is weak (|r| < 0.08). This isn't a red flag — claim
  risk here is driven by interactions (a young driver with a high-power car in a
  dense area) that a correlation matrix can't surface, which is the actual
  motivation for using gradient-boosted trees over a linear GLM baseline.

## Feature engineering

Categoricals are one-hot encoded below a cardinality threshold of 25 (everything in
this dataset qualifies) and would route to `sklearn.preprocessing.TargetEncoder`
above it — implemented generically even though nothing here exercises that branch,
so the code doesn't silently break if a higher-cardinality categorical shows up.

Three derived features, each justified because none is a standard off-the-shelf
feature for this dataset:

1. **`log_density = log1p(Density)`** — Density spans ~1 to ~27,000 and is heavily
   right-skewed; log-transforming keeps a handful of extreme-density regions from
   dominating tree splits.
2. **`vehage_drivage_ratio = VehAge / (DrivAge + 1)`** — a young driver in a very old
   car and an experienced driver in a new one are different risk profiles that
   neither raw feature alone conveys.
3. **`bonus_malus_per_year_licensed = BonusMalus / max(DrivAge - 17, 1)`** — this
   dataset is cross-sectional (one row per policy-period, no repeated-IDpol
   timeline), so there's no way to build genuine "claims in the last N years"
   history features. `BonusMalus` (France's no-claims bonus/malus score) is the only
   field that embeds real driving history, so normalizing it by an approximate
   years-licensed-since-18 proxy flags drivers whose malus is unusually high/low
   *given* plausible experience, rather than raw magnitude alone. This is an
   explicit approximation (assumes licensing age 18), not a measured value.

## Frequency model (XGBClassifier)

**Target: binary `HasClaim = (ClaimNb > 0)`**, not a Poisson count. A count model is
more standard actuarial practice in general, but the expected-loss combination this
project builds toward needs a probability, and PR-AUC/calibration (the specified
evaluation approach) are only meaningful for a binary target — the binary
formulation matches that architecture, not because it's the more "correct" choice
in isolation.

**Imbalance:** `scale_pos_weight = n_negative / n_positive`, computed on the
**train split only** (18.91), passed directly to `XGBClassifier`. No
SMOTE/undersampling — this is the simpler, leakage-safe, XGBoost-native mechanism.

**Tuning:** a plain loop over 4 hand-picked combinations of
`max_depth`/`learning_rate`/`subsample`/`colsample_bytree`, each fit with
`early_stopping_rounds=30` against a held-out validation set, selected by
validation PR-AUC (not `GridSearchCV`, since its fit/predict wrapper makes it
awkward to pass a per-fit `eval_set` for early stopping — a plain loop keeps that
explicit for four fits total). Best config: `max_depth=5, learning_rate=0.1,
subsample=0.8, colsample_bytree=1.0`, stopped at iteration 252.

**Results on test:**

| Metric | Value |
|---|---|
| PR-AUC | **0.139** |
| No-skill baseline PR-AUC (= positive rate) | 0.050 |
| Test claim rate | 5.02% |

A PR-AUC of 0.139 against a 0.050 baseline is roughly 2.75x lift — a modest but real
signal, consistent with how hard this dataset is known to be (claim occurrence in
motor insurance is dominated by chance, not policy characteristics).

**A finding worth calling out explicitly:** `scale_pos_weight` makes the raw score
well-*ranked* but badly *miscalibrated* in absolute terms — mean raw predicted
probability came out at **0.426**, roughly 8.5x the actual 5.02% test claim rate.
This is expected (it's what reweighting the loss function does), but it would make
`expected_loss = P(claim) × E[severity]` wildly overstate aggregate loss if used
directly. An isotonic regression calibrator, **fit on the validation set** (never
train, to avoid overfitting the calibration map to the same rows the tree was fit
on), brings the mean calibrated probability to **0.0503** — within 0.1% of the true
5.02% test rate. See `reports/figures/frequency_calibration.png` for the raw-vs-
calibrated comparison; the raw curve sits far below the diagonal (well-ranked, badly
calibrated) while the calibrated curve hugs it closely.

## Severity model (XGBRegressor)

**Trained only on rows with a confirmed claim payout** (`ClaimAmountTotal > 0`,
n=17,417 train / 3,755 val / 3,772 test — smaller than the frequency splits because
of the 27% freq/sev mismatch noted above).

**Target: `log1p(ClaimAmountTotal)`, fit with `reg:squarederror` — not
`reg:tweedie`.** Tweedie is the right choice when a *single* model must jointly
handle the zero-mass and the continuous positive tail (i.e. modeling pure premium
directly, without decomposing into frequency × severity). This pipeline already
separates the two, so the zero-inflation problem Tweedie solves has already been
handled by the frequency classifier — applying Tweedie to a zero-free subset would
be redundant, since its power parameter is tuned for a zero-inflated distribution
that no longer exists in this filtered training subset. The EDA's skew numbers
(106 → -0.5 after log1p) support squared-error loss on the log scale as a
reasonable, standard fit objective.

**Results on test (original EUR scale, after undoing the log transform):**

| Metric | Overall | Top 10% of claims (≥ €3,164) |
|---|---|---|
| RMSE | €10,811 | €33,838 |
| MAE | €2,302 | €10,378 |

The large gap between overall and top-10% error is expected, not a flaw — the tail
is inherently harder to predict precisely (a €50k claim and a €150k claim look
similar on the input features but are wildly different in cost), and this is
exactly why the evaluation reports the tail separately rather than letting one
blended RMSE hide it.

**A correction that mattered:** naively back-transforming with
`expm1(predicted_log_mean)` is a biased (systematically low) estimator of the true
mean under log-skewed residuals, by Jensen's inequality. Duan's smearing estimator
(`mean(exp(residuals))`, computed on the **validation** set) corrects for this — the
smearing factor came out to **2.438**, moving mean predicted severity on test from
44% of the actual mean to 108% of it. Left uncorrected, this bias would have made
the aggregate-loss validation below look badly wrong when the underlying ranking was
actually fine.

## Combining into expected loss

```
expected_loss = P(claim) × P(real payout | claim) × E[severity | payout]
              = calibrated_frequency_prob × payout_given_claim_rate × smeared_severity_pred
```

The middle term is not in the original formula — it was added after the first
end-to-end run produced a predicted-to-actual aggregate loss ratio of **1.51**,
which was too large to hand-wave away. Diagnosis: the frequency model's target
(`ClaimNb > 0`) and the severity model's training population (`ClaimAmountTotal >
0`) disagree on the same 27% of policies flagged during data cleaning — the
frequency model correctly predicts "a claim was counted," but the severity model,
trained only on payouts that actually exist, implicitly assumes every predicted
claim has a real cost attached. Multiplying the two directly overstates expected
loss by roughly `1 / P(real payout | claim)`.

The fix: `payout_given_claim_rate = P(ClaimAmountTotal > 0 | ClaimNb > 0)`,
estimated on the **train split only** (0.7305 — i.e. 73% of counted claims have a
real payout on record), multiplied into the combination. This brought the ratio down
to **1.10**:

| | Value |
|---|---|
| Total predicted expected loss (test) | €9,410,327 |
| Total actual loss (test) | €8,525,609 |
| Ratio | **1.10** |

The remaining 10% overshoot is consistent with the severity model's smearing
correction slightly overshooting on this particular test split (108% rather than
exactly 100%, see above) — a reasonable residual given the corrections involved, not
a sign of a further hidden bug.

The decile lift chart (`reports/figures/decile_lift_chart.png`) shows the model
correctly separates the highest-risk decile (mean predicted €312 vs actual €205 per
policy) from the rest, with expected noise in the middle deciles — any individual
policy's claim outcome is dominated by chance, not by its risk factors, so decile-
level aggregation is where the signal becomes visible.

## SHAP: what drives frequency vs. severity

Two separate `shap.TreeExplainer` instances (one per model, never shared), each run
on a fixed 5,000-row sample of the test set for speed.

| Frequency — top 5 by mean |SHAP| | Severity — top 5 by mean |SHAP| |
|---|---|
| Exposure (0.431) | BonusMalus (0.099) |
| BonusMalus (0.233) | Exposure (0.091) |
| VehAge (0.136) | DrivAge (0.048) |
| vehage_drivage_ratio (0.094) | vehage_drivage_ratio (0.037) |
| DrivAge (0.089) | VehBrand_B12 (0.023) |

Both models converge on the same two dominant features — `Exposure` and
`BonusMalus` — but their relative weight flips: Exposure dominates frequency by
roughly 2x over BonusMalus, while for severity BonusMalus edges out Exposure. This
makes actuarial sense: more time on the road means more *opportunity* to have an
accident (frequency), but *how bad* an accident is once it happens is more about
the driver's underlying risk profile (BonusMalus) than how long they've been
exposed.

A more surprising finding: high `Exposure` pushes frequency **up** but pushes
severity **down** (visible in `reports/figures/severity_shap_summary.png` — high
Exposure values, in red, sit on the negative-SHAP side). One plausible actuarial
story: full-year policies may represent a more stable, lower-turnover population,
while partial-year policies (new policies, mid-term switches) may skew toward
riskier late-onset or non-renewal-driven claims. This is exactly the kind of
frequency/severity divergence that a single combined model would hide, and is a
direct payoff of decomposing the problem into two separate SHAP analyses.

`VehBrand_B12` cracking the severity top-5 (but not frequency's) also suggests a
genuine brand-level repair-cost difference rather than a claim-likelihood
difference — the kind of thing a pricing team would want to investigate further
before trusting it in production.

## API

FastAPI service in `/api`, loading all model artifacts once at startup:

```bash
source .venv/bin/activate
uvicorn api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Exposure": 0.8, "VehPower": 7, "VehAge": 5, "DrivAge": 35,
    "BonusMalus": 60, "VehBrand": "B12", "VehGas": "Diesel",
    "Density": 1200, "Region": "R82", "Area": "D"
  }'
```

returns probability of claim, expected severity, expected loss, and the top-5 SHAP
feature contributions for each model. The request transform reuses
`src/features/engineer.py` directly (rather than reimplementing it in the API
layer) specifically to avoid train/serve skew.

### Docker

```bash
docker build -t claim-risk-api .
docker run -p 8000:8000 claim-risk-api
```

`python:3.11-slim` matches the pinned dev interpreter; `libgomp1` is installed
explicitly since xgboost's shared library needs OpenMP at runtime and the slim
image doesn't ship it by default (on macOS dev, the equivalent is `brew install
libomp`).

## Reproducing this from scratch

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.models.train_frequency   # fetches/cleans data, trains + calibrates frequency model
python -m src.models.train_severity    # trains severity model on claiming rows only
python -m src.models.train_combine     # combines into expected loss, validates aggregate
python -m src.interpret.run_shap       # SHAP summaries for both models

pytest tests/                          # unit tests, no network/data dependency
```

## What I'd do with more time

- **A Tweedie GLM baseline.** Right now the "why not Tweedie" argument is made from
  first principles; a real side-by-side comparison against a Tweedie-distributed
  single-model baseline would make the frequency/severity decomposition's benefit
  (or lack thereof) empirical rather than argued.
- **Recalibrate scale_pos_weight vs. class-weight alternatives.** Isotonic
  calibration fixes the symptom (miscalibrated probabilities); it'd be worth
  comparing against training without `scale_pos_weight` at all and calibrating from
  scratch, to see whether the reweighting is even buying anything once calibration
  is applied anyway.
- **Investigate the freq/sev extraction mismatch further.** The 27%
  claim-without-payout rate is treated here as an unavoidable data limitation, but
  it's large enough that in a real production setting I'd want to understand
  *why* — is it systematic to certain claim types, time periods, or regions? — rather
  than applying a single global correction factor.
- **Optuna or a wider hyperparameter search.** The 4-point manual grid was a
  deliberate simplicity choice for a solo project; a proper Bayesian search would
  likely find a better operating point, especially for the severity model, whose
  validation RMSE barely moved across the grid (1.1268–1.1270 in log space) —
  suggesting either the search space was too narrow or the ceiling on this data,
  given its inherent noise, has effectively been reached.
- **Monitoring/drift detection for a production deployment.** This project validates
  against a static test split; a deployed pricing model needs ongoing calibration
  monitoring, since the isotonic calibrator and smearing factor are both fit against
  a fixed historical distribution that will drift.
- **Group-aware cross-validation if the data ever gets a real time dimension.** The
  current split is a simple stratified random split, justified because each row is
  already a unique policy-period with no repeated-IDpol structure — but if this were
  extended to genuinely longitudinal data (the same policyholder across multiple
  years), a time-based or group-based split would become necessary to avoid leakage.
