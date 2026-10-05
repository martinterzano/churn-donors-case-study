# Donor Churn Prediction: Recurring Donors Case Study

> Monthly classification model that scores an NGO's active recurring donors by
> their probability of missing the next monthly gift. Random Forest with isotonic
> calibration, validated across 13 quarterly walk-forward folds over a four-year
> history, and operationalised as a four-segment retention playbook. Shipped as a
> scoring notebook feeding BigQuery for downstream retention outreach. One of
> two branches of a shared scoping phase. The other branch, propensity for a
> second gift from one-time donors, did not reach production (see §12).

**Sector:** non-profit · **Role:** end-to-end (data, feature engineering, modelling, validation, operational segmentation, handoff) · **Stack:** Python, scikit-learn, Random Forest, BigQuery, GCP · **Cadence:** monthly scoring · **Status:** validated, scoring notebook operational, handed off for production deployment

*Writeup prepared October 2026; the implementation is client property.*

The client owns the exact business metrics and internal identifiers. This case
study describes how the problem was framed, how the model was built, and how
the system reasons, without publishing client data, feature names, or raw
performance numbers. Every metric below is expressed as a range or order of
magnitude.

---

## Contents

1. [The problem](#1-the-problem)
2. [Scoping: recurring donors as a cleanly-labelled subset](#2-scoping-recurring-donors-as-a-cleanly-labelled-subset)
3. [Engineering architecture and shared pipeline](#3-engineering-architecture-and-shared-pipeline)
4. [Validation strategy: walk-forward over quarterly snapshots](#4-validation-strategy-walk-forward-over-quarterly-snapshots)
5. [The experiment grid: what we varied and why](#5-the-experiment-grid-what-we-varied-and-why)
6. [Why `active_threshold_days = 90` is the pivot parameter](#6-why-active_threshold_days--90-is-the-pivot-parameter)
7. [Model choice: Random Forest, and why not Logistic Regression](#7-model-choice-random-forest-and-why-not-logistic-regression)
8. [Probability calibration: isotonic over Platt](#8-probability-calibration-isotonic-over-platt)
9. [Results and operational segmentation](#9-results-and-operational-segmentation)
10. [Where the model actually adds value](#10-where-the-model-actually-adds-value)
11. [Known limitations and explicit gap list](#11-known-limitations-and-explicit-gap-list)
12. [Related work: the one-time donors branch](#12-related-work-the-one-time-donors-branch)
13. [Scoping decisions and repository contents](#13-scoping-decisions-and-repository-contents)
14. [What this case study is meant to show](#14-what-this-case-study-is-meant-to-show)

---

## 1. The problem

An NGO relies on recurring donors (people who set up an automated monthly
contribution) for predictable revenue. Every month a small share of those
donors disappear quietly, and the loss is immediate and compounding. One
missed gift is the start of a stream of missed gifts, not an isolated event.

Before this project, the client's retention workflow was running in two modes.
Reactive outreach after a donor had already lapsed, and broad communication to
the entire recurring base regardless of actual risk. Reactive outreach has low
conversion by the time it happens. Broad outreach dilutes the retention team's
capacity across donors who are not actually at risk.

The question the model was designed to answer. For each active recurring donor,
what is the probability that they will miss their next monthly gift, so the
retention team can prioritise outreach by risk?

The deliverable is a scored list refreshed monthly, with donors sorted into
operational segments that each have a documented retention action (see §9).

## 2. Scoping: recurring donors as a cleanly-labelled subset

The project started from a broader engagement with the client around donor
reengagement. During scoping, two distinct generative processes for
"donation events" were identified in the data, and the engagement bifurcated
into two parallel projects.

- **One-time donors making a voluntary second gift.** Driven by external stimulus (campaigns, cause salience, direct asks). Addressed as a separate propensity project. See §12.
- **Recurring donors fulfilling or breaking a prior monthly pledge.** Driven by internal circumstances (payment capacity, pledge change, life events). This project.

Keeping the two populations in separate models was not optional. A single
model trained on both would learn to discriminate across two different causal
stories at the same time, and the resulting ranking would be operationally
incoherent for both the retention team (which handles recurring donors) and
the acquisition team (which handles one-time donors).

The scoping decision also shaped the label. Because recurring donors give at a
near-monthly cadence (around 97% of the gifts observed in the training data
were monthly), the label for this project is binary and clean. Did the next
monthly gift arrive within the expected window, or not? The subscription-like
rhythm is what makes the churn signal detectable in the first place.

## 3. Engineering architecture and shared pipeline

The experiment grid in §5 ran on top of an infrastructure that was deliberately
designed before the first experiment. This part of the project predates the
model and outlasts the handoff: a modular sklearn `Pipeline` inherited from
the earlier propensity project (see §12), extended with a project-specific
config file and an experiment logger wired to BigQuery. The grid is only
cheap to run because the infrastructure is cheap to extend.

### 3.1 Modular sklearn Pipeline reused from the propensity project

The training pipeline is composed as a sklearn `Pipeline` with five stages,
each a separate module.

```
Pipeline([
    ('mapper',           FeatureMapper()),       # categorical normalisation
    ('feature_engineer', ChurnFeatureEngineer()),# churn-specific derived features
    ('outlier_clipper',  OutlierClipper()),      # per-column quantile clipping
    ('preprocess',       build_preprocessor()),  # impute + scale + one-hot
    ('model',            get_model(name)),       # LR / RF / XGBoost by name
])
```

Four of the five stages are **inherited verbatim** from the propensity
project (`FeatureMapper`, `OutlierClipper`, `build_preprocessor`,
`get_model`). One is **specialised**: `ChurnFeatureEngineer` replaces the
propensity-era `FeatureEngineer` because the derived features that matter for
churn (recency relative to historical cadence, gap patterns, recent-window
counts) are a different catalogue from the propensity-era features (log and
bucket of the first-gift amount, timing-within-year flags).

This split is deliberate. Shared infrastructure where behaviour is identical
across projects; project-specific modules where it should not be. Reusing
`OutlierClipper` meant the clipping policy was already battle-tested on the
previous project's data. Reusing `build_preprocessor` meant the imputation
and encoding behaviour had been validated. Writing a new `FeatureEngineer`
subclass meant the churn-specific features were not shoehorned into the
propensity shape.

### 3.2 Config file as single source of truth

A separate `churn_config.py` module holds every parameter that could change
between experiments or between quarters.

```python
# ── Population & training window ──────────────────────────────
ACTIVE_THRESHOLD_DAYS = 90       # see §6 for why this value
CHURN_WINDOW_DAYS     = 30
SNAPSHOT_START        = '2022-06-01'
SNAPSHOT_END          = '2025-12-31'

# ── Feature set ───────────────────────────────────────────────
NUMERIC_FEATURES     = [...]     # the churn-specific numeric catalogue
CATEGORICAL_FEATURES = [...]
FEATURE_COLS         = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Hyperparameter search grid (Random Forest) ────────────────
PARAM_GRID_RF = {...}

# ── Artifacts & BigQuery ──────────────────────────────────────
ARTIFACT_PATH = 'artifacts/churn_final_random_forest.pkl'
SCORES_TABLE  = '...churn_scores'

# ── Recency group thresholds ──────────────────────────────────
RECENCY_EARLY_MAX   = 30   # 0–30d : still in monthly cycle
RECENCY_MID_MAX     = 60   # 31–60d : missed 1 cycle
RECENCY_AT_RISK_MAX = 90   # 61–90d : missed 2 cycles
```

Every notebook (EDA, walk-forward training, final training, scoring)
imports from this file. Changing `ACTIVE_THRESHOLD_DAYS` from 90 to 60 is a
one-line edit and propagates to the full stack. The feature list is never
hardcoded in a notebook. The recency bands used by the segmentation strategy
(§9.3) are the same bands used by the training population filter, because
both read from the same constants.

The direct benefit is reproducibility of experiments. If an experiment's run
ID in BigQuery says `ACTIVE_THRESHOLD_DAYS = 60`, that is the only number
that mattered, and the notebook that produced it can be replayed by
flipping the config and re-running.

### 3.3 Two-table experiment logger in BigQuery

The experiment grid in §5 (five configurations × two model types × 13
walk-forward folds each) produced on the order of 130 model fits. Each one
logged to BigQuery through a dedicated module that writes to two tables.

| Table | Row granularity | What it holds |
|---|---|---|
| `experiment_runs` | One row per run (parameter combination + model type) | Run ID, timestamp, model name, parameters (`active_threshold_days`, `churn_window_days`, snapshot range), feature list, aggregate metrics (mean + std of ROC-AUC, PR-AUC, Lift@10% across folds) |
| `experiment_folds` | One row per fold per run | Run ID, fold index, validation snapshot date, train/validation size, baseline churn rate, per-fold metrics |

The two-table structure is what makes stability analysis a query, not a
re-run. "Which experiments had PR-AUC std above 0.15 across folds?" is a
`SELECT` on `experiment_runs`. "Which Q2 folds dragged down the Q2
performance of the winning configuration?" is a `JOIN` of
`experiment_folds` and `experiment_runs` filtered by `quarter = 2`.

This is a direct evolution of the propensity-era logger (one row per run,
no fold breakdown; see §7 of the propensity case). Walk-forward CV
multiplies the information volume per experiment by the number of folds, so
flattening everything into a single row per run would have hidden the exact
property the experiment grid was designed to reveal.

### 3.4 Why this infrastructure matters for a project that produces 5 experiments

Five experiments are not many. The infrastructure cost of running them
could reasonably have been one notebook with hard-coded parameters, swapped
by hand. The reason to do it differently is that the infrastructure makes
the *next* five experiments cheap, and the five after that cheaper still.
The grid in §5 only produced conclusive findings on `active_threshold_days`
because every configuration left a reproducible BigQuery row behind, which
made it possible to see the collapse pattern (90 → 60 → 35 days) at a
glance instead of as a comparison of three notebook outputs.

The infrastructure also made it defensible to the client to say "we've run
this combination before, here's the run ID, we don't need to re-fit". A
scoping decision on the recency bands could be made against logged
historical runs, not against vague recollection.

## 4. Validation strategy: walk-forward over quarterly snapshots

### 4.1 Why not a single train/test split

The donor base in this project is small by ML standards (around 7,000 unique
recurring donors in the four-year history, around 150,000 transactions). A
single random split would leave the training set thin, and would mix donors
and time periods on both sides of the split in a way that leaks implicit
information about the current state of the portfolio into the model.

A single temporal holdout (train on the first N years, test on the last) is
better for leakage, and it tests the model on exactly one period at the end.
If that period happens to be an unusual month, the number reported is noise.

### 4.2 Walk-forward over 14 quarterly snapshots

To expand the dataset and get a reliable estimate of out-of-time performance,
the training set is built from **14 quarterly snapshots** between 2022-Q3 and
2025-Q4. Each snapshot represents a cross-section of the active recurring
donor pool at a point in time. For each snapshot, the label is computed
looking forward 30 days (the churn window). Snapshots are stacked into a long
dataset of around **44,700 donor-snapshots**.

Walk-forward validation on top of this stack produces **13 folds**.

- **Fold k train set.** All snapshots from period 1 through k.
- **Fold k validation set.** Snapshot k+1 only.

K-fold applied naively would create temporal leakage. A snapshot from T+2
could end up in the training fold while T+1 is held out, meaning the model
trained on data from the future is being evaluated on the past. Walk-forward
preserves causal direction. Training always uses snapshots earlier than the
validation snapshot.

### 4.3 Fold-level reporting

Each fold produces its own set of metrics. The aggregate across folds is the
headline number; the per-fold distribution is what matters for stability
diagnosis. Two patterns surfaced from the fold-level view:

- **Q2 seasonal dip across all three years.** Folds 3, 7, and 11 (validation on April snapshots) show a lower churn rate than the surrounding folds (around 5% versus the ~10% population average). The dip is consistent across all three years in the training history, which rules out a data artefact and points at a seasonal giving pattern. The model performs less reliably on Q2 folds. Expect lower AUC-PR in April predictions.
- **Fold 13 (validation on October 2025) shows a higher baseline churn rate (~14%)** than the historical ~10% average. Worth monitoring whether this is a trend or a one-off spike.

The per-fold reporting is logged to BigQuery (see §13), so trend analysis
across folds is a query, not a replay.

## 5. The experiment grid: what we varied and why

Five systematic experiments were run on this walk-forward setup, varying three
parameters.

| Parameter | What it controls | Values tested |
|---|---|---|
| `active_threshold_days` | How many days a donor can go without giving before being included in the snapshot's at-risk pool | 35, 60, 90 |
| `churn_window_days` | How far forward the label looks for the next monthly gift | 30, 35 |
| `last_interval_days` feature | A derived feature capturing recency relative to the donor's own historical cadence | included / ablated |
| Model | Baseline vs. non-linear | Logistic Regression, Random Forest |

The grid is intentionally narrow. Each parameter is tested against a specific
hypothesis, not against a general "let's see what moves". The three parameters
were chosen because they drive, in order of magnitude:

1. The **cleanliness of the positive class** (how at-risk a donor has to be to count as a candidate).
2. The **sharpness of the label** (how tight the window around the expected next gift is).
3. The **expressiveness of the feature set** (whether a specific recency feature carries independent signal).

Each run was logged to a dedicated BigQuery table with run ID, timestamp, the
parameter combination, aggregate metrics, and per-fold metrics. Reproducing
any past experiment is a `SELECT` with a `WHERE run_id = ...`.

## 6. Why `active_threshold_days = 90` is the pivot parameter

The headline finding of the experiment grid is that `active_threshold_days` is
the single most sensitive parameter in the model. The AUC-PR (precision-recall
area, which is the right metric here, see §9) collapses hard as the threshold
is tightened.

| `active_threshold_days` | RF AUC-PR (35-day window, full features) | vs. 90-day |
|---|---|---|
| 90 | ~0.54 | baseline |
| 60 | ~0.38 | **−30%** |
| 35 | ~0.16 | **−70%** |

### 6.1 Why the parameter matters this much

At `active_threshold_days = 90`, donors still in the snapshot pool have missed
up to two monthly cycles. At that point, they look structurally different
from the non-churners. Their cadence has deviated from their own history,
their last-interval has grown, and the feature set has a lot of signal to
work with.

At `active_threshold_days = 35`, donors in the pool have barely gone past
their normal monthly cycle. The majority are still perfectly on-track from a
behavioural standpoint; a few are starting to deteriorate but the features
cannot tell them apart yet. The model is being asked to predict churn from
noise, and the AUC-PR of 0.16 reflects that.

### 6.2 Why we did not relax the parameter

A tighter threshold would have been operationally attractive. Catching a
deteriorating donor earlier gives the retention team more runway to act.
Attractive, and infeasible with this feature set. The pattern that distinguishes
future-churners from on-track donors is a cadence deviation that only becomes
visible after a couple of missed cycles. Pushing the threshold below 90 days
would mean producing scores that are structurally close to random, which is
worse than not producing scores at all.

This is a hard limit of the current feature set. A different feature set
(behavioural engagement data, email interaction history, cause-page visits)
might carry earlier-stage signal that lets the threshold be relaxed. That is
noted as a future-version path. In this version, 90 days is the correct call.

## 7. Model choice: Random Forest, and why not Logistic Regression

Both Random Forest and Logistic Regression were trained on every configuration
of the grid. Random Forest wins consistently.

| Config | Logistic AUC-PR | RF AUC-PR | Δ |
|---|---|---|---|
| 90d / 30d window | ~0.52 | ~0.66 | **+28%** |
| 90d / 35d window | ~0.44 | ~0.54 | **+22%** |
| 60d / 35d window | ~0.25 | ~0.38 | **+52%** |
| 35d / 35d window | ~0.11 | ~0.16 | **+48%** |

The pattern tells us something beyond "RF is better". The gap widens as the
problem gets harder, which is a signature of non-linearity. The relationship
between recency features and churn probability is not monotone. A donor who
is slightly late (last interval 32 days vs. their 30-day baseline) is
behaviourally similar to an on-time donor. A donor who is moderately late
(45 days) is a different animal. A donor who is very late (60+ days) is
different again. Trees capture these regime changes naturally; a linear model
has to approximate them with coefficients that get pulled in opposite
directions.

Random Forest was chosen over gradient-boosted trees (XGBoost, LightGBM) for
pragmatic reasons at this stage of the project. Lower training cost on the
walk-forward loop (13 folds × 5 experiments × 2 models = 130 fits), lower
tuning overhead, and performance already in a strong range. The experiment
analysis explicitly flagged gradient boosting as the natural next step. In
retrospect, the natural experiment is RF vs. LightGBM on the winning
configuration, which is a day of work and would resolve whether there is
additional signal left on the table.

### 7.1 Class imbalance handling

Across the full training stack, the churn rate is around 10%. Not extreme,
but imbalanced enough that an uncorrected model would collapse to
"predict not-churn for everyone" and achieve high accuracy with near-zero
recall on the positive class. `class_weight="balanced"` on the RF resolves
this in-place. The gradient contribution of positive samples is rescaled so
the loss is driven by both classes, and the full training set is preserved.

Resampling (SMOTE, undersampling) was not used. SMOTE on mixed numeric and
categorical features generates synthetic samples that are semantically
meaningless, and undersampling throws away information at a dataset size
where every positive sample matters.

## 8. Probability calibration: isotonic over Platt

Random Forest outputs a probability-like score that is a good ranker but a
poor probability estimator. Predicted probabilities tend to compress toward
the centre of the distribution (RF rarely predicts near 0 or 1), and the
raw scores do not match observed churn rates in the way a well-calibrated
model should.

For a pure-ranking use case (sort donors, pick the top K) this does not
matter. Any monotonic transformation preserves ranking. It matters here for
two downstream reasons:

- The segmentation strategy (§9) uses probability bands within recency groups ("top 20% of model probability inside the Early-recency band"). Band membership is directly sensitive to probability values, not just rankings.
- The retention team talks in probability language when prioritising cases ("this donor has an 80% chance of churning"). A probability that does not match reality is worse than no probability at all.

Calibration is applied with `CalibratedClassifierCV(method="isotonic", cv=5)`
as a wrapper around the base RF. Isotonic over Platt for two reasons:

| Method | Assumes | Fit here |
|---|---|---|
| Platt scaling (sigmoid) | A sigmoid relationship between raw score and true probability. Two parameters. | Fine on small validation sets; too stiff for non-sigmoid miscalibration. |
| Isotonic regression | Only that the mapping is monotonic. Non-parametric, piecewise-constant. | Fits the compressed-toward-centre pattern RF produces, without forcing it into a sigmoid shape. |

The validation set has enough positive samples per fold for isotonic's
flexibility to earn its extra parameters without overfitting. The pattern is
in the lapse-prediction case study's calibration snippet, which is a
model-agnostic implementation of the same wrapper logic.

### 8.1 Q2 calibration caveat

Monitoring the Brier Skill Score per fold reveals a slight degradation in Q2
folds (April snapshots). The model is still above the naive baseline, and the
margin narrows. This has been documented and not yet mitigated. Options for a
future version include fitting the isotonic calibration excluding Q2 folds,
adding a quarter-of-year feature so the model can learn the seasonality
directly, or adding a `calibration_confidence = "low"` flag on April scores
so the retention team treats them with lower weight. None is implemented in
this version; see §11.

## 9. Results and operational segmentation

*All numbers below are approximate. The client owns the exact metrics.*

### 9.1 Which metric drives decisions, and which doesn't

**ROC-AUC, reported but not optimised.** With a ~10% churn rate, the confusion
matrix is dominated by True Negatives. The model classifies ~90% of the
population as not-churn almost for free, which inflates ROC-AUC toward high
values even for mediocre models. ROC-AUC of 0.89 sounds strong and tells us
surprisingly little about operational performance on the 10% that actually
matter.

**AUC-PR, the model-selection metric.** AUC-PR is computed only over the
ranking of the positive class and is not diluted by True Negatives. It is the
correct metric under this class imbalance, and it is the one that moved most
across experiments (see §5). AUC-PR is what told us the winning configuration
(RF + 90-day threshold + 30-day window + full features) is actually better
than its neighbours in the grid.

**Recall@top-K and precision-within-segment, the operational metrics.** The
retention team acts on segments, not on the global ranking. The relevant
numbers are the capture rate and the precision inside each segment (see
§9.3).

### 9.2 Aggregate performance

Walk-forward across 13 quarterly folds, winning configuration (RF +
`active_threshold_days = 90` + `churn_window_days = 30` + full features):

| Metric | Value | Use |
|---|---|---|
| ROC-AUC | **~0.89 ± 0.06** | Sanity check. The model ranks churners above non-churners. |
| AUC-PR | **~0.66 ± 0.20** | Model selection. Strong given the ~10% prevalence. |
| Lift@10% | **~7x** | Operational. Top decile is around 7 times more enriched in churn than random. |

The AUC-PR standard deviation of ±0.20 is noticeable. The spread comes from
seasonal variation across folds (Q2 is weak, Q4 is strong), not from
instability of the model. The per-fold trajectory is monotonically improving
over the training horizon, which is consistent with a growing training set
each fold.

### 9.3 Four-segment operational playbook

The scored output is bucketed into four retention segments. Segment
membership is a combination of **recency band** (how long since the last
gift, as of the scoring date) and **model percentile within the recency
band**. Both dimensions matter. Recency determines the type of outreach that
makes sense; model percentile determines priority inside that outreach queue.

| Segment | Recency band | Model filter | Approx. monthly volume | Recommended action |
|---|---|---|---|---|
| **Critical** | At-Risk (>60 days since last gift) | All | ~230 | Personalised "we miss you" outreach by phone or direct mail. By this point departure is nearly certain without intervention. |
| **High Priority** | Mid (31–60 days) | Top 40% of model percentile within band | ~100–150 | Personal retention outreach within the week. Model precision ~75%. |
| **Medium Priority** | Early (0–30 days) | Top 20% of model percentile within band | ~200–300 | Light-touch automated email (thank-you, impact story, gentle check-in). This is where the model adds the most unique value. |
| **Monitor** | Early (0–30 days) | Below top 20% | ~600–800 | No active campaign, standard donor communication cadence. Reassess next month. |

### 9.4 Segment-level performance

| Segment | Baseline churn rate inside segment | Model precision inside segment | Lift vs. baseline inside segment |
|---|---|---|---|
| Critical (>60d) | ~86% | ~90% | ~1.05x |
| High Priority (31–60d) | ~48% | ~75% | ~1.56x |
| Medium Priority (0–30d) | ~6% | ~21% | **~3.4x** |

The pattern in the last column is the operational insight. See §10.

## 10. Where the model actually adds value

A naive reading of §9.4 is that the Critical segment is where the model wins.
It has the highest precision (~90%) and the highest base rate. That reading
is wrong, and getting this right was one of the operational contributions of
the project.

In the Critical segment, the baseline is already ~86% without the model.
Donors who have gone 60 days without a gift are already in trouble by any
reasonable rule. A pure rule-based system (`days_since_last_gift > 60 →
contact`) would reach similar precision at zero model overhead. The model's
~1.05x lift over the rule is small.

In the Medium Priority segment, the baseline is ~6%. These are donors still
on their monthly cycle, visibly on-track, who nobody would flag using a
simple rule. The model lifts precision from ~6% to ~21%, a ~3.4x factor.
This is where the model earns its keep.

The operational consequence of this analysis:

- **The model is not a replacement for the recency rule.** The recency rule handles the Critical and most of the High-Priority flow fine on its own.
- **The model is a surfacing tool for the Medium-Priority segment.** It takes a population that looks fine on the surface and extracts the ~20% that are quietly starting to disengage. These are the donors that recover best, because the intervention happens before the departure is nearly certain.

This also sets the priority for the next iteration of the model. Improving
performance on the Critical segment has low marginal operational value, since
the rule-based system is already near the ceiling there. Improving Medium
Priority precision has high marginal operational value. Any future feature
engineering effort should be targeted at the Early-recency band.

## 11. Known limitations and explicit gap list

The project reached handoff with validation complete and the scoring notebook
operational. A set of known gaps was explicitly documented at the handoff
point rather than hidden. They do not invalidate the model, and they shape
the roadmap for a v2.

### 11.1 Reproducibility and handoff friction

- **No `requirements.txt`, `environment.yml`, or `pyproject.toml`.** The runtime environment is implicit. For a project moving into production, this blocks the engineering handoff. Trivial to generate, flagged as the first thing to resolve.
- **A known logging bug in one experiment's `std_lift_10pct` column** (one experiment produced a std value ~4 orders of magnitude higher than any other). The bug has been identified and quarantined in the analysis document. The affected run's mean metrics are reliable; its std column is not. Resolution path is to fix the logger in `churn_experiment_logger.py` and re-log the affected run.

### 11.2 Observability gaps

- **No SHAP or per-donor explanation layer.** The model emits a probability, not a reason. Fundraisers calling flagged donors receive a score, which is less actionable than "this donor is flagged because her last-interval has grown to 47 days vs. her 29-day historical average, and her most recent gift was 40% below her mean". SHAP values on the calibrated model take a few hours to add and would produce significant operational value. Flagged as the highest-priority v2 feature.
- **No feature-level drift detection.** The run-level logger captures aggregate metrics, not the input distributions. If a feature's distribution shifts upstream (e.g., a change to the donation ingestion pipeline that affects how `last_payment_method` is recorded), the signal would surface in model performance months later, not at scoring time. PSI on key features is a standard next step.

### 11.3 Known seasonal weakness

- **Q2 calibration degradation (see §8.1).** The model performs slightly worse in April predictions. Documented, not mitigated. Three concrete mitigation options are listed in §8.1.

### 11.4 Engineering hygiene

- **`pickle` instead of `joblib` for model persistence.** `joblib` is sklearn's recommended serialisation for objects containing large numpy arrays (which an RF with 100 trees certainly is). Low risk at current scale, higher risk once the model is in a production pipeline and sklearn is upgraded.
- **No unit tests on the feature-engineering pipeline.** `build_snapshot_features()` and `build_scoring_features()` have no coverage. A silent error (a join that drops rows, a scaler that applies the wrong normalisation) would surface only when monthly scores look wrong. Minimum useful test battery: no-future-leak invariance, row-count invariance in/out of feature construction, no inf/nan in derived features.

These items are not surprises. They are tracked and prioritised. The point of
listing them here is that a case study that only documents what went well is
less useful than one that also documents what the next engineer picking up
the project needs to resolve first.

## 12. Related work: the one-time donors branch

The population excluded from this project in §2 (one-time donors) was
addressed as a separate propensity model. That project was trained and
validated, and the DS team recommended against production because the lift
at any operational cutoff did not justify the campaign cost. The client
accepted the recommendation.

The two projects share the client, the data engineering pipeline, parts of
the sklearn preprocessing architecture, and the scoping decision that
separated the two populations in the first place. They do not share the
modelling problem. The clean monthly cadence of recurring donors is exactly
what made the churn problem tractable in this feature set; the absence of a
cadence equivalent on the one-time side is exactly what made the propensity
problem harder.

Companion case study: [propensity-case-study](https://github.com/martinterzano/propensity-case-study).

## 13. Scoping decisions and repository contents

### 13.1 What was deliberately left out of the production system

- **No automated monthly scoring DAG.** The scoring notebook runs on demand and writes to a BigQuery table. Converting it to a scheduled job was scoped to the engineering handoff, since the business was comfortable running it manually at the start of each month during the ramp-up. Flagged as phase 2.
- **No automated retraining trigger.** The model is retrained manually when a quarterly snapshot's drift justifies it. A metric-based automated trigger is a reasonable next step and intentionally not implemented at the handoff point.
- **No A/B infrastructure.** The model's incrementality against the pre-existing retention workflow is measured retrospectively by comparing contacted donors' retention to a hold-out of uncontacted donors at the same risk level. A proper A/B test is on the roadmap once baseline operational metrics have stabilised.

### 13.2 What is in this repo

```
case-churn-donors/
├── README.md    ← this document
└── AUDIT.md     ← pre-publish checklist
```

The full training notebooks, feature-engineering modules
(`churn_feature_builders.py`, `churn_feature_engineer.py`,
`churn_feature_mapper.py`), the shared pipeline modules inherited from the
propensity project, the config file described in §3.2, the experiment
logger described in §3.3, the scoring code, and the BigQuery schemas are
all client property. The architecture is documented in §3 with the module
boundaries and composition patterns spelled out, which is enough to see
the shape of the engineering without exposing client logic or feature
taxonomies.

## 14. What this case study is meant to show

- **Walk-forward validation on snapshotted data as the default, not as an exotic choice.** Any time a dataset has temporal structure between observations, the validation design has to respect that structure. Walk-forward is the uncomplicated way to do that without hand-rolling a leakage-aware fold assignment.
- **A systematic experiment grid driven by hypotheses, not by curiosity.** Each parameter in §5 was varied to answer a specific question. The grid is small on purpose. The output is not a search for the best model but an explicit map of how each dimension of the problem affects performance.
- **A single parameter can gate the entire project.** `active_threshold_days = 90` is not a hyperparameter in the usual sense. It is a definition-of-the-problem parameter, and the model's AUC-PR drops by 70% when it is relaxed to 35 days. Treating it as "just another knob" would have hidden the fact that the feature set has a hard limit on how early it can detect churn.
- **AUC-PR over ROC-AUC under class imbalance.** With ~10% churn, ROC-AUC is dominated by True Negatives and inflates easily. AUC-PR evaluates the ranking of the positive class only, and it is the metric that moves between experiments.
- **Random Forest over Logistic Regression as a diagnostic, not just a performance choice.** The pattern of RF beating Logistic Regression by increasing margins as the problem gets harder is a signature of non-linearity in the feature-to-churn relationship. The model choice confirmed the hypothesis that recency features interact in a regime-dependent way.
- **Separating ranking and calibration as independent artefacts.** The RF ranks; the isotonic wrapper calibrates. They evolve independently, and a future recalibration pass can be run without retraining the ranker.
- **Operational segmentation that reflects where the model actually adds value.** The four-segment playbook does not treat all segments equally. The analysis in §10 makes explicit that the model's unique value is in the Early-recency band, where no rule-based system would flag anything, and that is where any future feature work should be targeted.
- **An honest gap list at handoff.** The §11 inventory is the kind of document that gets lost if it is not written down at the moment of handoff. Making it explicit is what distinguishes "finished for me" from "ready for someone else to pick up".
- **Shared infrastructure as a seniority signal.** The modular pipeline described in §3 was inherited from the earlier propensity project (§12), extended with a project-specific config file and a two-table experiment logger, and left in a shape that the next DS can keep extending. Reusable scaffolding across projects is a deliberate choice at a seniority level, not an accident of two projects running at the same time.

---

Martín Terzano · [helliumlab.com](https://helliumlab.com) · [LinkedIn](https://www.linkedin.com/in/martinterzano) · martin@helliumlab.com
