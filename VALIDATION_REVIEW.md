# Validation design: cross-check against the literature

A review of this benchmark's splitting and evaluation protocol against YieldSAT
(Miranda et al., CVPR 2026), "From MODIS to Sentinel-2" (IEEE, 2025), and Celik
et al. (IEEE GRSL 20, 2023, Art. 8500905).

---

# MEASURED RESULTS — the concerns below were tested, not assumed

Run on the real cohort with the two precomputed representations (AlphaEarth
and the 21-D Sentinel-2 index baseline). No extraction required. The harness
reproduces the benchmark's own numbers exactly under the existing protocol
(AlphaEarth R2 0.675 / RMSE 12.64; S2 indices 0.635 / 13.41), which is what
validates it.

**Regression heads.** The benchmark's canonical registry is four heads: Ridge
(alpha from `[0.01, 0.1, 1, 10, 100]`, selected on validation, refit on
train+validation), Random Forest (600 trees, squared error, unlimited depth,
`min_samples_leaf=2`, `max_features=1.0`, bootstrap), XGBoost (600 trees, lr 0.03,
depth 6, child weight 2, subsample/colsample 0.8, L2 1.0, hist), and EBM (1,000
rounds, 5 interactions, 128/64 bins, 14 outer bags, lr 0.04). RF/XGB/EBM use
seeds 0/1/2; Ridge is deterministic.

Sections 1-3 below use **Ridge**, matching `probe.py`, which is the benchmark's
representation-quality diagnostic. Section 2 (LOYO) uses **Random Forest**,
matching `loyo.py`. Section 4 additionally re-runs the headline comparison across
Ridge, RF and XGBoost to confirm the conclusion is not a property of the head.

## 1. The spatial leak is real, and it reverses the encoder ranking

| Protocol | AlphaEarth R2 | S2 indices R2 | Winner |
| --- | --- | --- | --- |
| County GroupKFold (as implemented) | **0.675** | 0.635 | AlphaEarth |
| Spatial-block CV (5 contiguous blocks) | 0.346 | **0.516** | **S2 indices** |
| Spatial-block + 30 km buffer | 0.346 | **0.516** | **S2 indices** |
| Spatial-block + 50 km buffer | 0.327 | **0.518** | **S2 indices** |

RMSE follows: AlphaEarth 12.64 -> 17.56 bu/acre; S2 indices 13.41 -> 15.17.

**The conclusion flips.** Under the current protocol AlphaEarth beats the
handcrafted baseline; under spatial blocking it loses to it, decisively. And the
two representations are affected very differently — AlphaEarth loses 0.33 R2,
the index baseline only 0.12. That asymmetry is the substantive finding: much of
AlphaEarth's apparent advantage is spatial interpolation between neighbouring
counties, not transferable skill.

Why the leak is this large: under the existing folds the median test county sits
**34 km** from the nearest training county, with 89% within 50 km and the closest
at 5.6 km. Corn Belt counties are roughly 40-50 km across, so test counties are
typically *directly adjacent* to training counties.

| Fold | Test counties | Nearest train county (median) | Within 30 km | Within 50 km |
| --- | --- | --- | --- | --- |
| 0 | 81 | 33.0 km | 35.8% | 91.4% |
| 1 | 81 | 35.3 km | 29.6% | 91.4% |
| 2 | 82 | 34.5 km | 32.9% | 86.6% |
| 3 | 81 | 34.3 km | 28.4% | 84.0% |
| 4 | 81 | 33.3 km | 38.3% | 91.4% |

Note the buffer adds almost nothing on top of blocking — the blocks already
separate the regions, so 0/30/50 km give the same answer. Blocking is what
matters, not the buffer width.

## 2. The LOYO look-ahead is worth 0.19 R2

Random Forest, seeds 0/1/2, the benchmark's own LOYO estimator:

| Held-out year | LOYO (all other years) | Forward-chaining (prior years only) |
| --- | --- | --- |
| 2019 | 0.267 | n/a — no prior years |
| 2020 | 0.080 | -0.036 |
| 2021 | 0.414 | -0.029 |
| 2022 | 0.340 | 0.340 — identical, the only true forecast |
| **mean** | **0.275** | **0.092** |

AlphaEarth is inflated by **+0.186 R2** on the three years where both are
defined; the S2 index baseline by +0.059. As predicted, 2022 is byte-identical
between the two protocols because it is the one fold with no future years to
leak.

Again the learned representation is inflated ~3x more than the handcrafted one,
so the look-ahead does not affect encoders equally either.

Also note how weak genuine forecasting is here: with one or two training years,
forward-chained R2 is around zero or negative. Four years is very thin for a
temporal claim, and this is the honest picture of it.

## 3. The encoder gap is not significant under the current protocol — but the reversal is

Paired county bootstrap, 2000 resamples, on identical held-out county-years:

| Protocol | Delta R2 (S2 indices - AlphaEarth) | 95% CI | Verdict |
| --- | --- | --- | --- |
| County GroupKFold | -0.040 | [-0.089, +0.011] | **not significant** |
| Spatial-block CV | **+0.142** | [+0.064, +0.227] | **significant** |

So the headline "AlphaEarth beats the Sentinel-2 index baseline" is **not
statistically supported** on the current protocol — the CI crosses zero. The
finding that survives is the opposite one, and only under spatial blocking: the
handcrafted 21-D baseline significantly outperforms AlphaEarth when the folds are
spatially separated.

## 3b. The same tests on the full 2,180 cohort — well powered

The results above use the 1,038-county-year GeoFM cohort (406 counties). Restricting
both tabular representations to their common complete cohort gives **2,180
county-years / 953 counties / 13 states**, which more than doubles the counties and
raises leave-one-state-out from 5 folds to 13. Ridge, alpha selected on a held-out
partition, refit on train+validation.

| Protocol | AlphaEarth | S2 indices | Paired delta (S2 - AE) | 95% CI | |
| --- | --- | --- | --- | --- | --- |
| County-grouped 5-fold | **0.770** | 0.758 | -0.012 | [-0.046, +0.030] | not significant |
| Leave-one-state-out, 13 states | 0.312 | **0.416** | **+0.067** | [+0.022, +0.118] | **significant** |

Three things follow.

**The county-grouped null is now a real result, not low power.** At 406 counties the
CI crossing zero could be dismissed as too small a sample. At 953 counties it still
crosses zero. Under spatially-interpolating folds, a 64-D learned annual embedding
and 21 handcrafted vegetation indices are statistically indistinguishable.

**Under spatial shift the handcrafted baseline wins, by a smaller margin than the
406-county estimate suggested.** The gap is +0.067, not the +0.173 measured on the
smaller cohort — more states and more data give a more modest, better-estimated
effect. It remains significant.

**Both collapse under spatial shift, and AlphaEarth is far less stable.** 0.770 ->
0.312 and 0.758 -> 0.416. Across the 13 held-out states the standard deviation is
**0.411 for AlphaEarth against 0.227 for the index baseline**, and AlphaEarth's
worst state is -0.601 versus -0.037. Per-state detail is in
`outputs/baselines/loso_2180_13states.csv`.

| State | n | AlphaEarth | S2 indices |
| --- | --- | --- | --- |
| Illinois | 259 | 0.484 | 0.416 |
| Indiana | 198 | 0.628 | 0.415 |
| Iowa | 292 | 0.499 | 0.523 |
| Kansas | 199 | 0.327 | 0.596 |
| Kentucky | 99 | **-0.507** | -0.037 |
| Michigan | 80 | **-0.601** | 0.131 |
| Minnesota | 158 | 0.689 | 0.752 |
| Missouri | 135 | 0.638 | 0.334 |
| Nebraska | 216 | 0.118 | **0.714** |
| North Dakota | 89 | **0.768** | 0.219 |
| Ohio | 197 | 0.332 | 0.229 |
| South Dakota | 112 | 0.470 | 0.655 |
| Wisconsin | 146 | 0.213 | 0.467 |

Neither representation wins everywhere — AlphaEarth takes 6 of 13 states — but it
fails much harder where it fails. Kentucky and Michigan are the smallest cohorts and
the most peripheral to the Corn Belt, which is where a learned embedding trained on
different geography would be expected to break down.

## 4. The recommended protocol, tested

### Leave-one-state-out — the cleanest option, and unanimous

States are natural agronomic regions, need no clustering step, and have **no
hyperparameter to justify**. This is YieldSAT's LORO in a form a reviewer cannot
argue with. Five states carry >=40 county-years (949 of 1,038 total):

| State | Test county-years | AlphaEarth R2 | S2 indices R2 |
| --- | --- | --- | --- |
| 19 Iowa | 281 | 0.217 | **0.479** |
| 17 Illinois | 233 | 0.393 | **0.420** |
| 18 Indiana | 176 | 0.400 | **0.461** |
| 31 Nebraska | 150 | 0.142 | **0.513** |
| 39 Ohio | 109 | 0.377 | **0.524** |
| **mean** | | **0.306** | **0.479** |

The index baseline wins in all five states under Ridge, by +0.173 R2 on average.
Both degrade from their county-GroupKFold scores, but very unequally: the index
baseline 0.635 -> 0.479, AlphaEarth 0.675 -> 0.306.

> **Correction.** An earlier version of this table reported the S2 column as
> 0.612 mean, with gaps of +0.31 to +0.45. Those numbers were wrong. Both frames
> contain the identical 1,038 county-years, but they were built in *different row
> orders*, and the state masks were derived from the AlphaEarth frame and applied
> positionally to the S2 frame. Features and labels stayed aligned within each
> frame, so the S2 model was valid — but its "held-out state" was an arbitrary
> subset rather than a state, which is an easier task and inflated the score. The
> harness now sorts both frames to a canonical `(county_id, year)` order and
> asserts key-order equality before indexing. The corrected numbers are above.
> Only the leave-one-state-out results were affected; the spatial-block,
> forward-holdout, combined and paired-bootstrap results all derived their indices
> per-frame and are unchanged.

#### The result is not an artefact of the regression head — but it is not unanimous either

The canonical registry has four heads. The tests above used **Ridge**, so the
obvious objection is that a linear head flatters the 21-D engineered indices and
handicaps the 128-D embedding. That objection does not hold — the gap is stable
across heads — but the per-state agreement is weaker than the means suggest:

| Head | AlphaEarth R2 | S2 indices R2 | Gap | S2 wins in |
| --- | --- | --- | --- | --- |
| Ridge (alpha selected on validation) | 0.306 | 0.479 | +0.173 | **5/5** states |
| Random Forest (600 trees, leaf 2, seeds 0/1/2) | 0.221 | 0.407 | +0.187 | 3/5 states |
| XGBoost (600, lr 0.03, depth 6, seeds 0/1/2) | 0.209 | 0.394 | +0.184 | 3/5 states |

The mean gap is consistent at +0.17 to +0.19 across all three heads, and
AlphaEarth is the weaker representation under every one. But the tree heads
disagree on two of five states, so this should be reported as a mean difference
with a paired interval, not as a clean sweep.

EBM was not run — its `shap`/`llvmlite` dependency was removed from the
verification sandbox for disk.

### Forward-in-time holdout — train 2019-2021, validate 2021, test 2022

| Representation | Test n | R2 | RMSE |
| --- | --- | --- | --- |
| AlphaEarth | 342 | 0.545 | 15.51 |
| S2 indices | 342 | **0.624** | **14.11** |

One clean operational number, and the S2 baseline wins here too.

### Both shifts at once — spatial blocks *and* forward time

| Representation | R2 |
| --- | --- |
| AlphaEarth | **-0.084** (worse than predicting the mean) |
| S2 indices | 0.260 |

### Block geometry is a scientific choice, not a nuisance

k-means seeds are stable (within a given k, three seeds agree to ~0.02 R2), but
the **number** of blocks moves the answer systematically, because it sets how far
test counties sit from training data:

| Protocol | Median test -> train distance | AlphaEarth | S2 indices |
| --- | --- | --- | --- |
| spatial blocks k=4 | 185 km | 0.247 | 0.502 |
| spatial blocks k=5 | 146 km | 0.346 | 0.516 |
| spatial blocks k=6 | 93 km | 0.446 | 0.541 |
| spatial blocks k=8 | 93 km | 0.486 | 0.476 |
| spatial blocks k=10 | 77 km | 0.410 | 0.473 |
| spatial blocks k=20 | 51 km | 0.431 | 0.478 |
| **county GroupKFold (current)** | **34 km** | **0.675** | 0.630 |

Read as a curve rather than a single number: AlphaEarth degrades steeply with
separation distance, the index baseline is nearly flat. Reporting this curve is
more honest than defending one arbitrary k, and it converts "which k?" into an
interpretable question about the spatial range over which yield is autocorrelated.

### Every protocol except the current one gives the same answer

| Protocol | AlphaEarth | S2 indices | Winner |
| --- | --- | --- | --- |
| County GroupKFold (current) | 0.675 | 0.635 | AlphaEarth (not significant) |
| Spatial blocks, k=5 | 0.346 | 0.516 | **S2 indices** |
| Leave-one-state-out | 0.306 | 0.479 | **S2 indices** (5/5 states, Ridge) |
| Forward holdout 2022 | 0.545 | 0.624 | **S2 indices** |
| Spatial + forward combined | -0.084 | 0.260 | **S2 indices** |

The current protocol is the only one under which AlphaEarth wins, and even there
the paired bootstrap CI crosses zero.

## Recommended protocol, in final form

1. **Primary: leave-one-state-out.** No hyperparameter, natural regions,
   interpretable, and it is YieldSAT's LORO. Use an inner state as validation.
   Only 5 of the 13 cohort states carry >=40 county-years, so this gives 5
   evaluations covering 949 of 1,038 county-years — state it as such.
2. **Secondary: forward holdout to 2022.** One honest operational number, and the
   protocol that makes your results comparable to the CNN-RNN / GNN-RNN line.
3. **Robustness appendix: the distance curve above**, showing R2 against median
   test-to-train separation, with the current county-GroupKFold as its leftmost
   point.
4. **Retire or rename the current LOYO** — 3 of its 4 folds train on future years.
5. **Report paired bootstrap CIs** for every A-vs-B claim.

Keep the existing county-GroupKFold manifest as the sensitivity analysis. It is
not wrong; it answers a narrower question (interpolation among neighbouring
counties) than the paper currently claims.

## What this means before you revise

These were run on the two representations available without extraction. The four
GeoFM encoders are unmeasured, and there is no reason to assume they behave like
AlphaEarth — AlphaEarth is a precomputed annual embedding with, plausibly, more
location-specific information baked in than a per-timestep Sentinel-2 encoder.
**Testing them the same way is the point**, and it is cheap once extraction is
done: the harness is protocol-only and needs no re-extraction.

The practical implication for sequencing: **decide the protocol before running
the four-head regression benchmark**, because the tables produced under the
current splits may not survive contact with a spatially blocked one.

---

## What this benchmark currently does

| Element | Implementation |
| --- | --- |
| Main protocol | 5-fold `GroupKFold`, grouped by county |
| Grouping guarantee | All years of a county stay in one fold role |
| Validation | The next outer fold; hyperparameters selected there, then refit on train+validation |
| Year handling | Every cohort year (2019-2022) required in *every* train/val/test partition |
| Temporal protocol | Separate climate-free LOYO, 2019-2022, Random Forest only |
| Seeds | 0/1/2 for stochastic regressors; averaged within fold, then mean and population sd across folds |
| Aggregation | Patch representations pooled to county-year *before* the loss |
| Cohort integrity | Data contracts with cohort and complete-patch identity hashes; runs fail on drift |
| Scale | 406 counties, 1,038 county-years, 4 years |

## Where it matches or exceeds normal practice

**County grouping is done correctly, and many papers get this wrong.** Because all
years of a county are forced into a single fold role, the same county never
appears in both train and test. That closes the most common leak in county-level
yield work, where a county-year is treated as an independent sample and the same
county's other years leak into training. Scores here are not inflated by that
mechanism.

**Hyperparameter hygiene is clean.** Ridge alpha is chosen on a held-out
validation partition, never on test, then refit on train+validation. Preprocessing
is fitted on training data only. This is stated explicitly in the canonical rules
and enforced in code.

**Aggregation happens before the loss.** Patch representations are pooled to
county-year and the error is computed there, matching the resolution of the label.
Computing per-patch losses against a repeated county label — common and wrong —
is explicitly forbidden.

**Seed handling is in the right order**: average over seeds within a fold, then
report spread across folds. Reversing that order conflates optimization noise with
fold difficulty.

**Cohort integrity checking is better than the field norm.** Identity hashes and
contract audits that hard-fail on cohort drift are unusual in published yield work
and worth mentioning as a methodological contribution.

**A separate LOYO protocol** matches YieldSAT's temporal-shift setup and is
standard best practice for yield papers.

## Gaps a reviewer would raise

### 1. No spatial-block or buffered cross-validation — the main one

`GroupKFold` assigns counties to folds without regard to geography, so a test
county can sit immediately adjacent to a training county. In the Corn Belt,
neighbouring counties share weather systems, soil associations, management
practice, and sometimes the same Sentinel-2 granule. That is textbook spatial
autocorrelation, and it means the main table measures interpolation between
neighbours more than generalization to new areas.

Confirmed absent in code: no match for buffer, spatial block, contiguity, or
adjacency anywhere in `benchmark_embeddings/`.

The standard remedy is spatial-block CV — group counties into blocks (~50 km),
hold out whole blocks, and drop training counties within an exclusion buffer
(~30 km) of any test block — with plain county-grouped `GroupKFold` reported
alongside as a sensitivity analysis. Expect scores to drop; that drop is the
honest estimate of spatial generalization.

### 2. No spatial-shift protocol at all (no LORO)

YieldSAT — which this repo cites as the inspiration for the 3D-ConvLSTM — runs
three protocols: CV10, **leave-one-region-out (LORO)** for spatial shift, and
LOYO for temporal shift. It reports that standard models degrade by **-22
percentage points of R2 under LORO** and **-19 p.p. under LOYO**.

This benchmark has the temporal half (LOYO) but no spatial analogue. Combined with
gap 1, there is currently *no* protocol here that tests generalization to a new
region. Given that the supervised model is explicitly framed as YieldSAT-inspired,
a reviewer familiar with that paper will look for LORO and notice its absence.

That LORO degradation is larger than the LOYO degradation is the relevant warning:
in YieldSAT's data, spatial shift hurt more than temporal shift, and spatial shift
is exactly the axis this benchmark does not test.

### 3. Uncertainty is reported as population SD over five folds

`ddof=0` over five numbers understates uncertainty, and fold-to-fold spread here is
large — AlphaEarth ranges R2 0.563 (fold 3) to 0.743 (fold 0). Reporting mean and
population SD is defensible and common, but a spatial bootstrap over counties would
give a defensible confidence interval instead of a spread statistic derived from
five points.

### 4. No paired comparison between encoders

Every encoder is evaluated on identical folds and identical held-out county-years,
so **paired** deltas are available essentially for free — per-county differences,
with a CI or sign test. The benchmark currently reports independent means with
spreads, which is a much weaker basis for "encoder A beats encoder B" when the
fold-to-fold spread (~0.08-0.18 R2) is comparable to the between-encoder gap
(AlphaEarth 0.675 vs S2 indices 0.635 = 0.04). On these numbers, an unpaired
comparison cannot support a claim of difference; a paired one might.

### 5. Name the main protocol for what it is

Because every year appears in every partition, the main table is a **spatial**
generalization test with year held constant in distribution. That is a deliberate
and defensible design — it isolates the encoder comparison from year effects — but
readers will otherwise assume generic CV. State it explicitly, and note that
temporal generalization is measured separately by LOYO.

## The wider deep-learning yield literature

The county-level DL yield lineage — You et al. (AAAI 2017), Khaki, Wang &
Archontoulis (CNN-RNN, Front. Plant Sci. 2020), Sun et al. (CNN-LSTM 2019), Fan
et al. (GNN-RNN, AAAI 2022) — converges on a protocol this benchmark does not use.

### Forward temporal holdout is the field default, and LOYO is not equivalent

Those papers evaluate by training on past years and testing on a *later* year.
Khaki et al. forecast 2016, 2017 and 2018 from prior years; You et al. do the same
forward split. The reason is operational: the task is predicting an upcoming
harvest, so any information from after the target season is unavailable at
prediction time.

This benchmark's LOYO is not that. `loyo.py:425` is

```python
train_indices = np.flatnonzero(row_years != held_out_year)
```

so every other year trains the model, future years included:

| Held-out year | Trains on | Future years in training |
| --- | --- | --- |
| 2019 | 2020, 2021, 2022 | **3** |
| 2020 | 2019, 2021, 2022 | **2** |
| 2021 | 2019, 2020, 2022 | **1** |
| 2022 | 2019, 2020, 2021 | 0 |

**Only the 2022 fold is a genuine forecast.** The other three use look-ahead
information — not leakage of the *target* (no county-year appears in both sides),
but leakage of the *era*: the model has seen how the growing seasons after the
held-out year behaved, including any technology, cultivar or management trend.

This is defensible if the claim is "temporal generalization" in the interpolation
sense, and it is what "leave-one-year-out" literally means. It is not defensible
if the paper frames it as forecasting skill. Two options: rename it explicitly as
temporal *interpolation*, or add a forward-chaining variant (train 2019-2021 ->
test 2022; train 2019-2020 -> test 2021) and report that as the operational
number. With only four years the forward-chaining version yields two or three
evaluations, which is thin but honest, and it is exactly what Khaki et al. report
on.

### Spatial structure is treated as signal elsewhere, not just as a leak

Fan et al.'s GNN-RNN adds a graph over neighbouring counties precisely because
neighbours are informative. That cuts both ways for this benchmark: it confirms
strong inter-county dependence exists (supporting the spatial-blocking concern
above), while also showing the field treats that dependence as exploitable
structure. The distinction is that GNN-RNN still evaluates on held-out *years*, so
the spatial dependence never becomes a train/test leak. Here, with random county
folds and years shared across partitions, it can.

### Loose accuracy context

Khaki et al. report RMSE at roughly 8-9% of average yield for Corn Belt corn and
soybean. Expressed the same way, on a cohort mean of **187.0 bu/acre**:

| Representation | RMSE (bu/acre) | as % of cohort mean |
| --- | --- | --- |
| AlphaEarth, 64-D | 12.64 | **6.8%** |
| Sentinel-2 indices, 21-D | 13.41 | **7.2%** |

Those are in a credible range for county corn, and if anything slightly better
than the CNN-RNN figures. Treat that with care: the protocols differ (random
county folds here versus forward temporal holdout there), and a
spatially-interpolating split should be expected to score better. The comparison is
context, not a claim of superiority — and it is another reason the
spatial-block/forward-chaining numbers matter, since those are the ones directly
comparable to this literature.

### Also worth reading

"Out-of-Distribution Generalization in Climate-Aware Crop Yield Prediction with
Earth Observation Data" (arXiv 2510.07350) is recent and addresses exactly the
spatial/temporal OOD question raised here.

## Comparability of reported numbers

"From MODIS to Sentinel-2" reports Sentinel-2 corn at **R2 0.79, RMSE 8.40
bu/acre** (versus MODIS 0.66 / 8.69). The baselines measured here are AlphaEarth
R2 0.675 / RMSE 12.64 and the 21-D S2 index baseline R2 0.635 / RMSE 13.41.

Do not read that as this pipeline underperforming. The cohorts, years, regions,
and CV designs all differ, and their study matched sensors within one workflow
rather than matching against this county cohort. Cross-paper metric comparison is
only meaningful with a shared cohort and protocol; if the paper cites those
numbers, it should be as context, explicitly labelled as non-matched.

The direction of their finding is relevant, though: finer spatial resolution
helped, which is consistent with this benchmark's premise that 10 m Sentinel-2
representations are worth evaluating at county scale.

## Celik et al. (2023) and the EBM

The EBM in the regressor registry traces to Celik et al., which predicts CONUS
cotton yield from multisource inputs — climate, soil, and biophysical remote
sensing — with an Explainable Boosting Machine, chosen for interpretability of
per-feature contributions.

Two observations:

- The EBM's value in that paper is **interpretability**, not raw accuracy. Here it
  is used purely as a fourth accuracy head, with `interactions=5` and no
  inspection of shape functions. If the EBM is retained, extracting its
  per-feature contributions would use it for what it is for and would add an
  explainability angle the other three regressors cannot provide.
- Celik et al. use **multisource** inputs including soil. This benchmark's main
  family is deliberately unfused, with climate confined to an auxiliary family and
  soil absent entirely. That is a defensible scoping decision, but it means the
  EBM is not being asked the question it was introduced to answer.

## Recommended changes, in priority order

1. **Add spatial-block-buffered CV as a second protocol** and report it alongside
   the existing county-grouped folds. Highest-value change; pre-empts the strongest
   reviewer objection.
2. **Add forward-chaining temporal evaluation** (train 2019-2021 -> test 2022, and
   train 2019-2020 -> test 2021) as the operational number, and either rename the
   current LOYO as temporal *interpolation* or report both. This is the protocol
   the CNN-RNN / GNN-RNN line of work uses, so it is what makes your numbers
   comparable to theirs. Cheap: it is a change to the year mask in `loyo.py`, no
   re-extraction.
3. **Add a leave-one-region-out protocol** (state, or USDA agricultural district)
   to match YieldSAT's spatial-shift setup.
4. **Report paired encoder deltas with CIs** on the shared held-out county-years,
   rather than comparing independent means.
5. **Replace the five-fold population SD** with a spatial bootstrap CI, or report
   both.
6. **State in the paper** that the main protocol holds years constant and therefore
   measures spatial generalization, with the temporal axis covered separately.

Items 1-3 change what is measured. Items 4-5 change only how it is reported and are
computable from predictions already written to disk (`predictions.csv` per fold),
with no re-extraction.

If only one thing changes, make it item 1. If two, add item 2 — it is the cheapest
of the three substantive ones and closes the clearest gap against the established
county-level DL yield literature.

## Sources

- YieldSAT: <https://yieldsat.github.io/> and <https://arxiv.org/abs/2604.00940>
- From MODIS to Sentinel-2: <https://ieeexplore.ieee.org/document/11214337/>
- Celik et al. 2023, IEEE GRSL 20, doi 10.1109/LGRS.2023.3303643; code at
  <https://github.com/mf-celik/yieldPred_EBM>
- Khaki, Wang & Archontoulis, "A CNN-RNN Framework for Crop Yield Prediction",
  Front. Plant Sci. 2020: <https://arxiv.org/abs/1911.09045>, code at
  <https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction>
- You et al., "Deep Gaussian Process for Crop Yield Prediction Based on Remote
  Sensing Data", AAAI 2017
- Sun et al., "County-Level Soybean Yield Prediction Using Deep CNN-LSTM Model":
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6832950/>
- Fan et al., "GNN-RNN: A Spatio-Temporal Approach for Crop Yield Prediction",
  AAAI 2022
- "Out-of-Distribution Generalization in Climate-Aware Crop Yield Prediction with
  Earth Observation Data": <https://arxiv.org/html/2510.07350>
