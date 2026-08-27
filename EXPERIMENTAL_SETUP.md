# Revised experimental setup

Written against the re-extracted cohort of August 2026. Every number here was
verified by `scripts/verify_cohort.py` or read from a run's provenance sidecar;
nothing is carried over from the original submission.

## 1. Cohort

The evaluation cohort is **2,076 county-years across 890 counties**, 2019–2022.
It is the intersection of three requirements, applied in this order:

| Stage | County-years | Counties |
| --- | --- | --- |
| Counties with USDA-NASS corn yield and a tabular record | 2,920 | — |
| AlphaEarth embeddings **and** all 21 Sentinel-2 index columns complete | 2,180 | 953 |
| Covered by a complete seven-composite Sentinel-2 tile sequence | **2,076** | **890** |

The binding constraint at the second stage is the Sentinel-2 index table: 2,180
of 2,920 rows have all 21 EVI/LAI/FPAR columns, while AlphaEarth and Daymet are
complete for all 2,920. The third stage loses a further 104 county-years for
which no complete tile sequence exists.

Per year: 440 (2019), 769 (2020), 443 (2021), 424 (2022). Thirteen states retain
at least 40 county-years, which is what makes the leave-one-state-out protocol
run at 13 folds rather than 5.

The cohort is written to `cohort_covered_keys.txt` and its SHA-256 is recorded in
`outputs/cohort_covered/cohort_contract.json`. Every downstream artefact keys off
that file, so a result can be traced to the exact cohort that produced it.

## 2. Imagery

427,049 Sentinel-2 L2A patches, forming **61,007 complete seven-composite
sequences**. Composites are 28-day windows from 15 April to 30 September; the
last two both fall in September, so the zero-based month vector Presto receives
is `[3, 4, 5, 6, 7, 8, 8]`.

Each patch is 256 × 256 pixels at 10 m — 6.554 km² of ground — with ten bands
(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12). Patch counts per county-year
follow the area-adaptive design: minimum 1, **median 32**, maximum 64.

That gives a median of 210 km² sampled per county-year against a median county
land area of 1,476 km², so **the patch encoders see about 13% of each county**
(10th percentile 5.9%, 90th 19.1%). Because the patch grid is fixed across years,
the 61,007 sequences occupy 25,425 distinct ground locations totalling
166,625 km². AlphaEarth and the Sentinel-2 index baseline are county-wide
aggregates and see all of it. The 13% is CDL corn-selected rather than random,
which is a defensible choice, but it is not the same view of a county and the
comparison should be read with that in mind.

173 county-years fall below 5% coverage and one is represented by a single patch.
In that last case the standard-deviation half of the pooled feature vector is
identically zero.

Both exclusion policies are set to `error` rather than `skip`. A 2,000-file
random sample of the corpus returned 2,000 patches at exactly 256 × 256, so no
patch is undersized and the strict policy costs nothing; `skip` would force a
zip-header read of all 427,049 files inside every extraction and training job.
The 4.4% undersize rate reported in the original setup was a property of an
earlier local export, not of this corpus.

## 3. Encoders

Six representations. Four are frozen geospatial foundation models run over the
patches; two are the tabular baselines.

| Representation | Pooling | Row scope | Dim |
| --- | --- | --- | --- |
| Clay v1.5 | CLS token | patch × timestep | 1024 |
| Prithvi-EO-2.0-300M-TL | per-timestep spatial mean | patch × timestep | 1024 |
| TerraMind v1 base, `s2_6_prithvi` | per timestep | patch × timestep | 768 |
| TerraMind v1 base, `s2_10_zero_pad` | per timestep | patch × timestep | 768 |
| Presto | learned global token mean (`eval_task=True`) | patch **sequence** | 128 |
| AlphaEarth | — | county-year | 64 |
| Sentinel-2 indices | — | county-year | 21 |

Presto is the only one of the four that models the temporal axis. It reduces each
composite to a band vector by spatial mean, feeds the resulting seven-step pixel
series through its encoder, and returns one 128-dimensional vector per patch
sequence — 61,007 rows. The other three emit one row per patch-timestep, 427,049
rows each.

Weights and revisions are recorded per run. Presto is nasaharvest/presto at
`ba88a3f` with the bundled `default_model.pt`. Clay uses the v1.5 checkpoint with
the official `configs/metadata.yaml`. Prithvi and TerraMind are loaded through
terratorch.

## 4. County-year aggregation

`run_main_table.pooled()` groups every row belonging to a county-year and
concatenates the mean and the standard deviation across those rows. Feature
widths are therefore Clay 2048, Prithvi 2048, TerraMind 1536, Presto 256.

AlphaEarth (64) and the Sentinel-2 indices (21) are already county-year
quantities and are used as-is, without pooling, so their widths are unchanged.

For Clay, Prithvi and TerraMind those rows are patches × timesteps, so the mean
and standard deviation pool over space **and time together**. April and September
observations contribute equally and their order is discarded; phenological
structure survives only as variance. Presto's rows are sequences, so its
aggregation is spatial only.

This is worth stating plainly because it shapes what the comparison measures.
Three of the four foundation models are being used as spatial encoders with
unweighted temporal averaging attached, against one that does learned temporal
modelling. A margin for or against Presto is not purely a statement about
representation quality.

With 2,076 samples and up to 2,048 features, the encoder rows sit in the p ≫ n
regime. That is the direct reason for the Ridge change in §6.

## 5. Splits

Built by `scripts/build_splits_tabular.py --restrict-keys cohort_covered_keys.txt`.

**Main table — county-grouped 5-fold.** `GroupKFold` on county FIPS, so no county
appears in more than one fold. Within each fold, validation is fold
`(fold + 1) mod 5`; the remainder is training. Every year must appear in all
three partitions of every fold, and the builder raises if it does not. The
manifest has 10,380 rows — 2,076 county-years × 5 folds.

Note that fold membership depends on the whole cohort, because `GroupKFold`
assigns groups greedily to balance fold sizes. The 2,076 manifest is therefore
not a subset of the 2,180 one: counties present in both can land in different
folds. Results from the two cohorts are not directly comparable row by row.

**Leave-one-state-out.** Thirteen states with at least 40 county-years. Each fold
holds out one state for test and the next in the list for validation. 26,988 rows
— 2,076 × 13.

**Leave-one-year-out.** Temporal generalisation, evaluated with Ridge and Random
Forest only, using all counties without spatial constraint. The spatial and
temporal axes are treated separately: the main table constrains space and pools
years, LOYO constrains time and pools counties.

## 6. Regression heads

Four heads on the frozen features, all on the county-grouped folds:

- **Ridge**, α selected from {0.01, 0.1, 1, 10, 100} on the validation fold, then
  refit on train + validation.
- **Random Forest**, 600 trees, `min_samples_leaf=2`, `max_features=1.0`.
- **XGBoost**, 600 rounds, learning rate 0.03, depth 6, subsample 0.8,
  `colsample_bytree` 0.8.
- **EBM**, library defaults.

The three tree heads are averaged over seeds 0, 1, 2 within each fold; the fold
scores are then reported as mean ± population standard deviation across the five
folds. Ridge is deterministic.

Two deliberate departures from the published configuration, both recorded in
`run_contract.json`:

**Ridge standardises features.** The published run did not, which makes the L2
penalty scale-dependent and therefore unequal across representations of different
magnitude. Unscaled, AlphaEarth's R² fell from 0.766 to 0.271 across the alpha
grid; standardised, it stays between 0.75 and 0.77. The published number was an
artefact of where the alpha grid happened to sit relative to the feature scale.

**EBM uses library defaults.** The registry configuration — `max_rounds=1000`,
`max_bins=128`, no early stopping — underfits by roughly 0.09 R².

Both changes were validated on the 2,180 cohort, where Random Forest and XGBoost
reproduce the published values to within 0.008.

## 7. Supervised baseline

A randomly initialised **3D-ConvLSTM**, the architecture the YieldSAT benchmark
reports and the closest available reference point. YieldSAT also evaluates a
3D-LSTM and a Transformer; the repository has a matched 3D-LSTM
(`supervised_s2_lstm`) and no Transformer equivalent. Only the ConvLSTM is run
here, and the omission should be stated rather than left for a reader to notice.

Architecture: a 3D convolutional stem with channels [24, 48, 64], 3 × 3 × 3
kernels, spatial stride 2 at each stage, group normalisation and GELU, dropout
0.15, followed by a single ConvLSTM layer with 64 hidden channels. Spatial hidden
and cell states are carried through the sequence and only the final hidden state
is pooled — which is why this variant rather than the GRU or LSTM ones, both of
which spatially average each timestep before the recurrence and discard the
patch structure.

The trainer uses the same county-grouped manifest, the same
`validation_fold_offset`, and the same county-year averaging before the loss is
computed. Band and target statistics are fitted on training county-years only.
Early stopping selects on validation RMSE and the test fold is scored once.

Three asymmetries against the frozen heads, all favouring the frozen side:

The frozen heads fit on train + validation, four folds of data. The supervised
model holds validation out for early stopping and trains on three. That is 25%
less data, and it is intrinsic — a linear probe with fixed hyperparameters needs
no validation signal and a neural network does. Refitting on train + validation
for the selected epoch count would close it, but the trainer does not support
that.

The frozen heads receive mean and standard deviation across patches; the
supervised model averages patch representations with no dispersion term.

Training samples a capped number of patches per county-year each epoch, while
extraction used every patch.

Because all three lean the same way, the supervised row is better described as a
lower bound on from-scratch performance than as a like-for-like contest.

Planned as 5 folds × seed 0 first, with seeds 1 and 2 added only if the fold
spread turns out to be wide relative to the margin being claimed.

## 8. Excluded from this revision

**Presto + ERA5-Land.** Encoder-input climate fusion, tagged
`auxiliary_climate_fusion` rather than `main_benchmark`. The ERA5 download for
all 427,049 tiles was incomplete at the time of writing; a 500-file sample of
what exists parses correctly, keys to real Sentinel-2 patches, and agrees on
dates, grid and units, so this is a matter of completion rather than validity.
The climate result reported in the paper is Daymet late fusion at county level,
which uses the 35 Daymet columns already present for all 2,076 county-years.

**Presto with pixel sampling.** The extractor now supports
`--spatial-mode sample`, which draws real pixel time series and averages their
embeddings rather than encoding a spatial mean — closer to what Presto was
pretrained on, since a spatial mean is a spectrum belonging to no pixel. The
published configuration (`mean`) is what the numbers here use. The comparison is
a follow-up.

## 9. Verification

`scripts/verify_cohort.py` checks the two things that fail silently: a short file
copy, and a representation missing columns for county-years the others have. It
counts files, rebuilds the seven-composite sequences from filenames, and tests
set membership in both directions against the cohort key list, then derives
AlphaEarth, Sentinel-2 index and Daymet keys with the same functions the manifest
builder uses. It exits non-zero on any mismatch.

On this cohort it reports 427,049 files, 0 unparsed names, 2,076 county-years
with complete tiles, no cohort county-year missing from the tiles and no tile
county-year outside the cohort, and all three tabular representations covering
the full 2,076.

Extraction ran on JUPITER (NVIDIA GH200, aarch64) under Python 3.13.5 and
PyTorch 2.13.0+cu130. The regression table is run locally, in the environment
that reproduced the published values, so that a change in the numbers has one
possible cause rather than two.
