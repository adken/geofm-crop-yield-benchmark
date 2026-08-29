# geofm-crop-yield-embeddings

Code and run artefacts for *From Pretraining to Prediction: Benchmarking
Geospatial Foundation Models for Corn Yield Estimation*.

Five geospatial foundation models — Clay v1.5, Prithvi-EO-2.0-300M-TL, TerraMind
v1 base, Presto and AlphaEarth — are run frozen over Sentinel-2 imagery of the
U.S. Corn Belt, pooled to county-year vectors, and compared against a
vegetation-index baseline and an end-to-end supervised 3D-ConvLSTM under one set
of splits.

The paper covers the method and the results. This file covers how to rerun it.

## Cohort

| | |
| --- | --- |
| Region and period | 13 Corn Belt states, 2019–2022 |
| County-years | 2,076 across 890 counties |
| Sentinel-2 | 61,007 complete sequences, 427,049 patch–timestep files |
| Composites | 7 × 28 days, 15 Apr to 30 Sep |
| Labels | USDA–NASS county corn yield, bushels per acre |

One county-grouped fold manifest is shared by every analysis. A county never
appears in more than one fold, and every year appears in the training,
validation and test partition of every fold.

## Setup

Python 3.12+, then:

```bash
pip install -e '.[tabular]'
```

Three encoders need upstream code that is not a PyPI dependency:

| | Needed for | Provide via |
| --- | --- | --- |
| `terratorch` | Prithvi, TerraMind | `pip install terratorch 'torchgeo==0.9.0'` |
| `nasaharvest/presto` checkout | Presto | `PRESTO_REPO`, the directory *containing* the `presto` package |
| Clay checkout + v1.5 checkpoint | Clay | `CLAY_REPO`, `CLAY_METADATA`, `CLAY_V15_CHECKPOINT` |

Copy `configs/local.env.example` to `configs/local.env`, set the paths for your
machine, and `source` it. Everything below reads from it; the file is untracked
because it differs per host.

The imagery is not distributed here — 427,049 NPZ files, 273 GB compressed and
1.04 TB raw. The Earth Engine exporters under `data/scripts/` regenerate it,
along with the yield, Daymet, CDL and AlphaEarth tables.

## Pipeline, in the order it runs

Each stage writes a `data_contract.json` beside its output recording the cohort
hash, inputs and settings. Later stages refuse to run against contracts that
disagree.

### 1. Cohort and splits

`prepare_alphaearth` converts the matched AlphaEarth table to the canonical
embedding schema; `build_splits` intersects the three input sources and writes
the authoritative five-fold `GroupKFold` manifest on county FIPS.

```bash
python -m benchmark_embeddings.prepare_alphaearth \
  --input "$ALPHAEARTH_CSV" --output "$ALPHAEARTH_OUTPUT"

python -m benchmark_embeddings.build_splits \
  --s2-dir "$COUNTY_PATCH_TIMESTEPS" --s2-daymet-merged "$S2_DAYMET_MERGED" \
  --s2-fips-map "$COUNTY_FIPS_MAP" --alphaearth "$ALPHAEARTH_OUTPUT" \
  --labels "$YIELD_LABELS" --output "$COMMON_SPLIT_MANIFEST" \
  --expected-input-count 427049
```

### 2. Frozen extraction

One adapter per encoder under `frozen/`, all writing the same table:
`county_id, year, patch_id, timestep, backbone, embedding`.

| Encoder | Input | Output | Rows |
| --- | --- | --- | --- |
| Clay v1.5 | 256², 10 bands, DN | 1024-d CLS token | 427,049 |
| Prithvi-EO-2.0-300M-TL | 224², 6 bands, standardised | 1024-d mean of spatial tokens | 427,049 |
| TerraMind S2-6 | 224², 6 bands, reflectance | 768-d mean of spatial tokens | 427,049 |
| TerraMind S2-10 | 224², 10 bands + 2 zero-padded | 768-d mean of spatial tokens | 427,049 |
| Presto | 256², spatially averaged, DN | 128-d per sequence | 61,007 |
| AlphaEarth | precomputed in Earth Engine | 64-d per county-year | 2,076 |

Clay, Prithvi and TerraMind encode each patch–timestep independently. Presto
takes the whole seven-step sequence and emits one row per patch with
`representation_scope=sequence`. AlphaEarth never passes through an encoder.

```bash
python -m benchmark_embeddings.frozen.clay \
  --npz-dir "$COUNTY_PATCH_TIMESTEPS" --metadata "$CLAY_METADATA" \
  --output "$EMBEDDINGS_ROOT/clay_v1_5_cls.parquet" --expected-input-count 427049

python -m benchmark_embeddings.frozen.prithvi \
  --npz-dir "$COUNTY_PATCH_TIMESTEPS" \
  --output "$EMBEDDINGS_ROOT/prithvi.parquet" --expected-input-count 427049

python -m benchmark_embeddings.frozen.terramind \
  --npz-dir "$COUNTY_PATCH_TIMESTEPS" --experiment s2_6_prithvi \
  --output "$EMBEDDINGS_ROOT/terramind_s2_6.parquet" --expected-input-count 427049

python -m benchmark_embeddings.frozen.presto \
  --s2-dir "$COUNTY_PATCH_TIMESTEPS" --presto-repo "$PRESTO_REPO" \
  --output "$EMBEDDINGS_ROOT/presto_s2.parquet" \
  --expected-input-count 427049 --s2-units auto
```

`--experiment s2_10_zero_pad` gives the ten-band TerraMind variant;
`--era5-dir "$ERA5_PATCH_TIMESTEPS"` gives Presto+ERA5.

Two practical notes. Every adapter takes `--preflight-only`, which validates the
cohort and tensor shapes without loading a model — worth running first, since a
full pass is hours. And extraction is I/O-bound rather than compute-bound: on a
shared filesystem set `BENCHMARK_IO_WORKERS=32`, where a single reader sees
about 25 ms per file against 5.9 ms at 32 concurrent readers.

### 3. County aggregation

Patch embeddings are pooled to one vector per county-year by concatenating the
mean and population standard deviation over all patch–timestep rows — the
*joint* form, used by the main benchmark, climate fusion, LOYO and LOSO. The
temporal ablation and the in-season and phenology analyses keep the per-timestep
sequence instead, because the temporal axis is what they measure. This happens
inside the analysis modules rather than as a separate step.

### 4. Benchmark tables

Four heads per representation over the five folds: Ridge as a linear probe with
the penalty selected on the validation fold, and Random Forest, XGBoost and an
Explainable Boosting Machine with fixed settings and seeds 0–2. The
climate-fusion family adds 35 county-level Daymet features after extraction, and
separately evaluates Presto with ERA5-Land supplied inside the encoder.

```bash
python scripts/run_main_table.py \
  --alphaearth-csv data/alphaearth_matched.csv \
  --s2-daymet-merged outputs/cohort_covered/s2_daymet_merged_covered.csv \
  --fips-map data/geometry/county_fips_map.csv \
  --split outputs/cohort_covered/group_kfold_county_tabular.csv \
  --embeddings clay=outputs/embeddings/clay_v1_5_cls.parquet \
  --out-dir outputs/main_table_covered

python -m benchmark_embeddings.regression_benchmark --family climate_fusion \
  --temporal-pool joint --ebm-interactions 0 \
  --out-dir outputs/climate_fusion_covered_joint   # plus the eight input paths
```

### 5. Transfer and ablations

`loyo` holds out one growing season at a time. `run_loso_joint.sh` holds out
each of 13 states in turn — 91 runs — and aggregates them with a 10,000-sample
state-cluster bootstrap. `temporal_ablation` compares mean pooling,
concatenation and a 1D convolution as temporal readouts.

```bash
python -m benchmark_embeddings.loyo --temporal-pool joint --regressor ridge \
  --out-dir outputs/loyo_covered_joint             # plus the six input paths

bash scripts/run_loso_joint.sh
python -m benchmark_embeddings.temporal_ablation ...
```

### 6. Phenology and in-season

`scripts/quantify_phenology_agreement.py` correlates each embedding's first
principal component against LAI, EVI and fPAR per county-year, and measures how
much of each index is linearly recoverable from the full embedding.
`scripts/run_inseason_forecast.py` refits on the first *k* composites for
*k* = 1…7, and in a leave-one-composite-out mode that isolates each growth
stage's marginal contribution. It also writes per-county predictions, from
which `scripts/plot_inseason_anomaly.py` computes the within-county anomaly —
the target with each county's mean removed, which separates recognising a
county from reading its current season:

```bash
python scripts/plot_inseason_anomaly.py \
  --predictions outputs/inseason_covered/inseason_results_predictions.csv \
  --metric r2 --out figures/inseason_anomaly.png
```

`--metric rmse` gives the same decomposition in bushels per acre.

### 7. Supervised reference

A randomly initialised 3D-ConvLSTM trained end-to-end on the same patches and
folds — a three-stage 3D convolutional stem feeding a ConvLSTM that carries
spatial hidden and cell states through the sequence.

```bash
python -m benchmark_embeddings.train --config configs/supervised_s2.yaml \
  --out-dir <run-dir> --fold 0 --seed 20260614 --model 3d_convlstm
```

### 8. Audit

```bash
python -m benchmark_embeddings.experiment_parity --output outputs/experiment_parity.json
```

This is the check that matters: it reads every `data_contract.json` and fails on
cohort, target or patch-identity drift rather than silently intersecting inputs.

## Cluster

`scripts/*.sbatch` are the Slurm equivalents, and `scripts/CLUSTER.md` documents
the JSC JUPITER layout. `scripts/run_all_local.sh` sequences everything without
Slurm and skips stages whose outputs already exist.

## Layout

```
benchmark_embeddings/
  data/          NPZ patch reader, spatial policy, band normalisation, splits
  frozen/        one adapter per foundation model, shared output schema
  models/        supervised networks (3D stem, then ConvLSTM or GRU/LSTM)
  *.py           CLI modules, each also a benchmark-* console script
configs/         experiment configuration
scripts/         Slurm entry points and analysis scripts
tests/           data, model, split and schema checks
outputs/         run contracts, summaries and results
```

`outputs/` carries the contracts, summary tables and result files for the runs
the paper reports, so the published numbers can be traced without rerunning
anything. Checkpoints and the 14 GB of embedding parquets are excluded for
size, as are most per-county prediction files; the in-season predictions are
kept because the appendix figure and table are computed from them.

## Tests

```bash
python -m pytest tests/ -q
```
