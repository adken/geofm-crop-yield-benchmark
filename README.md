# benchmark-embeddings

Benchmarking frozen geospatial foundation model (GeoFM) embeddings for
county-level corn yield estimation in the U.S. Corn Belt.

Five encoders — Clay v1.5, Prithvi-EO-2.0-300M-TL, TerraMind v1 base, Presto and
AlphaEarth — are run frozen over the same Sentinel-2 imagery, pooled to
county-year vectors, and compared against a handcrafted vegetation-index
baseline and an end-to-end supervised 3D-ConvLSTM under one set of splits.

## Cohort

| | |
| --- | --- |
| Region and period | 13 Corn Belt states, 2019–2022 |
| County-years | 2,076 across 890 counties |
| Sentinel-2 patches | 61,007 complete sequences, 427,049 patch–timestep files |
| Patch geometry | 256 × 256 px at 10 m, ten bands |
| Composites | 7 × 28 days: 15 Apr, 13 May, 10 Jun, 8 Jul, 5 Aug, 2 Sep, 30 Sep |
| Labels | USDA–NASS county corn yield, bushels per acre |

Every analysis uses this same cohort and the same county-grouped fold manifest.
A county never appears in more than one fold, and every year appears in the
training, validation and test partition of every fold.

## Pipeline, in the order it runs

**1. Cohort and splits.** `build_splits` intersects the AlphaEarth county-year
table, the Sentinel-2 index table and the patch corpus, then writes the
authoritative five-fold `GroupKFold` manifest on county FIPS. `prepare_alphaearth`
converts the matched AlphaEarth CSV into the canonical embedding schema. Both
write a `data_contract.json` recording the cohort hash.

**2. Frozen extraction.** One adapter per encoder under `frozen/`, each writing
the same table: `county_id, year, patch_id, timestep, backbone, embedding`.

| Encoder | Input | Output | Rows |
| --- | --- | --- | --- |
| Clay v1.5 | 256², 10 bands, DN | 1024-d CLS token per patch–timestep | 427,049 |
| Prithvi-EO-2.0-300M-TL | 224², 6 bands, standardised | 1024-d mean of 196 spatial tokens | 427,049 |
| TerraMind S2-6 | 224², 6 bands, reflectance | 768-d mean of spatial tokens | 427,049 |
| TerraMind S2-10 | 224², 10 bands + 2 zero-padded | 768-d mean of spatial tokens | 427,049 |
| Presto | 256², spatially averaged, DN | 128-d per complete sequence | 61,007 |
| AlphaEarth | precomputed in Earth Engine | 64-d per county-year | 2,076 |

Clay, Prithvi and TerraMind encode each patch–timestep independently. Presto
ingests the whole seven-step sequence and emits one row with
`representation_scope=sequence`. AlphaEarth never passes through an encoder.
Each adapter supports `--preflight-only`, which validates the cohort and shapes
without running the model.

**3. County aggregation.** Patch-level embeddings are pooled to one vector per
county-year by concatenating the mean and population standard deviation over all
patch–timestep rows — the *joint* form. The main benchmark, climate fusion,
leave-one-year-out and leave-one-state-out all use it. The temporal ablation and
the in-season and phenology analyses keep the per-timestep sequence instead,
because the temporal axis is what they measure.

**4. Main benchmark.** `regression_benchmark --family main` fits four heads on
each representation over the five folds: Ridge as a linear probe with the penalty
chosen on the validation fold, and Random Forest, XGBoost and an Explainable
Boosting Machine with fixed preregistered settings and seeds 0–2.

**5. Climate fusion.** `regression_benchmark --family climate_fusion` adds 35
county-level Daymet features after extraction (late fusion) and, separately,
evaluates Presto with ERA5-Land supplied inside the encoder. The two differ in
where climate enters, not only in its source.

**6. Temporal ablation.** `temporal_ablation` compares mean pooling,
concatenation and a 1D convolution as temporal readouts for Clay, Prithvi and
TerraMind, each followed by the same MLP head.

**7. Transfer.** `loyo` holds out one growing season at a time; `probe` with the
state manifest holds out one of 13 states at a time, aggregated by
`loso_aggregate` with a 10,000-sample state-cluster bootstrap.

**8. Phenology and in-season.** `scripts/quantify_phenology_agreement.py`
correlates each embedding's first principal component against LAI, EVI and fPAR
per county-year and measures how much of each index is linearly recoverable from
the full embedding. `scripts/run_inseason_forecast.py` refits on the first *k*
composites for *k* = 1…7, and in a leave-one-composite-out mode that isolates the
marginal contribution of each growth stage.

**9. Supervised reference.** `train` fits a randomly initialised 3D-ConvLSTM
end-to-end on the same patches and folds — a three-stage 3D convolutional stem
feeding a ConvLSTM that carries spatial hidden and cell states through the
sequence.

**10. Audit.** `experiment_parity` cross-checks every `data_contract.json` and
fails on cohort, target or patch-identity drift rather than silently
intersecting inputs.

## Entry points

Everything is a CLI module, each also exposed as a `benchmark-*` console script.

| Module | Role |
| --- | --- |
| `build_splits` | Five-fold county-grouped manifest. Run first. |
| `prepare_alphaearth` | AlphaEarth CSV to canonical embedding schema. |
| `frozen.clay` / `.prithvi` / `.terramind` / `.presto` | Frozen extraction, one per encoder. |
| `regression_benchmark` | Ridge / RF / XGBoost / EBM tables, main and climate-fusion families. |
| `probe` | Single-representation Ridge evaluation, used for LOSO. |
| `loso_aggregate` | Per-state aggregation with cluster bootstrap. |
| `temporal_ablation` (+ `_aggregate`) | mean vs concat vs Conv1D readout. |
| `loyo` | Leave-one-year-out. |
| `train` | Supervised 3D-ConvLSTM / GRU / LSTM. |
| `supervised_aggregate` | Aggregates the supervised fold × seed grid. |
| `experiment_parity` | Cross-family contract audit. |

`scripts/run_all_local.sh` sequences the whole thing without Slurm. The
`.sbatch` files under `scripts/` are the cluster equivalents.

## Repository layout

```
benchmark_embeddings/
  data/          NPZ patch reader, spatial policy, band normalisation, splits
  frozen/        one adapter per foundation model, shared output schema
  models/        supervised networks (3D stem, then ConvLSTM or GRU/LSTM)
  *.py           the CLI modules listed above
configs/         experiment configuration
scripts/         Slurm entry points and analysis scripts
tests/           data, model, split and schema checks
outputs/         run artefacts: contracts, summaries, results
```

## Setup

Python 3.12 or newer. Install the package and its tabular extras:

```bash
pip install -e '.[tabular]'
```

Three encoders need additional upstream packages: `terratorch` for Prithvi and
TerraMind, and a checkout of `nasaharvest/presto` for Presto — Presto's own
package is not on PyPI under that name and must be pointed at with
`PRESTO_REPO`. Clay needs its official checkout and the v1.5 checkpoint.

Copy `configs/local.env.example` to `configs/local.env` and set the paths for
your machine. That file is deliberately untracked because it differs per host;
everything else reads its paths from it.

## Reproducing

```bash
source configs/local.env

# 1. cohort and splits
python -m benchmark_embeddings.build_splits ...
python -m benchmark_embeddings.prepare_alphaearth ...

# 2. extraction, one encoder at a time (--preflight-only to validate first)
python -m benchmark_embeddings.frozen.clay --npz-dir "$COUNTY_PATCH_TIMESTEPS" ...

# 3. benchmark tables
python scripts/run_main_table.py ...
python -m benchmark_embeddings.regression_benchmark --family climate_fusion ...

# 4. transfer and ablations
python -m benchmark_embeddings.loyo --temporal-pool joint --regressor ridge ...
bash scripts/run_loso_joint.sh
python -m benchmark_embeddings.temporal_ablation ...
```

Each command writes a `data_contract.json` beside its results. Run
`experiment_parity` at the end to confirm the families agree on cohort, target
and patch identity.

## Data

The Sentinel-2 patches and the embedding parquets are not in this repository —
94 GB and 14 GB respectively. The Earth Engine download scripts under
`data/scripts/` regenerate the imagery, and the extraction modules regenerate
the embeddings from it. Small run artefacts, contracts and summary tables are
included so the reported numbers can be traced without rerunning anything.
