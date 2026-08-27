# Running on the cluster

## Current setup: JUPITER under `3d-abc`

**JUPITER compute nodes do not mount `/p` (JUST). Only `/e` (exasm).** A login
node sees both, which is why a path can work interactively and then fail inside
every job. Verified by `srun`: `/p/project1/3d-abc/adriko1` and
`/p/scratch/geofm4eo/adriko1/US/T7` are both invisible from a compute node.

Everything a job touches therefore lives under `/e`:

| | Path |
| --- | --- |
| Code | `/e/project1/3d-abc/adriko1/benchmark-embeddings` |
| Environment | `source /e/project1/3d-abc/adriko1/EODeepLearning/activate.sh` |
| Clay checkpoint | `/e/project1/3d-abc/adriko1/clay/` |
| HuggingFace cache | `/e/project1/3d-abc/adriko1/hf_cache` |
| Tiles (427,049 files, 273 GB) | `/e/project1/3d-abc/adriko1/datasets/US/T7` |
| Source corpus (read-only, being retired) | `/p/scratch/geofm4eo/adriko1/US/T7` |

SLURM account is `3d-abc`, not `geofm4eo`. JUPITER has a `booster` partition for
GPU work, so the four `extract_*.sbatch` files are correct as written. The
analysis scripts request `batch`, which is unverified here — those steps run
comfortably from the Parquets locally anyway, so confirm with `sinfo -s` only if
you intend to submit them.

All ten batch scripts now carry `--account=3d-abc`. Five of the analysis ones
previously had no account line at all and would have been rejected at submission.

Point `BENCHMARK_ENV_SETUP` at that activate script so jobs are self-contained:

```bash
export BENCHMARK_ENV_SETUP=/e/project1/3d-abc/adriko1/EODeepLearning/activate.sh
```

**`unset PYTHONPATH` first.** The retired `geofm4eo` venv under `/p` leaks in
through `PYTHONPATH`, which takes precedence over the active environment and
produces a NumPy C-extension error naming a `cpython-311` `.so` under a Python
3.13 interpreter. If that error appears, `PYTHONPATH` is the cause, not the
environment.

Group membership carries across projects, so reading `geofm4eo` data while
charging `3d-abc` compute is fine. Data transfers between `/p` and `/e` must run
somewhere that mounts both — a transfer partition if `sinfo -s` shows one, the
login node otherwise. Not a compute node.

The sections below marked *historical* record the geofm4eo scratch incident and
are kept for the reasoning, not as instructions.

---

**The canonical run order is in `README.md` under "Canonical run sequence"** —
seven steps, from the AlphaEarth conversion through to the experiment-parity
audit. That is authoritative. This file does not restate it; it covers only what
differs when the work runs on JSC rather than locally.

| README step | Local stage | On the cluster |
| --- | --- | --- |
| 1. AlphaEarth to canonical schema | `run_all_local.sh prepare` | login node, seconds |
| 2. Build the five-fold manifest | `run_all_local.sh prepare` | login node, ~10 s |
| 3. Four frozen-encoder preflights | `run_all_local.sh preflight` | login node, read-only |
| 4. Extract Clay / Presto / Presto+ERA5 / Prithvi / both TerraMind | `run_all_local.sh extract` | **`sbatch scripts/extract_*.sbatch` — the GPU work** |
| 5. Main + climate-fusion regressors | `main`, `climate` | `sbatch run_main_regression_benchmark.sbatch`, `run_climate_regression_benchmark.sbatch` |
| 6. Temporal ablation, LOYO, supervised grid | `temporal`, `loyo`, `supervised` | `sbatch run_temporal_ablation.sbatch`, `run_main_loyo.sbatch`, `submit_supervised_cv.sh` |
| 7. Aggregate + parity audit | `aggregate`, `parity` | login node |

Only **step 4** genuinely needs the cluster. Steps 5-7 run from the Parquet
embeddings and are comfortable locally — which is why the recommendation below is
to extract in place and bring back only the Parquets.

Tiles live at `/e/project1/3d-abc/adriko1/datasets/US/T7`. Keep them there: 273 GB
of imagery reduces to 9.1 GB of embeddings across the five encoder tables, a 31x
reduction overall and ~153x per encoder. Only the Parquets need to come home.

Those tables store one row per patch-timestep, which is why they are gigabytes
rather than megabytes. Once `run_main_table.pooled()` reduces them to county-year
mean+std features the whole benchmark is about 126 MB in memory.

## What changed in these scripts

Three fixes were needed before the four `extract_*.sbatch` files would run here:

- added `--account=3d-abc` and `--partition=booster` (they had neither, and the
  analysis sbatch files had no account line either)
- threaded `--undersize-policy` through, which had been added to the CLIs and to
  `run_all_local.sh` but not to these
- `*_EXPECTED_INPUT_COUNT` still defaults to **77813**, which aborts on any other
  corpus — override it with the count from the coverage scan below

## The target

**953 counties / 2,180 county-years** — the cohort where AlphaEarth and the 21-D
index baseline are both complete, written out as
`data/sources/cohort_2180_keys.txt`. Reaching it would let the six-encoder
comparison run at the same power as the two-representation result already
measured there (13-fold leave-one-state-out, significant effects).

Local tiles reach 406 of those 953 counties. The gap:

| | County-years | Counties |
| --- | --- | --- |
| Target cohort | 2,180 | 953 |
| Covered by local tiles | 1,038 | 406 |
| **Still needed** | **1,142** | **547** |

Concentrated in the western and peripheral states — Kansas (82 counties missing
of 96), Missouri (72 of 75), Minnesota (58 of 71), Kentucky (56 of 60), South
Dakota (50 of 58), Michigan (45 of 47), North Dakota (41 of 44). The Corn Belt
core is nearly complete: Iowa is missing 6 of 98, Illinois 12 of 95, Indiana 13
of 83.

That matters because the peripheral states are exactly where AlphaEarth failed in
the leave-one-state-out test (Kentucky -0.507, Michigan -0.601) — so the counties
still missing are the ones carrying the most information about transfer.

Roughly 53,000 further tiles, about 42 GB to process.

## 1. Check coverage and capture the cohort (read-only)

```bash
python scripts/match_tiles.py \
  --tile-dir /e/project1/3d-abc/adriko1/datasets/US/T7 \
  --keys data/sources/cohort_2180_keys.txt \
  --out-keys data/sources/cohort_covered_keys.txt
```

`--out-keys` writes the county-years the tiles actually support -- the scan
already reported 2,076 of 2,180 across 890 counties. That file is the definitive
cohort; everything downstream keys off it.

`--keys` takes the plain key list above, or any CSV/XLSX with county and year
columns. Using the key list is preferable here: it pins the target to the exact
2,180 cohort rather than the wider 2,920.

Scans filenames only, never opens an NPZ. Reports how much of the 2,180-county-year
target has complete seven-interval tiles, plus counties, per-year coverage, and
how many states clear 40 county-years — the last figure gates leave-one-state-out.

**Check the `unparsed names` count first.** If the cluster tiles use a different
naming convention it will report them as unparsed rather than guessing, and the
whole scan is meaningless until that is fixed. Locally: 0 unparsed of 77,813.

## 2. Curate a working directory

```bash
python scripts/match_tiles.py \
  --tile-dir /e/project1/3d-abc/adriko1/datasets/US/T7 \
  --keys data/sources/cohort_2180_keys.txt \
  --link-dir /e/project1/3d-abc/adriko1/datasets/US/T7_matched \
  --dry-run          # drop --dry-run to create
```

Hardlinks by default, so on the same filesystem it costs **no extra storage** —
the same inodes, a second name. It falls back to symlinks across devices, never
touches the source, is idempotent (re-running creates 0), and aborts if a flat
directory would collide two filenames.

Only tiles that are *complete* and *in the target cohort* are linked, so the
resulting directory is the cohort. Verified locally against a 2,920-key target:
66,582 files linked, and a Presto preflight over them returned exactly the
expected county-year count with `incomplete_spatial_patches_excluded: 0`.

Note the linked set carries every interval belonging to a selected tile, so a few
out-of-schedule interval-07 files come along. That is deliberate — the extractors
count then exclude them, which preserves the documented audit trail.

## 3. Extract  (README step 4 — the only step that needs the cluster)

Note the README asks for **six** extractions, not four: Clay, S2-only Presto,
Presto+ERA5, Prithvi, and *both* TerraMind spectral variants
(`s2_6_prithvi` and `s2_10_zero_pad`). Presto+ERA5 additionally needs
`--era5-dir`.

```bash
export COUNTY_PATCH_TIMESTEPS=/e/project1/3d-abc/adriko1/datasets/US/T7
export UNDERSIZE_POLICY=skip
export PYTHON_BIN=$(which python)

# the count printed by step 2 -- NOT the old 77813 default, which will abort
export CLAY_EXPECTED_INPUT_COUNT=<N>
export PRESTO_EXPECTED_INPUT_COUNT=<N>
export PRITHVI_EXPECTED_INPUT_COUNT=<N>
export TERRAMIND_EXPECTED_INPUT_COUNT=<N>

# order: everything that shares the terratorch/Clay environment first,
# Presto last because it needs a separate one (see below)
sbatch scripts/extract_clay.sbatch      "$COUNTY_PATCH_TIMESTEPS" clay_v1_5_cls.parquet
sbatch scripts/extract_prithvi.sbatch   "$COUNTY_PATCH_TIMESTEPS" prithvi.parquet
sbatch scripts/extract_terramind.sbatch "$COUNTY_PATCH_TIMESTEPS" terramind_s2_6.parquet
sbatch scripts/extract_terramind.sbatch "$COUNTY_PATCH_TIMESTEPS" terramind_s2_10.parquet
sbatch scripts/extract_presto.sbatch    "$COUNTY_PATCH_TIMESTEPS" presto_s2.parquet
```

The second TerraMind call needs its spectral variant set — `s2_6_prithvi` for
the first, `s2_10_zero_pad` for the second — so they are two runs, not one.

Clay additionally needs `CLAY_REPO`, `CLAY_V15_CHECKPOINT` and `CLAY_METADATA`:

```bash
R=/e/project1/3d-abc/adriko1/benchmark-embeddings
export CLAY_REPO=$R/clay/model               # NOT clay-foundation-model
export CLAY_METADATA=$R/clay/model/configs/metadata.yaml
export CLAY_V15_CHECKPOINT=<path to clay-v1.5.ckpt>
```

`CLAY_REPO` must be the directory containing `src/module.py` — `load_clay_encoder`
tries `claymodel.module` then `src.module`, and v1.5 provides the latter. A
sibling `clay-foundation-model` checkout has neither and fails with
`cannot import ClayMAEModule`.

Either metadata file works: `clay/metadata.yaml` is a trimmed copy holding only
the `sentinel-2-l2a` block, and that block is identical to the official
`configs/metadata.yaml`, which additionally carries seven other sensors.

Note `sync_to_cluster.sh` excludes `clay/*.ckpt`, so the 4.9 GB checkpoint is
never synced by default — pass `--with-clay-ckpt` if it is not already there.
Presto needs `PRESTO_REPO` plus the undeclared dependencies listed in
`README.md` §2b.

### HuggingFace weights: pre-download, compute nodes are offline

Three of the four encoders fetch weights from the Hub at construction time, and
JUPITER compute nodes have no outbound network. Clay is the surprising one: its
`ClayMAE.__init__` builds a DINOv2 ViT-L/14 *teacher* with
`timm.create_model(..., pretrained=True)` purely as a training artefact —
inference never uses it, but construction fails without it, surfacing as
`huggingface_hub.errors.LocalEntryNotFoundError`.

On the login node:

```bash
export HF_HOME=/e/project1/3d-abc/adriko1/hf_cache

python - <<'PY'
import timm
timm.create_model("vit_large_patch14_reg4_dinov2", pretrained=True, num_classes=0)
print("clay teacher cached")
PY
# Prithvi and TerraMind: fetch their checkpoints the same way before submitting,
# e.g. by running each extractor once with --max-files / --max-sequences small.
```

Then in every job:

```bash
export HF_HOME=/e/project1/3d-abc/adriko1/hf_cache
export HF_HUB_OFFLINE=1
```

`HF_HOME` must match between the download and the job, or the cache is invisible.
`HF_HUB_OFFLINE=1` turns a silent network stall into an immediate error.

### Presto: separate environment, and skip the ERA5 variant

Presto cannot share the terratorch environment — `openmapflow` pins
`pandas==1.5.3`. Point `PRESTO_ENV_SETUP` at its own activate script; the Presto
job prefers it over `BENCHMARK_ENV_SETUP`, so both can stay exported:

```bash
export PRESTO_ENV_SETUP=<presto environment>/activate.sh
sbatch scripts/extract_presto.sbatch "$COUNTY_PATCH_TIMESTEPS" $OUT/presto_s2.parquet
```

**Presto+ERA5 is out of scope for this revision.** It needs co-located ERA5-Land
patches for all 427,049 tiles, which do not exist and would require a fresh
Earth Engine export campaign on the scale of the Sentinel-2 one. It is also not
needed for the main table: that compares six *Sentinel-2* representations, and
the paper's climate experiment is Daymet late fusion at county level, handled by
the shared probe rather than by this extractor. Run `presto_s2` only.

### The four CLIs are not consistent

Worth keeping to hand — these differ per encoder and the errors are unhelpful:

| | Clay | Prithvi | TerraMind | Presto |
| --- | --- | --- | --- | --- |
| input flag | `--npz-dir` | `--npz-dir` | `--npz-dir` | `--s2-dir` |
| cap flag | `--max-files` | `--max-sequences` | `--max-files` | `--max-sequences` |
| units flag | *(auto)* | `--source-units` | `--source-units` | `--s2-units` |
| size flag | `--spatial-size` | `--source-size` | `--source-size` | `--spatial-size` |
| default batch | 8 | **1** | 8 | 256 |
| rows emitted | one per file | one per file | one per file | one per **sequence** |

Prithvi's default batch size of 1 is a performance trap on 427,049 inputs —
raise `PRITHVI_BATCH_SIZE`. And note Presto is the only extractor whose row count
is 61,007 rather than 427,049, because it pools the temporal sequence internally.

### Clay dependency chain

Beyond the repo itself, Clay v1.5 needs `lightning`, `python-box` (imports as
`box`), `timm`, `einops`, `torchvision`, and `jsonargparse[signatures]`. The last
is easy to miss: without the `[signatures]` extra Lightning binds
`ArgumentParser = object` and checkpoint loading fails with
`TypeError: object() takes no arguments`.

**Order: Clay, Prithvi, TerraMind, then Presto.** Clay, Prithvi and TerraMind
share one environment; Presto needs a separate one because `openmapflow` pins
`pandas==1.5.3` and drags in the whole Earth Engine training stack (see
"Presto dependencies" below). Doing the three together and Presto last means one
environment switch instead of several.

The tradeoff against the older "Presto first" advice: Presto is 0.82M parameters
and validates the whole path cheaply, whereas Clay is the most expensive job to
discover a path error in. Get the cheap validation another way — run Clay once
with `--max-sequences 8` and confirm it writes a Parquet before queueing the
full extraction:

Note Clay's CLI differs from Presto's: `--npz-dir` rather than `--s2-dir`,
`--metadata` is required, and the cap is `--max-files` (files, not sequences).

```bash
python -m benchmark_embeddings.frozen.clay \
  --npz-dir  "$COUNTY_PATCH_TIMESTEPS" \
  --metadata "$CLAY_METADATA" \
  --undersize-policy skip \
  --preflight-only
```

`--preflight-only` returns before the model loads, so it needs neither the
checkpoint nor a GPU — it builds the index, prints the contract and reads three
sample patches. Run it before anything else and read `county_years`.

That costs minutes and catches the failures that actually happen here: missing
checkpoint paths, HuggingFace with no outbound network, a wrong
`*_EXPECTED_INPUT_COUNT`, and an unreadable staged tile.

### Presto spatial mode

The published run reduced each composite to a spatial mean over the patch.
Presto is a pixel time-series model, so that mean is a spectrum belonging to no
pixel it was pretrained on. `--spatial-mode sample` instead encodes real pixel
sequences and averages the embeddings — encode-then-average rather than
average-then-encode, which differ because the encoder is nonlinear.

```bash
export PRESTO_SPATIAL_MODE=mean          # published configuration (default)

# or the faithful reading:
export PRESTO_SPATIAL_MODE=sample
export PRESTO_PIXEL_SAMPLES=64
export PRESTO_NONFINITE_POLICY=mask      # drop cloud-masked and all-zero fill
export PRESTO_BATCH_SIZE=8               # batch x K must fit in VRAM
```

Sampling multiplies encoder calls by K, so ~61k sequences becomes ~3.9M forward
passes. Seeds derive from `sha256(county-year-patch-seed)`, so a resumed job
draws the same pixels. Smoke-test first:

```bash
sbatch scripts/extract_presto.sbatch "$COUNTY_PATCH_TIMESTEPS" /tmp/smoke.parquet
# with PRESTO_SPATIAL_MODE=sample, PRESTO_PIXEL_SAMPLES=4, and
# --max-sequences 8 added, to confirm the encoder accepts B*K rows
```

Run both modes and compare before choosing what goes in the revised table.

## 4. Bring back only the embeddings

```bash
rsync -av user@cluster:/e/project1/3d-abc/adriko1/benchmark-embeddings/outputs/embeddings/*.parquet* \
  outputs/embeddings/
```

Everything downstream — probe, regression benchmark, temporal ablation, LOYO —
runs locally from the Parquets. No tiles needed.

## What to watch

| Measure | Local today | Target |
| --- | --- | --- |
| Counties in the GeoFM cohort | 406 | **953** |
| County-years | 1,038 | **2,180** |
| States with >=40 county-years | 5 | **13** |

The last row is the one that matters. On the 2,180 cohort the two tabular
representations already give a significant 13-fold leave-one-state-out result
(see `VALIDATION_REVIEW.md` section 3b). Getting the four GeoFM encoders onto the
same cohort is what puts them in that same well-powered comparison instead of the
underpowered 406-county one.

## Recovery sequence (after the scratch over-deletion) — *historical*

Superseded by the JUPITER move: the cohort now lives on `/e`, and
`cohort_covered_keys.txt` already exists, so the scan does not need repeating.
Kept because the reasoning about copy-vs-delete still applies.

Paths as they were on JUWELS/JUST:

| | Path |
| --- | --- |
| Project copy (intact, 976,434 files) | `/p/project1/geofm4eo/adriko1/datasets/county_npz/US/T7/` |
| Scratch working copy | `/p/scratch/geofm4eo/adriko1/US/T7_matched` |
| Scripts and key lists | `/p/scratch/geofm4eo/adriko1/scripts/` |

Order matters: scratch is the volume that ran out of space, so free it before
writing anything to it.

```bash
S=/p/scratch/geofm4eo/adriko1

# 1. clear the damaged partial copy -- every file is on project1
rm -rf $S/US/T7/

# 2. confirm room: need ~342 GB for 427,049 files
df -h $S

# 3. scan project1, write both outputs to scratch (login node: read-only, quick)
python $S/scripts/match_tiles.py \
  --tile-dir /e/project1/3d-abc/adriko1/datasets/US/T7 \
  --keys     $S/scripts/cohort_2180_keys.txt \
  --out-keys $S/scripts/cohort_covered_keys.txt \
  --out-list $S/scripts/cohort_files.txt

# expect: 2,076 covered county-years, 890 counties, 0 unparsed names

# 4. stage in batch, not on the login node
sbatch scripts/stage_tiles.sbatch $S/scripts/cohort_files.txt $S/US/T7_matched

# 5. verify before extracting
ls $S/US/T7_matched | wc -l              # 427,049
wc -l < $S/scripts/cohort_covered_keys.txt   # 2,076
```

Never `--link-mode move` against project1 -- that would relocate the only
remaining originals. Copy or symlink only.

If step 2 shows less than ~400 GB free, use `--link-mode symlink --link-dir
$S/US/T7_matched` instead of staging: same cohort scoping, no copy, extraction
reads through to project1.

## What went wrong the first time

A delete pass has to compute "everything *not* in the cohort" and act on it, so
any mismatch in that set removes real data. It ran on the login node for 48 hours
with no log and no completion check, and the corpus fell from ~1M files to
192,736 -- below the 427,049 the cohort needs. The scan then correctly reported
370 county-years instead of 2,076.

Two changes prevent a repeat: act only on files positively identified (`move`
leaves the remainder by construction; copy touches nothing else), and run it as a
batch job, which `stage_tiles.sbatch` does -- it logs, checks the final count and
exits non-zero on a short stage.

## Presto locally: don't

An earlier version of this file recommended running Presto on the local tiles
since it is CPU-cheap. **That was wrong**, and the cluster scan shows why.

| | Local | Cluster |
| --- | --- | --- |
| County-years of the 2,180 target | 1,038 (47.6%) | **2,076 (95.2%)** |
| Spatial tiles | 10,435 | 61,013 |
| **Tiles per county-year** | **6.3** | **29.4** |
| Patch-timestep files | 73,045 | 427,049 |

The local corpus is **~4.7x patch-subsampled**, not just missing counties. The
paper's area-adaptive design allows up to 16/32/48/64 patches per county by size;
locally the maximum observed is 15 with a median of 7. County features are the
spatial mean and standard deviation across patches, so pooling over ~5x fewer
patches changes the features themselves — a local run cannot reproduce the
published numbers even on the county-years the two corpora share.

Run all six extractions on the cluster, against the same curated directory.

## What the cluster scan means

**2,076 of 2,180 county-years, 95.2%.** That is the important number: the GeoFM
cohort can now be essentially the same as the tabular cohort, instead of the
1,038 the local tiles allow. It puts the six-encoder comparison into the same
well-powered protocol where the two tabular representations already give a
significant 13-fold leave-one-state-out result.

Check the counties and the states-with->=40 line in the `match_tiles.py` output —
those, not the county-year count, determine whether leave-one-state-out runs at
13 folds or 5.

Volume: 427,049 files, roughly 342 GB at ~0.8 MB each. It stays on the cluster;
only the Parquets come back.

## Presto dependencies (wherever you run it)

`import presto` pulls the entire Earth Engine training stack, none of it declared
in `pyproject.toml`. Verified chain:

```bash
git clone https://github.com/nasaharvest/presto /path/to/presto
pip install earthengine-api webdataset hurry.filesize geopandas \
            google-cloud-storage xarray einops
pip install --no-deps openmapflow     # --no-deps REQUIRED: it pins pandas==1.5.3
```

Verify — should report **0.82M params**:

```bash
python -c "import sys; sys.path.insert(0,'/path/to/presto'); import presto;
print(sum(p.numel() for p in presto.Presto.load_pretrained().parameters())/1e6,'M')"
```

## Before extraction: verify the tiles and the three representations agree

```bash
python scripts/verify_cohort.py \
  --alphaearth-csv   data/sources/embeddings_with_yield_matched.csv \
  --s2-daymet-merged data/sources/s2_daymet_merged_matched.xlsx \
  --fips-map         data/geometry/county_fips_map.csv \
  --cohort-keys      /e/project1/3d-abc/adriko1/scripts/cohort_covered_keys.txt \
  --tile-dir         /e/project1/3d-abc/adriko1/datasets/US/T7 \
  --expect-files     427049 \
  --expect-county-years 2076
```

Checks two things that fail silently: a short file copy, and a representation
missing columns for county-years the others have. Exits non-zero on either.
Filenames only, so it costs a metadata scan rather than reading 273 GB.

It reports the **paired cohort** — the intersection of the tile cohort with
AlphaEarth, S2 indices and Daymet — which is the N the main table will actually
use and the number the paper should quote. On the 2,180 cohort it reproduces
953 counties and 13 states with >=40 county-years.

Note S2 indices are the binding constraint: 2,180 complete rows against 2,920
for both AlphaEarth and Daymet. If the paired count comes back below 2,076, that
is where to look first.

## Between the scan and extraction: rebuild the manifest on the covered cohort

```bash
python scripts/build_splits_tabular.py \
  --alphaearth-csv   data/sources/embeddings_with_yield_matched.csv \
  --s2-daymet-merged data/sources/s2_daymet_merged_matched.xlsx \
  --fips-map         data/geometry/county_fips_map.csv \
  --restrict-keys    data/sources/cohort_covered_keys.txt \
  --out-dir          outputs/cohort_covered
```

This puts every representation -- the four encoders, AlphaEarth and the index
baseline -- on one identical cohort, which is what makes the main table paired
and removes the cross-cohort caveat. It writes both the county-grouped 5-fold
manifest (the main table) and the leave-one-state-out manifest.

Expected from 2,076 county-years / 890 counties: all 13 states retain >=40
county-years, so leave-one-state-out runs at 13 folds.

## After extraction

```bash
python scripts/run_main_table.py \
  --alphaearth-csv   data/sources/embeddings_with_yield_matched.csv \
  --s2-daymet-merged data/sources/s2_daymet_merged_matched.xlsx \
  --fips-map         data/geometry/county_fips_map.csv \
  --split            outputs/cohort_covered/group_kfold_county_tabular.csv \
  --embeddings presto=outputs/embeddings/presto_s2.parquet \
  --embeddings clay=outputs/embeddings/clay_v1_5_cls.parquet \
  --embeddings prithvi=outputs/embeddings/prithvi.parquet \
  --embeddings terramind=outputs/embeddings/terramind_s2_6.parquet \
  --out-dir    outputs/main_table
```

It intersects every supplied representation to one common cohort so the
comparison is paired, and runs all four heads. Verified against the known table:
reproduces AlphaEarth 0.770 / 0.812 / 0.830 and S2 indices 0.758 exactly.
