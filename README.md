# Local setup

Working setup steps for `benchmark-embeddings`, verified end to end on
2026-08-17. Run everything from the repo root.

## What this project is

A county-level corn-yield benchmark. It compares frozen geospatial foundation
model embeddings (Clay v1.5, Prithvi-EO-2.0-300M-TL, TerraMind v1 Base, Presto,
AlphaEarth) against a handcrafted 21-D Sentinel-2 vegetation-index baseline and
against three supervised Sentinel-2 networks trained from scratch (3D-ConvLSTM,
GRU, LSTM). Everything is scored in bushels per acre on one shared five-fold
county-grouped split manifest, so a result is always attributable to a specific
encoder plus a specific estimator.

## Prerequisites

| Requirement | Notes |
| --- | --- |
| **Python >= 3.12** in the conda base env | Not 3.10, despite `pyproject.toml`. See "Known issues". |
| Apple Silicon + MPS | `LOCAL_DEVICE=mps`. Neural stages only; the classical regressors stay on CPU. |
| ~4 GB disk for the environment | torch + terratorch + interpret are the bulk. |
| ~90 GB source data | Already present under `data/`. |
| Clay | Fully local under `clay/`: repo at `clay/model`, checkpoint `clay/clay-v1.5.ckpt`, metadata in the repo. Verified. |
| Presto repo | Existing local checkout under `yield-embeddings/presto/presto`. Bundles its own checkpoint. |
| Network to HuggingFace + ~3 GB cache | Prithvi and TerraMind weights auto-download via the TerraTorch registry. No checkpoint files and no `HF_TOKEN` needed — verified. |
| GPU | Optional. Extraction over 77,813 patches is slow on CPU; MPS is supported. |

No database, message queue, cloud credentials, or API keys are needed.

## 1. Environment

Target is the **conda base environment** with **Apple MPS**.

```bash
conda activate base
python -V                     # must be 3.12+; 3.13 is what has run this code
cd /Users/adriko/benchmark-embeddings

# torch first. The default macOS arm64 wheels carry MPS -- do NOT use the
# +cpu index here, that build has no MPS backend.
python -m pip install torch torchvision

python -m pip install -e '.[test,parquet,tabular]'
```

Extras map to: `test` -> pytest; `parquet` -> pyarrow; `tabular` -> xgboost and
interpret (the EBM regressor). All three are needed for the paper's four-head
regressor tables.

Confirm MPS before anything else:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Must print `True`. `configs/local.env` sets `LOCAL_DEVICE=mps` rather than
`auto`, so a missing MPS backend aborts the run immediately instead of quietly
falling back to CPU for a multi-hour job.

Two things to be aware of when installing into `base` rather than a dedicated
env. This stack is heavy — torch, terratorch, torchgeo, interpret — and pip
installing over conda-provided `numpy`/`pandas` can leave the env in a mixed
conda/pip state that is awkward to unpick. If anything conflicts, `conda create
-n benchmark python=3.13` and repeating these steps is the clean escape hatch.

## BLOCKER: 4.4% of the source patches are smaller than 256x256

This stops every workflow that reads raw patches, and the preflights do not
catch it. Found by attempting a real Presto extraction, which died on the very
first county:

```
county_17001_year_2019_x656030_y4403666_interval_00_2019-04-15.npz:
Sentinel-2 patch is 239x339, below benchmark expectation 256x256;
padding is disabled
```

A full header scan of all 77,813 files:

| Measure | Value |
| --- | --- |
| Files at or above 256x256 | 74,387 |
| Files **below** 256 in a dimension | **3,426 (4.4%)** |
| Smallest height / width seen | 186 / 235 |
| In-schedule spatial patches with >=1 undersized timestep | 491 of 11,092 (4.4%) |
| **Benchmark cohort county-years affected** | **288 of 1,038 (27.7%)** |
| Folds affected | 0, 1, 2, 3, 4 — all of them |

`undersize_policy` is hardcoded to `"error"` in all five readers
(`frozen/clay.py:502`, `frozen/presto.py:680`, `frozen/prithvi.py:570`,
`frozen/terramind.py:502`, and `data/county_patches.py:464`, which the
supervised trainer shares). No CLI flag exposes it — `--help` on all four
extractors and on `train` returns zero matches for "undersize".

So the following cannot currently run to completion: all four frozen
extractions, and the entire supervised grid. Unaffected: AlphaEarth and the
21-D Sentinel-2 index baseline, because both are precomputed tables that never
touch the NPZs — which is exactly why the probe runs in "Verified" succeeded.

Why the preflights pass anyway: they audit cohort counts, band metadata, dates,
and coordinates, and validate the spatial policy against only a small sample.
The undersized files are a 4.4% minority, so a sample misses them.

### Resolution: `--undersize-policy skip`

Implemented as an opt-in flag. **The default is unchanged (`error`)**, so nothing
silently starts dropping data; you must ask for the deviation explicitly.

```bash
python -m benchmark_embeddings.frozen.presto ... --undersize-policy skip
```

`configs/local.env` sets `UNDERSIZE_POLICY=skip`, and `run_all_local.sh` threads
it into all 12 extractor invocations (6 preflight, 6 extract). Override per run
with `UNDERSIZE_POLICY=error bash scripts/run_all_local.sh ...`. For the
supervised trainer it is a config key, `data.undersize_policy`, set to `skip` in
all three `configs/supervised_s2*.yaml`.

Available on all four extractors and on the shared reader the supervised trainer
uses. Design decisions worth knowing:

- **Whole spatial patches are dropped, not individual files.** Removing one
  undersized timestep would leave a ragged sequence, which the county
  aggregation and complete-patch accounting both assume cannot happen.
- **Screening happens at index-build time, not mid-iteration.** A multi-hour
  extraction can no longer die 40% of the way through on a 4.4% minority. The
  cost is a header-only pass (`npz_spatial_shape`), which reads NPY headers
  inside the zip without inflating pixels.
- **Every run records the deviation** in `provenance["dataset"]`:
  `undersize_policy`, `undersized_spatial_patches_excluded`,
  `undersized_files_excluded`.
- **The two exclusion reasons stay separate.** Undersize is no longer folded
  into `incomplete_spatial_patches_excluded`, which had briefly inflated it from
  177 to 657.

Result with `skip`, verified on the real cohort:

| Contract field | error | skip |
| --- | --- | --- |
| `source_spatial_patches` | 11,092 | 11,092 |
| `complete_spatial_patches` | 10,915 | **10,435** |
| `incomplete_spatial_patches_excluded` | 177 | 177 |
| `undersized_spatial_patches_excluded` | 0 | **480** |

All four extractors agree on 10,435. Presto and Prithvi report 480 patches /
3,360 files excluded; Clay and TerraMind report 491 / 3,426. **Both are
correct** — Presto and Prithvi screen only complete sequences (10,915), whereas
Clay and TerraMind emit per-timestep rows and so screen every in-schedule patch
(11,092). The 11-patch difference is undersized *and* schedule-incomplete, so it
is excluded either way. An independent header scan of the raw cohort
independently produced 491 / 3,426.

`skip` unblocks the pipeline, but it treats a symptom. Two better options follow:
crop to what the models actually need, and fix the export.

## Model input sizes, checked against official documentation

224 is not a hard requirement for Prithvi or TerraMind. Both accept 256; their
`img_size` attribute records the *pretraining* size rather than a constraint,
because their positional encodings are generated per input rather than stored as
a fixed learned table. Verified by forward pass.

| Encoder | Documented pretraining size | Accepts other sizes? | Verified by forward pass |
| --- | --- | --- | --- |
| **Clay v1.5** | **256**, `patch_size=8` (official spec) | Yes — `DynamicEmbedding`, `img_size=None` | 256 -> 1025 tokens, 224 -> 785, 192 -> 577 |
| **Prithvi-EO-2.0-300M-TL** | **224** (`img_size=(224,224)`; the model card states no size) | Yes — generated 3D sin/cos encodings | 224 -> 197 tokens, 256 -> 257 |
| **TerraMind v1 base** | **224** (`image_size=(224,224)`) | Yes | 224 -> 196 tokens, 256 -> 256 |
| **Presto** | n/a — temporal pixel model | n/a; `pixels.mean(axis=(-2,-1))` collapses the patch | — |

**There is no single size that is native for all three.** Clay was pretrained at
256; Prithvi and TerraMind at 224.

Which means the existing design is deliberate and defensible, not a bug: it runs
each encoder at its own documented pretraining resolution — Clay at 256, Prithvi
and TerraMind cropped 256 -> 224. The cost is that the ground footprint differs
across encoders (2,560 m for Clay and Presto, 2,240 m for Prithvi and TerraMind).
That is a real caveat worth stating in the write-up, but it is a considered
trade-off against running a frozen encoder off its pretraining scale.

### The defaults already use each encoder's design size

Worth stating plainly, because it is easy to miss: `--source-size 256` is a
*harmonization* step, not the model input. What each model actually receives is
already its own design size.

| Encoder | Design size | What it receives by default | Native? |
| --- | --- | --- | --- |
| Clay v1.5 | 256 | 256 (`target_size`) | yes |
| Prithvi-EO-2.0-300M-TL | 224 | 256 harmonized, then cropped to `model_size` 224 | yes |
| TerraMind v1 base | 224 | 256 harmonized, then cropped to `model_size` 224 | yes |
| Presto | n/a | spatial mean over the 256 window | n/a |

So "use the design size for each" is the current behaviour, and it is why the
recommendation is simply to leave the flags alone. The only thing that cannot be
per-encoder is the rejection gate, below.

### Ground sampling distance is not a confound

Prithvi-EO-2.0 was pretrained on 30 m HLS while this benchmark feeds it 10 m
Sentinel-2. That difference does not constrain the comparison, and the official
sources are explicit about it:

- **Prithvi-EO-2.0 is validated on 10 m Sentinel-2 in its own paper.** The
  Sen4Map downstream task is Sentinel-2 at native 10 m (or upsampled to 10 m from
  20 m); Prithvi-EO-2.0-300M and 600M were both fine-tuned on it, with the 600M
  reaching 76.1% F1 on Europe land cover. So 10 m S2 is a documented, supported
  input regime, not a deviation.
- **Sen4Map inputs are 64x64 pixels**, far below 224. Together with the model
  card's GEO-bench claim of tasks "from 0.1m to 15m", this shows Prithvi is
  explicitly intended to transfer across both resolution and input size. Any
  argument that Prithvi must be fed 224 px at 30 m does not survive its own
  evaluation suite.
- **TerraMind's official multitemporal crop config
  (`terramind_v1_base_multitemporal_crop.yaml`) feeds HLS 30 m data through the
  `S2L2A` modality**, that is, in the opposite direction. Its
  `backbone_bands` are exactly
  `["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]` — the same six as this
  benchmark's `s2_6_prithvi` experiment, which independently confirms the band
  interface here is used as intended.

So GSD is not a per-encoder constraint to solve, and it does not confound the
comparison. Both encoders are used across resolutions by their own authors.

Two further points from that TerraMind config confirm the extractor's
configuration: it selects encoder layer indices `[2,5,8,11]` for the base model,
so index 11 is indeed the final layer this benchmark pools; and
`remove_cls_token: False` is consistent with TerraMind having no CLS token, which
is why the extractor mean-pools all spatial tokens rather than inventing one.

One difference worth a deliberate note rather than a fix: that config uses
TerraTorch's temporal wrapper (`backbone_use_temporal: true`,
`backbone_temporal_pooling: concat`) to handle multi-temporal input inside the
backbone. This benchmark instead encodes each timestep independently and pools
afterwards, which is what makes the mean/concat/Conv1D temporal ablation a
controlled comparison. That is a defensible choice, but it is a divergence from
the official multitemporal recipe and is worth stating in the write-up.

### The gate must be identical across encoders

Decisive practical constraint: `regression_benchmark.py:370` and `loyo.py:244`
both raise on complete-patch identity mismatch between encoders. So the
undersize gate cannot be set per encoder — mixing a 224 gate for Prithvi with a
256 gate for Clay produces different cohorts (10,733 vs 10,435) and the benchmark
refuses to run. Whatever footprint you choose, apply it everywhere.

### One case where 224 is free

For **Prithvi and TerraMind only**, `--source-size 224` does not change what the
model sees at all. A center-crop of a center-crop is the same crop: source
256 -> model 224 selects exactly the same pixels as source 224 -> model 224. The
flag only relaxes the rejection threshold. Clay and Presto are different — there
`--spatial-size 224` genuinely changes the input (Clay 785 vs 1025 tokens, Presto
averaging over a smaller window).

### Configuring the footprint

Every extractor now takes an explicit footprint flag, so the harmonized crop is
an auditable argument instead of a buried constant:

```bash
python -m benchmark_embeddings.frozen.clay      ... --spatial-size 224
python -m benchmark_embeddings.frozen.presto    ... --spatial-size 224
python -m benchmark_embeddings.frozen.prithvi   ... --source-size  224
python -m benchmark_embeddings.frozen.terramind ... --source-size  224
```

**Defaults are unchanged at 256**, so nothing shifts unless you ask for it.

Two supporting fixes were required, both real bugs:

- **Clay's token count was hardcoded to 1025** (`1 + (256/8)^2`). It is now
  derived by `clay_token_count(size)`, so it tracks the configured footprint —
  785 at 224 — and still rejects genuine mismatches and non-multiples of 8.
- **`oversize_policy` was a hardcoded string** `"center_crop_to_256_then_center_crop_to_224"`,
  emitted regardless of the sizes actually used. It is now derived. The Prithvi
  test had been asserting that literal while its own fixture cropped 10 -> 8, so
  the recorded provenance was false whenever sizes differed from default; the
  test now asserts the true value.

Measured on the real cohort, with `--undersize-policy skip`:

| Footprint (all encoders) | Complete spatial patches | Undersized dropped |
| --- | --- | --- |
| **256 — recommended default** | 10,435 | 480 |
| 224 | 10,733 | 182 |

Verified by a real Presto extraction at `--spatial-size 224`
(`target_size: [224, 224]`, 128-D, all finite) and by Prithvi's preflight over the
full cohort. Width never blocks a 224 gate — the minimum width across all patches
is 235, so only height ever falls short.

### Recommendation: stay at 256, i.e. change nothing

Since ground sampling distance is not a confound, the case is simply that the
defaults already give
each encoder its design size and the alternative buys ~2.8% more patches.

The 256 defaults already give every encoder its design size, so this is a
"leave the flags alone" recommendation rather than a trade-off.

- It is Clay's documented pretraining size. Clay is the only encoder whose native
  resolution 256 satisfies, and a frozen encoder cannot adapt to a scale shift.
- Prithvi and TerraMind see identical pixels either way, so 224 buys them nothing
  in fidelity.
- The gain is 298 patches out of 10,733, roughly 2.8%, spread across 11,092
  spatial patches that already aggregate to county level. It is unlikely to move
  a county-level score.
- 256 keeps continuity with any embeddings already extracted and with the
  existing split manifest hashes.

Run 224 as a **sensitivity check**, not the primary configuration. It is cheap
now that the flag exists, and reporting that scores are stable across footprints
is a stronger claim than picking one silently. If you do report it, note that
Clay is off-native at 224 while Prithvi and TerraMind are unchanged, so a Clay
score shift there is expected and is not evidence about the other encoders.

## ROOT CAUSE (upstream): the export never honoured `dimensions`

The undersized patches are one tail of a much larger defect. A header scan of all
11,072 spatial patches:

| Measure | Value |
| --- | --- |
| Exactly 256x256, as documented | **394 (3.6%)** |
| Distinct delivered shapes | **997** |
| Height min / median / max | 186 / 262 / 268 |
| Width min / median / max | 235 / **344** / 366 |
| Oversized, so center-cropped | 10,581 |
| Median patch area discarded by that crop | **27.5%** (max 32.9%) |

Note the median aspect ratio: 344/262 = 1.30. For a patch at 40.06N,
1/cos(40.06) = 1.307. A measured patch is 262x341, ratio 1.3015. That is a
lat/lon round-trip signature, not a coincidence.

`data/scripts/gee-sentinel-2.py` built its download request as:

```python
params = {
    "region":       region4326,                     # lat/lon polygon
    "crs":          county_crs_str,                 # county UTM
    "dimensions":   f"{pixels}x{pixels}",           # 256x256
    "crsTransform": [PIX, 0, x0, 0, -PIX, y0],      # UTM grid
}
```

Earth Engine accepts *either* `region` (+ `scale`/`dimensions`) *or*
`crsTransform` + `dimensions` — not both. When `crsTransform` is present it
wins and **`dimensions` is ignored**; the extent is then taken from the bounds of
`region`. Because `region4326` is the UTM patch square reprojected to EPSG:4326,
its bounds transformed back onto the UTM grid are stretched in x by
1/cos(latitude). Hence ~344 columns where 256 were requested.

The undersized patches are the same bug from the other side. Three `.clip()`
calls set the image *footprint* to the county:
`get_cdl_corn_mask`, `get_cdl_corn_union_mask`, and `composite_interval`. For a
patch straddling the county boundary the footprint is smaller than the patch, and
the GeoTIFF export trims the raster to it — down to 186x235. That is why 89.8% of
undersized patches sit on their county's x/y extreme.

### The fix

Four changes, applied to `data/scripts/gee-sentinel-2.py`:

1. **Drop `region` from the download params.** `crs` + `crsTransform` +
   `dimensions` already pin the origin, the 10 m pixel size, the north-up
   orientation, and the 256x256 extent. Deterministic, and no reprojection
   distortion.
2. **Remove `.clip(region4326)` from `composite_interval`**, so an edge patch's
   footprint no longer truncates its raster.
3. **Remove `.clip(region)` from both CDL mask builders**, because the `And()`
   chain in `s2_mask_and_scale` inherits their footprint. Value masking is
   unaffected: `updateMask` still restricts pixels to corn/county/cloud-free.
4. **Assert the delivered shape** before writing, and thread the requested
   `pixels` through `meta` so the check is actually reachable.

Masked pixels already export as 0 rather than nodata — confirmed empirically:
every extracted sequence reports `valid_fraction_min = 1.0`, and a sampled patch
is 100% finite. So this fix does **not** trip `nonfinite_policy=error`. Newly
included out-of-county area will arrive as 0, exactly as non-corn pixels do today.

### Re-export is unnecessary

The area discarded by the centre crop is not lost coverage, so the tiles do not
need re-exporting:

| Quantity | Value |
| --- | --- |
| Grid stride (`stride_m = patch_size_m`) | 2,560 m — non-overlapping tiles |
| Intended tile, from NPZ `bbox` | 2,560 x 2,560 m, centred on the recorded centroid |
| Delivered raster | 3,410 x 2,620 m |
| Overspill | **+850 m in x, +60 m in y** |

The overspill is symmetric about the intended centre, so **center-cropping to
256x256 recovers exactly the intended 2,560 m tile.** And because the grid is
non-overlapping at 2,560 m, that extra 850 m is territory the *adjacent* tile
already covers. It is redundant at county level, not new information.

So the center-crop is not throwing data away — it is correcting the export's
overspill and restoring the intended grid. For every patch at or above 256 px,
the current pipeline already yields a uniform 2,560 m footprint. My earlier
"footprint varies from 186x235 to 268x366" concern is resolved by the crop for
all of them.

That leaves exactly one real defect: **edge tiles with less than 256 px of
in-county coverage.** And re-export is the wrong fix for those too. Today they
arrive trimmed to the county boundary and contain only real observations
(`valid_fraction = 1.0`). A "fixed" export would return a full 256x256 grid with
out-of-county pixels as 0 — injecting fabricated zeros into the encoder and into
the county spatial mean. Dropping them is cleaner than inventing them.

**Conclusion: use `--undersize-policy skip` and do not re-export.** Re-exporting
would cost a multi-day ~90 GB download, invalidate the split manifest hashes and
any existing embeddings, and buy no additional real observations.

The script fix is still worth keeping for any *future* export — deterministic
256x256 beats 997 silently varying shapes, and the shape assertion catches the
class of bug that hid here for months. But it does not justify redoing the
existing data.

`--undersize-policy skip` reduces per-county patch counts and changes the
complete-patch identity hashes, so do not mix `skip` and `error` embeddings in
one comparison. That is the one thing to keep straight.

## 2. terratorch (undeclared dependency)

`pyproject.toml` does not list `terratorch`, but
`benchmark_embeddings/frozen/prithvi.py` and `frozen/terramind.py` both import
`terratorch.registry`, and one test imports it too. Install it separately, and
**pin torchgeo**:

```bash
python -m pip install terratorch
python -m pip install 'torchgeo==0.9.0'
```

`torchgeo` 0.10.0 (what terratorch 1.2.11 resolves to by default) removed
`torchgeo.trainers.utils`, which terratorch still imports — importing
`terratorch` fails outright until you downgrade. Do **not** try `torchgeo<0.8`;
it depends on `fiona`, which needs system GDAL.

Also make sure torch and torchvision come from the same build channel. A
PyPI `torchvision` against a `+cpu` torch raises
`RuntimeError: operator torchvision::nms does not exist`.

Verify:

```bash
python -c "from terratorch.registry import BACKBONE_REGISTRY; print('ok')"
```

## 2b. Presto (undeclared dependencies, and a name collision)

**Do not `pip install presto`.** PyPI's `presto` 0.7.9 is pRESTO, an immune-
repertoire sequencing toolkit — it pulls `biopython` and is entirely unrelated.
`frozen/presto.py` defends against exactly this: it checks for `Presto` and
`construct_single_presto_input` and raises "imported `presto` is not the
nasaharvest/presto package". Use the checkout at `$PRESTO_REPO`, or
`pip install git+https://github.com/nasaharvest/presto`.

`--presto-repo` is optional in the Python (`presto_repo: str | Path | None`); it
only does `sys.path.insert`. If the package is importable you can omit it —
though `run_all_local.sh` is stricter and mandates the directory
(`require_dir` at line 274).

`import presto` needs considerably more than the repo. `presto/__init__.py`
imports `.dataops.utils`, which triggers `dataops/__init__.py` line 1
(`from .dataset import TAR_BUCKET`), and `dataset.py` pulls the whole Earth
Engine training stack. None of it is declared in this project's `pyproject.toml`:

```bash
pip install earthengine-api webdataset hurry.filesize geopandas google-cloud-storage
pip install --no-deps openmapflow      # --no-deps is REQUIRED
```

`--no-deps` on `openmapflow` is not optional: it pins `pandas==1.5.3`, which has
no cp313 wheels (so pip tries a source build and fails for want of Cython) and
directly contradicts this project's `pandas>=2.0`. Installed with `--no-deps`,
only `openmapflow.ee_boundingbox.EEBoundingBox` is needed at import time and
pandas stays on 2.x/3.x. Verified working.

Verify:

```bash
python -c "
import sys; sys.path.insert(0, '$PRESTO_REPO')
import presto, torch
m = presto.Presto.load_pretrained()
print('presto OK |', sum(p.numel() for p in m.parameters())/1e6, 'M params')
"
```

Expect ~0.82M parameters — Presto is a small temporal pixel model, so that
figure is correct, not a truncated download.

## 3. Config

`configs/local.env` is fully populated. Clay is entirely local and verified:

```
CLAY_REPO=${BENCHMARK_ROOT}/clay/model
CLAY_METADATA=${CLAY_REPO}/configs/metadata.yaml
CLAY_V15_CHECKPOINT=${BENCHMARK_ROOT}/clay/clay-v1.5.ckpt
```

`PRESTO_REPO` is the only value still outside this folder, so it is the only one
I could not check:

```
YIELD_EMBEDDINGS_ROOT=/Users/adriko/Phd/code/yield-embeddings
PRESTO_REPO=${YIELD_EMBEDDINGS_ROOT}/presto/presto
```

Confirm it before the `extract` stage:

```bash
source configs/local.env
for p in "$CLAY_REPO" "$CLAY_METADATA" "$CLAY_V15_CHECKPOINT" "$PRESTO_REPO"; do
  [ -e "$p" ] && echo "ok      $p" || echo "MISSING $p"
done
```

Metadata exists in two places — `clay/model/configs/metadata.yaml` (official,
8 platforms) and `clay/metadata.yaml` (832 B, `sentinel-2-l2a` only). Both carry
an identical `sentinel-2-l2a` block, and that is the only block the extractor
reads, so either works. The config uses the official one inside the repo.

Never `pip install claymodel` as a substitute for the checkout. It pins
`torch==2.4.0`, which has no cp313 wheels, so it would pull the environment back
to an old torch. Combined with `build_splits.py` needing 3.12+, that leaves
**3.12 as the only viable Python version**. The checkout avoids the pin entirely.


### Which encoders actually need checkpoint files

Only Clay and Presto. This was checked by running the extractors' own loaders,
not by reading the docs.

| Encoder | Checkpoint | Metadata / repo | Behaviour |
| --- | --- | --- | --- |
| **Clay v1.5** | **Required** — `clay/clay-v1.5.ckpt` | **Both required**: `--metadata` on every invocation including `--preflight-only`, and `--clay-repo` for extraction. Both local | `main()` calls `parser.error("--checkpoint, --clay-repo, and --output are required for extraction")` |
| **Presto** | Bundled in the repo checkout | `--presto-repo` required for extraction | Preflight runs without it |
| **Prithvi-EO-2.0-300M-TL** | Optional | none | Loader passes `pretrained=True` and downloads. `--checkpoint` only adds `ckpt_path` to override |
| **TerraMind v1** | Optional | none | `checkpoint_path is None` -> `pretrained=True`; supplying one flips to `pretrained=False` + `ckpt_path` |

`clay/clay-v1.5.ckpt` is 5.1 GB — a full Lightning *training*
checkpoint, not a stripped encoder: it carries optimizer and `epoch_loop` state
alongside 552 `model.encoder.*` tensors and a
`vit_large_patch14_reg4_dinov2` backbone, consistent with the
`model_size="large"` that `load_clay_encoder` passes. Budget RAM accordingly —
`load_from_checkpoint` reads the whole dict before discarding the optimizer
state.

Set `HF_HOME` if you want the ~3 GB of downloaded weights somewhere other than
`~/.cache/huggingface`. An `HF_TOKEN` is not required, but without one the Hub
warns about rate limits and downloads more slowly.

`COUNTY_FIPS_MAP` points at `data/geometry/county_fips_map.csv`, which I
generated from the bundled Census shapefile — see "Generated files" below.

## 4. Verify

```bash
conda activate base
python -m pytest -q
```

Expect **97 passed**.

Then confirm MPS is really being used, rather than assumed, with a short run
before committing to the 45-run supervised grid:

```bash
python -m benchmark_embeddings.train --synthetic-smoke \
  --device mps --out-dir /tmp/be_smoke_mps
```

Synthetic CPU smoke run of the supervised trainer:

```bash
python -m benchmark_embeddings.train --synthetic-smoke --out-dir /tmp/be_smoke
cat /tmp/be_smoke/result.json
```

Writes `best.pt`, `config_used.yaml`, `data_contract.json`, `log.json`,
`normalization.json`, `predictions.csv`, `result.json`.

## 5. Run against the real data

```bash
source configs/local.env

# Stage 1: canonical AlphaEarth parquet + the five-fold split manifest (~10 s)
bash scripts/run_all_local.sh prepare configs/local.env

# Stage 2: read-only audits of all four encoders over the 77,813-file cohort
bash scripts/run_all_local.sh preflight configs/local.env

# Stage 3: extraction. Confirm the CLAY_REPO / PRESTO_REPO paths first
# (see "Config"), and preview with DRY_RUN=1 before committing to it.
DRY_RUN=1 bash scripts/run_all_local.sh extract configs/local.env
bash scripts/run_all_local.sh extract configs/local.env
```

Resumable — completed outputs are skipped unless `FORCE=1`. Preview any stage
with `DRY_RUN=1`. Stages: `prepare`, `preflight`, `extract`, `main`,
`terramind10`, `climate`, `temporal`, `loyo`, `supervised`, `aggregate`,
`parity`, or `all`.

`DRY_RUN=1` and `FORCE=1` only work because `configs/local.env` declares its
tunables as `${VAR:-default}`. If you switch back to
`configs/local.env.example`, those prefixes are silently ignored and the stage
executes for real — see "Known issues" #7.

A quick single-representation Ridge probe, useful as an end-to-end sanity check
that needs no external checkout:

```bash
python -m benchmark_embeddings.probe \
  --embeddings outputs/embeddings/alphaearth.parquet \
  --labels data/labels/county_yield.csv \
  --split data/group_kfold_county_T7.csv \
  --fold 0 --out-dir /tmp/probe_ae_f0
```

## Where to pick up

Everything up to extraction is verified. Nothing downstream of extraction has
ever run on real embeddings — `regression_benchmark`, `temporal_ablation`,
`loyo`, and `experiment_parity` have only been exercised by unit tests and
`--help`. That is the gap.

No data regeneration is needed — see "Do NOT re-export". Go straight to
extraction with `--undersize-policy skip`.

### Step 1 — cheapest full end-to-end slice (~20 min)

Start here. It de-risks the expensive runs by exercising the whole
frozen -> probe chain on real data with a real encoder. Presto is 0.82M
parameters, so it is by far the cheapest:

```bash
source configs/local.env

python -m benchmark_embeddings.frozen.presto \
  --s2-dir "$COUNTY_PATCH_TIMESTEPS" \
  --presto-repo "$PRESTO_REPO" \
  --output "$EMBEDDINGS_ROOT/presto_s2.parquet" \
  --expected-input-count 77813 \
  --s2-units auto \
  --undersize-policy "$UNDERSIZE_POLICY" \
  --device "$LOCAL_DEVICE"

for fold in 0 1 2 3 4; do
  python -m benchmark_embeddings.probe \
    --embeddings "$EMBEDDINGS_ROOT/presto_s2.parquet" \
    --labels "$YIELD_LABELS" \
    --split "$COMMON_SPLIT_MANIFEST" \
    --fold "$fold" \
    --out-dir "$RESULTS_ROOT/probe_presto/fold_$fold"
done
```

This yields the first real GeoFM number on this cohort, and it is the only
single-encoder path — `regression_benchmark` requires Clay, Prithvi, TerraMind,
and Presto all present, so it cannot run until all four are extracted.

#### Baselines to check the Presto result against

Already computed on real data, all five folds, and saved to
`outputs/baselines/probe_baselines_5fold.csv` alongside the canonical AlphaEarth
Parquet. Ridge probe, county-grouped folds, bushels per acre:

| Representation | R2 per fold (0-4) | R2 mean | RMSE mean | RMSE sd |
| --- | --- | --- | --- | --- |
| AlphaEarth, 64-D | 0.743 / 0.671 / 0.673 / 0.563 / 0.725 | **0.675** | **12.64** | 0.84 |
| Sentinel-2 indices, 21-D | 0.733 / 0.568 / 0.653 / 0.574 / 0.647 | **0.635** | **13.41** | 0.72 |

Two things this establishes beyond the numbers themselves:

- **Fold coverage is correct.** The five test partitions sum to 1,038 county-years
  — exactly the cohort — so every county-year is tested exactly once, which is
  what the benchmark requires.
- **Fold 0 is the easiest fold.** It gives the best score for both
  representations, and fold 3 the worst (R2 0.563 / 0.574). Any single-fold
  number, including the fold-0 figures quoted earlier in this file, will look
  optimistic. Report five-fold means.

A Presto R2 in the 0.6-0.75 band is a plausible result. Far outside it means
something upstream is wrong — check `valid_fraction_min` and the unit detection
in the provenance before believing the score.

### Step 2 — the three heavy encoders

Only after Step 1 looks sane. These are the MPS-bound jobs; run one first and
check its provenance before launching the rest.

```bash
bash scripts/run_all_local.sh extract configs/local.env
```

Watch for: Clay loading the 5.1 GB checkpoint (never tested — see "Not
verified"), and MPS behaviour on real 256x256 cubes.

### Step 3 — the untested downstream half

```bash
bash scripts/run_all_local.sh main configs/local.env
bash scripts/run_all_local.sh temporal configs/local.env
bash scripts/run_all_local.sh loyo configs/local.env
bash scripts/run_all_local.sh supervised configs/local.env   # 45 runs, slowest
bash scripts/run_all_local.sh aggregate configs/local.env
bash scripts/run_all_local.sh parity configs/local.env
```

Expect `parity` to need attention: it has never seen a `skip`-policy contract, so
how it reports the changed complete-patch identity hashes is unknown.

## Entry points

There is no server and no daemon. Everything is a CLI module, each also exposed
as a console script (`benchmark-*`) by `pyproject.toml`.

| Module | Role |
| --- | --- |
| `build_splits` | Builds the authoritative five-fold county-grouped manifest. Run first. |
| `prepare_alphaearth` | Converts the matched AlphaEarth CSV to the canonical embedding schema. |
| `frozen.clay` / `.presto` / `.prithvi` / `.terramind` | Frozen encoder extraction. Each supports `--preflight-only`. |
| `regression_benchmark` | Canonical Ridge / RF / XGBoost / EBM five-fold tables. |
| `probe` | Single-input Ridge diagnostic. |
| `temporal_ablation` (+ `_aggregate`) | mean vs concat vs Conv1D temporal readout, shared MLP head. |
| `loyo` | Climate-free leave-one-year-out, Random Forest only. |
| `train` | Supervised 3D-ConvLSTM / GRU / LSTM trainer. |
| `supervised_aggregate` | Aggregates the 3-model x 5-fold x 3-seed grid. |
| `experiment_parity` | Final cross-family contract audit. |

`scripts/run_all_local.sh` sequences all of it without Slurm; the `.sbatch`
files are the cluster equivalents.

## Architecture

- Pure Python CLI package, setuptools build backend, no server component.
- `data/` — NPZ county-patch reader, the 256x256 spatial policy, band
  normalization, and split loading.
- `frozen/` — one adapter per foundation model behind a shared schema. Every
  adapter writes the same table:
  `county_id, year, patch_id, timestep, backbone, embedding`.
- Clay, Prithvi, and TerraMind encode each patch-timestep independently and emit
  one row per timestep. Presto ingests the whole seven-step sequence and emits a
  single row with `representation_scope=sequence`. AlphaEarth is precomputed
  annual. The probe handles all three scopes.
- `models/` — supervised networks: a shared 3D conv stem, then ConvLSTM (keeps
  spatial hidden state) or GRU/LSTM (spatial mean per timestep first).
- Every workflow reads the same `group_kfold_county_T7.csv` manifest. All years
  of a county stay in one fold role.
- Fixed aggregation order: county spatial mean + population std across complete
  patches per timestep, *then* temporal pooling.
- Each run writes a `data_contract.json`. `experiment_parity` cross-checks those
  contracts and fails on cohort, target, or patch-identity drift rather than
  silently intersecting inputs.
- Experiment families are deliberately walled off. Presto+ERA5 and all Daymet
  fusion variants belong to the auxiliary climate study and are rejected by the
  main and LOYO runners.

## Verified

Environment: Python 3.13.13, torch 2.13.0+cpu, torchvision 0.28.0+cpu,
numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, pyarrow 25.0.1, xgboost 3.4.1,
interpret 0.7.8, terratorch 1.2.11, torchgeo 0.9.0, pytest 9.1.1.

- `pytest` — 97 passed, 0 failed.
- `--help` on all 14 CLI modules.
- `train --synthetic-smoke` — full train/val/test loop, all seven artifacts.
- `prepare_alphaearth` on the real CSV — 2,180 county-years, 64-D.
- `build_splits` on the real 77,813-file cohort — 5.5 s; 1,038 county-years,
  406 counties. **The regenerated manifest is byte-identical to the checked-in
  `data/group_kfold_county_T7.csv`**, which also confirms the generated FIPS map
  is correct.
- All six `--preflight-only` audits pass over the real cohort: Clay, Presto
  S2-only, Presto+ERA5, Prithvi, TerraMind 6-band, TerraMind 10-band. The
  cohort resolves to 77,813 input files, 77,457 in-schedule (356 interval-07
  files excluded as documented), 11,092 spatial patches, 10,915 complete
  sequences, patch counts 1–15 per county-year.
- `probe` on real AlphaEarth embeddings, fold 0: R² 0.743, RMSE 11.84 bu/ac,
  MAE 8.87, n=208, alpha 1.0.
- `probe` on the real 21-D Sentinel-2 index baseline, fold 0: R² 0.733,
  RMSE 12.06 bu/ac, MAE 9.31, n=208, alpha 0.1.
- **Real pretrained weight downloads, through the extractors' own loaders, with
  no checkpoint argument and no `HF_TOKEN`:**
  - `load_prithvi(device=cpu)` -> `PrithviViT`, 303.9M params, embed_dim 1024,
    24 blocks. ~1.3 GB cached.
  - `load_terramind(model_name="terramind_v1_base", experiment="s2_6_prithvi")`
    -> `TerraMindViT`, 86.1M params, S2L2A patch adapter validated at 6
    channels. ~1.5 GB cached.
  
  Both loaders also run their internal contract assertions (embed dim, input
  channels, frame count, encoder depth, patch size, temporal/location encoding
  flags) against the downloaded weights and pass.
- **A real Presto extraction, producing a real Parquet.** `import presto`
  resolves the genuine nasaharvest package, `Presto.load_pretrained()` loads the
  bundled `data/default_model.pt` (0.82M params), and
  `benchmark_embeddings.frozen.presto --undersize-policy skip` ran against the
  actual patch directory and wrote a valid canonical table: 128-D embeddings, all
  finite, `backbone=presto_s2`, `representation_scope=sequence`, correct column
  set, with the undersize deviation recorded in the provenance sidecar.
- **The undersize skip policy**, on all four extractors plus the trainer's shared
  reader: default `error` still fails byte-identically on the same file, `skip`
  yields 10,435 complete patches consistently across all four, and the 97 tests
  still pass.
- **Clay wiring, short of the checkpoint read.**
  `sys.path.insert(0, "clay/model")` then `import src.module` resolves
  `src.module.ClayMAEModule` (`clay/model/src/module.py:11`); the
  `claymodel.module` branch is absent, so the `src.module` fallback is the one in
  use. `ClayMAEModule(model_size="large", metadata_path=...,
  dolls=[16,32,64,128,256,768,1024], mask_ratio=0.0, shuffle=False)` constructs
  and exposes `model.encoder` — the attribute `load_clay_encoder` asserts on —
  as an `Encoder` with 311.4M params and `dim=1024`, matching the documented
  1024-D CLS representation.
- All three `metadata.yaml` copies carry an identical `sentinel-2-l2a` block.

## Not verified

- **A full-cohort extraction for any encoder.** Presto ran for real but only on
  bounded `--max-sequences` slices (50 and 200), because a full 10,435-sequence
  run takes longer than the sandbox allowed per command. Prithvi's and
  TerraMind's weights load but no forward pass was run at all. Clay's checkpoint
  read is untested (see below).
- **Everything downstream of a full extraction.** With no complete embedding
  Parquet, `regression_benchmark`, `temporal_ablation`, `loyo`, and
  `experiment_parity` were exercised only via `--help` and unit tests. In
  particular the parity audit has never seen a `skip`-policy contract, so how it
  reports the changed patch-identity hashes is unverified.
- **The supervised trainer's skip path.** `county_patches.py` accepts
  `undersize_policy='skip'` and the tests pass, but it was never run against the
  real 256x256 cubes — only the synthetic smoke path.
- **Loading the Clay checkpoint.** Inspected structurally via the zip archive
  only. The verification sandbox has 3 GB of RAM against a 5.1 GB checkpoint, so
  `ClayMAEModule.load_from_checkpoint` was never actually called. This is the
  step most likely to surprise you — try it standalone before launching the
  extraction.
- **`PRESTO_REPO`.** Set to `${YIELD_EMBEDDINGS_ROOT}/presto/presto` by the
  documented convention, but that tree is outside the folder I could read, so
  it is inferred rather than confirmed. It is now the only unverified path in
  `configs/local.env`. Run the check in "Config" before the `extract` stage.
- **Downstream stages that consume extracted embeddings**: `regression_benchmark`
  (main, terramind10, climate), `temporal_ablation`, `loyo`,
  `supervised_aggregate`, `experiment_parity`. Their `--help` and unit tests
  pass, but no full-cohort run happened, because they all require the encoder
  parquets from `extract`.
- **Supervised training on real patches.** Only the synthetic smoke path ran; a
  real run reads 256x256x10x7 cubes and the full grid is 45 runs.
- **Apple MPS, and the conda base environment.** All verification ran on Linux
  CPU in a throwaway Python 3.13 venv, because I had no way to execute anything
  on your Mac. So the specific combination now configured — conda base plus
  `LOCAL_DEVICE=mps` — is the one part of the setup nobody has exercised. The
  code paths are sound (`train.py` and `temporal_ablation.py` both validate MPS
  explicitly and are covered by `tests/test_device_selection.py`; all four
  frozen extractors resolve `mps` in their own `_device()`), but the runtime
  behaviour of MPS kernels on this workload is unverified. Run the MPS smoke
  test in step 4 first.
- **`PYTHON_BIN` resolution.** `configs/local.env` derives the conda base from
  `CONDA_EXE`, falling back to `conda info --base` and then to the usual install
  prefixes. Verify with `source configs/local.env && "$PYTHON_BIN" -V`.
- **The `.sbatch` scripts.** No Slurm available.

## Code changes

Six source files were modified to add the undersize skip policy. All defaults are
unchanged, so existing behaviour is byte-identical unless you pass the new flag.

| File | Change |
| --- | --- |
| `data/io.py` | New `UNDERSIZE_POLICIES`, `normalise_undersize_policy()`, `npz_spatial_shape()` (header-only size probe), `screen_undersized_patches()` |
| `frozen/presto.py` | `undersize_policy` arg, index-time screening, `--undersize-policy` flag, contract fields, split the incomplete/undersize counters |
| `frozen/prithvi.py` | same as Presto |
| `frozen/clay.py` | same, adapted to its per-file index by grouping to spatial patches first |
| `frozen/terramind.py` | same as Clay |
| `data/county_patches.py` | `undersize_policy` arg, `_drop_undersized_entries()`, contract fields — this is the trainer's shared reader |
| `train.py` | passes `data.undersize_policy` through at both dataset construction sites |
| `scripts/run_all_local.sh` | `UNDERSIZE_POLICY` knob threaded into all 12 extractor invocations |
| `configs/supervised_s2{,_gru,_lstm}.yaml` | new `data.undersize_policy: skip` key |
| `frozen/clay.py` | `--spatial-size`; `clay_token_count()` replaces the 256-pinned constant |
| `frozen/presto.py` | `--spatial-size` |
| `frozen/prithvi.py`, `frozen/terramind.py` | `--source-size`; `oversize_policy` string now derived from the real sizes; `terramind_token_count()` replaces the 196-pinned constant |
| `tests/test_prithvi_extractor.py` | asserts the derived `oversize_policy` instead of a literal its own fixture contradicted |
| `data/scripts/gee-sentinel-2.py` | **untested, and not needed for the existing data** — drop `region` from the download params, remove three footprint `.clip()` calls, assert the exported grid shape, thread `pixels` through `meta`. Keep for future exports; see "Do NOT re-export" for why the current 90 GB should not be regenerated |

Regression evidence: 97/97 tests pass, and
`train --synthetic-smoke` reproduces its previous numbers exactly
(test county RMSE 8.874801, R2 -77.87156746752622) with
`undersize_policy: error` still recorded in the contract — confirming the default
path is untouched.

## Generated files

Two files were added.

1. `configs/local.env` — was an unedited copy of `local.env.example` with every
   path still `/path/to/...`. Now filled in.
2. `data/geometry/county_fips_map.csv` — 3,194 rows, columns `NAME,STATEFP,GEOID`,
   derived from `data/geometry/cb_2021_us_county_500k/`. Nothing in the repo
   satisfied `COUNTY_FIPS_MAP`, and `data/s2_daymet_merged.xlsx` carries only
   `County` (name) and `StateFP`, so the name->GEOID lookup is mandatory.
   Independent cities (LSAD 25) are excluded because six of them collide by name
   with a same-state county (Baltimore MD, St. Louis MO, Fairfax/Franklin/
   Richmond/Roanoke VA) and `_fips_lookup` rejects ambiguous rows. Correctness
   is established by the byte-identical manifest reproduction above.

To regenerate #2:

```bash
python - <<'PY'
import shapefile, pandas as pd   # pip install pyshp
r = shapefile.Reader("data/geometry/cb_2021_us_county_500k/cb_2021_us_county_500k.shp")
flds = [f[0] for f in r.fields[1:]]
df = pd.DataFrame([dict(zip(flds, rec)) for rec in r.records()])
df["STATEFP"] = df["STATEFP"].astype(int)
df = df[df["LSAD"] != "25"]
df[["NAME","STATEFP","GEOID"]].drop_duplicates().to_csv(
    "data/geometry/county_fips_map.csv", index=False)
PY
```

## Known issues

1. **`pyproject.toml` understates the Python floor.** It says
   `requires-python = ">=3.10"`, but `benchmark_embeddings/build_splits.py:204`
   has a backslash inside an f-string expression, valid only on 3.12+. Under
   3.10 the whole package fails to import with `SyntaxError: f-string
   expression part cannot include a backslash`. It is the only file affected —
   every other module compiles under 3.10. Fix by bumping the floor to
   `>=3.12`, or by hoisting the separator to a local variable if you want 3.10
   support back.
2. **`terratorch` is missing from `dependencies`.** Prithvi and TerraMind
   extraction, plus `tests/test_prithvi_extractor.py`, need it. Consider a
   `geofm = ["terratorch", "torchgeo==0.9.0"]` extra.
3. **`interpret` needs `shap` and `llvmlite`** on 3.13, which is a heavy
   install. Skip the `tabular` extra if you only want Ridge and Random Forest;
   `regression_benchmark` raises a clear `ImportError` naming the missing
   package when you select `xgboost` or `ebm`.
4. **`.DS_Store` files** are scattered through the tree, including inside
   `data/patches/`. They do not break the `*.npz` glob, but the file counts
   differ from the npz counts (77,816 entries vs 77,813 npz in
   `data/patches/sentinel-2-l2a`) — worth knowing when an audit count looks off.
5. **No git repository.** `git status` and `git log` both fail; there is no
   `.git` directory, no `CONTRIBUTING`, and no `docs/`. Nothing here is version
   controlled, so consider `git init` before making changes. If you do, add
   `clay/`, `data/patches/`, `data/YieldSAT/`, `outputs/`, and `.venv/` to
   `.gitignore` first — that is ~93 GB, and `clay/model` is itself a clone of
   another repo.
6. **`pip install claymodel` will downgrade your environment.** It pins
   `torch==2.4.0`, capping Python at 3.12. Use the `clay/model` checkout instead.
7. **`configs/local.env.example` silently disables `DRY_RUN` and `FORCE`.**
   `run_all_local.sh` sources the env file *before* reading those variables, but
   the example writes them as bare `export DRY_RUN=0` / `export FORCE=0`. So
   `DRY_RUN=1 bash scripts/run_all_local.sh extract configs/local.env.example`
   does not preview anything — it clobbers the override and **runs the real
   pipeline**. The same applies to `LOCAL_DEVICE` and every other tunable.
   `configs/local.env` fixes this by using the `${VAR:-default}` form
   throughout, so command-line prefixes win. Verified in both directions:
   overrides take effect, and the defaults still apply when nothing is set.
   Worth pushing back upstream into `local.env.example`.
