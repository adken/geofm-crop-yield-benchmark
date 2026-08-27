# Supervised Sentinel-2 county models

The package contains three randomly initialized, Sentinel-2-only alternatives:

- `supervised_s2_3d_convlstm`: 3D convolutional stem followed by spatial
  ConvLSTM states;
- `supervised_s2_gru`: the same stem, spatial mean per timestep, then a GRU;
- `supervised_s2_lstm`: the same stem, spatial mean per timestep, then an LSTM.

These models are evaluated only with the main county-grouped folds. Temporal
LOYO is a separate embedding experiment that uses a fixed Random Forest; the
supervised trainer does not expose a LOYO pathway.

The public YieldSAT project reports a 3D-ConvLSTM but does not expose the exact
implementation used for that result. The first model is therefore described as
**YieldSAT-inspired**, not an exact reproduction. The GRU and LSTM are explicit
matched architectural controls.

## Shared data and supervision contract

- Input: `[num_patches, time=7, channels=10, height=256, width=256]`.
- Bands: `B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12`.
- Source patches larger than 256×256 are center-cropped without interpolation.
- Source patches smaller than 256 in either dimension are rejected.
- Patch counts remain variable by county-year.
- Training samples a configured number of patches per county each epoch.
- Validation and test encode every patch in deterministic chunks.
- Patch representations are averaged once within county-year; MSE is computed
  only after this aggregation.
- Band and target statistics are fitted using training county-years only.
- Early stopping selects on validation RMSE. The test fold is evaluated once.
- `COUNTY_YIELD_CSV` is the authoritative county-year label table; its raw
  `yield` values are used directly as bushels per acre.
- Reported predictions are inverse-transformed to bushels per acre.

The GRU/LSTM variants pool the 3D stem's spatial maps per timestep before the
recurrent layer. ConvLSTM instead retains spatial hidden and cell maps through
the sequence and pools only its final hidden state.

## Commands

From `benchmark-embeddings`:

```bash
python -m pytest -q
```

Generated CPU smoke run:

```bash
python -m benchmark_embeddings.train \
  --synthetic-smoke \
  --out-dir /tmp/benchmark_embeddings_smoke
```

Primary split, one model/fold:

```bash
python -m benchmark_embeddings.train \
  --config configs/supervised_s2.yaml \
  --model 3d_convlstm \
  --fold 0 \
  --out-dir "${SCRATCH_ROOT}/results/benchmark_embeddings/convlstm_fold0"
```

Change `--model` to `gru` or `lstm`; alternatively use the complete
`configs/supervised_s2_gru.yaml` and `configs/supervised_s2_lstm.yaml` files.

Shape/split inspection without training:

```bash
python -m benchmark_embeddings.train \
  --config configs/supervised_s2.yaml \
  --fold 0 \
  --inspect-only
```

Each completed run writes `config_used.yaml`, `normalization.json`,
`data_contract.json`, `log.json`, `best.pt`, `predictions.csv`, and
`result.json`.

## Matched five-fold comparison

Use one base configuration for all three architectures. The submission helper
changes only the model name, outer fold, deterministic seed, and output path;
it schedules models `3d_convlstm`, `gru`, and `lstm` for folds 0--4 and seeds
0, 1, and 2:

```bash
bash scripts/submit_supervised_cv.sh \
  configs/supervised_s2.yaml \
  "$RESULTS_ROOT/supervised_cv"
```

After all 45 jobs finish, aggregate their run directories with
`benchmark_embeddings.supervised_aggregate`. The aggregator rejects a missing
model/fold/seed combination, any configuration or raw-cohort drift, different
partition keys within a fold, target drift, duplicate out-of-fold keys, and
metrics that do not reproduce from the saved predictions. It averages seeds
within each fold and then reports mean and population standard deviation over
the five folds.

```bash
python -m benchmark_embeddings.supervised_aggregate \
  --run-dirs "$RESULTS_ROOT"/supervised_cv/{3d_convlstm,gru,lstm}/fold_{0,1,2,3,4}/seed_{0,1,2} \
  --out-dir "$RESULTS_ROOT/supervised_cv_summary"
```
