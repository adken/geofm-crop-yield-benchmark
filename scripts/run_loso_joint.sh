#!/usr/bin/env bash
# Leave-one-state-out with joint patch-timestep pooling, matching the main table.
#
#   bash scripts/run_loso_joint.sh [OUT_DIR]
#
# Run with bash, not zsh: zsh does not word-split unquoted variables, so a
# space-separated state list arrives as a single argument and argparse rejects
# it. Arrays would work in both shells, but a script keeps the two behaviours
# out of the picture entirely.
#
# 91 probe invocations: seven representations across thirteen held-out states.
# Ridge with alpha selected on the validation state, so each is quick.
#
# Only the five patch-timestep encoders change under joint pooling. AlphaEarth,
# Presto and the Sentinel-2 indices have no temporal axis to pool over, so their
# numbers must reproduce the existing two-stage run exactly -- that is a free
# correctness check on the whole thing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

OUT="${1:-outputs/loso_all_encoders_covered_joint}"
SPLIT=outputs/cohort_covered/loso_state_tabular.csv
LABELS=data/labels/county_yield.csv
FIPS=data/geometry/county_fips_map.csv
STATES=(17 18 19 20 21 26 27 29 31 38 39 46 55)

for path in "$SPLIT" "$LABELS" "$FIPS"; do
  [[ -r "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

# name:parquet for everything that goes through the embedding branch.
EMBEDDINGS=(
  "clay:outputs/embeddings/clay_v1_5_cls.parquet"
  "prithvi:outputs/embeddings/prithvi.parquet"
  "terramind_s2_6:outputs/embeddings/terramind_s2_6.parquet"
  "terramind_s2_10:outputs/embeddings/terramind_s2_10.parquet"
  "presto:outputs/embeddings/presto_s2.parquet"
  "alphaearth:outputs/cohort_covered/alphaearth.parquet"
)

for entry in "${EMBEDDINGS[@]}"; do
  name="${entry%%:*}"
  file="${entry#*:}"
  [[ -r "$file" ]] || { echo "missing embedding: $file" >&2; exit 2; }
  for state in "${STATES[@]}"; do
    echo "== ${name} / state ${state}"
    python -m benchmark_embeddings.probe \
      --embeddings "$file" \
      --labels "$LABELS" \
      --split "$SPLIT" \
      --fold "$state" \
      --temporal-pool joint \
      --spatial-pool mean_std \
      --timesteps 7 \
      --out-dir "${OUT}/${name}/${state}"
  done
done

# The index baseline is preaggregated to county-year, so it takes the index
# branch and no pooling arguments apply.
for state in "${STATES[@]}"; do
  echo "== sentinel2_indices / state ${state}"
  python -m benchmark_embeddings.probe \
    --s2-indices outputs/cohort_covered/sentinel2_indices_covered.csv \
    --s2-indices-fips-map "$FIPS" \
    --labels "$LABELS" \
    --split "$SPLIT" \
    --fold "$state" \
    --out-dir "${OUT}/sentinel2_indices/${state}"
done

echo
echo "aggregating"
python -m benchmark_embeddings.loso_aggregate \
  --input-dir "$OUT" \
  --out-dir "${OUT}/aggregate"

echo
echo "done. Compare against the two-stage run:"
echo "  AlphaEarth, Presto and S2 indices must be identical;"
echo "  Clay, Prithvi and both TerraMind variants should differ."
