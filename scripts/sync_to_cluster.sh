#!/usr/bin/env bash
# Push the repository to the cluster.
#
#   bash scripts/sync_to_cluster.sh adriko1@HOST:/e/project1/3d-abc/adriko1/benchmark-embeddings
#
# Syncs the WHOLE tree except the bulk inputs, which already live on JSC:
#   data/patches   88 GB   Sentinel-2 NPZs -- the cluster copy is authoritative
#   data/YieldSAT  6.5 GB  field-level labels, not used by the county benchmark
#   clay/*.ckpt    4.9 GB  send explicitly with --with-clay-ckpt if absent there
# What remains is roughly 70 MB: code, configs, tests, manifests, the tabular
# sources, the county geometry, and outputs/.
#
# Flags (any order, after the destination):
#   --with-clay-ckpt   also send the 4.9 GB Clay checkpoint (resumable)
#   --minimal          send only the files extraction strictly needs (~29 MB)
#
# Set DRY_RUN=1 to preview without transferring.
#
# NOTE: JUPITER compute nodes do not mount /p (JUST) -- only /e (exasm). A venv
# or checkout under /p is invisible to jobs. Sync to /e/project1/3d-abc/...

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/sync_to_cluster.sh USER@HOST:/dest/path [--with-clay-ckpt] [--minimal]" >&2
  exit 2
fi

dest=$1
shift
with_ckpt=0
minimal=0
for arg in "$@"; do
  case "$arg" in
    --with-clay-ckpt) with_ckpt=1 ;;
    --minimal)        minimal=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

DRY=""
[[ "${DRY_RUN:-0}" == "1" ]] && DRY="--dry-run"

# Generated, environment-specific, or already-on-cluster paths.
excludes=(
  --exclude 'data/patches/'
  --exclude 'data/YieldSAT/'
  --exclude 'clay/*.ckpt'
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '*.egg-info/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude '.pytest_cache/'
  --exclude '.DS_Store'
)

if [[ "$minimal" == "1" ]]; then
  includes=(
    benchmark_embeddings
    scripts
    configs
    pyproject.toml
    README.md
    SETUP.md
    VALIDATION_REVIEW.md
    data/sources/embeddings_with_yield_matched.csv
    data/sources/s2_daymet_merged_matched.xlsx
    data/sources/cohort_2180_keys.txt
    data/geometry/county_fips_map.csv
    data/labels/county_yield.csv
    clay/model
    clay/metadata.yaml
  )
  echo "syncing the minimal extraction set to ${dest}${DRY:+  (dry run)}"
  rsync -av $DRY --relative "${excludes[@]}" "${includes[@]}" "${dest}/"
else
  echo "syncing the full tree (minus bulk inputs) to ${dest}${DRY:+  (dry run)}"
  du -sh --exclude=data/patches --exclude=data/YieldSAT --exclude='*.ckpt' \
         --exclude=.git . 2>/dev/null | sed 's/^/  local size: /' || true
  rsync -av $DRY "${excludes[@]}" ./ "${dest}/"
fi

if [[ "$with_ckpt" == "1" ]]; then
  echo
  echo "sending the Clay checkpoint (4.9 GB) -- resumable, safe to re-run"
  rsync -av --partial --progress clay/clay-v1.5.ckpt "${dest}/clay/"
else
  echo
  echo "Clay checkpoint NOT sent. Clay extraction needs it:"
  echo "  bash scripts/sync_to_cluster.sh ${dest} --with-clay-ckpt"
  echo "Check first whether it already exists on the cluster."
fi

cat <<EOF

On the cluster:

  unset PYTHONPATH        # the retired geofm4eo venv leaks in through it
  source /e/project1/3d-abc/adriko1/EODeepLearning/activate.sh
  export BENCHMARK_ENV_SETUP=/e/project1/3d-abc/adriko1/EODeepLearning/activate.sh

  # only if the environment is missing them:
  #   pip install -e '.[test,parquet,tabular]'
  #   pip install terratorch 'torchgeo==0.9.0'        # see SETUP.md section 2
  #   pip install earthengine-api webdataset hurry.filesize geopandas \\
  #               google-cloud-storage xarray einops
  #   pip install --no-deps openmapflow                # pins pandas==1.5.3 otherwise

  export BENCHMARK_ROOT=${dest##*:}
  export COUNTY_PATCH_TIMESTEPS=/e/project1/3d-abc/adriko1/datasets/US/T7
  # leave UNDERSIZE_POLICY unset: the cluster corpus sampled 2000/2000 at
  # 256x256, so 'error' costs nothing and halts loudly if that ever changes.
  # 'skip' forces a zip-header read of all 427,049 files in every job.
  export PRESTO_EXPECTED_INPUT_COUNT=427049          # and CLAY_/PRITHVI_/TERRAMIND_

Presto spatial mode (see scripts/extract_presto.sbatch):

  export PRESTO_SPATIAL_MODE=mean                    # published configuration
  # or, encoding real pixel sequences rather than a spatial mean:
  export PRESTO_SPATIAL_MODE=sample
  export PRESTO_PIXEL_SAMPLES=64
  export PRESTO_NONFINITE_POLICY=mask
  export PRESTO_BATCH_SIZE=8                         # batch x K must fit in VRAM

Then build the manifests and start with Presto -- see scripts/CLUSTER.md.
EOF
