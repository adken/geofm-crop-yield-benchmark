#!/bin/bash -l
#SBATCH --job-name=presto-embed
#SBATCH --account=3d-abc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --output=presto-%j.out
#SBATCH --error=presto-%j.err

set -euo pipefail

# Activate the project environment inside the job. SLURM exports the submitting
# environment by default, but a conda/venv activation does not survive reliably --
# sourcing here makes the job self-contained and reproducible from the log.
# Presto needs its own environment: openmapflow pins pandas==1.5.3 and pulls in
# the Earth Engine training stack, which cannot coexist with the terratorch
# environment the other three encoders use.
#
# Deliberately NOT falling back to BENCHMARK_ENV_SETUP: jupiter_env.sh exports
# that for the terratorch encoders, and silently inheriting it here would run
# Presto against the wrong interpreter. Override with PRESTO_ENV_SETUP only.
BENCHMARK_ENV_SETUP="${PRESTO_ENV_SETUP:-/e/project1/3d-abc/adriko1/Presto/activate.sh}"

# sbatch exports the submitting environment, so a shell that sourced
# jupiter_env.sh arrives here with the terratorch venv on PATH. Presto's
# activate script changes the module stage but does not displace that venv,
# leaving a venv built against one Python running under another -- which
# surfaces as "Failed to load PyTorch C extensions". Drop it first.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "dropping inherited virtualenv: ${VIRTUAL_ENV}"
  PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd: -)"
  export PATH
  unset VIRTUAL_ENV
fi

# PYTHONPATH outranks the active environment, so an inherited one pointing at
# the terratorch site-packages puts a Python 3.13 torch build on sys.path under
# whichever interpreter Presto's stage provides. That is the actual cause of
# "Failed to load PyTorch C extensions", not the venv on PATH. Clear it
# outright: Presto has its own venv and needs no PYTHONPATH.
#
# Note `${PYTHONPATH:-}` does NOT clear it -- that expands to the existing
# value whenever the variable is set, which is the case here.
if [[ -n "${PYTHONPATH:-}" ]]; then
  echo "clearing inherited PYTHONPATH: ${PYTHONPATH}"
fi
export PYTHONPATH=
if [[ -r "${BENCHMARK_ENV_SETUP}" ]]; then
  echo "sourcing ${BENCHMARK_ENV_SETUP}"
  # Bind PYTHONPATH before sourcing: the activate script references it, and an
  # unbound variable under `set -u` would abort the job before any work starts.
  # Empty rather than unset, so the retired /p geofm4eo venv cannot shadow this.
  export PYTHONPATH="${PYTHONPATH:-}"
  set +u
  # shellcheck disable=SC1090
  source "${BENCHMARK_ENV_SETUP}"
  set -u
else
  echo "ERROR: cannot read BENCHMARK_ENV_SETUP=${BENCHMARK_ENV_SETUP}" >&2
  echo "Export it to the correct activate.sh, or fix the default in this file." >&2
  exit 2
fi
echo "python: $(command -v python)  ($(python -V 2>&1))"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch extract_presto.sbatch S2_NPZ_DIR OUTPUT_PARQUET [ERA5_NPZ_DIR]"
  exit 2
fi

: "${PRESTO_REPO:?Set PRESTO_REPO to the official nasaharvest/presto checkout}"

s2_dir=$1
output_path=$2
era5_dir=${3:-}
expected_count=${PRESTO_EXPECTED_INPUT_COUNT:-77813}
python_bin=${PYTHON_BIN:-python}

# Presto picks its own default device at import time -- roughly
#   device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# -- and builds index tensors there regardless of what we pass. On a GPU node a
# CPU run therefore gets a cuda index against cpu tables:
#   "Expected all tensors to be on the same device ... index is on cuda:0"
# Hiding the GPU makes Presto's internal default agree with ours.
if [[ "${PRESTO_DEVICE:-cuda}" == "cpu" ]]; then
  echo "PRESTO_DEVICE=cpu: hiding GPUs so Presto's internal default matches"
  export CUDA_VISIBLE_DEVICES=""
fi

extra_args=()
if [[ -n "$era5_dir" ]]; then
  extra_args+=(--era5-dir "$era5_dir")
  extra_args+=(
    --expected-era5-input-count
    "${PRESTO_EXPECTED_ERA5_INPUT_COUNT:-$expected_count}"
  )
fi

srun "$python_bin" -m benchmark_embeddings.frozen.presto \
  --s2-dir "$s2_dir" \
  --presto-repo "$PRESTO_REPO" \
  --output "$output_path" \
  --expected-input-count "$expected_count" \
  --s2-units "${PRESTO_S2_UNITS:-auto}" \
  --era5-source-bands \
    "${PRESTO_ERA5_BAND_1:-total_precipitation}" \
    "${PRESTO_ERA5_BAND_2:-temperature_2m}" \
  --batch-size "${PRESTO_BATCH_SIZE:-256}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --undersize-policy "${UNDERSIZE_POLICY:-error}" \
  --nonfinite-policy "${PRESTO_NONFINITE_POLICY:-error}" \
  --spatial-mode "${PRESTO_SPATIAL_MODE:-mean}" \
  --pixel-samples "${PRESTO_PIXEL_SAMPLES:-64}" \
  --sample-seed "${PRESTO_SAMPLE_SEED:-0}" \
  --min-valid-fraction "${PRESTO_MIN_VALID_FRACTION:-0}" \
  --device "${PRESTO_DEVICE:-cuda}" \
  "${extra_args[@]}"

# PRESTO_DEVICE=cpu is a legitimate choice, not a fallback. Presto is 0.82M
# parameters against Clay's ~200M, and the job is bound by reading 427,049 NPZ
# files rather than by the encoder: at batch 256 the whole cohort is 239
# encoder batches. Presto also builds at least one constant tensor inside its
# forward pass without a device argument, so a CUDA run can fail with
# "Expected all tensors to be on the same device". If you set cpu, submit to a
# CPU partition too -- a GPU sitting idle still bills GPU hours.

# Spatial-mode note: PRESTO_SPATIAL_MODE=sample encodes PRESTO_PIXEL_SAMPLES
# real pixel sequences per patch instead of one spatial mean, so the encoder
# batch is multiplied by that factor. Drop PRESTO_BATCH_SIZE proportionally
# (e.g. 256 -> 8 at 64 samples) or the job will run out of device memory.
#
# PRESTO_NONFINITE_POLICY=mask excludes cloud-masked and all-zero fill pixels
# from the statistics rather than folding them in as zeros; it is required
# before PRESTO_MIN_VALID_FRACTION has any effect.
