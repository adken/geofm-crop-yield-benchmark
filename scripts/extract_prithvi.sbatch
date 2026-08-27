#!/bin/bash -l
#SBATCH --job-name=prithvi-embed
#SBATCH --account=3d-abc
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=11:59:00
#SBATCH --output=prithvi-%j.out
#SBATCH --error=prithvi-%j.err

set -euo pipefail

# Activate the project environment inside the job. SLURM exports the submitting
# environment by default, but a conda/venv activation does not survive reliably --
# sourcing here makes the job self-contained and reproducible from the log.
# Defaults to /e/project1/3d-abc/adriko1/EODeepLearning/activate.sh; export
# BENCHMARK_ENV_SETUP to override. Fails loudly if the path is unreadable.
BENCHMARK_ENV_SETUP="${BENCHMARK_ENV_SETUP:-/e/project1/3d-abc/adriko1/EODeepLearning/activate.sh}"
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

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: sbatch extract_prithvi.sbatch S2_NPZ_DIR OUTPUT_PARQUET"
  exit 2
fi

s2_dir=$1
output_path=$2
expected_count=${PRITHVI_EXPECTED_INPUT_COUNT:-77813}
python_bin=${PYTHON_BIN:-python}

checkpoint_args=()
if [[ -n "${PRITHVI_CHECKPOINT:-}" ]]; then
  checkpoint_args+=(--checkpoint "$PRITHVI_CHECKPOINT")
fi

srun "$python_bin" -m benchmark_embeddings.frozen.prithvi \
  --npz-dir "$s2_dir" \
  --output "$output_path" \
  --pooling per_timestep_spatial_mean \
  --expected-input-count "$expected_count" \
  --expected-timesteps 7 \
  --source-units "${PRITHVI_S2_UNITS:-auto}" \
  --batch-size "${PRITHVI_BATCH_SIZE:-1}" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --undersize-policy "${UNDERSIZE_POLICY:-error}" \
  --device cuda \
  "${checkpoint_args[@]}"
