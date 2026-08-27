#!/bin/bash
#SBATCH --account=3d-abc
#SBATCH --job-name=benchmark-s2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=booster
#SBATCH --time=11:59:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#
# Usage (resources remain standard sbatch arguments):
#   sbatch --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#     scripts/train_supervised.sbatch CONFIG OUT_DIR SEED FOLD MODEL
#
# Positional arguments:
#   1 configuration path
#   2 output directory
#   3 seed
#   4 primary outer fold
#   5 model: 3d_convlstm, gru, or lstm

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# sbatch copies the submitted script into a spool directory before running it,
# so ${BASH_SOURCE[0]} points at the spool copy and HERE/.. is not the repo.
# That resolved to a path with no benchmark_embeddings package and produced
# "No module named 'benchmark_embeddings'". Locate the root by looking for the
# package, preferring SLURM_SUBMIT_DIR, and allow an explicit override.
if [[ -z "${BENCHMARK_ROOT:-}" ]]; then
  for candidate in "${SLURM_SUBMIT_DIR:-}" "$(cd "${HERE}/.." 2>/dev/null && pwd)" "${PWD}"; do
    if [[ -n "${candidate}" && -d "${candidate}/benchmark_embeddings" ]]; then
      BENCHMARK_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BENCHMARK_ROOT:-}" || ! -d "${BENCHMARK_ROOT}/benchmark_embeddings" ]]; then
  echo "ERROR: cannot locate the benchmark_embeddings package root." >&2
  echo "Submit from the repository root, or export BENCHMARK_ROOT." >&2
  exit 2
fi
echo "benchmark root: ${BENCHMARK_ROOT}"

# The configs use ${COUNTY_PATCH_TIMESTEPS} and friends, expanded by the
# training CLI from the environment. sbatch exports the submitting shell, so a
# job submitted from a shell that had sourced configs/local.env works and one
# submitted from a fresh shell fails with the literal placeholder as a path.
# Source it here so the job does not depend on how it was launched. Only fills
# variables that are unset, so an explicit export on the command line still
# wins.
LOCAL_ENV="${BENCHMARK_LOCAL_ENV:-${BENCHMARK_ROOT}/configs/local.env}"
if [[ -z "${COUNTY_PATCH_TIMESTEPS:-}" && -r "${LOCAL_ENV}" ]]; then
  echo "sourcing ${LOCAL_ENV}"
  # configs/local.env is machine-specific and may set BENCHMARK_ROOT for the
  # machine it was written on. Restore the root resolved above afterwards, or a
  # file synced from a workstation sends the job to a path that does not exist
  # on this cluster.
  _resolved_root="${BENCHMARK_ROOT}"
  set +u
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
  set -u
  if [[ "${BENCHMARK_ROOT}" != "${_resolved_root}" ]]; then
    echo "local.env changed BENCHMARK_ROOT to ${BENCHMARK_ROOT}; restoring ${_resolved_root}" >&2
    export BENCHMARK_ROOT="${_resolved_root}"
  fi
fi
for required in COUNTY_PATCH_TIMESTEPS YIELD_LABELS COMMON_SPLIT_MANIFEST; do
  if [[ -z "${!required:-}" ]]; then
    echo "ERROR: ${required} is unset and could not be read from ${LOCAL_ENV}." >&2
    echo "Create configs/local.env from configs/local.env.example." >&2
    exit 2
  fi
done

# Site-specific module/virtual-environment setup. Defaults to the JUPITER
# project environment; export BENCHMARK_ENV_SETUP to override.
BENCHMARK_ENV_SETUP="${BENCHMARK_ENV_SETUP:-/e/project1/3d-abc/adriko1/EODeepLearning/activate.sh}"
if [[ -r "${BENCHMARK_ENV_SETUP}" ]]; then
    echo "sourcing ${BENCHMARK_ENV_SETUP}"
    # Bind PYTHONPATH before sourcing: the activate script references it, and an
    # unbound variable under `set -u` would abort the job before any work starts.
    # ${PYTHONPATH:-} does NOT clear a value that is already set, and an
    # inherited PYTHONPATH outranks the active environment -- that is what
    # produced "Failed to load PyTorch C extensions" in the extractors.
    if [[ -n "${PYTHONPATH:-}" ]]; then
      echo "clearing inherited PYTHONPATH: ${PYTHONPATH}"
    fi
    export PYTHONPATH=
    set +u
    # shellcheck disable=SC1090
    source "${BENCHMARK_ENV_SETUP}"
    set -u
else
    echo "ERROR: cannot read BENCHMARK_ENV_SETUP=${BENCHMARK_ENV_SETUP}" >&2
    exit 2
fi
echo "python: $(command -v python)  ($(python -V 2>&1))"

CONFIG_PATH="${1:?configuration path is required}"
OUT_DIR="${2:?output directory is required}"
SEED="${3:-20260614}"
FOLD="${4:-0}"
MODEL="${5:-3d_convlstm}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Configuration not found: ${CONFIG_PATH}" >&2
    exit 2
fi
if [[ -s "${OUT_DIR}/result.json" && "${YE_FORCE:-0}" != "1" ]]; then
    echo "Completed result already exists: ${OUT_DIR}/result.json"
    exit 0
fi

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Without this a job killed at the wall discards its buffered stdout, so the
# .out file is empty and says nothing about how far training got. log.json is
# written every epoch regardless, but the console log is the first thing anyone
# looks at.
export PYTHONUNBUFFERED=1
# Concurrent readers for the per-(patch, composite) NPZ files. Measured on this
# filesystem from a compute node: 1 reader sees 25 ms per file, 8 sees 7.1 ms,
# 32 sees 5.9 ms, and 64 is no better than 32. At 32 the cost is zlib
# decompression rather than the filesystem, so that is the useful setting.
export BENCHMARK_IO_WORKERS="${BENCHMARK_IO_WORKERS:-32}"

ARGS=(
    --config "${CONFIG_PATH}"
    --out-dir "${OUT_DIR}"
    --seed "${SEED}"
    --fold "${FOLD}"
    --model "${MODEL}"
    --device cuda
)
# Score an existing validation-selected checkpoint without training. Set
# YE_EVAL_ONLY=1 when a fold was stopped by the wall clock: best.pt is usable
# but the run never reached the test evaluation that writes result.json.
if [[ "${YE_EVAL_ONLY:-0}" == "1" ]]; then
    ARGS+=(--eval-only)
fi
srun --ntasks=1 python -m benchmark_embeddings.train "${ARGS[@]}"
