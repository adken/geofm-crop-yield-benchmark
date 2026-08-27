#!/bin/bash
#SBATCH --account=3d-abc
#SBATCH --job-name=temporal-ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=booster
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#
# Usage:
#   sbatch scripts/run_temporal_ablation.sbatch \
#     CLAY_PARQUET PRITHVI_PARQUET TERRAMIND_PARQUET \
#     LABELS_CSV COMMON_SPLIT_CSV FOLD OUT_DIR
#
# Optional environment variables:
#   TEMPORAL_SEEDS="0 1 2"
#   TEMPORAL_PREFLIGHT=1
#   TEMPORAL_MAX_EPOCHS=300
#   TEMPORAL_PATIENCE=30

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${HERE}/.." && pwd)"

BENCHMARK_ENV_SETUP="${BENCHMARK_ENV_SETUP:-/e/project1/3d-abc/adriko1/EODeepLearning/activate.sh}"
if [[ -r "${BENCHMARK_ENV_SETUP}" ]]; then
    echo "sourcing ${BENCHMARK_ENV_SETUP}"
    export PYTHONPATH="${PYTHONPATH:-}"
    set +u
    # shellcheck disable=SC1090
    source "${BENCHMARK_ENV_SETUP}"
    set -u
else
    echo "ERROR: cannot read BENCHMARK_ENV_SETUP=${BENCHMARK_ENV_SETUP}" >&2
    exit 2
fi
echo "python: $(command -v python)  ($(python -V 2>&1))"

CLAY_PARQUET="${1:?Clay embedding Parquet is required}"
PRITHVI_PARQUET="${2:?Prithvi embedding Parquet is required}"
TERRAMIND_PARQUET="${3:?TerraMind embedding Parquet is required}"
LABELS_CSV="${4:?yield-label CSV is required}"
COMMON_SPLIT_CSV="${5:?common split manifest is required}"
FOLD="${6:?fold is required}"
OUT_DIR="${7:?output directory is required}"

for INPUT_PATH in \
    "${CLAY_PARQUET}" \
    "${PRITHVI_PARQUET}" \
    "${TERRAMIND_PARQUET}" \
    "${LABELS_CSV}" \
    "${COMMON_SPLIT_CSV}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Input not found: ${INPUT_PATH}" >&2
        exit 2
    fi
done

if [[ -s "${OUT_DIR}/summary.csv" && "${YE_FORCE:-0}" != "1" ]]; then
    echo "Completed temporal ablation already exists: ${OUT_DIR}/summary.csv"
    exit 0
fi

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

read -r -a SEEDS <<< "${TEMPORAL_SEEDS:-0 1 2}"
ARGS=(
    --clay "${CLAY_PARQUET}"
    --prithvi "${PRITHVI_PARQUET}"
    --terramind "${TERRAMIND_PARQUET}"
    --labels "${LABELS_CSV}"
    --split "${COMMON_SPLIT_CSV}"
    --fold "${FOLD}"
    --out-dir "${OUT_DIR}"
    --seeds "${SEEDS[@]}"
    --strategies mean concat conv1d
    --spatial-pool mean_std
    --timesteps 7
    --readout-dim 256
    --conv-channels 128
    --conv-kernel-size 3
    --mlp-hidden 128
    --dropout 0.2
    --learning-rate 0.001
    --weight-decay 0.0001
    --batch-size 32
    --max-epochs "${TEMPORAL_MAX_EPOCHS:-300}"
    --patience "${TEMPORAL_PATIENCE:-30}"
    --device cuda
    --mixed-precision
)
if [[ "${TEMPORAL_PREFLIGHT:-0}" == "1" ]]; then
    ARGS+=(--preflight-only)
fi

srun --ntasks=1 python -m benchmark_embeddings.temporal_ablation "${ARGS[@]}"
