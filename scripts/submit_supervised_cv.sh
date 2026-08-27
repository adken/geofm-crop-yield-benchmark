#!/bin/bash
# Submit the matched 3-model x 5-fold x 3-seed supervised comparison.
# Usage: bash scripts/submit_supervised_cv.sh CONFIG RESULTS_ROOT

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?shared supervised configuration is required}"
RESULTS_ROOT="${2:?supervised result root is required}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Configuration not found: ${CONFIG}" >&2
    exit 2
fi

MODELS=(3d_convlstm gru lstm)
FOLDS=(0 1 2 3 4)
SEEDS=(0 1 2)

for MODEL in "${MODELS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            OUT_DIR="${RESULTS_ROOT}/${MODEL}/fold_${FOLD}/seed_${SEED}"
            sbatch "${HERE}/train_supervised.sbatch" \
                "${CONFIG}" "${OUT_DIR}" "${SEED}" "${FOLD}" "${MODEL}"
        done
    done
done
