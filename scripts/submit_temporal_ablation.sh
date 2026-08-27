#!/bin/bash
# Submit the matched 3-encoder x 3-pooling x 5-fold temporal MLP ablation.
# Usage:
#   bash scripts/submit_temporal_ablation.sh \
#     CLAY PRITHVI TERRAMIND LABELS COMMON_SPLIT RESULTS_ROOT

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAY="${1:?Clay embedding Parquet is required}"
PRITHVI="${2:?Prithvi embedding Parquet is required}"
TERRAMIND="${3:?TerraMind embedding Parquet is required}"
LABELS="${4:?yield-label CSV is required}"
SPLIT="${5:?common five-fold manifest is required}"
RESULTS_ROOT="${6:?temporal-ablation result root is required}"

for FOLD in 0 1 2 3 4; do
    sbatch "${HERE}/run_temporal_ablation.sbatch" \
        "${CLAY}" \
        "${PRITHVI}" \
        "${TERRAMIND}" \
        "${LABELS}" \
        "${SPLIT}" \
        "${FOLD}" \
        "${RESULTS_ROOT}/fold_${FOLD}"
done
