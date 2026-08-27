#!/bin/bash
#SBATCH --account=3d-abc
#SBATCH --job-name=main-regression
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#
# Usage:
#   sbatch scripts/run_main_regression_benchmark.sbatch \
#     CLAY PRITHVI TERRAMIND_6B PRESTO_S2 ALPHAEARTH S2_INDICES \
#     LABELS COMMON_SPLIT OUT_DIR [S2_FIPS_MAP]

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

CLAY="${1:?Clay Parquet is required}"
PRITHVI="${2:?Prithvi Parquet is required}"
TERRAMIND="${3:?six-band TerraMind Parquet is required}"
PRESTO="${4:?S2-only Presto Parquet is required}"
ALPHAEARTH="${5:?AlphaEarth Parquet is required}"
S2_INDICES="${6:?Sentinel-2 index table is required}"
LABELS="${7:?yield-label table is required}"
SPLIT="${8:?common five-fold manifest is required}"
OUT_DIR="${9:?output directory is required}"
S2_FIPS_MAP="${10:-}"

for INPUT_PATH in "${CLAY}" "${PRITHVI}" "${TERRAMIND}" "${PRESTO}" \
    "${ALPHAEARTH}" "${S2_INDICES}" "${LABELS}" "${SPLIT}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Input not found: ${INPUT_PATH}" >&2
        exit 2
    fi
done
if [[ -n "${S2_FIPS_MAP}" && ! -f "${S2_FIPS_MAP}" ]]; then
    echo "Sentinel-2 FIPS map not found: ${S2_FIPS_MAP}" >&2
    exit 2
fi
if [[ -s "${OUT_DIR}/summary_across_folds.csv" && "${YE_FORCE:-0}" != "1" ]]; then
    echo "Completed benchmark already exists: ${OUT_DIR}/summary_across_folds.csv"
    exit 0
fi

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

ARGS=(
    --family main
    --clay "${CLAY}"
    --prithvi "${PRITHVI}"
    --terramind "${TERRAMIND}"
    --presto "${PRESTO}"
    --alphaearth "${ALPHAEARTH}"
    --s2-indices "${S2_INDICES}"
    --labels "${LABELS}"
    --split "${SPLIT}"
    --out-dir "${OUT_DIR}"
    --folds 0 1 2 3 4
    --regressors ridge random_forest xgboost ebm
    --seeds 0 1 2
    --ridge-alphas 0.01 0.1 1 10 100
    --timesteps 7
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"
)
if [[ -n "${S2_FIPS_MAP}" ]]; then
    ARGS+=(--s2-indices-fips-map "${S2_FIPS_MAP}")
fi
if [[ "${BENCHMARK_PREFLIGHT:-0}" == "1" ]]; then
    ARGS+=(--preflight-only)
fi

srun --ntasks=1 python -m benchmark_embeddings.regression_benchmark "${ARGS[@]}"
