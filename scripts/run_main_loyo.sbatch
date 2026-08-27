#!/bin/bash
#SBATCH --account=3d-abc
#SBATCH --job-name=main-loyo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#
# Usage:
#   sbatch scripts/run_main_loyo.sbatch \
#     CLAY_PARQUET PRITHVI_PARQUET TERRAMIND_6B_PARQUET PRESTO_S2_PARQUET \
#     ALPHAEARTH_PARQUET S2_INDICES LABELS_CSV OUT_DIR [S2_FIPS_MAP]
#
# Optional environment variables:
#   LOYO_SEEDS="0 1 2"
#   LOYO_N_ESTIMATORS=600
#   LOYO_PREFLIGHT=1

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
TERRAMIND_PARQUET="${3:?six-band TerraMind embedding Parquet is required}"
PRESTO_PARQUET="${4:?S2-only Presto embedding Parquet is required}"
ALPHAEARTH_PARQUET="${5:?AlphaEarth embedding Parquet is required}"
S2_INDICES="${6:?Sentinel-2 index table is required}"
LABELS_CSV="${7:?yield-label CSV is required}"
OUT_DIR="${8:?output directory is required}"
S2_FIPS_MAP="${9:-}"

for INPUT_PATH in \
    "${CLAY_PARQUET}" \
    "${PRITHVI_PARQUET}" \
    "${TERRAMIND_PARQUET}" \
    "${PRESTO_PARQUET}" \
    "${ALPHAEARTH_PARQUET}" \
    "${S2_INDICES}" \
    "${LABELS_CSV}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Input not found: ${INPUT_PATH}" >&2
        exit 2
    fi
done
if [[ -n "${S2_FIPS_MAP}" && ! -f "${S2_FIPS_MAP}" ]]; then
    echo "Sentinel-2 FIPS map not found: ${S2_FIPS_MAP}" >&2
    exit 2
fi
if [[ -s "${OUT_DIR}/summary_across_years.csv" && "${YE_FORCE:-0}" != "1" ]]; then
    echo "Completed LOYO result already exists: ${OUT_DIR}/summary_across_years.csv"
    exit 0
fi

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

read -r -a SEEDS <<< "${LOYO_SEEDS:-0 1 2}"
ARGS=(
    --clay "${CLAY_PARQUET}"
    --prithvi "${PRITHVI_PARQUET}"
    --terramind "${TERRAMIND_PARQUET}"
    --presto "${PRESTO_PARQUET}"
    --alphaearth "${ALPHAEARTH_PARQUET}"
    --s2-indices "${S2_INDICES}"
    --labels "${LABELS_CSV}"
    --out-dir "${OUT_DIR}"
    --years 2019 2020 2021 2022
    --seeds "${SEEDS[@]}"
    --timesteps 7
    --n-estimators "${LOYO_N_ESTIMATORS:-600}"
    --min-samples-leaf 2
    --max-features 1.0
    --n-jobs "${SLURM_CPUS_PER_TASK:-16}"
)
if [[ -n "${S2_FIPS_MAP}" ]]; then
    ARGS+=(--s2-indices-fips-map "${S2_FIPS_MAP}")
fi
if [[ "${LOYO_PREFLIGHT:-0}" == "1" ]]; then
    ARGS+=(--preflight-only)
fi

srun --ntasks=1 python -m benchmark_embeddings.loyo "${ARGS[@]}"
