#!/bin/bash
#SBATCH --account=3d-abc
#SBATCH --job-name=climate-regression
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=booster
#SBATCH --time=11:59:00
# JUPITER exposes only `booster` (largebooster is down), so this CPU-bound job
# runs on a GPU partition and the GH200 sits idle. That is a deliberate trade:
# the alternative is days of runtime on a workstation. The QOS wall limit is
# exactly 12:00:00 and rejects equality, hence 11:59:00.
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#
# Usage:
#   sbatch scripts/run_climate_regression_benchmark.sbatch \
#     CLAY PRITHVI TERRAMIND_6B PRESTO_S2 PRESTO_ERA5 S2_DAYMET_MERGED \
#     LABELS COMMON_SPLIT OUT_DIR [COUNTY_FIPS_MAP]

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
PRESTO_ERA5="${5:?native Presto+ERA5 Parquet is required}"
S2_DAYMET_MERGED="${6:?merged Sentinel-2 and Daymet table is required}"
LABELS="${7:?yield-label table is required}"
SPLIT="${8:?common five-fold manifest is required}"
OUT_DIR="${9:?output directory is required}"
COUNTY_FIPS_MAP="${10:-}"

for INPUT_PATH in "${CLAY}" "${PRITHVI}" "${TERRAMIND}" "${PRESTO}" \
    "${PRESTO_ERA5}" "${S2_DAYMET_MERGED}" "${LABELS}" "${SPLIT}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Input not found: ${INPUT_PATH}" >&2
        exit 2
    fi
done
if [[ -n "${COUNTY_FIPS_MAP}" && ! -f "${COUNTY_FIPS_MAP}" ]]; then
    echo "FIPS map not found: ${COUNTY_FIPS_MAP}" >&2
    exit 2
fi
if [[ -s "${OUT_DIR}/summary_across_folds.csv" && "${YE_FORCE:-0}" != "1" ]]; then
    echo "Completed benchmark already exists: ${OUT_DIR}/summary_across_folds.csv"
    exit 0
fi

# Fail before the queue rather than after: these two packages are imported
# lazily inside the fitting loop, and the environment on aarch64 may not carry
# them. require_regressor_dependencies() checks this too, but only after the
# embeddings have been read, which is twenty minutes in.
python - <<'PYCHECK' || exit 2
import importlib, sys
missing = [m for m in ("xgboost", "interpret") if importlib.util.find_spec(m) is None]
if missing:
    sys.stderr.write(f"ERROR: missing regressor packages {missing}\n")
    sys.stderr.write("Install them in this environment or pass --regressors "
                     "without the heads that need them.\n")
    sys.exit(1)
PYCHECK

mkdir -p "${OUT_DIR}"
cd "${BENCHMARK_ROOT}"
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

ARGS=(
    --family climate_fusion
    --clay "${CLAY}"
    --prithvi "${PRITHVI}"
    --terramind "${TERRAMIND}"
    --presto "${PRESTO}"
    --presto-era5 "${PRESTO_ERA5}"
    --s2-daymet-merged "${S2_DAYMET_MERGED}"
    --labels "${LABELS}"
    --split "${SPLIT}"
    --out-dir "${OUT_DIR}"
    --folds 0 1 2 3 4
    --regressors ridge random_forest xgboost ebm
    --seeds 0 1 2
    --ridge-alphas 0.01 0.1 1 10 100
    --timesteps 7
    --n-jobs "${SLURM_CPUS_PER_TASK:-32}"
    # County pooling must match the main table, LOYO and LOSO, all of which use
    # the joint form. Leaving this at the module default would silently produce
    # two-stage features and make the fusion table the only analysis pooled
    # differently -- which is the defect this run exists to correct.
    --temporal-pool "${CLIMATE_TEMPORAL_POOL:-joint}"
    # Library defaults with no pairwise interactions, matching the main table.
    # The earlier preregistered EBM configuration underfits by roughly 0.09 R2,
    # so keeping it here would make the fusion EBM column incomparable to the
    # main-table EBM column for reasons unrelated to climate.
    --ebm-interactions "${CLIMATE_EBM_INTERACTIONS:-0}"
)
if [[ -n "${COUNTY_FIPS_MAP}" ]]; then
    ARGS+=(
        --s2-indices-fips-map "${COUNTY_FIPS_MAP}"
        --daymet-fips-map "${COUNTY_FIPS_MAP}"
    )
fi
if [[ "${BENCHMARK_PREFLIGHT:-0}" == "1" ]]; then
    ARGS+=(--preflight-only)
fi

srun --ntasks=1 python -m benchmark_embeddings.regression_benchmark "${ARGS[@]}"
