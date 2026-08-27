#!/usr/bin/env bash
# Run the complete benchmark sequentially on one workstation.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_all_local.sh [STAGE] [ENV_FILE]

Stages:
  all           prepare, preflight, extract, regressions, ablations, aggregate, parity
  prepare       canonical AlphaEarth Parquet and county-grouped split manifest
  preflight     full-cohort frozen-encoder input audits
  extract       Clay, Presto, Prithvi, and both TerraMind extractions
  main          unfused main five-fold regression benchmark
  terramind10   separate ten-observed-band TerraMind regression run
  climate       Presto+ERA5 and all Daymet late-fusion regressions
  temporal      five temporal-pooling ablation folds
  loyo          climate-free main-benchmark LOYO
  supervised    sequential 3-model x 5-fold x 3-seed supervised grid
  aggregate     temporal and supervised aggregation
  parity        final cross-experiment contract audit

ENV_FILE is optional when variables are already exported. Start from
configs/local.env.example. Existing completed outputs are skipped; set FORCE=1
to rerun them. Set DRY_RUN=1 to print commands without executing them.
EOF
}

STAGE="${1:-all}"
ENV_FILE="${2:-}"
if [[ "${STAGE}" == "-h" || "${STAGE}" == "--help" || "${STAGE}" == "help" ]]; then
    usage
    exit 0
fi
case "${STAGE}" in
    all|prepare|preflight|extract|main|terramind10|climate|temporal|loyo|supervised|aggregate|parity) ;;
    *)
        echo "Unknown stage: ${STAGE}" >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -n "${ENV_FILE}" ]]; then
    if [[ ! -f "${ENV_FILE}" ]]; then
        echo "Environment file not found: ${ENV_FILE}" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi
cd "${BENCHMARK_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EXPECTED_INPUT_COUNT="${EXPECTED_INPUT_COUNT:-77813}"
LOCAL_DEVICE="${LOCAL_DEVICE:-auto}"
LOCAL_NUM_WORKERS="${LOCAL_NUM_WORKERS:-0}"
LOCAL_N_JOBS="${LOCAL_N_JOBS:-4}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
UNDERSIZE_POLICY="${UNDERSIZE_POLICY:-error}"
export PYTORCH_ENABLE_MPS_FALLBACK
export PYTHONPATH="${BENCHMARK_ROOT}:${PYTHONPATH:-}"

EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-${BENCHMARK_ROOT}/outputs/embeddings}"
RESULTS_ROOT="${RESULTS_ROOT:-${BENCHMARK_ROOT}/outputs/results}"
ALPHAEARTH_CSV="${ALPHAEARTH_CSV:-${BENCHMARK_ROOT}/data/alphaearth_matched.csv}"
ALPHAEARTH_OUTPUT="${ALPHAEARTH_OUTPUT:-${BENCHMARK_ROOT}/data/alphaearth.parquet}"
COMMON_SPLIT_MANIFEST="${COMMON_SPLIT_MANIFEST:-${BENCHMARK_ROOT}/data/group_kfold_county_T7.csv}"
S2_DAYMET_MERGED="${S2_DAYMET_MERGED:-${BENCHMARK_ROOT}/data/s2_daymet_merged.xlsx}"

if [[ -n "${YIELD_EMBEDDINGS_ROOT:-}" ]]; then
    CLAY_REPO="${CLAY_REPO:-${YIELD_EMBEDDINGS_ROOT}/clay/model}"
    CLAY_METADATA="${CLAY_METADATA:-${YIELD_EMBEDDINGS_ROOT}/clay/model/configs/metadata.yaml}"
    PRESTO_REPO="${PRESTO_REPO:-${YIELD_EMBEDDINGS_ROOT}/presto/presto}"
fi

CLAY_OUTPUT="${CLAY_OUTPUT:-${EMBEDDINGS_ROOT}/clay_v1_5_cls.parquet}"
PRESTO_OUTPUT="${PRESTO_OUTPUT:-${EMBEDDINGS_ROOT}/presto_s2.parquet}"
PRESTO_ERA5_OUTPUT="${PRESTO_ERA5_OUTPUT:-${EMBEDDINGS_ROOT}/presto_s2_era5.parquet}"
PRITHVI_OUTPUT="${PRITHVI_OUTPUT:-${EMBEDDINGS_ROOT}/prithvi_eo_v2_300_tl_per_timestep_spatial_mean.parquet}"
TERRAMIND_6_MODEL="${TERRAMIND_6_MODEL:-terramind_v1_base}"
TERRAMIND_10_MODEL="${TERRAMIND_10_MODEL:-terramind_v1_base}"
TERRAMIND_6_OUTPUT="${TERRAMIND_6_OUTPUT:-${EMBEDDINGS_ROOT}/${TERRAMIND_6_MODEL}_s2_6.parquet}"
TERRAMIND_10_OUTPUT="${TERRAMIND_10_OUTPUT:-${EMBEDDINGS_ROOT}/${TERRAMIND_10_MODEL}_s2_10_zp12.parquet}"
SUPERVISED_CONFIG="${SUPERVISED_CONFIG:-${BENCHMARK_ROOT}/configs/supervised_s2.yaml}"

read -r -a FOLDS <<< "${LOCAL_FOLDS:-0 1 2 3 4}"
read -r -a SEEDS <<< "${LOCAL_SEEDS:-0 1 2}"
read -r -a REGRESSORS <<< "${LOCAL_REGRESSORS:-ridge random_forest xgboost ebm}"
read -r -a RIDGE_ALPHAS <<< "${LOCAL_RIDGE_ALPHAS:-0.01 0.1 1 10 100}"

mkdir -p "${EMBEDDINGS_ROOT}" "${RESULTS_ROOT}"

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_cmd() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

require_value() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Required setting is missing: ${name}" >&2
        exit 2
    fi
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "${label} not found: ${path}" >&2
        exit 2
    fi
}

require_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "${path}" ]]; then
        echo "${label} not found: ${path}" >&2
        exit 2
    fi
}

run_unless_complete() {
    local marker="$1"
    shift
    if [[ -s "${marker}" && "${FORCE}" != "1" ]]; then
        log "Skipping completed output ${marker}"
        return 0
    fi
    run_cmd "$@"
}

resolve_device() {
    "${PYTHON_BIN}" -c '
import sys, torch
requested = sys.argv[1].strip().lower()
if requested == "auto":
    if torch.cuda.is_available():
        requested = "cuda"
    elif torch.backends.mps.is_available():
        requested = "mps"
    else:
        requested = "cpu"
elif requested == "cuda" and not torch.cuda.is_available():
    raise SystemExit("LOCAL_DEVICE=cuda, but CUDA is unavailable")
elif requested == "mps" and not torch.backends.mps.is_available():
    raise SystemExit("LOCAL_DEVICE=mps, but MPS is unavailable")
if requested not in {"cpu", "cuda", "mps"}:
    raise SystemExit(f"Unsupported LOCAL_DEVICE={requested!r}")
print(requested)
' "${LOCAL_DEVICE}"
}

DEVICE="$(resolve_device)"
log "Local device: ${DEVICE}; MPS fallback=${PYTORCH_ENABLE_MPS_FALLBACK}"

export EMBEDDINGS_ROOT RESULTS_ROOT COMMON_SPLIT_MANIFEST
export COUNTY_PATCH_TIMESTEPS="${COUNTY_PATCH_TIMESTEPS:-}"
export YIELD_LABELS="${YIELD_LABELS:-}"

prepare_inputs() {
    require_file "${ALPHAEARTH_CSV}" "AlphaEarth source CSV"
    require_value COUNTY_PATCH_TIMESTEPS
    require_value YIELD_LABELS
    require_value COUNTY_FIPS_MAP
    require_dir "${COUNTY_PATCH_TIMESTEPS}" "Sentinel-2 patch directory"
    require_file "${S2_DAYMET_MERGED}" "merged Sentinel-2/Daymet table"
    require_file "${COUNTY_FIPS_MAP}" "county FIPS map"
    require_file "${YIELD_LABELS}" "yield labels"

    log "Preparing canonical AlphaEarth embeddings"
    run_unless_complete "${ALPHAEARTH_OUTPUT}" \
        "${PYTHON_BIN}" -m benchmark_embeddings.prepare_alphaearth \
        --input "${ALPHAEARTH_CSV}" \
        --output "${ALPHAEARTH_OUTPUT}"

    log "Building the authoritative county-grouped five-fold manifest"
    run_unless_complete "${COMMON_SPLIT_MANIFEST}.contract.json" \
        "${PYTHON_BIN}" -m benchmark_embeddings.build_splits \
        --s2-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --s2-daymet-merged "${S2_DAYMET_MERGED}" \
        --s2-fips-map "${COUNTY_FIPS_MAP}" \
        --alphaearth "${ALPHAEARTH_OUTPUT}" \
        --labels "${YIELD_LABELS}" \
        --output "${COMMON_SPLIT_MANIFEST}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}"
}

preflight_encoders() {
    require_value COUNTY_PATCH_TIMESTEPS
    require_value ERA5_PATCH_TIMESTEPS
    require_value CLAY_METADATA
    require_dir "${COUNTY_PATCH_TIMESTEPS}" "Sentinel-2 patch directory"
    require_dir "${ERA5_PATCH_TIMESTEPS}" "ERA5 patch directory"
    require_file "${CLAY_METADATA}" "Clay metadata"

    log "Preflighting Clay"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.clay \
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --metadata "${CLAY_METADATA}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --preflight-only

    log "Preflighting S2-only Presto"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.presto \
        --s2-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --s2-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --preflight-only

    log "Preflighting native Presto+ERA5"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.presto \
        --s2-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --era5-dir "${ERA5_PATCH_TIMESTEPS}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --expected-era5-input-count "${EXPECTED_INPUT_COUNT}" \
        --s2-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --era5-source-bands total_precipitation temperature_2m \
        --preflight-only

    log "Preflighting Prithvi-EO-2.0-300M-TL"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.prithvi \
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --source-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --preflight-only

    log "Preflighting six-band TerraMind"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.terramind \
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --experiment s2_6_prithvi \
        --model "${TERRAMIND_6_MODEL}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --source-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --preflight-only

    log "Preflighting ten-observed-band TerraMind"
    run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.frozen.terramind \
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --experiment s2_10_zero_pad \
        --model "${TERRAMIND_10_MODEL}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --source-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --preflight-only
}

extract_embeddings() {
    require_value COUNTY_PATCH_TIMESTEPS
    require_value ERA5_PATCH_TIMESTEPS
    require_value CLAY_REPO
    require_value CLAY_METADATA
    require_value CLAY_V15_CHECKPOINT
    require_value PRESTO_REPO
    require_dir "${COUNTY_PATCH_TIMESTEPS}" "Sentinel-2 patch directory"
    require_dir "${ERA5_PATCH_TIMESTEPS}" "ERA5 patch directory"
    require_dir "${CLAY_REPO}" "Clay repository"
    require_file "${CLAY_METADATA}" "Clay metadata"
    require_file "${CLAY_V15_CHECKPOINT}" "Clay checkpoint"
    require_dir "${PRESTO_REPO}" "Presto repository"

    log "Extracting Clay embeddings"
    run_unless_complete "${CLAY_OUTPUT}" \
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.clay \
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --metadata "${CLAY_METADATA}" \
        --clay-repo "${CLAY_REPO}" \
        --checkpoint "${CLAY_V15_CHECKPOINT}" \
        --output "${CLAY_OUTPUT}" \
        --pooling cls \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --batch-size "${CLAY_BATCH_SIZE:-1}" \
        --num-workers "${LOCAL_NUM_WORKERS}" \
        --device "${DEVICE}"

    log "Extracting S2-only Presto embeddings"
    run_unless_complete "${PRESTO_OUTPUT}" \
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.presto \
        --s2-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --presto-repo "${PRESTO_REPO}" \
        --output "${PRESTO_OUTPUT}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --s2-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --batch-size "${PRESTO_BATCH_SIZE:-64}" \
        --num-workers "${LOCAL_NUM_WORKERS}" \
        --device "${DEVICE}"

    log "Extracting native Presto+ERA5 embeddings"
    run_unless_complete "${PRESTO_ERA5_OUTPUT}" \
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.presto \
        --s2-dir "${COUNTY_PATCH_TIMESTEPS}" \
        --era5-dir "${ERA5_PATCH_TIMESTEPS}" \
        --presto-repo "${PRESTO_REPO}" \
        --output "${PRESTO_ERA5_OUTPUT}" \
        --expected-input-count "${EXPECTED_INPUT_COUNT}" \
        --expected-era5-input-count "${EXPECTED_INPUT_COUNT}" \
        --s2-units auto \
        --undersize-policy "${UNDERSIZE_POLICY}" \
        --era5-source-bands total_precipitation temperature_2m \
        --batch-size "${PRESTO_BATCH_SIZE:-64}" \
        --num-workers "${LOCAL_NUM_WORKERS}" \
        --device "${DEVICE}"

    local prithvi_args=(
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.prithvi
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}"
        --output "${PRITHVI_OUTPUT}"
        --pooling per_timestep_spatial_mean
        --expected-input-count "${EXPECTED_INPUT_COUNT}"
        --expected-timesteps 7
        --source-units auto
        --undersize-policy "${UNDERSIZE_POLICY}"
        --batch-size "${PRITHVI_BATCH_SIZE:-1}"
        --num-workers "${LOCAL_NUM_WORKERS}"
        --device "${DEVICE}"
    )
    if [[ -n "${PRITHVI_CHECKPOINT:-}" ]]; then
        require_file "${PRITHVI_CHECKPOINT}" "Prithvi checkpoint"
        prithvi_args+=(--checkpoint "${PRITHVI_CHECKPOINT}")
    fi
    log "Extracting Prithvi-EO-2.0-300M-TL embeddings"
    run_unless_complete "${PRITHVI_OUTPUT}" "${prithvi_args[@]}"

    local terramind_6_args=(
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.terramind
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}"
        --experiment s2_6_prithvi
        --model "${TERRAMIND_6_MODEL}"
        --output "${TERRAMIND_6_OUTPUT}"
        --expected-input-count "${EXPECTED_INPUT_COUNT}"
        --source-units auto
        --undersize-policy "${UNDERSIZE_POLICY}"
        --batch-size "${TERRAMIND_BATCH_SIZE:-1}"
        --num-workers "${LOCAL_NUM_WORKERS}"
        --device "${DEVICE}"
    )
    if [[ -n "${TERRAMIND_6_CHECKPOINT:-}" ]]; then
        require_file "${TERRAMIND_6_CHECKPOINT}" "six-band TerraMind checkpoint"
        terramind_6_args+=(--checkpoint "${TERRAMIND_6_CHECKPOINT}")
    fi
    log "Extracting six-band TerraMind embeddings"
    run_unless_complete "${TERRAMIND_6_OUTPUT}" "${terramind_6_args[@]}"

    local terramind_10_args=(
        "${PYTHON_BIN}" -m benchmark_embeddings.frozen.terramind
        --npz-dir "${COUNTY_PATCH_TIMESTEPS}"
        --experiment s2_10_zero_pad
        --model "${TERRAMIND_10_MODEL}"
        --output "${TERRAMIND_10_OUTPUT}"
        --expected-input-count "${EXPECTED_INPUT_COUNT}"
        --source-units auto
        --undersize-policy "${UNDERSIZE_POLICY}"
        --batch-size "${TERRAMIND_BATCH_SIZE:-1}"
        --num-workers "${LOCAL_NUM_WORKERS}"
        --device "${DEVICE}"
    )
    if [[ -n "${TERRAMIND_10_CHECKPOINT:-}" ]]; then
        require_file "${TERRAMIND_10_CHECKPOINT}" "ten-band TerraMind checkpoint"
        terramind_10_args+=(--checkpoint "${TERRAMIND_10_CHECKPOINT}")
    fi
    log "Extracting ten-observed-band TerraMind embeddings"
    run_unless_complete "${TERRAMIND_10_OUTPUT}" "${terramind_10_args[@]}"
}

require_analysis_inputs() {
    require_value YIELD_LABELS
    require_value COUNTY_FIPS_MAP
    require_file "${CLAY_OUTPUT}" "Clay embeddings"
    require_file "${PRITHVI_OUTPUT}" "Prithvi embeddings"
    require_file "${TERRAMIND_6_OUTPUT}" "six-band TerraMind embeddings"
    require_file "${PRESTO_OUTPUT}" "S2-only Presto embeddings"
    require_file "${ALPHAEARTH_OUTPUT}" "AlphaEarth embeddings"
    require_file "${S2_DAYMET_MERGED}" "merged Sentinel-2/Daymet table"
    require_file "${YIELD_LABELS}" "yield labels"
    require_file "${COMMON_SPLIT_MANIFEST}" "common split manifest"
    require_file "${COUNTY_FIPS_MAP}" "county FIPS map"
}

run_regression_family() {
    local family="$1"
    local terramind_path="$2"
    local out_dir="$3"
    local marker="${out_dir}/summary_across_folds.csv"
    if [[ -s "${marker}" && "${FORCE}" != "1" ]]; then
        log "Skipping completed output ${marker}"
        return 0
    fi
    local args=(
        "${PYTHON_BIN}" -m benchmark_embeddings.regression_benchmark
        --family "${family}"
        --clay "${CLAY_OUTPUT}"
        --prithvi "${PRITHVI_OUTPUT}"
        --terramind "${terramind_path}"
        --presto "${PRESTO_OUTPUT}"
        --labels "${YIELD_LABELS}"
        --split "${COMMON_SPLIT_MANIFEST}"
        --out-dir "${out_dir}"
        --folds "${FOLDS[@]}"
        --regressors "${REGRESSORS[@]}"
        --seeds "${SEEDS[@]}"
        --ridge-alphas "${RIDGE_ALPHAS[@]}"
        --timesteps 7
        --n-jobs "${LOCAL_N_JOBS}"
    )
    if [[ "${family}" == "main" ]]; then
        args+=(
            --alphaearth "${ALPHAEARTH_OUTPUT}"
            --s2-indices "${S2_DAYMET_MERGED}"
            --s2-indices-fips-map "${COUNTY_FIPS_MAP}"
        )
    else
        require_file "${PRESTO_ERA5_OUTPUT}" "Presto+ERA5 embeddings"
        args+=(
            --presto-era5 "${PRESTO_ERA5_OUTPUT}"
            --s2-daymet-merged "${S2_DAYMET_MERGED}"
            --s2-indices-fips-map "${COUNTY_FIPS_MAP}"
            --daymet-fips-map "${COUNTY_FIPS_MAP}"
        )
    fi
    log "Preflighting ${family} regression family"
    run_cmd "${args[@]}" --preflight-only
    log "Running ${family} regression family"
    run_cmd "${args[@]}"
}

run_main_regression() {
    require_analysis_inputs
    run_regression_family main "${TERRAMIND_6_OUTPUT}" "${RESULTS_ROOT}/main_regression"
}

run_terramind10_regression() {
    require_analysis_inputs
    require_file "${TERRAMIND_10_OUTPUT}" "ten-observed-band TerraMind embeddings"
    run_regression_family main "${TERRAMIND_10_OUTPUT}" "${RESULTS_ROOT}/terramind_10_regression"
}

run_climate_regression() {
    require_analysis_inputs
    run_regression_family climate_fusion "${TERRAMIND_6_OUTPUT}" "${RESULTS_ROOT}/climate_regression"
}

run_temporal() {
    require_analysis_inputs
    local fold
    for fold in "${FOLDS[@]}"; do
        local out_dir="${RESULTS_ROOT}/temporal_ablation/fold_${fold}"
        local marker="${out_dir}/summary.csv"
        if [[ -s "${marker}" && "${FORCE}" != "1" ]]; then
            log "Skipping completed output ${marker}"
            continue
        fi
        local args=(
            "${PYTHON_BIN}" -m benchmark_embeddings.temporal_ablation
            --clay "${CLAY_OUTPUT}"
            --prithvi "${PRITHVI_OUTPUT}"
            --terramind "${TERRAMIND_6_OUTPUT}"
            --labels "${YIELD_LABELS}"
            --split "${COMMON_SPLIT_MANIFEST}"
            --fold "${fold}"
            --out-dir "${out_dir}"
            --seeds "${SEEDS[@]}"
            --device "${DEVICE}"
            --batch-size "${TEMPORAL_BATCH_SIZE:-32}"
            --max-epochs "${TEMPORAL_MAX_EPOCHS:-300}"
            --patience "${TEMPORAL_PATIENCE:-30}"
        )
        log "Preflighting temporal ablation fold ${fold}"
        run_cmd "${args[@]}" --preflight-only
        log "Running temporal ablation fold ${fold}"
        run_cmd "${args[@]}"
    done
}

run_loyo() {
    require_analysis_inputs
    local out_dir="${RESULTS_ROOT}/main_loyo"
    local marker="${out_dir}/summary_across_years.csv"
    if [[ -s "${marker}" && "${FORCE}" != "1" ]]; then
        log "Skipping completed output ${marker}"
        return 0
    fi
    local args=(
        "${PYTHON_BIN}" -m benchmark_embeddings.loyo
        --clay "${CLAY_OUTPUT}"
        --prithvi "${PRITHVI_OUTPUT}"
        --terramind "${TERRAMIND_6_OUTPUT}"
        --presto "${PRESTO_OUTPUT}"
        --alphaearth "${ALPHAEARTH_OUTPUT}"
        --s2-indices "${S2_DAYMET_MERGED}"
        --s2-indices-fips-map "${COUNTY_FIPS_MAP}"
        --labels "${YIELD_LABELS}"
        --out-dir "${out_dir}"
        --years 2019 2020 2021 2022
        --seeds "${SEEDS[@]}"
        --n-jobs "${LOCAL_N_JOBS}"
    )
    log "Preflighting climate-free LOYO"
    run_cmd "${args[@]}" --preflight-only
    log "Running climate-free LOYO"
    run_cmd "${args[@]}"
}

run_supervised() {
    require_value COUNTY_PATCH_TIMESTEPS
    require_value YIELD_LABELS
    require_file "${SUPERVISED_CONFIG}" "supervised configuration"
    require_file "${COMMON_SPLIT_MANIFEST}" "common split manifest"
    require_dir "${COUNTY_PATCH_TIMESTEPS}" "Sentinel-2 patch directory"
    require_file "${YIELD_LABELS}" "yield labels"

    local fold model seed out_dir marker
    for fold in "${FOLDS[@]}"; do
        log "Inspecting supervised inputs for fold ${fold}"
        run_cmd "${PYTHON_BIN}" -m benchmark_embeddings.train \
            --config "${SUPERVISED_CONFIG}" \
            --out-dir "${RESULTS_ROOT}/supervised/preflight/fold_${fold}" \
            --fold "${fold}" \
            --model 3d_convlstm \
            --device "${DEVICE}" \
            --inspect-only
    done
    for model in 3d_convlstm gru lstm; do
        for fold in "${FOLDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                out_dir="${RESULTS_ROOT}/supervised/${model}/fold_${fold}/seed_${seed}"
                marker="${out_dir}/result.json"
                log "Supervised ${model}, fold ${fold}, seed ${seed}"
                run_unless_complete "${marker}" \
                    "${PYTHON_BIN}" -m benchmark_embeddings.train \
                    --config "${SUPERVISED_CONFIG}" \
                    --out-dir "${out_dir}" \
                    --seed "${seed}" \
                    --fold "${fold}" \
                    --model "${model}" \
                    --device "${DEVICE}"
            done
        done
    done
}

aggregate_results() {
    local temporal_dirs=()
    local supervised_dirs=()
    local fold model seed
    for fold in "${FOLDS[@]}"; do
        temporal_dirs+=("${RESULTS_ROOT}/temporal_ablation/fold_${fold}")
    done
    for model in 3d_convlstm gru lstm; do
        for fold in "${FOLDS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                supervised_dirs+=("${RESULTS_ROOT}/supervised/${model}/fold_${fold}/seed_${seed}")
            done
        done
    done

    log "Aggregating temporal ablation folds"
    run_unless_complete "${RESULTS_ROOT}/temporal_ablation/aggregate/summary_across_folds.csv" \
        "${PYTHON_BIN}" -m benchmark_embeddings.temporal_ablation_aggregate \
        --fold-dirs "${temporal_dirs[@]}" \
        --expected-folds "${FOLDS[@]}" \
        --out-dir "${RESULTS_ROOT}/temporal_ablation/aggregate"

    log "Aggregating supervised runs"
    run_unless_complete "${RESULTS_ROOT}/supervised_cv_summary/summary_across_folds.csv" \
        "${PYTHON_BIN}" -m benchmark_embeddings.supervised_aggregate \
        --run-dirs "${supervised_dirs[@]}" \
        --folds "${FOLDS[@]}" \
        --seeds "${SEEDS[@]}" \
        --out-dir "${RESULTS_ROOT}/supervised_cv_summary"
}

audit_parity() {
    local temporal_contracts=()
    local fold
    for fold in "${FOLDS[@]}"; do
        temporal_contracts+=("${RESULTS_ROOT}/temporal_ablation/fold_${fold}/data_contract.json")
    done
    log "Auditing final experiment parity"
    run_unless_complete "${RESULTS_ROOT}/experiment_parity.json" \
        "${PYTHON_BIN}" -m benchmark_embeddings.experiment_parity \
        --main-contract "${RESULTS_ROOT}/main_regression/data_contract.json" \
        --climate-contract "${RESULTS_ROOT}/climate_regression/data_contract.json" \
        --temporal-contracts "${temporal_contracts[@]}" \
        --loyo-contract "${RESULTS_ROOT}/main_loyo/data_contract.json" \
        --supervised-contract "${RESULTS_ROOT}/supervised_cv_summary/data_contract.json" \
        --expected-folds "${FOLDS[@]}" \
        --required-regressors "${REGRESSORS[@]}" \
        --output "${RESULTS_ROOT}/experiment_parity.json"
}

case "${STAGE}" in
    all)
        prepare_inputs
        preflight_encoders
        extract_embeddings
        run_main_regression
        run_terramind10_regression
        run_climate_regression
        run_temporal
        run_loyo
        run_supervised
        aggregate_results
        audit_parity
        ;;
    prepare) prepare_inputs ;;
    preflight) preflight_encoders ;;
    extract) extract_embeddings ;;
    main) run_main_regression ;;
    terramind10) run_terramind10_regression ;;
    climate) run_climate_regression ;;
    temporal) run_temporal ;;
    loyo) run_loyo ;;
    supervised) run_supervised ;;
    aggregate) aggregate_results ;;
    parity) audit_parity ;;
esac

log "Stage '${STAGE}' completed"
