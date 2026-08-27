#!/usr/bin/env bash
# Source this in every new shell on JUPITER before extraction:
#
#   source scripts/jupiter_env.sh
#
# Deliberately not executable and not `set -e`: it is meant to be sourced, and
# an exit inside a sourced file would close the login shell.
#
# HF_HUB_OFFLINE is NOT set here because it differs by context: leave it unset
# on the login node so weights can be fetched, export it to 1 for batch jobs so
# a missing cache fails immediately instead of stalling on a dead network.

R=/e/project1/3d-abc/adriko1
B=$R/benchmark-embeddings

# The retired geofm4eo venv under /p leaks in through PYTHONPATH and shadows the
# active environment. Bind it empty rather than unsetting it -- the activate
# script references it, and the batch scripts run under `set -u`.
export PYTHONPATH=

export BENCHMARK_ENV_SETUP=$R/EODeepLearning/activate.sh
if [[ -r "$BENCHMARK_ENV_SETUP" ]]; then
  # shellcheck disable=SC1090
  source "$BENCHMARK_ENV_SETUP"
else
  echo "WARNING: cannot read $BENCHMARK_ENV_SETUP" >&2
fi

# Presto has its own environment and its own repo; extract_presto.sbatch reads
# PRESTO_ENV_SETUP and never falls back to BENCHMARK_ENV_SETUP above, because
# openmapflow pins pandas==1.5.3 and cannot share the terratorch environment.
export PRESTO_ENV_SETUP=$R/Presto/activate.sh
# $R/Presto is a wrapper (activate.sh, venv, modules.sh); the nasaharvest clone
# is one level in, and the importable package is nested inside that again at
# Presto/presto/presto. Pointing at the wrapper makes Python find the clone
# directory as a namespace package, which imports but has no attributes --
# surfacing as "module 'presto' has no attribute 'Presto'".
export PRESTO_REPO=$R/Presto/presto

export COUNTY_PATCH_TIMESTEPS=$R/datasets/US/T7
export HF_HOME=$R/hf_cache
export OUT=$B/outputs/embeddings
mkdir -p "$OUT"

# Clay needs the tree containing src/module.py -- that is clay/model inside this
# repo, not a sibling clay-foundation-model checkout.
export CLAY_REPO=$B/clay/model
export CLAY_METADATA=$B/clay/model/configs/metadata.yaml
if [[ -z "${CLAY_V15_CHECKPOINT:-}" ]]; then
  CLAY_V15_CHECKPOINT=$(find "$R" -maxdepth 4 -name 'clay-v1.5*.ckpt' 2>/dev/null | head -1)
  export CLAY_V15_CHECKPOINT
fi

# 427,049 files / 61,007 seven-interval sequences. The extractors abort if the
# corpus does not match, which is the point: a short stage cannot pass silently.
export CLAY_EXPECTED_INPUT_COUNT=427049
export PRITHVI_EXPECTED_INPUT_COUNT=427049
export TERRAMIND_EXPECTED_INPUT_COUNT=427049
export PRESTO_EXPECTED_INPUT_COUNT=427049

# Prithvi ships a default of 1, which would be 427,049 sequential passes.
export CLAY_BATCH_SIZE=32
export PRITHVI_BATCH_SIZE=16
export TERRAMIND_BATCH_SIZE=16
export PRESTO_BATCH_SIZE=256

# Leave UNDERSIZE_POLICY unset: 2000/2000 sampled tiles are 256x256, so the
# default 'error' costs nothing, whereas 'skip' reads 427,049 zip headers in
# every job.

# Fail loudly here rather than letting an empty variable reach an extractor,
# where it becomes `--npz-dir .` and surfaces as "no NPZ files found below .".
if [[ ! -d "$COUNTY_PATCH_TIMESTEPS" ]]; then
  echo "ERROR: tile directory not found: $COUNTY_PATCH_TIMESTEPS" >&2
fi
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo >&2
  echo "ERROR: this file was executed, not sourced -- every export just went to" >&2
  echo "a subshell and was discarded. Run:  source scripts/jupiter_env.sh" >&2
  echo >&2
fi

echo "jupiter_env: python=$(command -v python)"
echo "             tiles=$COUNTY_PATCH_TIMESTEPS"
echo "             out=$OUT"
echo "             clay ckpt=${CLAY_V15_CHECKPOINT:-<NOT FOUND -- set it manually>}"
echo "             HF_HOME=$HF_HOME  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"
