#!/bin/bash -l
#SBATCH --account=3d-abc
#SBATCH --partition=batch
#SBATCH --job-name=stage-tiles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=11:59:00
#SBATCH --output=stage-tiles-%j.out
#SBATCH --error=stage-tiles-%j.err

# Stage the cohort tiles from project storage to scratch.
#
#   sbatch scripts/stage_tiles.sbatch FILE_LIST DEST_DIR [JOBS]
#
# FILE_LIST is the --out-list output of match_tiles.py, generated against the
# PROJECT copy. DEST_DIR is created if absent.
#
# Read-only with respect to the source. `cp -n` skips files already present, so
# the job is resumable: if it hits the wall clock, resubmit and it continues.
#
# Why batch rather than the login node: this is hours of metadata-heavy I/O on
# shared infrastructure, and a batch job gives you a log, an exit status, and a
# record of what actually happened.

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

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch stage_tiles.sbatch FILE_LIST DEST_DIR [JOBS]" >&2
  exit 2
fi

file_list=$1
dest_dir=$2
jobs=${3:-${SLURM_CPUS_PER_TASK:-16}}

[[ -r "$file_list" ]] || { echo "cannot read file list: $file_list" >&2; exit 2; }

total=$(wc -l < "$file_list")
mkdir -p "$dest_dir"

echo "staging $total files -> $dest_dir  (${jobs} parallel)"
echo "source list: $file_list"
df -h "$dest_dir" | tail -1
echo

start=$(date +%s)
# -n so re-runs skip what is already there; the job is safe to resubmit.
xargs -P "$jobs" -I{} cp -n {} "$dest_dir"/ < "$file_list"
elapsed=$(( $(date +%s) - start ))

placed=$(find "$dest_dir" -maxdepth 1 -name '*.npz' | wc -l)
echo
echo "elapsed        ${elapsed}s"
echo "files expected ${total}"
echo "files present  ${placed}"
df -h "$dest_dir" | tail -1

if [[ "$placed" -ne "$total" ]]; then
  echo
  echo "MISMATCH: ${placed} present against ${total} expected." >&2
  echo "Resubmit to continue (cp -n resumes), or investigate before extracting --" >&2
  echo "a short stage silently shrinks the cohort every downstream number keys off." >&2
  exit 1
fi

echo
echo "complete. Point extraction at it:"
echo "  export COUNTY_PATCH_TIMESTEPS=${dest_dir}"
echo "  export CLAY_EXPECTED_INPUT_COUNT=${total}   # and PRESTO_/PRITHVI_/TERRAMIND_"
