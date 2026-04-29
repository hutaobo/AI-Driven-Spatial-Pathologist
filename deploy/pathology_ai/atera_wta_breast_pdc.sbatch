#!/bin/bash -l
#SBATCH -A naiss2026-4-680
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH -J atera-spatho

set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/cfs/klemming/projects/supr/naiss2025-22-606/data/WTA_Preview_FFPE_Breast_Cancer_outs}"
RUN_ROOT="${RUN_ROOT:-/cfs/klemming/projects/supr/naiss2025-22-606/results/ai-driven-spatial-pathologist/atera_wta_breast_pdc_20260429}"
REPO_DIR="${REPO_DIR:-/cfs/klemming/home/h/hutaobo/AI-Driven-Spatial-Pathologist}"
PYXENIUM_DIR="${PYXENIUM_DIR:-/cfs/klemming/home/h/hutaobo/pyXenium}"
SFPLOT_ROOT="${SFPLOT_ROOT:-${RUN_ROOT}/deps/sfplot}"
PATHOLOGY_AI_BASE_URL="${PATHOLOGY_AI_BASE_URL:-http://nid002802:8000}"
VENV_DIR="${VENV_DIR:-${RUN_ROOT}/venv}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/deps"
exec > >(tee -a "${RUN_ROOT}/logs/slurm-${SLURM_JOB_ID:-manual}.log") 2>&1

echo "[atera-spatho] started on $(hostname) at $(date -Is)"
echo "[atera-spatho] dataset=${DATASET_ROOT}"
echo "[atera-spatho] run_root=${RUN_ROOT}"
echo "[atera-spatho] pathology_ai=${PATHOLOGY_AI_BASE_URL}"

module load python/3.12.3 || module load cray-python/3.11.7

curl -fsS "${PATHOLOGY_AI_BASE_URL%/}/health"
echo

if [ ! -d "${VENV_DIR}" ]; then
  python -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade \
  numpy scipy pandas pyarrow h5py zarr numcodecs tifffile imagecodecs \
  scikit-image scikit-learn matplotlib seaborn statsmodels scanpy shapely \
  fsspec requests pyyaml aiohttp click
python -m pip install --editable "${PYXENIUM_DIR}"
python -m pip install --editable "${REPO_DIR}"

python - <<'PY'
import histoseg
import pydantic
import pyXenium
import spatho
print("imports_ok", spatho.__version__)
PY

if [ ! -d "${SFPLOT_ROOT}/.git" ]; then
  git clone --depth 1 https://github.com/hutaobo/sfplot.git "${SFPLOT_ROOT}"
else
  git -C "${SFPLOT_ROOT}" fetch origin
  if git -C "${SFPLOT_ROOT}" show-ref --verify --quiet refs/remotes/origin/main; then
    git -C "${SFPLOT_ROOT}" checkout main
    git -C "${SFPLOT_ROOT}" pull --ff-only origin main
  else
    current_branch="$(git -C "${SFPLOT_ROOT}" branch --show-current)"
    if [ -n "${current_branch}" ]; then
      git -C "${SFPLOT_ROOT}" pull --ff-only origin "${current_branch}" || true
    fi
  fi
fi

export SFPLOT_ROOT
python "${REPO_DIR}/scripts/pdc_atera_breast_workflow.py" \
  --dataset-root "${DATASET_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --sfplot-root "${SFPLOT_ROOT}" \
  --pathology-ai-base-url "${PATHOLOGY_AI_BASE_URL}"

echo "[atera-spatho] finished at $(date -Is)"
