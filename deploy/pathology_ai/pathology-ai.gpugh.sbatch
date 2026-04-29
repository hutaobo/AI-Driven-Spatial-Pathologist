#!/usr/bin/env bash
#SBATCH -A naiss2026-4-680-gpu
#SBATCH -J pathology-ai
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -o pathology-ai-%j.out
#SBATCH -e pathology-ai-%j.err

set -euo pipefail

DEFAULT_ROOT="/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai"
PDC_PATHOLOGY_AI_ROOT="${PDC_PATHOLOGY_AI_ROOT:-$DEFAULT_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${PATHOLOGY_AI_ENV_FILE:-$SCRIPT_DIR/pathology-ai.gpugh.env}"
if [ ! -f "$ENV_FILE" ]; then
  ENV_FILE="$SCRIPT_DIR/pathology-ai.gpugh.env.example"
fi

die() {
  echo "ERROR: $*" >&2
  exit 1
}

load_module_function() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi
  for init_script in /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
    if [ -r "$init_script" ]; then
      # shellcheck disable=SC1090
      source "$init_script"
      break
    fi
  done
}

load_apptainer() {
  if command -v apptainer >/dev/null 2>&1; then
    return 0
  fi
  load_module_function
  if ! command -v module >/dev/null 2>&1; then
    die "Environment modules are unavailable; cannot load apptainer."
  fi
  module load apptainer/1.4.4 >/dev/null 2>&1 && return 0
  module load PDC/25.03 >/dev/null 2>&1 && module load apptainer/1.4.4 >/dev/null 2>&1 && return 0
  die "Could not load apptainer/1.4.4 on this gpugh node."
}

require_file_or_dir() {
  local path="$1"
  local label="$2"
  if [ ! -e "$path" ]; then
    die "$label is missing at $path. Run deploy/pathology_ai/pdc_prepare_gh200.sh from logingh first."
  fi
}

export_runtime_env() {
  export APPTAINERENV_HF_HOME="$HF_HOME"
  export APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  export APPTAINERENV_TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export APPTAINERENV_PYTHONPATH="$REPO_ROOT/src"
  export APPTAINERENV_LLM_BASE_URL="$LLM_BASE_URL"
  export APPTAINERENV_LLM_MODEL="$LLM_MODEL"
  export APPTAINERENV_EMBED_BASE_URL="$EMBED_BASE_URL"
  export APPTAINERENV_EMBED_MODEL="$EMBED_MODEL"
  export APPTAINERENV_RERANK_BASE_URL="$RERANK_BASE_URL"
  export APPTAINERENV_RERANK_MODEL="$RERANK_MODEL"
  export APPTAINERENV_VECTOR_DB="$VECTOR_DB"
  export APPTAINERENV_QDRANT_BASE_URL="$QDRANT_BASE_URL"
  export APPTAINERENV_QDRANT_COLLECTION="$QDRANT_COLLECTION"
  export APPTAINERENV_DEFAULT_TOP_K="$DEFAULT_TOP_K"
  export APPTAINERENV_STRICT_JSON="$STRICT_JSON"
  export APPTAINERENV_REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS"
  export APPTAINERENV_MAX_CHUNK_CHARS="$MAX_CHUNK_CHARS"
  export APPTAINERENV_CHUNK_OVERLAP_CHARS="$CHUNK_OVERLAP_CHARS"
  if [ -n "${HF_TOKEN:-}" ]; then
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
    export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  fi
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local log_file="$3"
  local attempts="${4:-180}"
  local interval="${5:-10}"
  echo "Waiting for $name at $url"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready"
      return 0
    fi
    check_processes
    sleep "$interval"
  done
  echo "Last $name log lines:" >&2
  tail -n 80 "$log_file" >&2 || true
  die "Timed out waiting for $name at $url"
}

wait_for_pathology_ai() {
  local url="http://127.0.0.1:8000/health"
  echo "Waiting for pathology-ai readiness at $url"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" 2>/dev/null | grep -q '"ready": true'; then
      echo "pathology-ai is ready"
      return 0
    fi
    check_processes
    sleep 5
  done
  echo "Last pathology-ai health payload:" >&2
  curl -fsS "$url" >&2 || true
  echo >&2
  tail -n 80 "$LOG_DIR/pathology-ai.log" >&2 || true
  die "Timed out waiting for pathology-ai ready=true"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ "${#PIDS[@]}" -gt 0 ]; then
    echo "Stopping service processes: ${PIDS[*]}"
    kill "${PIDS[@]}" >/dev/null 2>&1 || true
    sleep 3
    kill -9 "${PIDS[@]}" >/dev/null 2>&1 || true
  fi
  exit "$status"
}

check_processes() {
  local index=0
  for pid in "${PIDS[@]}"; do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      die "Service process ${NAMES[$index]} exited early. Logs are in $LOG_DIR."
    fi
    index=$((index + 1))
  done
}

start_service() {
  local name="$1"
  local log_file="$2"
  shift 2
  echo "Starting $name; log: $log_file"
  "$@" >"$log_file" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$name")
}

load_apptainer

RUNTIME_ENV="$PDC_PATHOLOGY_AI_ROOT/runtime.env"
require_file_or_dir "$RUNTIME_ENV" "Runtime env"
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
require_file_or_dir "$PDC_PATHOLOGY_AI_VLLM_IMAGE" "vLLM sandbox"
require_file_or_dir "$PDC_PATHOLOGY_AI_QDRANT_IMAGE" "Qdrant sandbox"
require_file_or_dir "$ENV_FILE" "pathology-ai env file"

: "${PDC_PATHOLOGY_AI_ACCELERATOR:=rocm}"
: "${PDC_PATHOLOGY_AI_APPTAINER_GPU_FLAG:=--rocm}"
case "$PDC_PATHOLOGY_AI_ACCELERATOR" in
  cuda)
    [ "$(uname -m)" = "aarch64" ] || die "CUDA/GH200 runtime requires an aarch64 node. Submit to gpugh if that partition is available."
    ;;
  rocm)
    [ "$(uname -m)" = "x86_64" ] || die "ROCm runtime requires an x86_64 Dardel GPU node. Submit to gpu."
    ;;
  *)
    die "Unsupported PDC_PATHOLOGY_AI_ACCELERATOR=$PDC_PATHOLOGY_AI_ACCELERATOR. Use cuda or rocm."
    ;;
esac

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${LLM_BASE_URL:=http://127.0.0.1:8001/v1}"
: "${LLM_MODEL:=openai/gpt-oss-120b}"
: "${EMBED_BASE_URL:=http://127.0.0.1:8002}"
: "${EMBED_MODEL:=BAAI/bge-m3}"
: "${RERANK_BASE_URL:=http://127.0.0.1:8003}"
: "${RERANK_MODEL:=BAAI/bge-reranker-v2-m3}"
: "${VECTOR_DB:=qdrant}"
: "${QDRANT_BASE_URL:=http://127.0.0.1:6333}"
: "${QDRANT_COLLECTION:=pathology_reference}"
: "${DEFAULT_TOP_K:=6}"
: "${STRICT_JSON:=true}"
: "${REQUEST_TIMEOUT_SECONDS:=60}"
: "${MAX_CHUNK_CHARS:=1200}"
: "${CHUNK_OVERLAP_CHARS:=120}"

JOB_ID="${SLURM_JOB_ID:-manual}"
LOG_DIR="$PDC_PATHOLOGY_AI_ROOT/logs/$JOB_ID"
mkdir -p "$LOG_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$QDRANT_STORAGE" "$PDC_PATHOLOGY_AI_ROOT/tmp"
export APPTAINER_CACHEDIR="$PDC_PATHOLOGY_AI_ROOT/tmp/apptainer-cache"
export TMPDIR="$PDC_PATHOLOGY_AI_ROOT/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"

export_runtime_env

PIDS=()
NAMES=()
trap cleanup EXIT INT TERM

start_service qdrant "$LOG_DIR/qdrant.log" env \
  APPTAINERENV_QDRANT__SERVICE__HOST=127.0.0.1 \
  APPTAINERENV_QDRANT__SERVICE__HTTP_PORT=6333 \
  APPTAINERENV_QDRANT__SERVICE__GRPC_PORT=6334 \
  APPTAINERENV_QDRANT__STORAGE__STORAGE_PATH="$QDRANT_STORAGE" \
  apptainer exec --bind /cfs/klemming "$PDC_PATHOLOGY_AI_QDRANT_IMAGE" \
  bash -lc 'if command -v qdrant >/dev/null 2>&1; then exec qdrant; fi; exec /qdrant/qdrant'
wait_for_http qdrant "http://127.0.0.1:6333/collections" "$LOG_DIR/qdrant.log" 60 5

start_service vllm "$LOG_DIR/vllm.log" env \
  CUDA_VISIBLE_DEVICES=0,1 \
  HIP_VISIBLE_DEVICES=0,1 \
  APPTAINERENV_CUDA_VISIBLE_DEVICES=0,1 \
  APPTAINERENV_HIP_VISIBLE_DEVICES=0,1 \
  APPTAINERENV_LLM_MODEL="$LLM_MODEL" \
  apptainer exec "$PDC_PATHOLOGY_AI_APPTAINER_GPU_FLAG" --bind /cfs/klemming "$PDC_PATHOLOGY_AI_VLLM_IMAGE" \
  bash -lc 'if command -v vllm >/dev/null 2>&1; then exec vllm serve "$LLM_MODEL" --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 --served-model-name "$LLM_MODEL" --gpu-memory-utilization 0.9 --max-model-len 32768; fi; exec python -m vllm.entrypoints.openai.api_server --host 127.0.0.1 --port 8001 --model "$LLM_MODEL" --tensor-parallel-size 2 --served-model-name "$LLM_MODEL" --gpu-memory-utilization 0.9 --max-model-len 32768'
wait_for_http vllm "http://127.0.0.1:8001/v1/models" "$LOG_DIR/vllm.log" 360 10

start_service embedder "$LOG_DIR/embedder.log" env \
  CUDA_VISIBLE_DEVICES=2 \
  HIP_VISIBLE_DEVICES=2 \
  APPTAINERENV_CUDA_VISIBLE_DEVICES=2 \
  APPTAINERENV_HIP_VISIBLE_DEVICES=2 \
  APPTAINERENV_EMBED_MODEL="$EMBED_MODEL" \
  apptainer exec "$PDC_PATHOLOGY_AI_APPTAINER_GPU_FLAG" --bind /cfs/klemming "$PDC_PATHOLOGY_AI_VLLM_IMAGE" \
  python -m pathology_ai_service.tei_compat_server --mode embed --model-id "$EMBED_MODEL" --host 127.0.0.1 --port 8002 --device cuda --dtype auto --max-length 8192 --batch-size 4
wait_for_http embedder "http://127.0.0.1:8002/health" "$LOG_DIR/embedder.log" 180 10

start_service reranker "$LOG_DIR/reranker.log" env \
  CUDA_VISIBLE_DEVICES=3 \
  HIP_VISIBLE_DEVICES=3 \
  APPTAINERENV_CUDA_VISIBLE_DEVICES=3 \
  APPTAINERENV_HIP_VISIBLE_DEVICES=3 \
  APPTAINERENV_RERANK_MODEL="$RERANK_MODEL" \
  apptainer exec "$PDC_PATHOLOGY_AI_APPTAINER_GPU_FLAG" --bind /cfs/klemming "$PDC_PATHOLOGY_AI_VLLM_IMAGE" \
  python -m pathology_ai_service.tei_compat_server --mode rerank --model-id "$RERANK_MODEL" --host 127.0.0.1 --port 8003 --device cuda --dtype auto --max-length 8192 --batch-size 4
wait_for_http reranker "http://127.0.0.1:8003/health" "$LOG_DIR/reranker.log" 180 10

start_service pathology-ai "$LOG_DIR/pathology-ai.log" \
  apptainer exec --bind /cfs/klemming "$PDC_PATHOLOGY_AI_VLLM_IMAGE" \
  python -m pathology_ai_service --host 0.0.0.0 --port 8000
wait_for_pathology_ai

echo "pathology-ai stack is running on $(hostname)"
echo "Health endpoint: http://$(hostname):8000/health"
echo "Logs: $LOG_DIR"

wait -n "${PIDS[@]}"
die "A service process exited. Logs are in $LOG_DIR."
