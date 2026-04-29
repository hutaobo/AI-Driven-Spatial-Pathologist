#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="/cfs/klemming/projects/supr/naiss2023-23-563/pathology-ai"
PDC_PATHOLOGY_AI_ROOT="${PDC_PATHOLOGY_AI_ROOT:-$DEFAULT_ROOT}"
PDC_PATHOLOGY_AI_ACCELERATOR="${PDC_PATHOLOGY_AI_ACCELERATOR:-auto}"

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
  module load PDC/24.11 >/dev/null 2>&1 && module load apptainer/1.4.0-cpeGNU-24.11 >/dev/null 2>&1 && return 0
  die "Could not load apptainer. On PDC GH200, try: module load apptainer/1.4.4"
}

detect_accelerator() {
  local arch
  arch="$(uname -m)"
  if [ "$PDC_PATHOLOGY_AI_ACCELERATOR" != "auto" ]; then
    echo "$PDC_PATHOLOGY_AI_ACCELERATOR"
    return 0
  fi
  if [ "$arch" = "aarch64" ]; then
    echo "cuda"
    return 0
  fi
  if [ "$arch" = "x86_64" ]; then
    echo "rocm"
    return 0
  fi
  die "Unsupported architecture: $arch"
}

build_sandbox() {
  local name="$1"
  local source="$2"
  local target="$PDC_PATHOLOGY_AI_ROOT/images/$name"
  if [ -d "$target/.singularity.d" ] || [ -d "$target/.apptainer.d" ]; then
    echo "Sandbox exists: $target"
    return 0
  fi
  if [ -e "$target" ]; then
    die "$target exists but is not an apptainer sandbox. Move it aside and retry."
  fi
  echo "Building $target from $source"
  apptainer build --sandbox "$target" "$source"
}

ACCELERATOR="$(detect_accelerator)"
case "$ACCELERATOR" in
  cuda)
    VLLM_SANDBOX_NAME="${PDC_PATHOLOGY_AI_VLLM_SANDBOX_NAME:-vllm-openai-cuda-latest}"
    VLLM_SOURCE="${PDC_PATHOLOGY_AI_VLLM_SOURCE:-docker://vllm/vllm-openai:latest}"
    APPTAINER_GPU_FLAG="--nv"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      die "nvidia-smi is unavailable. For CUDA/GH200 preparation, run this on logingh or inside a gpugh allocation."
    fi
    ;;
  rocm)
    VLLM_SANDBOX_NAME="${PDC_PATHOLOGY_AI_VLLM_SANDBOX_NAME:-vllm-openai-rocm-latest}"
    VLLM_SOURCE="${PDC_PATHOLOGY_AI_VLLM_SOURCE:-docker://vllm/vllm-openai-rocm:latest}"
    APPTAINER_GPU_FLAG="--rocm"
    ;;
  *)
    die "Unsupported PDC_PATHOLOGY_AI_ACCELERATOR=$ACCELERATOR. Use auto, cuda, or rocm."
    ;;
esac

load_apptainer

mkdir -p \
  "$PDC_PATHOLOGY_AI_ROOT/images" \
  "$PDC_PATHOLOGY_AI_ROOT/hf-cache" \
  "$PDC_PATHOLOGY_AI_ROOT/qdrant" \
  "$PDC_PATHOLOGY_AI_ROOT/logs" \
  "$PDC_PATHOLOGY_AI_ROOT/tmp"

export APPTAINER_CACHEDIR="$PDC_PATHOLOGY_AI_ROOT/tmp/apptainer-cache"
export TMPDIR="$PDC_PATHOLOGY_AI_ROOT/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"

build_sandbox "$VLLM_SANDBOX_NAME" "$VLLM_SOURCE"
build_sandbox "qdrant-latest" "docker://qdrant/qdrant:latest"

cat > "$PDC_PATHOLOGY_AI_ROOT/runtime.env" <<EOF
export PDC_PATHOLOGY_AI_ROOT="$PDC_PATHOLOGY_AI_ROOT"
export PDC_PATHOLOGY_AI_ACCELERATOR="$ACCELERATOR"
export PDC_PATHOLOGY_AI_APPTAINER_GPU_FLAG="$APPTAINER_GPU_FLAG"
export PDC_PATHOLOGY_AI_VLLM_IMAGE="$PDC_PATHOLOGY_AI_ROOT/images/$VLLM_SANDBOX_NAME"
export PDC_PATHOLOGY_AI_QDRANT_IMAGE="$PDC_PATHOLOGY_AI_ROOT/images/qdrant-latest"
export HF_HOME="$PDC_PATHOLOGY_AI_ROOT/hf-cache"
export HUGGINGFACE_HUB_CACHE="$PDC_PATHOLOGY_AI_ROOT/hf-cache/hub"
export QDRANT_STORAGE="$PDC_PATHOLOGY_AI_ROOT/qdrant"
EOF

echo "Prepared PDC pathology-ai runtime at $PDC_PATHOLOGY_AI_ROOT"
echo "Runtime env: $PDC_PATHOLOGY_AI_ROOT/runtime.env"
