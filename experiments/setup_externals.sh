#!/usr/bin/env bash
# Fetch and build the third-party baselines used in the experiments.
#
#   bash experiments/setup_externals.sh               # fetch, build and pip-install everything
#   bash experiments/setup_externals.sh --fetch-only  # download sources only, no pip installs
#   bash experiments/setup_externals.sh --no-cuosqp   # skip the CUDA 11.8 toolkit and cuOSQP
#
# Everything is pinned to the commits and versions used in the paper (see
# EXTERNALS.md) and written to experiments/external/, which is not tracked by
# git. Rerunning the script skips anything already in place.

set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$EXP")"
EXT="${FAME_EXTERNAL_DIR:-$EXP/external}"
PYTHON="${PYTHON:-python}"

FETCH_ONLY=0
WITH_CUOSQP=1
for arg in "$@"; do
  case "$arg" in
    --fetch-only) FETCH_ONLY=1 ;;
    --no-cuosqp)  WITH_CUOSQP=0 ;;
    -h|--help)    sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

CUOSQP_URL=https://github.com/osqp/cuosqp.git
CUOSQP_COMMIT=2a045f8c79e8c70823ab65106e1ce67ad42e7d9f
CUPIQP_URL=https://github.com/PREDICT-EPFL/cupiqp.git
CUPIQP_COMMIT=d538dda8c93a76a16fcddc2f98c24c98133a7937
CUPIQP_CUDA="${CUPIQP_CUDA:-cuda13}"          # cupiqp extra matching the driver: cuda12 or cuda13
PIMPC_URL=https://github.com/SOLARIS-JHU/PiMPC.jl.git
PIMPC_COMMIT=6aa0950d42b2ee78616bcbfe2f3dc6f12d51ae67
KQ_URL=https://github.com/santoshrajkumar/kq-lmpc-quadrotor-updated.git
KQ_COMMIT=60fb4c0b68bcb3d6bfb414071ed919086ba1a23a
MENAGERIE_URL=https://github.com/google-deepmind/mujoco_menagerie.git
MENAGERIE_COMMIT=cda5236f98383f6fbaa4820c6b18dba892cd087f

CUDA_REDIST=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_COMPONENTS="cuda_nvcc cuda_cudart cuda_cccl libcublas libcusparse"
SAMPLES_URL=https://raw.githubusercontent.com/NVIDIA/cuda-samples/v12.5/Common
HELPER_CUDA_SHA256=33738a061a07baa3e0f93f33b7f5fd37ca1203703777fbca7b100608cc688830
HELPER_STRING_SHA256=616ac8b2588a6e73aed207f85faf0da5e617404390df1790f6637b1042dbcc94

log() { printf '\n==> %s\n' "$*"; }

# clone_at <url> <commit> <dir>: clone the repository and check out the commit.
clone_at() {
  local url=$1 commit=$2 dir=$3
  if [ -d "$dir/.git" ]; then
    if [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ]; then
      echo "$(basename "$dir") already at ${commit:0:7}"
      return
    fi
    echo "error: $dir exists at a different commit; remove it and rerun" >&2
    exit 1
  fi
  git clone --quiet --filter=blob:none --no-checkout "$url" "$dir"
  git -C "$dir" -c advice.detachedHead=false checkout --quiet "$commit"
  echo "$(basename "$dir") at ${commit:0:7}"
}

# sha256_check <file> <expected>
sha256_check() {
  local got
  got=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then
    echo "error: checksum mismatch for $1" >&2
    exit 1
  fi
}

# compute capability of GPU 0, e.g. 86
gpu_arch() {
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. '
}

mkdir -p "$EXT"

log "Source repositories"
clone_at "$PIMPC_URL" "$PIMPC_COMMIT" "$EXT/PiMPC.jl"
clone_at "$KQ_URL" "$KQ_COMMIT" "$EXT/kq-lmpc-quadrotor-updated"
clone_at "$CUPIQP_URL" "$CUPIQP_COMMIT" "$EXT/cupiqp"

log "Unitree G1 model (MuJoCo Menagerie)"
if [ ! -d "$EXT/mujoco_menagerie/.git" ]; then
  git clone --quiet --filter=blob:none --no-checkout "$MENAGERIE_URL" "$EXT/mujoco_menagerie"
  git -C "$EXT/mujoco_menagerie" sparse-checkout set unitree_g1
fi
git -C "$EXT/mujoco_menagerie" -c advice.detachedHead=false checkout --quiet "$MENAGERIE_COMMIT"
ln -sfn mujoco_menagerie/unitree_g1 "$EXT/unitree_g1"
echo "unitree_g1 at ${MENAGERIE_COMMIT:0:7}"

if [ "$WITH_CUOSQP" = 1 ]; then
  log "cuOSQP"
  clone_at "$CUOSQP_URL" "$CUOSQP_COMMIT" "$EXT/cuosqp"
  git -C "$EXT/cuosqp" submodule update --quiet --init --recursive

  # cuOSQP calls cuSPARSE functions removed in CUDA 12, so it is built with
  # the CUDA 11.8 redistributables from NVIDIA, checked against NVIDIA's index.
  log "CUDA 11.8 toolkit (for cuOSQP)"
  TK="$EXT/cuda11/toolkit"
  DL="$EXT/cuda11/downloads"
  if [ ! -x "$TK/bin/nvcc" ]; then
    mkdir -p "$TK" "$DL"
    curl -fsSL -o "$DL/redistrib_11.8.0.json" "$CUDA_REDIST/redistrib_11.8.0.json"
    for comp in $CUDA_COMPONENTS; do
      read -r rel sha < <("$PYTHON" -c "import json, sys
e = json.load(open(sys.argv[1]))[sys.argv[2]]['linux-x86_64']
print(e['relative_path'], e['sha256'])" "$DL/redistrib_11.8.0.json" "$comp")
      file="$DL/$(basename "$rel")"
      [ -f "$file" ] || curl -fL --progress-bar -o "$file" "$CUDA_REDIST/$rel"
      sha256_check "$file" "$sha"
      tar -xJf "$file" -C "$TK" --strip-components=1
      echo "$comp ok"
    done
    mkdir -p "$TK/include/samples"
    curl -fsSL -o "$TK/include/samples/helper_cuda.h" "$SAMPLES_URL/helper_cuda.h"
    curl -fsSL -o "$TK/include/samples/helper_string.h" "$SAMPLES_URL/helper_string.h"
    sha256_check "$TK/include/samples/helper_cuda.h" "$HELPER_CUDA_SHA256"
    sha256_check "$TK/include/samples/helper_string.h" "$HELPER_STRING_SHA256"
    rm -rf "$DL"
  fi
  ln -sfn lib "$TK/lib64"            # nvcc.profile looks for the libraries in lib64
  echo "CUDA 11.8 toolkit in $TK"
fi

log "CUDA MPC (experiments/baselines/cudampc)"
ARCH=$(gpu_arch || true)
# nvcc: $NVCC, then $CUDA_HOME, then PATH, then the CUDA 11.8 toolkit above
NVCC="${NVCC:-}"
for c in "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" "$(command -v nvcc || true)" "$EXT/cuda11/toolkit/bin/nvcc"; do
  if [ -z "$NVCC" ] && [ -n "$c" ] && [ -x "$c" ]; then NVCC="$c"; fi
done
if [ -n "$ARCH" ] && [ -n "$NVCC" ] && "$PYTHON" -c "import numpy" 2>/dev/null; then
  NVCC="$NVCC" "$PYTHON" -c "import sys; sys.path.insert(0, sys.argv[1])
from cudampc import build_library
print(build_library(force=True, arch='sm_' + sys.argv[2]))" "$EXP/baselines" "$ARCH" \
    || echo "warning: CUDA MPC build failed; set NVCC or CUDA_HOME to a toolkit that supports sm_$ARCH"
else
  echo "skipped (no GPU, nvcc or numpy found); it is built on first use"
fi

if [ "$FETCH_ONLY" = 1 ]; then
  log "Done (sources only). Run without --fetch-only to install the Python packages."
  exit 0
fi

log "Python packages"
"$PYTHON" -m pip install -e "$ROOT"
"$PYTHON" -m pip install -r "$EXP/requirements-bench.txt"
"$PYTHON" -m pip install "$EXT/cupiqp[$CUPIQP_CUDA]"

if [ "$WITH_CUOSQP" = 1 ]; then
  log "Building cuOSQP"
  ARCH=${CUDAARCHS:-$(gpu_arch)}
  if [ "${ARCH:-0}" -gt 90 ]; then
    echo "warning: CUDA 11.8 does not support sm_$ARCH; cuOSQP may not build on this GPU"
  fi
  (
    cd "$EXT/cuosqp"
    export CUDA_PATH="$EXT/cuda11/toolkit"
    export CUDACXX="$CUDA_PATH/bin/nvcc" CUDAARCHS="$ARCH" PATH="$CUDA_PATH/bin:$PATH"
    export CPATH="$CUDA_PATH/include/samples${CPATH:+:$CPATH}"
    "$PYTHON" -m pip install --no-build-isolation .
  )
fi

log "Done. Run 'source experiments/env.sh' before benchmarking."
