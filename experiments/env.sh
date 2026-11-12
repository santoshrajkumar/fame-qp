# Source this before running benchmarks that use cuOSQP or any JAX baseline:
#
#     source experiments/env.sh
#
# Both load CUDA shared libraries at runtime (libcublas, libcudnn, ...) and need
# them on the loader path.  The libraries ship inside the `nvidia-*` pip
# packages, which may live in more than one site-packages directory, so every
# one on sys.path is scanned rather than the first.
_LIBS=$(python - <<'PY'
import os, sys
out = []
for p in sys.path:
    nv = os.path.join(p, "nvidia")
    if not os.path.isdir(nv):
        continue
    for sub in sorted(os.listdir(nv)):
        d = os.path.join(nv, sub, "lib")
        if os.path.isdir(d):
            out.append(d)
try:
    import torch, os as _os
    d = _os.path.join(_os.path.dirname(torch.__file__), "lib")
    if _os.path.isdir(d):
        out.append(d)
except Exception:
    pass
print(":".join(dict.fromkeys(out)))
PY
)
[ -n "$_LIBS" ] && export LD_LIBRARY_PATH="$_LIBS:$LD_LIBRARY_PATH"

# cuOSQP uses cuSPARSE calls that CUDA 12 removed, so it is built against and
# run with the CUDA 11.8 libraries fetched by setup_externals.sh.
_C11="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/external/cuda11/toolkit/lib"
[ -d "$_C11" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_C11"
unset _C11

# a local CUDA toolkit, if CUDA_HOME points to one
for _c in "$CUDA_HOME"; do
  [ -n "$_c" ] && [ -d "$_c/lib64" ] && export LD_LIBRARY_PATH="$_c/lib64:$LD_LIBRARY_PATH"
done

# keep JAX from grabbing the whole device; torch needs it too
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset _LIBS _c
