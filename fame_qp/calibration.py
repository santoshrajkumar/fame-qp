"""Device calibration and per-problem kernel-mode selection.

Per-iteration time is modelled as

    t_iter ~= n_kernels * tau_node + t_stream(operator bytes)

Calibration (once per device and precision, cached on disk) measures the cost
of a graph node, the host read of the stopping test, streaming time against
operator size, and the register limit of the persistent kernel.

Mode selection (once per problem shape, cached) predicts the single- and
two-kernel forms, drops candidates that do not fit in memory or are predicted
more than ``PRUNE`` times slower than the best, and times the rest on
placeholder operators. The single kernel is kept unless the two-kernel form is
faster by more than ``MARGIN``.

The cache lives in ``$FAME_QP_CACHE``, ``$XDG_CACHE_HOME/fame_qp`` or
``~/.cache/fame_qp``.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import math
import os
import tempfile
import time
from typing import Optional

import torch
import triton

from . import kernels as K

__all__ = ["device_key", "cache_dir", "get_calibration", "select_kernel_mode",
           "predict_iter_us", "persistent_info", "clear_cache"]

PRUNE = 1.5
MARGIN = 0.10
STREAM_MB = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
PERSISTENT_TILES = (16, 32, 64, 128, 256)
PERSISTENT_WARPS = {16: (1, 2), 32: (1, 2), 64: (2, 4), 128: (4, 8), 256: (8,)}

_mem = {}


def cache_dir() -> str:
    d = os.environ.get("FAME_QP_CACHE")
    if not d:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        d = os.path.join(base, "fame_qp")
    return d


def _path(name):
    return os.path.join(cache_dir(), name)


def _load(name) -> dict:
    if name not in _mem:
        try:
            with open(_path(name)) as fh:
                _mem[name] = json.load(fh)
        except Exception:
            _mem[name] = {}
    return _mem[name]


def _store(name, key, value):
    """Merge ``key`` into the cache file under a lock, with an atomic rename."""
    _load(name)[key] = value
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        with open(_path(name) + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(_path(name)) as fh:
                    disk = json.load(fh)
            except Exception:
                disk = {}
            disk[key] = value
            fd, tmp = tempfile.mkstemp(dir=cache_dir(), prefix=name, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(disk, fh, indent=1)
            os.replace(tmp, _path(name))
    except Exception:
        pass


def clear_cache():
    """Remove all cached calibrations and mode choices."""
    _mem.clear()
    for name in ("calibration.json", "modes.json"):
        try:
            os.remove(_path(name))
        except FileNotFoundError:
            pass


def _driver_version() -> str:
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        lib.cuDriverGetVersion(ctypes.byref(v))
        return str(v.value)
    except Exception:
        return "unknown"


def device_key(device=None) -> str:
    """Identify the GPU and software stack the timings depend on."""
    from . import __version__
    p = torch.cuda.get_device_properties(device or torch.cuda.current_device())
    return (f"{p.name}|sm_{p.major}{p.minor}|driver {_driver_version()}|"
            f"cuda {torch.version.cuda}|triton {triton.__version__}|fame_qp {__version__}")


def _placeholder_box(n, dtype, dev):
    t = lambda k: torch.zeros(k, device=dev, dtype=dtype)  # noqa: E731
    W = torch.randn(n, n, device=dev, dtype=dtype) * 1e-3
    return dict(op=W, b=t(n), rin=t(n), rout=t(n), z=t(n), s=t(n), lam=t(n),
                l=t(n) - 1, h=t(n) + 1, rho=t(n) + 1, rho_inv=t(n) + 1, d=t(n) + 1)


def _time_tau_node(dtype, dev):
    x = torch.zeros(1, device=dev, dtype=dtype)
    return K.graph_time(lambda: x.add_(1.0), reps=100, outer=5) * 1e3


def _time_tau_read(dtype, dev):
    res = torch.zeros(5, device=dev, dtype=dtype)
    host = torch.zeros(5, dtype=dtype).pin_memory()
    ts = []
    for _ in range(60):
        res.add_(1.0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        host.copy_(res)
        host.tolist()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e6


def _time_stream(dtype, dev, tau_node):
    b = torch.finfo(dtype).bits // 8
    out = []
    for mb in STREAM_MB:
        n = max(32, int(math.sqrt(mb * 2 ** 20 / b)))
        try:
            p = _placeholder_box(n, dtype, dev)
        except torch.cuda.OutOfMemoryError:
            break
        best = float("inf")
        for tile in (((16, 128), 4), ((4, 256), 2)):
            f = lambda: K.single_kernel_iter(  # noqa: E731
                p["op"], p["b"], p["rin"], p["rout"], p["z"], p["s"], p["lam"],
                p["l"], p["h"], p["rho"], p["rho_inv"], p["d"], n, 0, 1e-6, 1.6, tile)
            try:
                best = min(best, K.graph_time(f) * 1e3)
            except Exception:
                continue
        out.append([n * n * b, max(best - tau_node, 0.01)])
        del p
    return out


def _time_persistent(dtype, dev):
    """Best warp count, registers and spills of the persistent kernel per tile."""
    table = {}
    for blk in PERSISTENT_TILES:
        n = blk
        p = _placeholder_box(n, dtype, dev)
        one = torch.ones(n, device=dev, dtype=dtype)
        info = torch.zeros(4, device=dev, dtype=torch.float32)
        rec = None
        for w in PERSISTENT_WARPS[blk]:
            def run(iters):
                return K.persistent_solve(p["op"], p["b"], p["l"], p["h"], p["rho"],
                                          p["rho_inv"], one, one, p["d"], p["z"],
                                          p["s"], p["lam"], info, n, 1.0, 1e-6, 1.6,
                                          0.0, 0.0, iters, 10 ** 9, num_warps=w,
                                          block=blk)
            try:
                h = run(10)
                torch.cuda.synchronize()
                regs = int(getattr(h, "n_regs", -1))
                spills = int(getattr(h, "n_spills", 0))
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                run(200)
                e0.record()
                run(1000)
                e1.record()
                torch.cuda.synchronize()
                us = e0.elapsed_time(e1)  # ms per 1000 iterations = us per iteration
            except Exception:
                continue
            cand = dict(warps=w, us_per_iter=us, regs=regs, spills=spills)
            if rec is None or (cand["spills"], cand["us_per_iter"]) < (rec["spills"], rec["us_per_iter"]):
                rec = cand
        if rec is not None:
            table[str(blk)] = rec
        del p
    return table


def get_calibration(dtype=torch.float32, device=None, force=False) -> dict:
    """Return the (cached) calibration for this device and precision."""
    dev = torch.device(device or "cuda")
    key = f"{device_key(dev)}|{str(dtype).replace('torch.', '')}"
    tab = _load("calibration.json")
    if not force and key in tab:
        return tab[key]
    tau = _time_tau_node(dtype, dev)
    cal = dict(tau_node_us=tau,
               tau_read_us=_time_tau_read(dtype, dev),
               stream=_time_stream(dtype, dev, tau),
               persistent=_time_persistent(dtype, dev),
               measured=time.strftime("%Y-%m-%dT%H:%M:%S"))
    ok = [int(b) for b, r in cal["persistent"].items() if r["spills"] == 0]
    cal["persistent_max_tile"] = max(ok) if ok else 0
    _store("calibration.json", key, cal)
    return cal


def stream_us(cal, nbytes) -> float:
    """Interpolate streaming time log-log; extrapolate at constant bandwidth."""
    pts = cal["stream"]
    if nbytes <= pts[0][0]:
        return pts[0][1] * nbytes / pts[0][0]
    if nbytes >= pts[-1][0]:
        return pts[-1][1] * nbytes / pts[-1][0]
    for (b0, t0), (b1, t1) in zip(pts, pts[1:]):
        if b0 <= nbytes <= b1:
            f = (math.log(nbytes) - math.log(b0)) / (math.log(b1) - math.log(b0))
            return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))
    return pts[-1][1]


def predict_iter_us(cal, mode, n, m_c, dtype) -> float:
    b = torch.finfo(dtype).bits // 8
    R = n + m_c
    if mode == "single_kernel":
        return cal["tau_node_us"] + stream_us(cal, R * R * b)
    return 2 * cal["tau_node_us"] + stream_us(cal, (n * R + m_c * n) * b)


def persistent_info(cal, n) -> Optional[dict]:
    """Persistent-kernel timings for size n, or None if the tile would spill."""
    blk = K.next_pow2(n)
    if blk > cal.get("persistent_max_tile", 0):
        return None
    rec = cal["persistent"].get(str(blk))
    if rec is None or rec["spills"] != 0:
        return None
    return dict(tile=blk, **rec)


def _placeholders(mode, n, m_c, has_diag, dtype, dev):
    t = lambda k: torch.zeros(k, device=dev, dtype=dtype)  # noqa: E731
    R, m = n + m_c, (n if has_diag else 0) + m_c
    rows = R if mode == "single_kernel" else n
    p = dict(op=torch.randn(rows, R, device=dev, dtype=dtype) * 1e-3,
             b=t(rows), rin=t(R), rout=t(R), zt=t(n), z=t(n), s=t(m), lam=t(m),
             l=t(m) - 1, h=t(m) + 1, rho=t(m) + 1, rho_inv=t(m) + 1,
             d=(t(n) + 1) if has_diag else None,
             C=torch.randn(m_c, n, device=dev, dtype=dtype) * 1e-3 if m_c else None)
    return p


def _time_mode(mode, n, m_c, has_diag, dtype, dev):
    """Return (best tile, us per iteration) on placeholder operators."""
    p = _placeholders(mode, n, m_c, has_diag, dtype, dev)
    best_t, best_tile = float("inf"), K.DEFAULT_TILE
    for bm, bn, w in K.TILE_CANDIDATES:
        tile = ((bm, bn), w)
        if mode == "single_kernel":
            f = lambda: K.single_kernel_iter(  # noqa: E731
                p["op"], p["b"], p["rin"], p["rout"], p["z"], p["s"], p["lam"],
                p["l"], p["h"], p["rho"], p["rho_inv"], p["d"], n, m_c, 1e-6, 1.6, tile)
        else:
            f = lambda: K.two_kernel_iter(  # noqa: E731
                p["op"], p["C"], p["b"], p["rin"], p["rout"], p["zt"], p["z"],
                p["s"], p["lam"], p["l"], p["h"], p["rho"], p["rho_inv"], p["d"],
                n, m_c, 1e-6, 1.6, tile)
        try:
            t = K.graph_time(f) * 1e3
        except Exception:
            continue
        if t < best_t:
            best_t, best_tile = t, tile
    del p
    return best_tile, best_t


def _time_check(n, m_c, has_diag, dtype, dev):
    """Time one stopping test in us."""
    m = (n if has_diag else 0) + m_c
    t = lambda k: torch.zeros(k, device=dev, dtype=dtype)  # noqa: E731
    P = torch.randn(n, n, device=dev, dtype=dtype) * 1e-3
    C = torch.randn(m_c, n, device=dev, dtype=dtype) * 1e-3 if m_c else None
    z, s, lam, q = t(n), t(m), t(m), t(n)
    Pz, Gz, Gtlam, part, res = t(n), t(m), t(n), t(K.n_partials(n, m)), t(5)
    one_n, one_m = t(n) + 1, t(m) + 1
    nd = n if has_diag else 0

    def f():
        torch.mv(P, z, out=Pz)
        if nd:
            torch.mul(one_n, z, out=Gz[:nd])
            torch.mul(one_n, lam[:nd], out=Gtlam)
        if m_c:
            torch.mv(C, z, out=Gz[nd:])
            if nd:
                Gtlam.addmv_(C.T, lam[nd:])
            else:
                torch.mv(C.T, lam, out=Gtlam)
        K.residual_norms(Gz, s, Pz, q, Gtlam, one_m, one_n, part, res, 1.0, 1e-4, 1e-4)

    return K.graph_time(f, reps=4) * 1e3


def select_kernel_mode(structure, n, m_c, dtype, n_rho=5, device=None,
                       force=False, allowed=("single_kernel", "two_kernel")) -> dict:
    """Choose single_kernel or two_kernel for this problem shape (cached)."""
    dev = torch.device(device or "cuda")
    cal = get_calibration(dtype, dev)
    has_diag = structure in ("box", "split")
    key = (f"{device_key(dev)}|{str(dtype).replace('torch.', '')}|{structure}|"
           f"n={n}|mc={m_c}|{','.join(allowed)}")
    tab = _load("modes.json")
    if not force and key in tab:
        return tab[key]

    cands = ["single_kernel"] if structure == "box" else list(allowed)
    b = torch.finfo(dtype).bits // 8
    free, _ = torch.cuda.mem_get_info(dev)
    pred, reason = {}, []
    for c in cands:
        rows = n + m_c if c == "single_kernel" else n
        if n_rho * rows * (n + m_c) * b > 0.6 * free:
            reason.append(f"{c}: bank does not fit in memory")
            continue
        pred[c] = predict_iter_us(cal, c, n, m_c, dtype)
    if not pred:
        raise MemoryError("no kernel mode fits in GPU memory for this problem")
    best_pred = min(pred.values())
    survivors = [c for c, v in pred.items() if v <= PRUNE * best_pred]
    for c in pred:
        if c not in survivors:
            reason.append(f"{c}: predicted {pred[c]:.1f} us > {PRUNE} x best")

    timed = {}
    for c in survivors:
        tile, us = _time_mode(c, n, m_c, has_diag, dtype, dev)
        timed[c] = (tile, us)
    if "single_kernel" in timed and "two_kernel" in timed:
        t1, t2 = timed["single_kernel"][1], timed["two_kernel"][1]
        mode = "two_kernel" if t2 < (1.0 - MARGIN) * t1 else "single_kernel"
    else:
        mode = next(iter(timed))
    reason.append(f"timed: " + ", ".join(f"{c} {v[1]:.2f} us" for c, v in timed.items()))
    rec = dict(mode=mode, tile=timed[mode][0], us_per_iter=timed[mode][1],
               us_check=_time_check(n, m_c, has_diag, dtype, dev),
               predicted=pred, timed={c: v[1] for c, v in timed.items()},
               reason="; ".join(reason))
    _store("modes.json", key, rec)
    return rec
