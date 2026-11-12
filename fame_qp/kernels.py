"""Triton kernels.

single_kernel   one ADMM iteration in one kernel using the stacked operator;
                two r buffers alternate between input and output
two_kernel      pass 1 computes z~ and the diagonal rows, pass 2 computes C z~
                and the dense rows
persistent      a complete box-constrained solve in one thread block
residual_norms  the stopping test, evaluated on the device
unscale         undo scaling and restore the user's row order
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

DEFAULT_TILE = ((16, 128), 4)  # ((BLOCK_M, BLOCK_N), num_warps)
TILE_CANDIDATES = [
    (8, 128, 8), (8, 128, 4), (16, 128, 4), (16, 256, 4),
    (4, 256, 2), (4, 512, 2), (2, 512, 2), (1, 512, 2), (32, 256, 4),
]


def acc_type(t: torch.Tensor):
    return tl.float64 if t.dtype == torch.float64 else tl.float32


def next_pow2(n: int, lo: int = 16) -> int:
    p = lo
    while p < n:
        p *= 2
    return p


@triton.jit
def _single_kernel_iter(
    W, B, RIN, ROUT, Z, S, LAM, L, H, RHO, RHOINV, D,
    n, mc, nd, sigma, alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACC: tl.constexpr,
    HAS_DIAG: tl.constexpr,
):
    pid = tl.program_id(0)
    tot = n + mc
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < tot

    acc = tl.zeros((BLOCK_M,), dtype=ACC)
    rbase = rows.to(tl.int64) * tot
    for j0 in range(0, tot, BLOCK_N):
        cols = j0 + tl.arange(0, BLOCK_N)
        cmask = cols < tot
        tile = tl.load(W + rbase[:, None] + cols[None, :],
                       mask=rmask[:, None] & cmask[None, :], other=0.0)
        rv = tl.load(RIN + cols, mask=cmask, other=0.0)
        acc += tl.sum(tile.to(ACC) * rv.to(ACC)[None, :], axis=1)
    out = acc + tl.load(B + rows, mask=rmask, other=0.0).to(ACC)

    is_z = rows < n
    mz = rmask & is_z
    iz = tl.where(is_z, rows, 0)
    z = tl.load(Z + iz, mask=mz, other=0.0).to(ACC)
    z_new = alpha * out + (1.0 - alpha) * z
    tl.store(Z + iz, z_new, mask=mz)

    # Constraint index: diagonal row i -> i, dense row n + j -> nd + j.
    if HAS_DIAG:
        mk = rmask
    else:
        mk = rmask & (is_z == 0)
    k = tl.where(mk, tl.where(is_z, rows, nd + rows - n), 0)
    s = tl.load(S + k, mask=mk, other=0.0).to(ACC)
    lam = tl.load(LAM + k, mask=mk, other=0.0).to(ACC)
    lo = tl.load(L + k, mask=mk, other=0.0).to(ACC)
    hi = tl.load(H + k, mask=mk, other=0.0).to(ACC)
    rho = tl.load(RHO + k, mask=mk, other=1.0).to(ACC)
    rinv = tl.load(RHOINV + k, mask=mk, other=1.0).to(ACC)
    if HAS_DIAG:
        dd = tl.load(D + iz, mask=mz, other=1.0).to(ACC)
        st = tl.where(is_z, dd * out, out)
    else:
        st = out
    zeta = alpha * st + (1.0 - alpha) * s + rinv * lam
    s_new = tl.minimum(tl.maximum(zeta, lo), hi)
    lam_new = rho * (zeta - s_new)
    tl.store(S + k, s_new, mask=mk)
    tl.store(LAM + k, lam_new, mask=mk)

    rc = rho * s_new - lam_new
    if HAS_DIAG:
        rz = sigma * z_new + dd * rc
    else:
        rz = sigma * z_new
    tl.store(ROUT + rows, tl.where(is_z, rz, rc), mask=rmask)


def single_kernel_iter(op, b, rin, rout, z, s, lam, l, h, rho, rho_inv, d,
                       n, mc, sigma, alpha, tile=DEFAULT_TILE):
    (bm, bn), warps = tile
    has_diag = d is not None
    _single_kernel_iter[(triton.cdiv(n + mc, bm),)](
        op, b, rin, rout, z, s, lam, l, h, rho, rho_inv, d if has_diag else rho,
        n, mc, n if has_diag else 0, sigma, alpha,
        BLOCK_M=bm, BLOCK_N=bn, ACC=acc_type(z), HAS_DIAG=has_diag,
        num_warps=warps)


@triton.jit
def _two_kernel_pass1(
    W, B, RIN, ROUT, ZT, Z, S, LAM, L, H, RHO, RHOINV, D,
    n, mc, sigma, alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACC: tl.constexpr,
    HAS_DIAG: tl.constexpr,
):
    pid = tl.program_id(0)
    tot = n + mc
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < n
    acc = tl.zeros((BLOCK_M,), dtype=ACC)
    rbase = rows.to(tl.int64) * tot
    for j0 in range(0, tot, BLOCK_N):
        cols = j0 + tl.arange(0, BLOCK_N)
        cmask = cols < tot
        tile = tl.load(W + rbase[:, None] + cols[None, :],
                       mask=rmask[:, None] & cmask[None, :], other=0.0)
        rv = tl.load(RIN + cols, mask=cmask, other=0.0)
        acc += tl.sum(tile.to(ACC) * rv.to(ACC)[None, :], axis=1)
    zt = acc + tl.load(B + rows, mask=rmask, other=0.0).to(ACC)
    tl.store(ZT + rows, zt, mask=rmask)

    z = tl.load(Z + rows, mask=rmask, other=0.0).to(ACC)
    z_new = alpha * zt + (1.0 - alpha) * z
    tl.store(Z + rows, z_new, mask=rmask)
    if HAS_DIAG:
        s = tl.load(S + rows, mask=rmask, other=0.0).to(ACC)
        lam = tl.load(LAM + rows, mask=rmask, other=0.0).to(ACC)
        lo = tl.load(L + rows, mask=rmask, other=0.0).to(ACC)
        hi = tl.load(H + rows, mask=rmask, other=0.0).to(ACC)
        rho = tl.load(RHO + rows, mask=rmask, other=1.0).to(ACC)
        rinv = tl.load(RHOINV + rows, mask=rmask, other=1.0).to(ACC)
        dd = tl.load(D + rows, mask=rmask, other=1.0).to(ACC)
        zeta = alpha * (dd * zt) + (1.0 - alpha) * s + rinv * lam
        s_new = tl.minimum(tl.maximum(zeta, lo), hi)
        lam_new = rho * (zeta - s_new)
        tl.store(S + rows, s_new, mask=rmask)
        tl.store(LAM + rows, lam_new, mask=rmask)
        rz = sigma * z_new + dd * (rho * s_new - lam_new)
    else:
        rz = sigma * z_new
    tl.store(ROUT + rows, rz, mask=rmask)


@triton.jit
def _two_kernel_pass2(
    C, ZT, ROUT, S, LAM, L, H, RHO, RHOINV,
    n, mc, nd, alpha,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ACC: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < mc
    acc = tl.zeros((BLOCK_M,), dtype=ACC)
    rbase = rows.to(tl.int64) * n
    for j0 in range(0, n, BLOCK_N):
        cols = j0 + tl.arange(0, BLOCK_N)
        cmask = cols < n
        tile = tl.load(C + rbase[:, None] + cols[None, :],
                       mask=rmask[:, None] & cmask[None, :], other=0.0)
        zv = tl.load(ZT + cols, mask=cmask, other=0.0)
        acc += tl.sum(tile.to(ACC) * zv.to(ACC)[None, :], axis=1)
    k = nd + rows
    s = tl.load(S + k, mask=rmask, other=0.0).to(ACC)
    lam = tl.load(LAM + k, mask=rmask, other=0.0).to(ACC)
    lo = tl.load(L + k, mask=rmask, other=0.0).to(ACC)
    hi = tl.load(H + k, mask=rmask, other=0.0).to(ACC)
    rho = tl.load(RHO + k, mask=rmask, other=1.0).to(ACC)
    rinv = tl.load(RHOINV + k, mask=rmask, other=1.0).to(ACC)
    zeta = alpha * acc + (1.0 - alpha) * s + rinv * lam
    s_new = tl.minimum(tl.maximum(zeta, lo), hi)
    lam_new = rho * (zeta - s_new)
    tl.store(S + k, s_new, mask=rmask)
    tl.store(LAM + k, lam_new, mask=rmask)
    tl.store(ROUT + n + rows, rho * s_new - lam_new, mask=rmask)


def two_kernel_iter(op, C, b, rin, rout, zt, z, s, lam, l, h, rho, rho_inv, d,
                    n, mc, sigma, alpha, tile=DEFAULT_TILE, tile2=None):
    (bm, bn), warps = tile
    has_diag = d is not None
    _two_kernel_pass1[(triton.cdiv(n, bm),)](
        op, b, rin, rout, zt, z, s, lam, l, h, rho, rho_inv, d if has_diag else rho,
        n, mc, sigma, alpha,
        BLOCK_M=bm, BLOCK_N=bn, ACC=acc_type(z), HAS_DIAG=has_diag,
        num_warps=warps)
    (bm2, bn2), w2 = tile2 or tile
    _two_kernel_pass2[(triton.cdiv(mc, bm2),)](
        C, zt, rout, s, lam, l, h, rho, rho_inv,
        n, mc, n if has_diag else 0, alpha,
        BLOCK_M=bm2, BLOCK_N=bn2, ACC=acc_type(z), num_warps=w2)


@triton.jit
def _persistent_solve(
    M, Q, L, H, RHO, RHOINV, EINV, DINV, DD, Z, S, LAM, INFO,
    n, cinv, sigma, alpha, eps_abs, eps_rel, max_iter, check_every,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    Mt = tl.load(M + offs[:, None] * n + offs[None, :],
                 mask=mask[:, None] & mask[None, :], other=0.0)
    q = tl.load(Q + offs, mask=mask, other=0.0)
    lo = tl.load(L + offs, mask=mask, other=0.0)
    hi = tl.load(H + offs, mask=mask, other=0.0)
    rho = tl.load(RHO + offs, mask=mask, other=1.0)
    rinv = tl.load(RHOINV + offs, mask=mask, other=1.0)
    ei = tl.load(EINV + offs, mask=mask, other=1.0)
    di = tl.load(DINV + offs, mask=mask, other=1.0)
    dd = tl.load(DD + offs, mask=mask, other=1.0)
    z = tl.load(Z + offs, mask=mask, other=0.0)
    s = tl.load(S + offs, mask=mask, other=0.0)
    lam = tl.load(LAM + offs, mask=mask, other=0.0)

    it = 0
    done = 0
    rp_out = 0.0
    rd_out = 0.0
    while (it < max_iter) and (done == 0):
        r = sigma * z + dd * (rho * s - lam) - q
        zt = tl.sum(Mt * r[None, :], axis=1)
        st = dd * zt
        zeta = alpha * st + (1.0 - alpha) * s + rinv * lam
        s_new = tl.minimum(tl.maximum(zeta, lo), hi)
        lam_new = rho * (zeta - s_new)
        z_new = alpha * zt + (1.0 - alpha) * z
        it += 1
        if (it % check_every) == 0:
            # P z~ + q + G'lam = sigma (z - z~) + d o rho o (s - d o z~),
            # so P is never loaded.
            rd_s = sigma * (z - zt) + dd * rho * (s - st)
            Pzt = rd_s - q - dd * lam
            rp = tl.max(tl.where(mask, tl.abs(ei * (st - s_new)), 0.0))
            rp_ref = tl.maximum(tl.max(tl.where(mask, tl.abs(ei * st), 0.0)),
                                tl.max(tl.where(mask, tl.abs(ei * s_new), 0.0)))
            rd = cinv * tl.max(tl.where(mask, tl.abs(di * rd_s), 0.0))
            rd_ref = cinv * tl.maximum(
                tl.max(tl.where(mask, tl.abs(di * Pzt), 0.0)),
                tl.maximum(tl.max(tl.where(mask, tl.abs(di * dd * lam), 0.0)),
                           tl.max(tl.where(mask, tl.abs(di * q), 0.0))))
            rp_out = rp.to(tl.float32)
            rd_out = rd.to(tl.float32)
            if (rp <= eps_abs + eps_rel * rp_ref) and (rd <= eps_abs + eps_rel * rd_ref):
                done = 1
        z = z_new
        s = s_new
        lam = lam_new

    tl.store(Z + offs, z, mask=mask)
    tl.store(S + offs, s, mask=mask)
    tl.store(LAM + offs, lam, mask=mask)
    tl.store(INFO + 0, it.to(tl.float32))
    tl.store(INFO + 1, rp_out)
    tl.store(INFO + 2, rd_out)
    tl.store(INFO + 3, done.to(tl.float32))


def persistent_solve(M, q, l, h, rho, rho_inv, Einv, Dinv, d, z, s, lam, info,
                     n, cinv, sigma, alpha, eps_abs, eps_rel, max_iter,
                     check_every, num_warps=4, block=None):
    """Run a box-constrained solve in one launch; returns the compiled kernel."""
    return _persistent_solve[(1,)](
        M, q, l, h, rho, rho_inv, Einv, Dinv, d, z, s, lam, info,
        n, cinv, sigma, alpha, eps_abs, eps_rel, max_iter, check_every,
        BLOCK=block or next_pow2(n), num_warps=num_warps, num_stages=1)


@triton.jit
def _partial_norms(GZ, S, PZ, Q, GTLAM, EINV, DINV, PART, n, m, nblk,
                   BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mm = offs < m
    gz = tl.load(GZ + offs, mask=mm, other=0.0)
    ss = tl.load(S + offs, mask=mm, other=0.0)
    ei = tl.load(EINV + offs, mask=mm, other=0.0)
    p0 = tl.max(tl.where(mm, tl.abs(ei * (gz - ss)), 0.0))
    p1 = tl.max(tl.where(mm, tl.abs(ei * gz), 0.0))
    p2 = tl.max(tl.where(mm, tl.abs(ei * ss), 0.0))
    mn = offs < n
    pz = tl.load(PZ + offs, mask=mn, other=0.0)
    qq = tl.load(Q + offs, mask=mn, other=0.0)
    gt = tl.load(GTLAM + offs, mask=mn, other=0.0)
    di = tl.load(DINV + offs, mask=mn, other=0.0)
    p3 = tl.max(tl.where(mn, tl.abs(di * (pz + qq + gt)), 0.0))
    p4 = tl.max(tl.where(mn, tl.abs(di * pz), 0.0))
    p5 = tl.max(tl.where(mn, tl.abs(di * gt), 0.0))
    p6 = tl.max(tl.where(mn, tl.abs(di * qq), 0.0))
    tl.store(PART + 0 * nblk + pid, p0)
    tl.store(PART + 1 * nblk + pid, p1)
    tl.store(PART + 2 * nblk + pid, p2)
    tl.store(PART + 3 * nblk + pid, p3)
    tl.store(PART + 4 * nblk + pid, p4)
    tl.store(PART + 5 * nblk + pid, p5)
    tl.store(PART + 6 * nblk + pid, p6)


@triton.jit
def _reduce_norms(PART, RES, nblk, cinv, eps_abs, eps_rel, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < nblk
    rp = tl.max(tl.load(PART + 0 * nblk + offs, mask=mask, other=0.0))
    a1 = tl.max(tl.load(PART + 1 * nblk + offs, mask=mask, other=0.0))
    a2 = tl.max(tl.load(PART + 2 * nblk + offs, mask=mask, other=0.0))
    rd = tl.max(tl.load(PART + 3 * nblk + offs, mask=mask, other=0.0))
    a4 = tl.max(tl.load(PART + 4 * nblk + offs, mask=mask, other=0.0))
    a5 = tl.max(tl.load(PART + 5 * nblk + offs, mask=mask, other=0.0))
    a6 = tl.max(tl.load(PART + 6 * nblk + offs, mask=mask, other=0.0))
    rp_ref = tl.maximum(a1, a2)
    rd = rd * cinv
    rd_ref = cinv * tl.maximum(a4, tl.maximum(a5, a6))
    ok = (rp <= eps_abs + eps_rel * rp_ref) & (rd <= eps_abs + eps_rel * rd_ref)
    tl.store(RES + 0, rp)
    tl.store(RES + 1, rp_ref)
    tl.store(RES + 2, rd)
    tl.store(RES + 3, rd_ref)
    tl.store(RES + 4, tl.where(ok, 1.0, 0.0))


NORM_BLOCK = 512


def n_partials(n, m):
    return 7 * triton.cdiv(max(n, m, 1), NORM_BLOCK)


def residual_norms(Gz, s, Pz, q, Gtlam, Einv, Dinv, part, res, cinv, eps_abs, eps_rel):
    """Write [r_prim, eta_p, r_dual, eta_d, stop_flag] to ``res``."""
    n, m = q.numel(), s.numel()
    nblk = triton.cdiv(max(n, m, 1), NORM_BLOCK)
    _partial_norms[(nblk,)](Gz, s, Pz, q, Gtlam, Einv, Dinv, part, n, m, nblk,
                            BLOCK=NORM_BLOCK, num_warps=4)
    _reduce_norms[(1,)](part, res, nblk, cinv, eps_abs, eps_rel,
                        BLOCK=next_pow2(nblk), num_warps=4)


@triton.jit
def _unscale(D, E, PERM, Z, S, LAM, ZO, SO, LAMO, n, m, cinv, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mn = offs < n
    tl.store(ZO + offs, tl.load(D + offs, mask=mn, other=1.0)
             * tl.load(Z + offs, mask=mn, other=0.0), mask=mn)
    mm = offs < m
    p = tl.load(PERM + offs, mask=mm, other=0)
    e = tl.load(E + offs, mask=mm, other=1.0)
    tl.store(SO + p, tl.load(S + offs, mask=mm, other=0.0) / e, mask=mm)
    tl.store(LAMO + p, cinv * e * tl.load(LAM + offs, mask=mm, other=0.0), mask=mm)


def unscale(D, E, perm, z, s, lam, zo, so, lamo, cinv, block=256):
    n, m = z.numel(), s.numel()
    _unscale[(triton.cdiv(max(n, m, 1), block),)](
        D, E, perm, z, s, lam, zo, so, lamo, n, m, cinv, BLOCK=block)


def graph_time(fn, reps=10, outer=10, repeats=3):
    """Time one call of ``fn`` in ms, replayed inside a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(outer):
            g.replay()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) / outer / reps)
    del g
    out.sort()
    return out[len(out) // 2]
