"""ADMM solver with automatic execution-mode and precision selection.

Each iteration, in the form used by OSQP, is

    z~    = M (sigma z - q + G'(rho o s - lam)),   M = (P + sigma I + G' diag(rho) G)^-1
    z     = alpha z~ + (1 - alpha) z
    zeta  = alpha G z~ + (1 - alpha) s + rho^-1 o lam
    s     = proj_[l, h](zeta)
    lam   = rho o (zeta - s)

and the solve stops when ||G z - s|| and ||P z + q + G'lam|| (unscaled,
infinity norm) fall below eps_abs + eps_rel times their scales.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import calibration as cal_mod
from . import kernels as K
from .operator import build_bank, fill_bank, rho_base_vector, rho_grid
from .problem import BIG, QP, to_tensor
from .scaling import ruiz_equilibrate
from .warmstart import check_segments, shift_stages

__all__ = ["Settings", "Result", "Solver"]

MODES = ("auto", "persistent", "single_kernel", "two_kernel")
U32 = 2.0 ** -24
U64 = 2.0 ** -53


@dataclass
class Settings:
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    max_iter: int = 4000
    sigma: float = 1e-6
    alpha: float = 1.6
    mode: str = "auto"
    precision: str = "float32"
    precision_margin: float = 0.2
    check_interval: int = 10
    persistent_check: int = 10
    persistent_budget: int = 200
    use_graph: bool = True
    rho0: float = 0.1
    n_rho: int = 5
    rho_ratio: float = 10.0
    adaptive_rho: bool = True
    adaptive_rho_tol: float = 5.0
    probe_rho: bool = True
    lazy_update: bool = False
    probe_budget: int = 400
    probe_tighten: float = 0.01
    probe_recentre: int = 2
    rho_group_ratio: Optional[float] = None
    rho_group_ratios: tuple = (0.001, 0.01, 0.1, 1.0, 10.0)
    scaling_iters: int = 10
    stall_checks: int = 12
    stall_accept_factor: float = 100.0
    compute_objective: bool = False
    retune: bool = False


@dataclass
class Result:
    z: torch.Tensor
    s: torch.Tensor
    lam: torch.Tensor
    info: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.info.get("status", "unknown")

    @property
    def iters(self) -> int:
        return self.info.get("iter", -1)


class _Engine:
    """Problem data, iterate, active operator and CUDA graph in one precision."""

    def __init__(self, sv: "Solver", dtype, kmode: dict):
        self.sv, self.dtype = sv, dtype
        self.kmode = kmode["mode"]
        self.tile = (tuple(kmode["tile"][0]), int(kmode["tile"][1]))
        self.us_iter, self.us_check = kmode["us_per_iter"], kmode["us_check"]
        dev, n, m, mc, nd = sv.device, sv.n, sv.m, sv.m_c, sv.n_d
        c = lambda t: None if t is None else t.to(dtype).contiguous()  # noqa: E731
        self.P, self.q, self.d, self.C = c(sv.Ps), c(sv.qs), c(sv.ds), c(sv.Cs)
        self.l, self.h = c(sv.ls), c(sv.hs)
        self.D, self.E = c(sv.sc.D), c(sv.sc.E)
        self.Dinv, self.Einv = c(sv.sc.Dinv), c(sv.sc.Einv)
        z = lambda k, d=dtype: torch.zeros(k, device=dev, dtype=d)  # noqa: E731
        R = n + mc
        rows = R if self.kmode == "single_kernel" else n
        self.op = torch.zeros(rows, R, device=dev, dtype=dtype)
        self.b, self.rho, self.rho_inv = z(rows), z(m), z(m)
        self.z, self.s, self.lam = z(n), z(m), z(m)
        self.rin, self.rout, self.zt, self.rc = z(R), z(R), z(n), z(m)
        self.Pz, self.Gz, self.Gtlam = z(n), z(m), z(n)
        self.part, self.res = z(K.n_partials(n, m)), z(5)
        self.res_host = torch.zeros(5, dtype=dtype).pin_memory()
        self.info_buf = z(4, torch.float32)
        self.zo, self.so, self.lamo = z(n), z(m), z(m)
        self.bank, self.idx, self.graph, self.persistent = None, 0, None, None
        self.stale = set()
        self.n_lazy = 0

    def set_bank(self, bank, idx):
        self.bank = bank
        self.stale = set()
        self.select(idx, force=True)

    def select(self, idx, force=False) -> bool:
        idx = int(max(0, min(self.bank.n_rho - 1, idx)))
        if idx == self.idx and not force:
            return False
        if idx in self.stale:
            sv, b = self.sv, self.bank
            fill_bank(b.W_bank, sv.Ps, sv.ds, sv.Cs, b.rho_base.to(torch.float64),
                      b.rhos, sv.st.sigma, b.form, only=[idx])
            self.stale.discard(idx)
            self.n_lazy += 1
        self.idx = idx
        self.op.copy_(self.bank.W_bank[idx])
        torch.mul(self.bank.rho_base, self.bank.rhos[idx], out=self.rho)
        torch.reciprocal(self.rho, out=self.rho_inv)
        self.bias()
        return True

    def bias(self):
        torch.mv(self.op[:, : self.sv.n], self.q, out=self.b)
        self.b.neg_()

    @property
    def rho_value(self):
        return self.bank.rhos[self.idx]

    def iterate(self, rin, rout, tile=None):
        sv, st = self.sv, self.sv.st
        if self.kmode == "single_kernel":
            K.single_kernel_iter(self.op, self.b, rin, rout, self.z, self.s, self.lam,
                                 self.l, self.h, self.rho, self.rho_inv, self.d,
                                 sv.n, sv.m_c, st.sigma, st.alpha, tile or self.tile)
        else:
            K.two_kernel_iter(self.op, self.C, self.b, rin, rout, self.zt, self.z,
                              self.s, self.lam, self.l, self.h, self.rho, self.rho_inv,
                              self.d, sv.n, sv.m_c, st.sigma, st.alpha, tile or self.tile)

    def prime(self):
        """Form r from (z, s, lam) for the active penalty."""
        n, nd = self.sv.n, self.sv.n_d
        torch.mul(self.rho, self.s, out=self.rc)
        self.rc.sub_(self.lam)
        rz = self.rin[:n]
        torch.mul(self.z, self.sv.st.sigma, out=rz)
        if nd:
            rz.addcmul_(self.d, self.rc[:nd])
        self.rin[n:].copy_(self.rc[nd:])

    def residuals(self):
        sv, st = self.sv, self.sv.st
        nd = sv.n_d
        torch.mv(self.P, self.z, out=self.Pz)
        if nd:
            torch.mul(self.d, self.z, out=self.Gz[:nd])
            torch.mul(self.d, self.lam[:nd], out=self.Gtlam)
        if sv.m_c:
            torch.mv(self.C, self.z, out=self.Gz[nd:])
            if nd:
                self.Gtlam.addmv_(self.C.T, self.lam[nd:])
            else:
                torch.mv(self.C.T, self.lam, out=self.Gtlam)
        K.residual_norms(self.Gz, self.s, self.Pz, self.q, self.Gtlam, self.Einv,
                         self.Dinv, self.part, self.res, sv.cinv, st.eps_abs, st.eps_rel)

    def read(self):
        self.res_host.copy_(self.res)
        return self.res_host.tolist()

    def block(self):
        for _ in range(self.sv.st.check_interval // 2):
            self.iterate(self.rin, self.rout)
            self.iterate(self.rout, self.rin)
        self.residuals()

    def persistent_launch(self, budget, warps):
        sv, st = self.sv, self.sv.st
        K.persistent_solve(self.op, self.q, self.l, self.h, self.rho, self.rho_inv,
                           self.Einv, self.Dinv, self.d, self.z, self.s, self.lam,
                           self.info_buf, sv.n, sv.cinv, st.sigma, st.alpha,
                           st.eps_abs, st.eps_rel, budget, st.persistent_check,
                           num_warps=warps)

    def _snap(self):
        return [t.clone() for t in (self.z, self.s, self.lam, self.rin, self.rout)]

    def _restore(self, snap):
        for buf, v in zip((self.z, self.s, self.lam, self.rin, self.rout), snap):
            buf.copy_(v)

    def capture(self, fn, warm=3):
        snap = self._snap()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warm):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        self._restore(snap)
        return g

    def zero(self):
        torch._foreach_zero_([self.z, self.s, self.lam, self.rin, self.rout])

    def run_block(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.block()


class Solver:
    """Set up once, then call ``update`` and ``solve`` in a loop."""

    def __init__(self, qp: QP, settings: Optional[Settings] = None, **kwargs):
        self.st = settings or Settings()
        for k, v in kwargs.items():
            if not hasattr(self.st, k):
                raise TypeError(f"unknown setting '{k}'")
            setattr(self.st, k, v)
        st = self.st
        if st.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if st.precision not in ("float32", "float64", "auto"):
            raise ValueError("precision must be float32, float64 or auto")
        if st.check_interval % 2:
            st.check_interval += 1  # the two r buffers alternate in pairs
        self.qp = qp
        self.device = qp.device
        self.n, self.m, self.n_d, self.m_c = qp.n, qp.m, qp.n_d, qp.m_c
        self.structure = qp.structure
        if st.mode == "persistent" and self.structure != "box":
            raise ValueError("mode 'persistent' needs the box structure")
        if st.mode == "two_kernel" and self.structure == "box":
            raise ValueError("mode 'two_kernel' needs dense rows (split or general)")
        self._setup()

    def _setup(self):
        st, qp = self.st, self.qp
        t0 = time.perf_counter()
        (self.Ps, self.qs, self.ds, self.Cs, self.ls, self.hs,
         self.sc) = ruiz_equilibrate(qp.P, qp.q, qp.d, qp.C, qp.l, qp.h,
                                     iters=st.scaling_iters)
        self.perm = qp.perm
        self.cinv = float(self.sc.cinv)
        self.gamma = 1.0 if st.rho_group_ratio is None else float(st.rho_group_ratio)
        self.selection = {}

        self.kappa = None
        if st.precision == "auto":
            self.kappa = self._kappa_estimate()
            eps = min(e for e in (st.eps_abs, st.eps_rel) if e > 0)
            r32 = U32 * self.kappa
            if r32 <= 0.01 * eps:
                cands, why = [torch.float32], "stage 1: u32*kappa <= 0.01 eps"
            elif r32 >= eps:
                cands, why = [torch.float64], "stage 1: u32*kappa >= eps"
            else:
                cands, why = [torch.float32, torch.float64], "stage 2: trial"
            self.selection["precision_stage"] = why
        else:
            cands = [getattr(torch, st.precision)]

        engines = {}
        for i, dt in enumerate(cands):
            eng = self._make_engine(dt)
            if i == 0:
                self._penalty_search(eng)
            else:
                first = engines[cands[0]]
                rho = first.rho_value
                eng.set_bank(self._make_bank(rho_grid(rho, st.n_rho, st.rho_ratio),
                                             eng), 0)
                eng.select(eng.bank.nearest(rho), force=True)
                eng.graph = eng.capture(eng.block) if st.use_graph else None
                eng.trial = self._trial(eng, st.probe_budget)
                eng.idx0 = eng.bank.nearest(rho)
            engines[dt] = eng

        plans = {dt: self._plan(eng) for dt, eng in engines.items()}
        if len(plans) == 1:
            dt = cands[0]
        else:
            ok = {d: p for d, p in plans.items() if p["reached"]} or plans
            t32 = ok.get(torch.float32, {}).get("T_us", math.inf)
            t64 = ok.get(torch.float64, {}).get("T_us", math.inf)
            dt = torch.float64 if t64 <= (1 + st.precision_margin) * t32 else torch.float32
            self.selection["precision_T_us"] = {"float32": t32, "float64": t64}
        self.eng = engines[dt]
        self.dtype = dt
        for d in list(engines):
            if d != dt:
                del engines[d]
        plan = plans[dt]
        self.mode = plan["mode"]
        if self.mode == "persistent" and st.probe_rho:
            # The persistent solve adapts rho only between launches, so its
            # starting rho is probed again for that schedule.
            eng = self.eng
            graph_state = (eng.idx0, eng.trial, eng.rho_scores, plan)
            k = st.check_interval
            per = max(k, (min(st.max_iter, st.persistent_budget) // k) * k)
            self._probe(eng, adapt_every=per)
            pplan = self._plan(eng)
            pplan["T_graph_us"] = plan["T_graph_us"]
            if st.mode == "persistent" or pplan.get("T_persistent_us", math.inf) <= plan["T_graph_us"]:
                plan = dict(pplan, mode="persistent", T_us=pplan["T_persistent_us"])
            else:
                eng.idx0, eng.trial, eng.rho_scores, plan = graph_state
                plan = dict(plan, mode=eng.kmode, T_us=plan["T_graph_us"],
                            persistent_rejected=f"needs {pplan['iters_est']} iterations "
                                                f"without per-block rho adaptation")
            self.mode = plan["mode"]
        self.selection.update(plan)

        eng = self.eng
        eng.reset_idx = eng.idx0
        eng.select(eng.idx0, force=True)
        eng.zero()
        if self.mode == "persistent":
            pi = plan["persistent"]
            self._pwarps = pi["warps"]
            self._pbudget = min(st.max_iter, st.persistent_budget)
            eng.graph = (eng.capture(lambda: eng.persistent_launch(self._pbudget, self._pwarps), warm=2)
                         if st.use_graph else None)
        elif eng.graph is None and st.use_graph:
            eng.graph = eng.capture(eng.block)
        torch.cuda.synchronize()
        self.setup_time = time.perf_counter() - t0

    def _make_engine(self, dtype):
        st = self.st
        allowed = (("single_kernel", "two_kernel") if st.mode in ("auto", "persistent")
                   else (st.mode,))
        km = cal_mod.select_kernel_mode(self.structure, self.n, self.m_c, dtype,
                                        st.n_rho, self.device, force=st.retune,
                                        allowed=allowed)
        self.selection.setdefault("kernel_mode", {})[str(dtype).replace("torch.", "")] = km
        return _Engine(self, dtype, km)

    def _make_bank(self, rhos, eng, gamma=None):
        return build_bank(self.Ps, self.ds, self.Cs, self.ls, self.hs, self.st.sigma,
                          rhos, self.gamma if gamma is None else gamma,
                          eng.kmode, eng.dtype)

    def _kappa_estimate(self, iters=30):
        """Estimate kappa(K) at rho0 by power iterations on K and K^-1."""
        n, nd, st = self.n, self.n_d, self.st
        base = rho_base_vector(self.ls, self.hs, nd, 1.0) * st.rho0
        Kmat = self.Ps + st.sigma * torch.eye(n, device=self.device, dtype=torch.float64)
        if nd:
            Kmat = Kmat + torch.diag(base[:nd] * self.ds ** 2)
        if self.m_c:
            Kmat = Kmat + self.Cs.T @ (base[nd:].unsqueeze(1) * self.Cs)
        L, info = torch.linalg.cholesky_ex(Kmat)
        if bool(info.any()):
            return math.inf
        v = torch.randn(n, 1, device=self.device, dtype=torch.float64)
        w = v.clone()
        lmax = lmin_inv = 1.0
        for _ in range(iters):
            v = Kmat @ v
            lmax = float(v.norm())
            v /= lmax
            w = torch.cholesky_solve(w, L)
            lmin_inv = float(w.norm())
            w /= lmin_inv
        return lmax * lmin_inv

    def _levels(self):
        t = self.st.probe_tighten
        return [t, math.sqrt(t), 1.0] if t < 1.0 else [1.0]

    def _trial(self, eng, budget, adapt_every=None):
        """Solve from zero and record when each tolerance level is first met."""
        st = self.st
        adapt_every = adapt_every or st.check_interval
        levels = self._levels()
        hits = [None] * len(levels)
        eng.zero()
        eng.prime()
        it, r = 0, [math.nan] * 5
        while it < budget:
            eng.run_block()
            it += st.check_interval
            r = eng.read()
            for i, f in enumerate(levels):
                if hits[i] is None and (r[0] <= f * (st.eps_abs + st.eps_rel * r[1])
                                        and r[2] <= f * (st.eps_abs + st.eps_rel * r[3])):
                    hits[i] = it
            if hits[0] is not None:
                break
            if it % adapt_every == 0 and self._adapt(eng, r):
                eng.prime()
        rel = r[0] / max(r[1], 1e-12) + r[2] / max(r[3], 1e-12)
        return dict(hits=hits, rel=rel if np.isfinite(rel) else 1e9)

    @staticmethod
    def _pick(trials, budget):
        nlev = len(trials[0]["hits"])
        level = next((i for i in range(nlev)
                      if any(t["hits"][i] is not None for t in trials)), None)
        scores = [float(t["hits"][level]) if level is not None and t["hits"][level] is not None
                  else float(budget) + t["rel"] for t in trials]
        return int(np.argmin(scores)), scores

    def _probe(self, eng, adapt_every=None):
        st = self.st
        trials = []
        for i in range(eng.bank.n_rho):
            eng.select(i, force=True)
            trials.append(self._trial(eng, st.probe_budget, adapt_every))
        best, scores = self._pick(trials, st.probe_budget)
        eng.select(best, force=True)
        eng.idx0, eng.trial, eng.rho_scores = best, trials[best], scores
        return best

    def _probe_recentre(self, eng):
        st = self.st
        self._probe(eng)
        for _ in range(max(0, st.probe_recentre)):
            i = eng.idx
            if eng.bank.n_rho < 3 or 0 < i < eng.bank.n_rho - 1:
                break
            centre = eng.rho_value
            eng.set_bank(self._make_bank(rho_grid(centre, st.n_rho, st.rho_ratio), eng), 0)
            eng.select(eng.bank.nearest(centre), force=True)
            self._probe(eng)

    def _penalty_search(self, eng):
        """Build the bank and graph, probe rho and, for split problems, gamma."""
        st = self.st
        eng.set_bank(self._make_bank(rho_grid(st.rho0, st.n_rho, st.rho_ratio), eng), 0)
        eng.select(eng.bank.nearest(st.rho0), force=True)
        eng.graph = eng.capture(eng.block) if st.use_graph else None
        eng.idx0, eng.trial = eng.idx, None
        if not st.probe_rho:
            eng.trial = self._trial(eng, st.probe_budget)
            return
        self._probe_recentre(eng)
        if self.structure == "split" and st.rho_group_ratio is None:
            rho = eng.rho_value
            trials = []
            for g in st.rho_group_ratios:
                eng.set_bank(self._make_bank(rho_grid(rho, st.n_rho, st.rho_ratio), eng, g), 0)
                eng.select(eng.bank.nearest(rho), force=True)
                trials.append(self._trial(eng, st.probe_budget))
            i, scores = self._pick(trials, st.probe_budget)
            self.gamma = st.rho_group_ratios[i]
            self.selection["gamma_scores"] = dict(zip(st.rho_group_ratios, scores))
            eng.set_bank(self._make_bank(rho_grid(rho, st.n_rho, st.rho_ratio), eng), 0)
            eng.select(eng.bank.nearest(rho), force=True)
            self._probe_recentre(eng)

    def _plan(self, eng) -> dict:
        st = self.st
        hits = eng.trial["hits"] if eng.trial else [None]
        I = hits[-1]
        reached = I is not None
        I = I if reached else st.probe_budget
        k = st.check_interval
        cal = cal_mod.get_calibration(eng.dtype, self.device)
        T_graph = math.ceil(I / k) * (k * eng.us_iter + eng.us_check + cal["tau_read_us"])
        plan = dict(iters_est=I, reached=reached, T_graph_us=T_graph,
                    kernel=eng.kmode, precision=str(eng.dtype).replace("torch.", ""))
        mode, T = eng.kmode, T_graph
        pinfo = cal_mod.persistent_info(cal, self.n) if self.structure == "box" else None
        if pinfo is not None and st.mode in ("auto", "persistent"):
            launches = math.ceil(I / st.persistent_budget)
            Ip = math.ceil(I / st.persistent_check) * st.persistent_check
            T_p = Ip * pinfo["us_per_iter"] + launches * (cal["tau_node_us"] + cal["tau_read_us"])
            plan.update(T_persistent_us=T_p, persistent=pinfo)
            if st.mode == "persistent" or T_p < T_graph:
                mode, T = "persistent", T_p
        elif st.mode == "persistent":
            raise ValueError("persistent mode: the tile for this n spills on this device")
        plan.update(mode=mode, T_us=T)
        return plan

    def _adapt(self, eng, r) -> bool:
        """OSQP residual balancing, snapped to the nearest bank entry."""
        st = self.st
        if not st.adaptive_rho or eng.bank.n_rho <= 1:
            return False
        num = r[0] / max(r[1], 1e-10)
        den = r[2] / max(r[3], 1e-10)
        if den <= 0.0 or not np.isfinite(num / den):
            return False
        cur = eng.rho_value
        new = cur * math.sqrt(max(num / den, 1e-12))
        if max(new / cur, cur / new) < st.adaptive_rho_tol:
            return False
        idx = eng.bank.nearest(new)
        return idx != eng.idx and eng.select(idx)

    def _rows(self, v):
        v = to_tensor(v, self.device).reshape(-1)
        if v.numel() == 1:
            v = v.expand(self.m)
        return torch.clamp(v, -BIG, BIG)[self.perm]

    def update(self, q=None, l=None, h=None):
        """Update q and/or the bounds (user row order)."""
        sc, eng = self.sc, self.eng
        if q is not None:
            self.qs = to_tensor(q, self.device).reshape(-1) * sc.D * sc.c
            eng.q.copy_(self.qs)
            eng.bias()
        if l is not None:
            self.ls = self._rows(l) * sc.E
            eng.l.copy_(self.ls)
        if h is not None:
            self.hs = self._rows(h) * sc.E
            eng.h.copy_(self.hs)

    def update_matrices(self, P=None, G=None):
        """Replace P and/or G, keeping the constraint structure.

        The setup scaling, rho, the iterate and the CUDA graph are kept. The
        bank is re-inverted, or only its active entry when ``lazy_update`` is set.
        """
        sc, eng, nd = self.sc, self.eng, self.n_d
        if P is not None:
            P = to_tensor(P, self.device)
            self.Ps = 0.5 * (P + P.T) * sc.c * sc.D.unsqueeze(1) * sc.D.unsqueeze(0)
            eng.P.copy_(self.Ps)
        if G is not None:
            Gi = to_tensor(G, self.device)[self.perm]
            if nd:
                self.ds = sc.E[:nd] * torch.diagonal(Gi[:nd]) * sc.D
                eng.d.copy_(self.ds)
            if self.m_c:
                self.Cs = sc.E[nd:].unsqueeze(1) * Gi[nd:] * sc.D.unsqueeze(0)
                eng.C.copy_(self.Cs)
        b = eng.bank
        if self.st.lazy_update:
            fill_bank(b.W_bank, self.Ps, self.ds, self.Cs,
                      b.rho_base.to(torch.float64), b.rhos, self.st.sigma, b.form,
                      only=[eng.idx])
            eng.stale = set(range(b.n_rho)) - {eng.idx}
        else:
            fill_bank(b.W_bank, self.Ps, self.ds, self.Cs,
                      b.rho_base.to(torch.float64), b.rhos, self.st.sigma, b.form)
            eng.stale = set()
        eng.select(eng.idx, force=True)

    def warm_start(self, z=None, lam=None):
        """Set the starting point (unscaled, user row order); s = proj(Gz)."""
        sc, eng, nd = self.sc, self.eng, self.n_d
        if z is not None:
            zt = to_tensor(z, self.device).reshape(-1) * sc.Dinv
            eng.z.copy_(zt)
            if nd:
                torch.mul(eng.d, eng.z, out=eng.s[:nd])
            if self.m_c:
                torch.mv(eng.C, eng.z, out=eng.s[nd:])
            eng.s.clamp_(min=eng.l, max=eng.h)
        if lam is not None:
            lamt = to_tensor(lam, self.device).reshape(-1)[self.perm] * sc.Einv * sc.c
            eng.lam.copy_(lamt)

    def warm_start_shifted(self, z_segments=(), lam_segments=(), tail="hold",
                           shift_dual=True):
        """Warm start from the last solution shifted one stage along the horizon.

        Segments are ``(start, size, count)`` stage blocks of z and of the rows
        of G; see :mod:`fame_qp.warmstart`. Call after ``update`` so that s is
        rebuilt from the current data.
        """
        if not getattr(self, "_has_solution", False):
            return
        check_segments(z_segments, self.n, "z")
        check_segments(lam_segments, self.m, "lam")
        z = shift_stages(self.eng.zo.to(torch.float64), z_segments, tail)
        if shift_dual:
            lam = shift_stages(self.eng.lamo.to(torch.float64), lam_segments, tail)
        else:
            lam = torch.zeros(self.m, device=self.device, dtype=torch.float64)
        self.warm_start(z=z, lam=lam)

    def reset(self):
        """Zero the iterate and restore the probed penalty."""
        self.eng.zero()
        self.eng.select(self.eng.reset_idx)

    def _stall_status(self, r):
        st, k = self.st, self.st.stall_accept_factor
        near = (r[0] <= k * (st.eps_abs + st.eps_rel * r[1])
                and r[2] <= k * (st.eps_abs + st.eps_rel * r[3]))
        return "solved_inaccurate" if near else "stalled"

    def _solve_graph(self):
        st, eng = self.st, self.eng
        eng.prime()
        it, r = 0, [math.nan] * 5
        best, stall = math.inf, 0
        while it < st.max_iter:
            eng.run_block()
            it += st.check_interval
            r = eng.read()
            if r[4] > 0.5:
                return it, "solved", r[0], r[2]
            tot = r[0] + r[2]
            if tot < 0.9 * best:
                best, stall = tot, 0
            else:
                stall += 1
                if stall >= st.stall_checks:
                    return it, self._stall_status(r), r[0], r[2]
            if self._adapt(eng, r):
                eng.prime()
                best, stall = math.inf, 0
        return it, "max_iter_reached", r[0], r[2]

    def _solve_persistent(self):
        st, eng = self.st, self.eng
        it, rp, rd = 0, math.nan, math.nan
        best, stall = math.inf, 0
        while it < st.max_iter:
            if eng.graph is not None:
                eng.graph.replay()
            else:
                eng.persistent_launch(min(st.max_iter - it, self._pbudget), self._pwarps)
            info = eng.info_buf.tolist()
            it += int(info[0])
            rp, rd = info[1], info[2]
            if info[3] > 0.5:
                return it, "solved", rp, rd
            eng.residuals()
            r = eng.read()
            tot = r[0] + r[2]
            if tot < 0.9 * best:
                best, stall = tot, 0
            else:
                stall += 1
                if stall >= max(2, st.stall_checks // 4):
                    return it, self._stall_status(r), rp, rd
            if self._adapt(eng, r):
                best, stall = math.inf, 0
        return it, "max_iter_reached", rp, rd

    def solve(self) -> Result:
        t0 = time.perf_counter()
        if self.mode == "persistent":
            it, status, rp, rd = self._solve_persistent()
        else:
            it, status, rp, rd = self._solve_graph()
        run_time = time.perf_counter() - t0
        eng = self.eng
        K.unscale(eng.D, eng.E, self.perm, eng.z, eng.s, eng.lam,
                  eng.zo, eng.so, eng.lamo, self.cinv)
        self._has_solution = True
        info = dict(status=status, iter=it, run_time=run_time, pri_res=rp, dua_res=rd,
                    rho=eng.rho_value, gamma=self.gamma, mode=self.mode,
                    precision=str(self.dtype).replace("torch.", ""),
                    structure=self.structure, setup_time=self.setup_time)
        if self.st.compute_objective:
            info["obj_val"] = self.qp.objective(eng.zo)
        return Result(z=eng.zo, s=eng.so, lam=eng.lamo, info=info)
