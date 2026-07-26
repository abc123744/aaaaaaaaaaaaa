#!/usr/bin/env python
"""Parallel Monte-Carlo estimation of the approximate-GDP filter PLD.

The implementation uses streaming histogram accumulators and evaluates

    delta(eps) = E[(1 - exp(eps - Lambda))_+],

a bounded expectation in [0,1]. Empirical-Bernstein bounds quantify
Monte-Carlo uncertainty in the privacy curve. The CDF discrepancy from the
Gaussian reference is also recorded.

Unification of the two arms.  Let Lambda = +L under P and Lambda = -L under Q.
Then in BOTH arms the reference is N(B, 2B) and
    delta_arm(eps) = E[(1 - e^{eps - Lambda})_+],
so one code path serves both, and the reported delta(eps) is the max of the
two arms.

Mechanism.  Poisson subsampling at rate q, remove-adjacency:
    P = (1-q) N(0, s^2) + q N(mu, s^2),   Q = N(0, s^2),   s = sigma_t * C_t,
    L = ln(P/Q) = ln(1 - q + q e^z),  z = mu (y - mu/2) / s^2.
Privacy depends on (q_t, sigma_t) only, through mu/s = 1/sigma_t; the clip C_t
cancels.  It still enters the *policy* dynamics (AdaGrad reads sum of y^2).

Filter (Algorithm 1).  Per-step charge Budg(qt,~q,sigma,1), the last step is
clipped to exactly exhaust the remaining budget, and the run halts there.

Usage
    python pld_mc.py --selftest
    python pld_mc.py --q 0.1 --N 1000000 --policy adagrad
    python pld_mc.py --q 0.1 --N 1000000000 --policy maxspend --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.special import ndtr

# ---------------------------------------------------------------- constants
SIGMA0, SIGMA_MIN, CLIP0 = 8.0, 2.0, 1.0

# Numerical-study configurations: (q, T, B_mult).
CONFIGS = {
    0.01:  dict(T=1200, B_mult=20000),
    0.1:   dict(T=120,  B_mult=2000),
    0.199: dict(T=65,   B_mult=1000),
    0.801: dict(T=20,   B_mult=200),
    0.95:  dict(T=20,   B_mult=150),
}

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def sample_size_tag(N):
    """Compact public filename tag for a trajectory count."""
    return "1e10" if int(N) == 10_000_000_000 else str(int(N))

# histogram grid: [B - LO_SD*sd, B + HI_SD*sd] with NBINS bins.
# Grid resolution enters the KS statistic only through the reference CDF
# increment per bin, <= 0.4 * width; reported as `grid_ks_slack`.
NBINS, LO_SD, HI_SD = 700_000, 15.0, 20.0
CURVE_PTS = 1500          # points kept in the exported delta(eps) curve
CDF_PTS = 3000            # points kept in the exported empirical CDF


def budget_step(q, sigma, mu_over_s, regime):
    """Budg(regime, q, sigma, mu): per-step budget charge.

    mu_over_s = mu / s is the normalised sensitivity actually used (= 1/sigma
    unless the last step was clipped down to exhaust the budget)."""
    if regime == 0:
        return 0.5 * q ** 2 * (np.exp(mu_over_s ** 2) - 1.0)
    return 0.5 * q ** 2 * mu_over_s ** 2


def inv_budget(q, B_rem, regime):
    """invBudg: the largest mu/s whose charge is exactly B_rem."""
    if regime == 0:
        return np.sqrt(np.log1p(2.0 * B_rem / q ** 2))
    return np.sqrt(2.0 * B_rem / q ** 2)


def regime_of(q):
    if q <= 0.3:
        return 0
    if q >= 0.7:
        return 1
    raise ValueError(f"q={q} is in neither asymptotic regime")


def budget_of(q, B_mult):
    """B0 = first-step charge at sigma0; B = B_mult * B0."""
    r = regime_of(q)
    B0 = budget_step(q, SIGMA0, 1.0 / SIGMA0, r)
    return float(B0), float(B_mult * B0)


def plrv(u, q, m):
    """L = ln(1 - q + q e^z) with z = m (u - m/2), in NORMALISED units:
        u = y / s   (standard normal under Q),   m = mu / s = 1/sigma_t.
    The clip C_t cancels from z exactly, so working in u avoids the huge
    intermediate values the AdaGrad rule produces when sum(y^2) is small.

    Stability: for z <= 0 use log1p(q expm1(z)); for z > 0 factor out e^z
    first.  Neither branch can overflow and neither cancels catastrophically.
    q may be scalar or per-trajectory."""
    z = m * (u - 0.5 * m)
    q = np.asarray(q, dtype=float)
    out = np.empty_like(z)
    neg = z <= 0.0
    qn = q if q.ndim == 0 else q[neg]
    qp = q if q.ndim == 0 else q[~neg]
    out[neg] = np.log1p(qn * np.expm1(z[neg]))
    zp = z[~neg]
    out[~neg] = zp + np.log1p((1.0 - qp) * np.expm1(-zp))
    return out


# ---------------------------------------------------------------- policies
#
# A policy maps the observed history to (q_t, sigma_t, C_t).  Everything it
# reads -- sum of y^2, the running privacy loss, remaining budget, step index
# -- is a function of y_{1:t-1}, so every policy here is legitimately adapted
# to the output filtration, which is exactly the class Algorithm 1 allows.
# Constraints enforced by the filter: sigma_t >= SIGMA_MIN, q_t <= qbar.

def policy_params(name, t, qbar, sum_sq, L_run, B_rem, B):
    """Return (q_t, sigma_t, C_t) as arrays/scalars for the active batch."""
    root = np.sqrt(sum_sq)

    if name == "adagrad":
        # AdaGrad-inspired rule:
        # sigma_t = clamp(sigma0/sqrt(sum y^2), [sig_min, sig0])
        sig = np.clip(SIGMA0 / root, SIGMA_MIN, SIGMA0)
        return qbar, sig, CLIP0 / root

    if name == "sigma0":
        # minimum spend per step -> most composed steps -> best CLT (bracket)
        return qbar, np.full_like(root, SIGMA0), CLIP0 / root

    if name == "maxspend":
        # Maximum spend per step, yielding fewer composed steps.
        return qbar, np.full_like(root, SIGMA_MIN), np.full_like(root, CLIP0)

    if name.startswith("losschase"):
        # History-dependent rule that spends maximally once the realized
        # privacy loss is large and minimally otherwise.
        theta = float(name.split(":")[1]) if ":" in name else 1.0
        hot = L_run > theta * (B - B_rem)          # ahead of its own budget
        sig = np.where(hot, SIGMA_MIN, SIGMA0)
        return qbar, sig, np.full_like(root, CLIP0)

    if name.startswith("qadapt"):
        # History-dependent sampling-rate rule: q_t alternates between qbar
        # and qbar/f depending on the realized privacy loss.
        f = float(name.split(":")[1]) if ":" in name else 4.0
        hot = L_run > (B - B_rem)
        q_t = np.where(hot, qbar, qbar / f)
        sig = np.full_like(root, SIGMA_MIN)
        return q_t, sig, np.full_like(root, CLIP0)

    raise ValueError(f"unknown policy {name}")


POLICIES = ["adagrad", "sigma0", "maxspend", "losschase", "qadapt"]


# ---------------------------------------------------------------- one chunk
def simulate_L(n, q, T, B, policy, arm, seed):
    """Simulate trajectories; return (L, halted, halt_step, budget_remaining).

    L is the total privacy loss ln(P/Q) of the released transcript, sampled
    under P (arm='P') or under Q (arm='Q')."""
    rng = np.random.default_rng(seed)
    regime = regime_of(q)
    B0 = budget_step(q, SIGMA0, 1.0 / SIGMA0, regime)

    # --- step 1 (sigma0, C0, uncapped, pre-charged B0) ---
    s = SIGMA0 * CLIP0
    m = 1.0 / SIGMA0                      # = mu/s with mu = C0, s = sigma0*C0
    u = rng.normal(0.0, 1.0, n)
    if arm == "P":
        u += np.where(rng.random(n) < 1.0 - q, 0.0, m)
    L = plrv(u, q, m)
    y = u * s
    sum_sq = y * y
    B_rem = np.full(n, B - B0)
    halted = np.zeros(n, dtype=bool)
    halt_step = np.zeros(n, dtype=np.int32)

    # --- steps 2..T ---------------------------------------------------------
    for t in range(2, T + 1):
        q_t, sig, C = policy_params(policy, t, q, sum_sq, L, B_rem, B)
        q_t = np.broadcast_to(np.asarray(q_t, dtype=float), (n,))
        s = sig * C
        # sensitivity actually used: min(own clip, budget-exhausting clip)
        mu_cap = inv_budget(q_t, B_rem, regime)          # in units of s
        inv_sig = 1.0 / sig
        m_over_s = np.minimum(inv_sig, mu_cap)
        live = ~halted
        u = rng.normal(0.0, 1.0, n)
        if arm == "P":
            u += np.where(rng.random(n) < 1.0 - q_t, 0.0, m_over_s)
        y = u * s
        step_L = plrv(u, q_t, m_over_s)
        L += np.where(live, step_L, 0.0)
        sum_sq += y * y
        charge = budget_step(q_t, sig, m_over_s, regime)
        B_rem = np.where(live, np.maximum(B_rem - charge, 0.0), B_rem)
        # Algorithm 1's own halting rule: "if C_{B,t} <= C_t then break",
        # i.e. the budget-capped sensitivity binds.  This is an exact
        # comparison; testing B_rem against 0 instead is unreliable because
        # the binding-step charge equals B_rem only up to ~1e-16 rounding,
        # so the residual can land either side of zero.
        newly = live & (mu_cap <= inv_sig)
        halt_step[newly] = t
        halted |= newly
        if halted.all():
            break
    return L, halted, halt_step, B_rem


def run_chunk(args):
    """Simulate `n` trajectories; return streaming accumulators only.

    Lambda = +L (arm P) or -L (arm Q); the reference is N(B,2B) for both, so
    one code path and one certificate serve both arms."""
    (n, q, T, B, policy, arm, seed, grid_lo, grid_hi) = args
    L, halted, halt_step, B_rem = simulate_L(
        n, q, T, B, policy, arm, seed)
    lam = L if arm == "P" else -L
    spend = B - B_rem

    # --- streaming accumulators --------------------------------------------
    width = (grid_hi - grid_lo) / NBINS
    idx = np.floor((lam - grid_lo) / width).astype(np.int64)
    under = int(np.count_nonzero(idx < 0))
    over = int(np.count_nonzero(idx >= NBINS))
    inside = (idx >= 0) & (idx < NBINS)
    idx_in = idx[inside]
    lam_in = lam[inside]
    # weights only where Lambda > 0: delta(eps) is only ever evaluated at
    # eps > 0, and this keeps e^{-Lambda} from overflowing in the left tail.
    pos = lam_in > 0.0
    w1 = np.zeros_like(lam_in)
    w2 = np.zeros_like(lam_in)
    w1[pos] = np.exp(-lam_in[pos])
    w2[pos] = np.exp(-2.0 * lam_in[pos])

    return dict(
        n=int(n),
        sum_L=float(lam.sum()), sum_L2=float((lam ** 2).sum()),
        sum_L3=float((lam ** 3).sum()), sum_L4=float((lam ** 4).sum()),
        sum_spend=float(spend.sum()), sum_spend2=float((spend ** 2).sum()),
        sum_spend3=float((spend ** 3).sum()),
        sum_spend4=float((spend ** 4).sum()),
        hist=np.bincount(idx_in, minlength=NBINS),
        hw1=np.bincount(idx_in, weights=w1, minlength=NBINS),
        hw2=np.bincount(idx_in, weights=w2, minlength=NBINS),
        halt_hist=np.bincount(halt_step, minlength=T + 1),
        under=under, over=over,
        n_halted=int(halted.sum()), sum_halt=float(halt_step.sum()),
        max_L=float(lam.max()), min_L=float(lam.min()),
        max_spend=float(spend.max()), min_spend=float(spend.min()),
    )


def merge(accs):
    out = dict(accs[0])
    out["hist"] = out["hist"].copy()
    out["hw1"] = out["hw1"].copy()
    out["hw2"] = out["hw2"].copy()
    out["halt_hist"] = out["halt_hist"].copy()
    for a in accs[1:]:
        for k in ("n", "sum_L", "sum_L2", "sum_L3", "sum_L4",
                  "sum_spend", "sum_spend2", "sum_spend3", "sum_spend4",
                  "under", "over", "n_halted", "sum_halt"):
            out[k] += a[k]
        out["hist"] += a["hist"]
        out["hw1"] += a["hw1"]
        out["hw2"] += a["hw2"]
        out["halt_hist"] += a["halt_hist"]
        out["max_L"] = max(out["max_L"], a["max_L"])
        out["min_L"] = min(out["min_L"], a["min_L"])
        out["max_spend"] = max(out["max_spend"], a["max_spend"])
        out["min_spend"] = min(out["min_spend"], a["min_spend"])
    return out


# ---------------------------------------------------------------- estimates
def dkw_slack(n, alpha=0.05):
    """Massart's DKW: P(sup|F_n - F| > e) <= 2 exp(-2 n e^2)."""
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def emp_bernstein(mean, sample_var, n, alpha=0.05):
    """Maurer--Pontil Thm. 4 upper bound; sample_var has denominator n-1."""
    lg = math.log(2.0 / alpha)
    return (mean + math.sqrt(2.0 * sample_var * lg / n)
            + 7.0 * lg / (3.0 * (n - 1)))


def analyse(acc, B, grid_lo, grid_hi, alpha=0.05):
    """KS statistic + certified Delta, and delta(eps) with certified bound."""
    n = acc["n"]
    sd = math.sqrt(2.0 * B)
    width = (grid_hi - grid_lo) / NBINS
    edges = grid_lo + width * np.arange(1, NBINS + 1)   # right edges

    cum = acc["under"] + np.cumsum(acc["hist"])
    F = cum / n
    ref = ndtr((edges - B) / sd)
    ks = float(np.max(np.abs(F - ref)))
    grid_slack = 0.4 * width / sd          # max reference-CDF jump per bin

    def _tail(h):
        """sum of h[j+1:] for each j, accumulated FROM THE TOP.

        The obvious `h.sum() - cumsum(h)` cancels catastrophically once the
        partial sum approaches the total: the residual is pure roundoff
        (~1e-16 of the total), and delta(eps) multiplies it by e^eps, which
        at eps ~ 30 manufactures a spurious delta ~ 1e-2 out of nothing.
        Reverse accumulation keeps every far-tail sum exact."""
        rev = np.cumsum(h[::-1])[::-1]
        return np.concatenate([rev[1:], [0.0]])

    tail_n = n - cum                                   # #{Lambda > edge}
    tail_w1 = _tail(acc["hw1"])
    tail_w2 = _tail(acc["hw2"])

    e = np.exp(edges)
    d_hat = tail_n / n - e * (tail_w1 / n)             # E[(1-e^{eps-L})_+]
    m2 = tail_n / n - 2.0 * e * (tail_w1 / n) + (e * e) * (tail_w2 / n)

    # Downsampled delta(eps) curve with a two-sided empirical-Bernstein band.
    # This is what Figure 2(c) plots; exporting it lets the figure be redrawn
    # from measurement instead of from a tail extrapolation.
    #
    # COVERAGE: delta_lo/delta_hi are POINTWISE (1-alpha) bounds.  They are NOT
    # a simultaneous band over the grid. See audit_coverage.py for the
    # simultaneous version. Label any plotted band built from these as
    # "pointwise".
    lg = math.log(2.0 / alpha)
    mean_c = np.maximum(d_hat, 0.0)
    var_c = np.maximum(m2 - mean_c * mean_c, 0.0)
    sample_var_c = var_c * n / (n - 1)
    half = (np.sqrt(2.0 * sample_var_c * lg / n)
            + 7.0 * lg / (3.0 * (n - 1)))
    sel = np.nonzero((edges > 0.0) & (tail_n > 0) & (mean_c > 0.0))[0]
    if len(sel) > CURVE_PTS:
        sel = sel[np.linspace(0, len(sel) - 1, CURVE_PTS).astype(int)]
    curve = dict(eps=edges[sel].tolist(),
                 delta=mean_c[sel].tolist(),
                 delta_lo=np.maximum(mean_c[sel] - half[sel], 0.0).tolist(),
                 delta_hi=(mean_c[sel] + half[sel]).tolist(),
                 sample_var=sample_var_c[sel].tolist(),
                 n_exceed=tail_n[sel].astype(np.int64).tolist())

    # Empirical CDF, downsampled, for panels (a)-(d). The histogram represents
    # the same object at 700k-bin resolution and is subsampled to ~3k points
    # before plotting.
    i_lo = int(np.searchsorted(F, 0.002))
    i_hi = int(np.searchsorted(F, 0.998))
    span = np.arange(max(i_lo - 1, 0), min(i_hi + 2, NBINS))
    if len(span) > CDF_PTS:
        span = span[np.linspace(0, len(span) - 1, CDF_PTS).astype(int)]
    j_ks = int(np.argmax(np.abs(F - ref)))
    cdf = dict(x=edges[span].tolist(), F_emp=F[span].tolist(),
               F_ref=ref[span].tolist(),
               disc=np.abs(F - ref)[span].tolist(),
               x_lo=float(edges[i_lo]), x_hi=float(edges[i_hi]),
               x_at_ks=float(edges[j_ks]))

    return dict(
        n=n, B=B, sqrt_2B=sd, curve=curve, cdf=cdf,
        mean_L=acc["sum_L"] / n,
        var_L=acc["sum_L2"] / n - (acc["sum_L"] / n) ** 2,
        ks_hat=ks,
        dkw=dkw_slack(n, alpha),
        grid_ks_slack=float(grid_slack),
        ks_certified_upper=ks + dkw_slack(n, alpha) + float(grid_slack),
        halt_ratio=acc["n_halted"] / n,
        mean_halt_step=acc["sum_halt"] / max(acc["n_halted"], 1),
        n_under=acc["under"], n_over=acc["over"],
        max_L=acc["max_L"], min_L=acc["min_L"],
        _edges=edges, _delta=d_hat, _m2=m2, _tail_n=tail_n,
    )


# ---------------------------------------------------------------- driver
def run(q, N, policy, workers=8, chunk=2_000_000, seed=1729, alpha=0.05,
        arms=("P", "Q"), verbose=True):
    cfg = CONFIGS[q]
    B0, B = budget_of(q, cfg["B_mult"])
    sd = math.sqrt(2.0 * B)
    grid_lo, grid_hi = B - LO_SD * sd, B + HI_SD * sd
    n_chunks = max(1, math.ceil(N / chunk))
    sizes = [N // n_chunks] * n_chunks
    for i in range(N - sum(sizes)):
        sizes[i] += 1

    out = {}
    for arm in arms:
        ss = np.random.SeedSequence([seed, ord(arm), int(q * 1000)])
        seeds = ss.spawn(n_chunks)
        jobs = [(sizes[i], q, cfg["T"], B, policy, arm, seeds[i],
                 grid_lo, grid_hi) for i in range(n_chunks)]
        t0 = time.time()
        if workers == 1:
            accs = [run_chunk(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                accs = list(ex.map(run_chunk, jobs))
        res = analyse(merge(accs), B, grid_lo, grid_hi, alpha)
        res["runtime_s"] = round(time.time() - t0, 1)
        res["arm"], res["q"], res["policy"] = arm, q, policy
        res["T"], res["B_mult"] = cfg["T"], cfg["B_mult"]
        out[arm] = res
        if verbose:
            print(f"  [{arm}] n={res['n']:.3g} mean={res['mean_L']:.6g} "
                  f"var={res['var_L']:.6g} KS={res['ks_hat']:.6g} "
                  f"(+DKW {res['dkw']:.2g} -> <= {res['ks_certified_upper']:.6g}) "
                  f"halt={res['halt_ratio']*100:.2f}% "
                  f"[{res['runtime_s']}s]", flush=True)
    return B, out


def save(q, policy, N, B, out):
    RESULTS_DIR.mkdir(exist_ok=True)
    rec = dict(q=q, policy=policy, N=N, B=B, sqrt_2B=math.sqrt(2 * B),
               arms={})
    for a, r in out.items():
        rec["arms"][a] = {k: v for k, v in r.items() if not k.startswith("_")}
    p = RESULTS_DIR / (
        f"mc_q{q}_{policy.replace(':', '_')}_N{sample_size_tag(N)}.json"
    )
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"  wrote {p}")
    return rec


# ---------------------------------------------------------------- self-test
def selftest():
    """Known-answer anchors.  Each is checkable independently of this code."""
    ok = True

    def rep(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")

    print("anchor battery (pld_mc)")

    # A1: stable PLRV == naive formula (scipy densities), incl. large |z|
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    worst = 0.0
    for q_ in (0.01, 0.1, 0.5, 0.95):
        for m_ in (0.125, 0.5, 2.0):
            u = rng.normal(0.0, 4.0, 20000)          # normalised units, s = 1
            naive = np.log(1 - q_ + q_ * norm.pdf(u, m_, 1.0)
                           / norm.pdf(u, 0.0, 1.0))
            worst = max(worst, float(np.max(np.abs(plrv(u, q_, m_) - naive))))
    rep("A1 stable PLRV == naive log(1-q+q d1/d0)", worst < 1e-11,
        f"max|diff|={worst:.2e}")

    # A2: budget/inv-budget are exact inverses in both regimes
    w = 0.0
    for r_, q_ in ((0, 0.1), (1, 0.9)):
        for Br in (1e-6, 1e-3, 0.5, 3.0):
            m = inv_budget(q_, Br, r_)
            w = max(w, abs(budget_step(q_, None, m, r_) - Br) / Br)
    rep("A2 invBudg inverts Budg", w < 1e-12, f"max rel err={w:.2e}")

    # A3: KNOWN ANSWER.  One un-subsampled Gaussian step (q=1, T=1) has
    # delta(eps) = Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2), mu=1/sigma.
    # Run the histogram estimator against that closed form.
    sig, n = 2.0, 4_000_000
    rg = np.random.default_rng(7)
    mu_g = 1.0 / sig                                  # normalised sensitivity
    lam = plrv(rg.normal(0.0, 1.0, n) + mu_g, 1.0, mu_g)   # arm P, q=1
    for eps in (0.05, 0.2, 0.5):
        emp = float(np.mean(np.maximum(1.0 - np.exp(eps - lam), 0.0)))
        exact = float(ndtr(-eps / mu_g + mu_g / 2)
                      - math.exp(eps) * ndtr(-eps / mu_g - mu_g / 2))
        rel = abs(emp - exact) / exact
        rep(f"A3 delta(eps={eps}) vs exact Gaussian-mechanism formula",
            rel < 0.02, f"emp={emp:.6g} exact={exact:.6g} rel={rel:.2%}")

    # A4: histogram-based delta == direct per-sample delta (same samples)
    B_ = 0.5
    sd_ = math.sqrt(2 * B_)
    glo, ghi = B_ - LO_SD * sd_, B_ + HI_SD * sd_
    width = (ghi - glo) / NBINS
    idx = np.floor((lam - glo) / width).astype(np.int64)
    keep = (idx >= 0) & (idx < NBINS)
    p = lam[keep] > 0
    w1 = np.zeros(int(keep.sum())); w2 = np.zeros(int(keep.sum()))
    w1[p] = np.exp(-lam[keep][p]); w2[p] = np.exp(-2 * lam[keep][p])
    h = np.bincount(idx[keep], minlength=NBINS)
    hw1 = np.bincount(idx[keep], weights=w1, minlength=NBINS)
    edges = glo + width * np.arange(1, NBINS + 1)
    tail_n = n - np.cumsum(h)
    tail_w1 = hw1.sum() - np.cumsum(hw1)
    d_hist = tail_n / n - np.exp(edges) * tail_w1 / n
    j = int(np.searchsorted(edges, 0.2))
    d_dir = float(np.mean(np.maximum(1.0 - np.exp(edges[j] - lam), 0.0)))
    rel = abs(d_hist[j] - d_dir) / d_dir
    rep("A4 histogram delta == direct delta", rel < 1e-9,
        f"hist={d_hist[j]:.8g} direct={d_dir:.8g} rel={rel:.1e}")

    # A5: regression anchors for the N=1e6 AdaGrad run at q=0.1.
    B_, res = run(0.1, 1_000_000, "adagrad", workers=1, chunk=1_000_000,
                  arms=("P",), verbose=False)
    r = res["P"]
    rep("A5 q=0.1 mean_L anchor (0.1526)", abs(r["mean_L"] - 0.1526) < 0.004,
        f"got {r['mean_L']:.4f}")
    rep("A5 q=0.1 var_L anchor (0.314)", abs(r["var_L"] - 0.314) < 0.01,
        f"got {r['var_L']:.4f}")
    rep("A5 q=0.1 empirical Delta_P anchor (0.01247)",
        abs(r["ks_hat"] - 0.01247) < 0.003, f"got {r['ks_hat']:.5f}")

    print("ANCHORS:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--policy", default="adagrad")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    print(f"q={a.q} policy={a.policy} N={a.N:.3g} workers={a.workers}")
    B, out = run(a.q, a.N, a.policy, workers=a.workers, chunk=a.chunk,
                 seed=a.seed, alpha=a.alpha)
    print(f"  B={B:.6g}  sqrt(2B)={math.sqrt(2*B):.4f}  "
          f"max per-direction upper Delta = "
          f"{max(out[x]['ks_certified_upper'] for x in out):.6g}")
    save(a.q, a.policy, a.N, B, out)


if __name__ == "__main__":
    main()
