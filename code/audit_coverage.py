#!/usr/bin/env python
"""Audit Monte-Carlo coverage and build privacy-scale verification curves.

No trajectories are rerun. If ``sample_var`` is absent, it is recovered from
the stored pointwise empirical-Bernstein radius.

At failure probability ``beta`` (default 1e-9), upper and lower
empirical-Bernstein curves are simultaneous over the full deterministic
700,000-point epsilon grid, all five q values, both DP directions, and both
sides of the interval. Only 1,500 points per curve are exported for display.

For X_i(eps)=(1-exp(eps-Lambda_i))_+ in [0,1], Maurer--Pontil Theorem 4 gives
one-sided radius

    r = sqrt(2 S^2 log(2/eta) / N) + 7 log(2/eta)/(3(N-1)),

where S^2 uses denominator N-1. We allocate eta over all
5 x 2 x 700,000 fixed candidate points and both interval sides. A union bound
therefore covers the entire deterministic grid with probability at least
1-beta. Any data-dependent subset may then be displayed without an additional
selection penalty.

The resulting confidence statement is not Wang et al.'s full
Estimate--Verify--Release (EVR) construction.  For a generic randomized
verification/release construction whose verification randomness is included
in the mechanism, a beta-probability verification failure can instead be
absorbed as

    delta_release(eps) <= U_beta(eps) + beta.

"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from pld_mc import NBINS, RESULTS_DIR, sample_size_tag

QS = [0.01, 0.1, 0.199, 0.801, 0.95]
ARMS = ("P", "Q")
STORED_POINTWISE_ALPHA = 0.05


def eb_add(c: float, n: int) -> float:
    return 7.0 * c / (3.0 * (n - 1))


def recover_sample_var(delta, delta_hi, n: int, c_stored: float):
    """Recover the N-1 sample variance from the stored radius.

    The production code that made the existing JSON used the denominator-N
    variance.  Inverting its radius recovers that quantity exactly; multiplying
    by N/(N-1) gives the sample variance required by Maurer--Pontil.
    """
    half = np.maximum(np.asarray(delta_hi) - np.asarray(delta), 0.0)
    root = np.maximum(half - eb_add(c_stored, n), 0.0)
    var_n = n * root * root / (2.0 * c_stored)
    return var_n * n / (n - 1)


def eb_radius(sample_var, n: int, c: float):
    return (np.sqrt(2.0 * np.asarray(sample_var) * c / n)
            + eb_add(c, n))


def load_curves(N: int):
    curves = []
    for q in QS:
        p = RESULTS_DIR / f"mc_q{q}_adagrad_N{sample_size_tag(N)}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"missing production result {p}; all 5 q x 2 arms are required")
        rec = json.loads(p.read_text(encoding="utf-8"))
        for arm in ARMS:
            a = rec["arms"][arm]
            cur = a["curve"]
            curves.append(dict(
                q=q, arm=arm, B=float(rec["B"]), n=int(a["n"]),
                eps=np.asarray(cur["eps"], dtype=float),
                delta=np.asarray(cur["delta"], dtype=float),
                delta_lo_pointwise=np.asarray(cur["delta_lo"], dtype=float),
                delta_hi_pointwise=np.asarray(cur["delta_hi"], dtype=float),
                sample_var=(np.asarray(cur["sample_var"], dtype=float)
                            if "sample_var" in cur else None),
            ))
    return curves


def audit_delta(N: int, beta: float):
    curves = load_curves(N)
    exported_points = sum(len(c["eps"]) for c in curves)
    n_curves = len(curves)
    if n_curves != len(QS) * len(ARMS):
        raise RuntimeError(f"expected 10 curves, found {n_curves}")
    candidate_points = n_curves * NBINS

    # The stored visualization bands were generated pointwise at alpha=0.05.
    c_stored = math.log(2.0 / STORED_POINTWISE_ALPHA)
    # Privacy-scale allocation over the full deterministic candidate grid.
    # The exported points are a possibly data-dependent subset of this grid,
    # so allocating over all candidates makes the subsequent selection safe.
    local_failure = beta / (2.0 * candidate_points)
    c_beta = math.log(2.0 / local_failure)

    serial_curves = []
    monotonicity = []
    for c in curves:
        eps, d, n = c["eps"], c["delta"], c["n"]
        sample_var = c["sample_var"]
        variance_source = "stored sample_var"
        if sample_var is None:
            sample_var = recover_sample_var(
                d, c["delta_hi_pointwise"], n, c_stored)
            variance_source = "recovered from stored pointwise radius"

        rad_beta = eb_radius(sample_var, n, c_beta)
        lower_beta = np.maximum(d - rad_beta, 0.0)
        upper_beta = np.minimum(d + rad_beta, 1.0)
        upper_release = np.minimum(upper_beta + beta, 1.0)

        diff = np.diff(c["delta_hi_pointwise"])
        monotonicity.append(dict(
            q=c["q"], arm=c["arm"], M=len(eps),
            pointwise_upper_increases=int(np.count_nonzero(diff > 0)),
            largest_pointwise_upper_increase=(
                float(np.max(diff)) if len(diff) else 0.0)))

        serial_curves.append(dict(
            q=c["q"], arm=c["arm"], n=n, M=len(eps),
            variance_source=variance_source,
            eps=eps.tolist(), delta_hat=d.tolist(),
            lower_beta=lower_beta.tolist(),
            upper_beta=upper_beta.tolist(),
            upper_plus_beta=upper_release.tolist()))

    methodology = dict(
        theorem="Maurer-Pontil (2009), Theorem 4",
        theorem_status=(
            "EVR-inspired simultaneous Monte-Carlo verification; not the "
            "relaxed-FP EVR verifier/release theorem of Wang et al. (2023)"),
        N=N, alpha_visualization=STORED_POINTWISE_ALPHA, beta=beta,
        q_values=len(QS), dp_directions=len(ARMS),
        curves=n_curves, candidate_points_per_curve=NBINS,
        grid_points_total=candidate_points,
        exported_points_total=exported_points,
        sided_events=2 * candidate_points,
        local_one_sided_failure=local_failure,
        c_stored_pointwise=c_stored, c_beta_two_sided=c_beta,
        generic_release_conversion="delta_release(eps) <= U_beta(eps) + beta")
    return methodology, monotonicity, serial_curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=1e-9,
                    help="privacy-scale failure probability")
    ap.add_argument("--N", type=int, default=10_000_000_000)
    ap.add_argument("--results-dir", type=pathlib.Path, default=None,
                    help="input result directory (default: repository results)")
    ap.add_argument("--output", type=pathlib.Path, default=None,
                    help="output JSON (default: RESULTS_DIR/coverage_audit.json)")
    a = ap.parse_args()
    global RESULTS_DIR
    if a.results_dir is not None:
        RESULTS_DIR = a.results_dir
    if not (0 < a.beta < 1):
        ap.error("beta must lie in (0,1)")

    methodology, mono, curves = audit_delta(a.N, a.beta)

    print("=" * 88)
    print("PRIVACY-SCALE SIMULTANEOUS EMPIRICAL-BERNSTEIN VERIFICATION")
    print("=" * 88)
    print(f"curves={methodology['curves']}  candidate grid points="
          f"{methodology['grid_points_total']}  exported points="
          f"{methodology['exported_points_total']}  sided events="
          f"{methodology['sided_events']}")
    print(f"beta={a.beta:.1e}  local one-sided failure="
          f"{methodology['local_one_sided_failure']:.3e}  "
          f"c_beta={methodology['c_beta_two_sided']:.6f}")
    print("generic release conversion: U_beta(eps)+beta")

    out_data = dict(methodology=methodology, monotonicity=mono, curves=curves)
    out = a.output or (RESULTS_DIR / "coverage_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
