#!/usr/bin/env python
"""Regenerate every figure and table in the Numerical Validation section
and Appendix G from the frozen N=1e10 measurements.

The figures use the ``tueplots`` NeurIPS context and the same layouts,
labels, line styles, and legends as the manuscript:

  * panels (a)-(d)  empirical CDF and CDF discrepancy, now from N=1e10
  * panel  (e)/(c)  the EVR-style pessimistic upper (eps,delta) curve with
                    the beta=1e-9 simultaneous empirical-Bernstein band
  * tables          the same three paper-facing numerical tables

Both DP directions are computed. `--direction max` (the default) reports
max{H(P||Q), H(Q||P)}, while `--direction P` provides a one-direction
diagnostic.

Usage
    python reproduce_paper.py                       # all figures + tables
    python reproduce_paper.py --direction P         # one-direction diagnostic
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from tueplots import axes as tue_axes, bundles        # noqa: E402

from pld_mc import CONFIGS, RESULTS_DIR, sample_size_tag  # noqa: E402

QS = [0.01, 0.1, 0.199, 0.801, 0.95]
TARGETS = np.logspace(-5, -7, 15)
FIGDIR = RESULTS_DIR / "figures"
PUBLIC_ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_METADATA = {"CreationDate": None, "ModDate": None}

# Per-q label placement in the discrepancy panels.
DISC_LXY = {
    0.01:  (0.97, 0.03, "right", "bottom"),
    0.1:   (0.03, 0.03, "left", "bottom"),
    0.199: (0.03, 0.03, "left", "bottom"),
    0.801: (0.03, 0.97, "left", "top"),
    0.95:  (0.03, 0.97, "left", "top"),
}

# Hand-tuned in-panel locations for the four-entry epsilon-delta legend.
# Each sits in the logarithmic gap between the RDP and privacy-curve lines.
EPS_LEGEND_KW = {
    0.01:  {"loc": "center", "bbox_to_anchor": (0.63, 0.71)},
    0.1:   {"loc": "center", "bbox_to_anchor": (0.63, 0.71)},
    0.199: {"loc": "center", "bbox_to_anchor": (0.63, 0.71)},
    0.801: {"loc": "center", "bbox_to_anchor": (0.63, 0.71)},
    0.95:  {"loc": "center", "bbox_to_anchor": (0.63, 0.71)},
}


# ------------------------------------------------------------------ baselines
def delta_gdp(eps, B):
    """delta(eps) for mu-GDP with mu = sqrt(2B) -- the Theorem 4.2 guarantee."""
    mu = math.sqrt(2.0 * B)
    return (norm.cdf(-eps / mu + mu / 2.0)
            - math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0))


def delta_rdp1(eps, B):
    """(1,B)-RDP -> (eps,delta) via Theorem G.1: delta = cB, 1/c + ln c = 1+eps."""
    c = brentq(lambda x: 1.0 / x + math.log(x) - 1.0 - eps, 1e-12, 1 - 1e-12)
    return c * B


def _fmt(x, sig=4):
    return f"{x:.{sig}g}"


# ------------------------------------------------------------------ loading
def load(q, N, policy="adagrad"):
    p = RESULTS_DIR / f"mc_q{q}_{policy}_N{sample_size_tag(N)}.json"
    if not p.exists():
        raise SystemExit(f"missing frozen result {p}")
    return json.loads(p.read_text())


def eps_at(eps, delta, target):
    """Invert a decreasing delta(eps) curve at `target`, log-linearly."""
    e, d = np.asarray(eps, float), np.asarray(delta, float)
    m = d > 0
    e, d = e[m], d[m]
    if len(d) == 0 or target >= d[0]:
        return float("nan")
    if target <= d[-1]:
        return float("nan")
    j = int(np.searchsorted(-d, -target))
    x0, x1 = e[j - 1], e[j]
    y0, y1 = math.log(d[j - 1]), math.log(d[j])
    t = (math.log(target) - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def combined_curve(rec, direction):
    """delta(eps) on a common grid: P-direction only, or the max of both."""
    cP = rec["arms"]["P"]["curve"]
    eps = np.array(cP["eps"])
    dP, loP, hiP = (np.array(cP[k]) for k in ("delta", "delta_lo", "delta_hi"))
    if direction == "P":
        return eps, dP, loP, hiP
    cQ = rec["arms"]["Q"]["curve"]
    eQ = np.array(cQ["eps"])
    dQ = np.interp(eps, eQ, cQ["delta"], left=cQ["delta"][0], right=0.0)
    loQ = np.interp(eps, eQ, cQ["delta_lo"], left=cQ["delta_lo"][0], right=0.0)
    hiQ = np.interp(eps, eQ, cQ["delta_hi"], left=cQ["delta_hi"][0], right=0.0)
    return eps, np.maximum(dP, dQ), np.maximum(loP, loQ), np.maximum(hiP, hiQ)


def simultaneous_band(audit, q, direction, eps):
    """The beta=1e-9 band, simultaneous over all grid points and directions."""
    by_arm = {
        row["arm"]: row
        for row in audit["curves"]
        if float(row["q"]) == float(q)
    }
    required = ("P",) if direction == "P" else ("P", "Q")
    missing = [arm for arm in required if arm not in by_arm]
    if missing:
        raise KeyError(f"audit is missing q={q}, arms={missing}")

    def values(arm, key):
        row = by_arm[arm]
        source_eps = np.asarray(row["eps"], dtype=float)
        source_values = np.asarray(row[key], dtype=float)
        if key == "upper_beta":
            # delta(epsilon) is non-increasing. Between stored grid points,
            # the upper bound at the nearest point to the left remains valid.
            idx = np.searchsorted(source_eps, eps, side="right") - 1
            out = np.ones_like(eps, dtype=float)
            inside = idx >= 0
            out[inside] = source_values[np.minimum(idx[inside],
                                                   len(source_values) - 1)]
            return out
        # For a lower bound, use the nearest point to the right. Beyond the
        # stored range, zero is always a valid lower bound.
        idx = np.searchsorted(source_eps, eps, side="left")
        out = np.zeros_like(eps, dtype=float)
        inside = idx < len(source_values)
        out[inside] = source_values[idx[inside]]
        return out

    lower = values("P", "lower_beta")
    upper = values("P", "upper_beta")
    if direction == "max":
        lower = np.maximum(lower, values("Q", "lower_beta"))
        upper = np.maximum(upper, values("Q", "upper_beta"))
    beta = float(audit["methodology"]["beta"])
    released = np.minimum(upper + beta, 1.0)
    return lower, upper, released


def arm_plot_arrays(rec, arm):
    """(x, F_emp, F_ref, disc, xlo, xhi) in the paper's L coordinates.

    The accumulators store Lambda = +L under P and Lambda = -L under Q, both
    referenced to N(B,2B).  The Q panels plot L against N(-B,2B), so the Q
    arm is mapped back:  x -> -x,  F -> 1-F  (the discrepancy is
    invariant), and the arrays are reversed so x stays increasing."""
    c = rec["arms"][arm]["cdf"]
    x = np.array(c["x"])
    F, R, D = (np.array(c[k]) for k in ("F_emp", "F_ref", "disc"))
    if arm == "Q":
        x, F, R, D = -x[::-1], 1.0 - F[::-1], 1.0 - R[::-1], D[::-1]
        return x, F, R, D, -c["x_hi"], -c["x_lo"]
    return x, F, R, D, c["x_lo"], c["x_hi"]


# ------------------------------------------------------------------ panels
def plot_cdf_panel(ax, x, F, R, lo, hi, title, legend_kw=None):
    ax.plot(x, F, lw=1.2, label="Simulated PLD")
    ax.plot(x, R, lw=1.2, ls="--", label=r"PLD of $\sqrt{2B}$-GDP")
    ax.set_ylabel("CDF")
    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=plt.rcParams["font.size"])
    ax.legend(**(legend_kw or {"loc": "upper left"}))
    ax.tick_params(labelbottom=False)


def plot_disc_panel(ax, x, D, lo, hi, ks, label, legend_kw=None,
                    delta_only=False, label_xy=None):
    ax.plot(x, D, "b-", lw=1.5, label="" if delta_only else "CDF Discrepancy")
    dline = ax.axhline(ks, color="red", lw=0.8, ls="--", alpha=0.7,
                       label=rf"$\widehat{{\Delta}} = {ks:.4f}$")
    ax.set_xlabel("Privacy Loss")
    ax.set_ylabel("CDF Error")
    ax.set_xlim(lo, hi)
    if delta_only:
        ax.legend(handles=[dline], **(legend_kw or
                  {"loc": "lower right", "bbox_to_anchor": (1.02, 0.65)}))
    else:
        ax.legend(**(legend_kw or
                  {"loc": "lower right", "bbox_to_anchor": (1.02, 0.65)}))
    lx, ly, lha, lva = label_xy or (0.97, 0.03, "right", "bottom")
    ax.text(lx, ly, label, transform=ax.transAxes, fontweight="bold",
            va=lva, ha=lha)


def plot_eps_delta(ax, B, eps, released, label, band=None,
                   legend_kw=None):
    ax.plot(eps, [delta_rdp1(x, B) for x in eps], ":", lw=1.2,
            label=r"$(1,B)$-RDP")
    ax.plot(eps, released, "-", lw=1.2,
            label=r"EVR-style pessimistic upper $\delta$")
    if band is not None:
        ax.fill_between(band[0], band[1], band[2], alpha=0.25, lw=0, zorder=0,
                        color="C1",
                        label=r"$1-10^{-9}$ MC confidence band")
    ax.plot(eps, [delta_gdp(x, B) for x in eps], "--", lw=1.2,
            label=r"$\sqrt{2B}$-GDP")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\delta$")
    span = float(eps[-1] - eps[0])
    pad = 0.015 * span if span > 0 else 0.05
    ax.set_xlim(float(eps[0] - pad), float(eps[-1] + pad))
    ax.legend(**(legend_kw or {"loc": "center right",
                               "bbox_to_anchor": (1.0, 0.58)}))
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.text(0.03, 0.03, label, transform=ax.transAxes, fontweight="bold",
            va="bottom", ha="left")


# ------------------------------------------------------------------ figures
def make_figures(N, direction, band, audit):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    style = bundles.neurips2024()
    style["text.usetex"] = False
    style.pop("text.latex.preamble", None)
    summary = {}

    for q in QS:
        rec = load(q, N)
        B = rec["B"]
        eps, d, _, _ = combined_curve(rec, direction)
        eps_stars = np.array([eps_at(eps, d, t) for t in TARGETS])
        bandarg = None
        release_eps_stars = eps_stars
        releasearg = (eps, d)
        if band:
            lo, hi, released = simultaneous_band(audit, q, direction, eps)
            release_eps_stars = np.array(
                [eps_at(eps, released, t) for t in TARGETS]
            )
            valid = release_eps_stars[~np.isnan(release_eps_stars)]
            m = (eps >= valid.min()) & (eps <= valid.max())
            if np.count_nonzero(m) < 2:
                raise ValueError(
                    f"insufficient certified plotting range for q={q}"
                )
            bandarg = (
                eps[m],
                np.maximum(lo[m], 1e-12),
                np.maximum(hi[m], 1e-12),
            )
            releasearg = (eps[m], released[m])
        xP, FP, RP, DP, loP, hiP = arm_plot_arrays(rec, "P")
        xQ, FQ, RQ, DQ, loQ, hiQ = arm_plot_arrays(rec, "Q")
        ksP = rec["arms"]["P"]["ks_hat"]
        ksQ = rec["arms"]["Q"]["ks_hat"]
        eps0, delta0 = float(eps_stars[0]), float(TARGETS[0])

        with plt.rc_context(style):
            plt.rcParams.update(tue_axes.spines(right=False, top=False))
            plt.rcParams.update(tue_axes.legend(frameon=False))

            # ---- Figure 2 (main body), q = 0.1 only ----------------------
            if q == 0.1:
                fig, axd = plt.subplot_mosaic(
                    [["a", "c"], ["b", "c"]], figsize=(6.75, 3.6),
                    layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
                plot_cdf_panel(axd["a"], xP, FP, RP, loP, hiP, r"Under $P$")
                axd["a"].text(0.97, 0.03, "(a)", transform=axd["a"].transAxes,
                              fontweight="bold", va="bottom", ha="right")
                plot_disc_panel(axd["b"], xP, DP, loP, hiP, ksP, "(b)")
                axd["b"].axvline(eps0, color="C3", lw=1.0, ls=":")
                axd["b"].annotate(rf"$\delta\!=\!{delta0:.0e}$",
                                  xy=(eps0, ksP * 0.55),
                                  xytext=(eps0 + 0.3, ksP * 0.8),
                                  fontsize=plt.rcParams["font.size"] - 1,
                                  color="C3",
                                  arrowprops=dict(arrowstyle="->", color="C3",
                                                  lw=0.7))
                plot_eps_delta(axd["c"], B, releasearg[0], releasearg[1], "(c)",
                               band=bandarg,
                               legend_kw=EPS_LEGEND_KW[q])
                fig.savefig(
                    FIGDIR / "CLT_validation.pdf",
                    bbox_inches="tight",
                    metadata=PDF_METADATA,
                )
                fig.savefig(
                    FIGDIR / "CLT_validation.png",
                    bbox_inches="tight",
                    dpi=180,
                    metadata={"Software": "Matplotlib"},
                )
                plt.close(fig)

            # ---- Appendix G, five-panel figure per q ---------------------
            fig, axd = plt.subplot_mosaic(
                [["a", "c", "e"], ["b", "d", "e"]], figsize=(9.5, 3.6),
                layout="constrained", gridspec_kw={"width_ratios": [1, 1, 2]})
            best = {"loc": "best"}
            lxy = DISC_LXY[q]
            plot_cdf_panel(axd["a"], xP, FP, RP, loP, hiP, r"Under $P$",
                           legend_kw=best)
            axd["a"].text(0.03, 0.97, "(a)", transform=axd["a"].transAxes,
                          fontweight="bold", va="top", ha="left")
            plot_disc_panel(axd["b"], xP, DP, loP, hiP, ksP, "(b)",
                            legend_kw=best, delta_only=True, label_xy=lxy)
            axd["b"].axvline(eps0, color="C3", lw=1.0, ls=":")
            axd["b"].annotate(rf"$\delta\!=\!{delta0:.0e}$",
                              xy=(eps0, ksP * 0.55),
                              xytext=(eps0 + 0.3, ksP * 0.8),
                              fontsize=plt.rcParams["font.size"] - 1,
                              color="C3",
                              arrowprops=dict(arrowstyle="->", color="C3",
                                              lw=0.7))
            plot_cdf_panel(axd["c"], xQ, FQ, RQ, loQ, hiQ, r"Under $Q$",
                           legend_kw=best)
            axd["c"].set_ylabel("")
            axd["c"].text(0.03, 0.97, "(c)", transform=axd["c"].transAxes,
                          fontweight="bold", va="top", ha="left")
            plot_disc_panel(axd["d"], xQ, DQ, loQ, hiQ, ksQ, "(d)",
                            legend_kw=best, delta_only=True, label_xy=lxy)
            axd["d"].set_ylabel("")
            plot_eps_delta(axd["e"], B, releasearg[0], releasearg[1], "(e)",
                           band=bandarg,
                           legend_kw=EPS_LEGEND_KW[q])
            fig.savefig(
                FIGDIR / f"CLT_validation_{q}.pdf",
                bbox_inches="tight",
                metadata=PDF_METADATA,
            )
            plt.close(fig)

        summary[q] = dict(
            B=B, sqrt_2B=rec["sqrt_2B"], ks_P=ksP, ks_Q=ksQ,
            ks_P_cert=rec["arms"]["P"]["ks_certified_upper"],
            ks_Q_cert=rec["arms"]["Q"]["ks_certified_upper"],
            eps_stars=eps_stars.tolist(),
            **{f"{k}_{a}": rec["arms"][a][k] for a in ("P", "Q")
               for k in ("mean_L", "var_L", "halt_ratio", "mean_halt_step")})
    return summary


# ------------------------------------------------------------------ tables
def make_tables(s_, N, direction, audit):
    out = []
    a = out.append

    a("% ---- Table 1 (main body, tab:delta-gdp-conv) ----")
    a(r"\begin{tabular}{c c c}")
    a(r"\toprule")
    a(r"$q$  & Empirical $\widehat{\Delta}$ & $\sqrt{2B}$-GDP \\")
    a(r"\midrule")
    for q in QS:
        d = max(s_[q]["ks_P"], s_[q]["ks_Q"])
        a(rf"${q}$   & ${_fmt(d)}$ & ${_fmt(s_[q]['sqrt_2B'])}$ \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a("")

    a("% ---- Table 2 (tab:plrv-moments) ----")
    a(r"\begin{tabular}{c c cc cc cc}")
    a(r"\toprule")
    a(r" & & \multicolumn{2}{c}{Theoretical} & \multicolumn{2}{c}{Empirical under $P$} & \multicolumn{2}{c}{Empirical under $Q$} \\")
    a(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    a(r"$q$ & $B$ & Mean & Variance & Mean & Variance & Mean & Variance \\")
    a(r"\midrule")
    for q in QS:
        B = s_[q]["B"]
        a(rf"${q}$ & ${_fmt(B)}$ & $\pm{_fmt(B)}$ & ${_fmt(2*B)}$ & "
          rf"${_fmt(s_[q]['mean_L_P'])}$ & ${_fmt(s_[q]['var_L_P'])}$ & "
          rf"${_fmt(-s_[q]['mean_L_Q'])}$ & ${_fmt(s_[q]['var_L_Q'])}$ \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a("")

    a("% ---- Table 3 (tab:delta-gdp) ----")
    a(r"\begin{tabular}{c c cc c}")
    a(r"\toprule")
    a(r"$q$ & $\widehat{\Delta}$ under $P$ & $\widehat{\Delta}$ under $Q$ & Final $\widehat{\Delta}$ & $\sqrt{2B}$-GDP \\")
    a(r"\midrule")
    for q in QS:
        d = max(s_[q]["ks_P"], s_[q]["ks_Q"])
        a(rf"${q}$ & ${_fmt(s_[q]['ks_P'])}$ & ${_fmt(s_[q]['ks_Q'])}$ & "
          rf"${_fmt(d)}$ & ${_fmt(s_[q]['sqrt_2B'])}$ \\")
    a(r"\bottomrule")
    a(r"\end{tabular}")
    a("")

    a("% ---- caption numbers, per figure ----")
    for q in QS:
        for arm, tag in (("P", "(a)--(b)"), ("Q", "(c)--(d)")):
            hr = s_[q][f"halt_ratio_{arm}"] * 100
            hs = s_[q][f"mean_halt_step_{arm}"]
            # accumulators store Lambda = -L under Q; the paper's captions
            # quote the PLD mean in L coordinates, i.e. negative under Q
            mn = s_[q][f"mean_L_{arm}"] * (1 if arm == "P" else -1)
            vr = s_[q][f"var_L_{arm}"]
            ks = s_[q]["ks_P"] if arm == "P" else s_[q]["ks_Q"]
            a(f"% q={q} {tag} Under {arm}: halts {hr:.2f}\\%, "
              f"mean {hs:.1f} steps ({hs*q:.2f} epochs), "
              f"PLD mean {_fmt(mn)}, variance {_fmt(vr)}, "
              f"empirical Delta {_fmt(ks)}")
    return "\n".join(out)


def main():
    global QS, FIGDIR, RESULTS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10_000_000_000)
    ap.add_argument("--direction", default="max", choices=["max", "P"])
    ap.add_argument("--only", type=float, default=None,
                    help="restrict to a single q (for testing)")
    ap.add_argument("--output-dir", type=pathlib.Path, default=None,
                    help="artifact root (default: public repository root)")
    ap.add_argument("--results-dir", type=pathlib.Path, default=None,
                    help="raw input result directory (default: repository results)")
    ap.add_argument("--audit", type=pathlib.Path, default=None,
                    help="coverage audit JSON (default: results-dir/coverage_audit.json)")
    a = ap.parse_args()

    if a.results_dir is not None:
        RESULTS_DIR = a.results_dir
    output_dir = a.output_dir or PUBLIC_ROOT
    FIGDIR = output_dir / "figures"
    if a.only is not None:
        QS = [a.only]
    audit_path = a.audit or (RESULTS_DIR / "coverage_audit.json")
    if not audit_path.exists():
        raise FileNotFoundError(
            f"{audit_path} is required; run audit_coverage.py first")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    s = make_figures(a.N, a.direction, True, audit)
    tex = make_tables(s, a.N, a.direction, audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables.tex").write_text(tex, encoding="utf-8")
    print(tex)
    print(f"\nfigures -> {FIGDIR}")
    print(f"tables  -> {output_dir / 'tables.tex'}")


if __name__ == "__main__":
    main()
