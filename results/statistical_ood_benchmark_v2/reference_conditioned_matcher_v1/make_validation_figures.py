#!/usr/bin/env python3
"""Validation figures for the reference-conditioned matcher.

Design follows the dataviz method: form picked per data job, status colours for the
accept/reject state (green=matched/good, red=unmatched/critical) always paired with a
text label, one axis, thin marks, recessive grid, direct labels, colourblind-safe hues
from the validated palette. Static PNGs for report/slides.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
CUT = 0.40

# validated palette
GOOD = "#0ca30c"      # matched / accept
CRIT = "#d03b3b"      # unmatched / reject
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#b9b8b4"; GRID = "#e7e6e3"; SURF = "#fcfcfb"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.edgecolor": INK2, "axes.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans",
})


def cutline(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(CUT, color=INK2, ls=(0, (4, 3)), lw=1.3, zorder=1)


# ---------- Fig 1: CsCl holdout ----------
def fig_holdout():
    h = pd.read_csv(HERE / "cscl_holdout_results.csv")
    full = h[h.condition == "full_library"].D_best.values
    rem = h[h.condition == "cscl_removed"].D_best.values
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axhspan(0, CUT, color=GOOD, alpha=0.06, zorder=0)
    ax.axhspan(CUT, 0.85, color=CRIT, alpha=0.06, zorder=0)
    cutline(ax)
    rng = np.random.default_rng(0)
    for x, vals, col in [(0, full, GOOD), (1, rem, CRIT)]:
        jit = x + rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(jit, vals, s=42, color=col, edgecolor=SURF, linewidth=0.8, zorder=3, alpha=0.9)
        ax.plot([x - 0.19, x + 0.19], [np.median(vals)] * 2, color=INK, lw=2.4, zorder=4)
        ax.annotate(f"median {np.median(vals):.2f}", (x + 0.22, np.median(vals)),
                    va="center", fontsize=10, color=INK)
    ax.text(0, 0.05, "23/23 matched CsCl", ha="center", color=GOOD, fontweight="bold", fontsize=11)
    ax.text(1, 0.80, "23/23 library_unmatched", ha="center", color=CRIT, fontweight="bold", fontsize=11)
    ax.text(1.46, CUT, "accept cutoff D=0.40", va="center", ha="right", fontsize=9.5,
            color=INK2, style="italic")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CsCl IN library", "CsCl REMOVED"])
    ax.set_xlim(-0.4, 1.5); ax.set_ylim(0, 0.85)
    ax.set_ylabel("match distance  D  (lower = closer)")
    ax.set_title("Leave-CsCl-out holdout: the same 23 CsCl domains\nmatch when CsCl is present, go unknown when it is removed",
                 fontsize=12, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(FIG / "1_cscl_holdout.png", dpi=200); plt.close(fig)


# ---------- Fig 2: Steven identification ----------
def fig_steven():
    st = pd.read_csv(HERE / "steven_match_results.csv")
    fams = ["CsCl", "Th3P4", "Cu3Au", "Ni3Sn", "zangenite", "Cr3Si_A15", "rocksalt_B1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (_, r) in zip(axes, st.iterrows()):
        d = {f: r[f"D_{f}"] for f in fams}
        order = sorted(d, key=d.get)
        vals = [d[f] for f in order]
        cols = [GOOD if f == r.matched else MUTED for f in order]
        y = np.arange(len(order))[::-1]
        ax.barh(y, vals, color=cols, height=0.62, zorder=3)
        cutline(ax, horizontal=False)
        for yi, f, v in zip(y, order, vals):
            ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9,
                    color=INK if f == r.matched else INK2)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10)
        ax.get_yticklabels()[list(order).index(r.matched)].set_fontweight("bold")
        ax.set_title(f"{r.candidate}  →  {r.matched}", fontsize=12, loc="left", color=INK)
        ax.set_xlabel("distance D to each reference"); ax.set_xlim(0, 1.35)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    axes[1].text(CUT + 0.01, 6.3, "accept cutoff", fontsize=9, color=INK2, style="italic")
    fig.suptitle("Steven examples: matched to their nearest reference, well inside the cutoff, with a clear margin",
                 fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG / "2_steven_identification.png", dpi=200); plt.close(fig)


# ---------- Fig 3: open-set decision summary (all validation cases) ----------
def fig_openset():
    h = pd.read_csv(HERE / "cscl_holdout_results.csv")
    a = pd.read_csv(HERE / "amorphous_negative_control.csv")
    st = pd.read_csv(HERE / "steven_match_results.csv")
    rows = [
        ("CsCl in library", h[h.condition == "full_library"].D_best.values, GOOD, "matched"),
        ("Steven 1 / 2", st.D_best.values, GOOD, "matched"),
        ("amorphous", a.D_best.values, CRIT, "unmatched"),
        ("CsCl removed", h[h.condition == "cscl_removed"].D_best.values, CRIT, "unmatched"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.7))
    ax.axvspan(0, CUT, color=GOOD, alpha=0.06); ax.axvspan(CUT, 0.85, color=CRIT, alpha=0.06)
    cutline(ax, horizontal=False)
    rng = np.random.default_rng(1)
    for i, (label, vals, col, _) in enumerate(rows):
        y = len(rows) - 1 - i + rng.uniform(-0.11, 0.11, len(vals))
        ax.scatter(vals, y, s=46, color=col, edgecolor=SURF, linewidth=0.8, zorder=3, alpha=0.9)
    ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xlim(0, 0.85); ax.set_ylim(-0.5, 4.0); ax.set_xlabel("match distance  D")
    ax.text(CUT, 3.72, "cutoff 0.40", ha="center", va="bottom", fontsize=9, color=INK2, style="italic")
    ax.text(CUT - 0.015, 3.32, "MATCHED  ◄", ha="right", color=GOOD, fontweight="bold", fontsize=10)
    ax.text(CUT + 0.015, 3.32, "►  UNKNOWN / OOD", ha="left", color=CRIT, fontweight="bold", fontsize=10)
    ax.set_title("Open-set decision across all validation cases:\nknown structures accept, novel / amorphous abstain",
                 fontsize=12, loc="left", pad=12)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout(); fig.savefig(FIG / "3_openset_decision.png", dpi=200); plt.close(fig)


# ---------- Fig 4: why not per-family std-dev (envelope-width confound) ----------
def fig_confound():
    t = pd.read_csv(HERE / "calibration_thresholds.csv")
    t = t.sort_values("tau_sk_shape")
    fams = t.family.values; vals = t.tau_sk_shape.values
    # colour by whether it's a well-ordered (tight) vs marginally-stable (wide) family
    cols = [CRIT if v > 0.6 else GOOD for v in vals]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    y = np.arange(len(fams))[::-1]
    ax.barh(y, vals, color=cols, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(fams, fontsize=10)
    ax.set_xlabel("per-family acceptance-envelope width  τ  (S(k) channel)")
    ax.set_title("Why a per-family std-dev fails: envelope width is confounded with stability\n"
                 "well-ordered CsCl is tight; marginally-stable Cr3Si/rocksalt are ~10× wider",
                 fontsize=11.5, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout(); fig.savefig(FIG / "4_envelope_confound.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_holdout(); fig_steven(); fig_openset(); fig_confound()
    print("wrote:", *[p.name for p in sorted(FIG.glob("*.png"))])
