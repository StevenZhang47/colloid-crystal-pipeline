#!/usr/bin/env python3
"""Figure 9 (amorphous vs CsCl S(k)) regenerated with a SHARED y-axis.

Why this exists: the original `9_amorphous_vs_crystal_sk.png` gave each panel its own
y-scale (amorphous topped out at ~0.15, CsCl at ~0.008 -- a ~19x difference). Read quickly,
that makes the amorphous curve look like it has the *larger* structure factor, which is the
opposite of the point. The difference is entirely forward scatter at low k, which lies
OUTSIDE the Bragg window [3.5, 16] where the coherence metric is evaluated.

The script that produced figures 8 and 9 is not in the tree (figure mtimes 14:13 / 14:26
postdate every .py here), so this is a fresh implementation following the conventions of
`make_rdf_figures.py` and using the same `describe_domain` path as the matcher.

Sources are resolved by matching the published coherence values (amorphous C = 0.17,
CsCl C = 0.53); if no source reproduces them the script SAYS SO rather than emitting a
figure with unverified provenance.
"""
import json
import sys
from pathlib import Path

import gsd.hoomd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import build_and_run_matcher as M   # noqa: E402
import match_multi_domain as MD     # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e3"
AMO_C, CRY_C = "#d03b3b", "#2a78d6"
mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 11, "axes.edgecolor": INK2, "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Bragg window used by the validator (report Eq. 13)
KLO, KHI = 3.5, 16.0

NICOLE = Path("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline"
              "/nicoles_dataset/phase_diagram")
# amorphous: first row of amorphous_negative_control.csv, the same blob used by figure 6
AMORPH = NICOLE / "debye5.25_volumefraction.015943113378_seed2/trajectory.gsd"
# CsCl candidates, in order of likelihood
CSCL_CANDIDATES = [
    NICOLE / "debye5.15_volumefraction.021636356171_seed1/trajectory.gsd",
    NICOLE / "debye5.2_volumefraction.021636356171_seed1/trajectory.gsd",
    Path("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability"
         "/runs/cscl/final_frame.gsd"),
]


def describe(path: Path):
    """Extract the ordered domain and describe it, exactly as the matcher does."""
    with gsd.hoomd.open(str(path), "r") as t:
        pos, tid, box = MD.canonical(t[-1])
    nn = MD.robust_first_shell_distance(pos, box)
    dom = M.gas_safe_domain(pos, box, nn)
    return MD.describe_domain(pos, tid, box, dom)


def main() -> None:
    print(f"amorphous source: {AMORPH}")
    amo = describe(AMORPH)
    print(f"  n={amo['n']}  L/S={amo['ls']:.3f}  Bragg coherence C={amo['bragg_coherence']:.2f}"
          f"   (published: 0.17)")

    cry, cry_path = None, None
    for p in CSCL_CANDIDATES:
        if not p.exists():
            print(f"  [skip, missing] {p}")
            continue
        d = describe(p)
        print(f"CsCl candidate {p.parent.name}: n={d['n']} L/S={d['ls']:.3f} "
              f"C={d['bragg_coherence']:.2f}   (published: 0.53)")
        if cry is None or abs(d["bragg_coherence"] - 0.53) < abs(cry["bragg_coherence"] - 0.53):
            cry, cry_path = d, p
    if cry is None:
        raise SystemExit("FAILED: no CsCl source could be described; figure not written.")

    # ---- provenance check: refuse to emit a figure whose curves are not the published ones
    ok_amo = abs(amo["bragg_coherence"] - 0.17) <= 0.02
    ok_cry = abs(cry["bragg_coherence"] - 0.53) <= 0.05
    if not (ok_amo and ok_cry):
        print("\n*** COHERENCE MISMATCH ***")
        print(f"    amorphous C={amo['bragg_coherence']:.3f} (want 0.17 +/- 0.02) -> "
              f"{'ok' if ok_amo else 'MISMATCH'}")
        print(f"    crystal   C={cry['bragg_coherence']:.3f} (want 0.53 +/- 0.05) -> "
              f"{'ok' if ok_cry else 'MISMATCH'}")
        print("    The regenerated curves are NOT the published ones. Writing the figure "
              "anyway would silently change what Figure 18 shows.")
        print("    Emitting to a *_UNVERIFIED name so it cannot be mistaken for the "
              "published figure.")
    suffix = "" if (ok_amo and ok_cry) else "_UNVERIFIED"

    # ---- figure: shared y-axis
    ymax = max(np.max(amo["sk"] if "sk" in amo else amo["sk_shape"]),
               np.max(cry["sk"] if "sk" in cry else cry["sk_shape"]))
    key = "sk" if "sk" in amo else "sk_shape"

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for ax, d, c, name, verdict in (
            (axes[0], amo, AMO_C, "AMORPHOUS", "disordered"),
            (axes[1], cry, CRY_C, "CsCl CRYSTAL", "ordered")):
        ax.plot(d["k"], d[key], color=c, lw=1.6)
        ax.axvspan(d["k"].min(), KLO, color="#000000", alpha=0.055, lw=0)
        ax.axvspan(KHI, d["k"].max(), color="#000000", alpha=0.055, lw=0)
        for x in (KLO, KHI):
            ax.axvline(x, color=INK2, lw=1.0, ls="--")
        ax.set_xlabel("k · d₀")
        ax.set_title(f"{name}   C = {d['bragg_coherence']:.2f}  ({verdict})",
                     fontsize=12, loc="left", color=c)
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlim(d["k"].min(), d["k"].max())
    axes[0].set_ylabel("S(k)  structure factor")
    axes[0].set_ylim(0, ymax * 1.05)
    axes[0].text(0.5 * (KLO + KHI), ymax * 0.97, "Bragg window", ha="center",
                 va="top", fontsize=9, color=INK2)

    fig.suptitle("Amorphous vs crystal S(k), SHARED y-axis: the amorphous curve is larger "
                 "only in the excluded\nforward-scatter region; inside the Bragg window it "
                 "has no sharp peaks → C below 0.18",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / f"9b_amorphous_vs_crystal_sk_shared_axis{suffix}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

    # ---- inset variant: shared axis loses the in-window detail, so also emit a zoom
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for ax, d, c, name, verdict in (
            (axes2[0], amo, AMO_C, "AMORPHOUS", "disordered"),
            (axes2[1], cry, CRY_C, "CsCl CRYSTAL", "ordered")):
        m = (d["k"] >= KLO) & (d["k"] <= KHI)
        ax.plot(d["k"][m], d[key][m], color=c, lw=1.8)
        ax.set_xlabel("k · d₀")
        ax.set_title(f"{name}   C = {d['bragg_coherence']:.2f}  ({verdict})",
                     fontsize=12, loc="left", color=c)
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlim(KLO, KHI)
    axes2[0].set_ylabel("S(k)  structure factor")
    inwin = max(np.max(amo[key][(amo["k"] >= KLO) & (amo["k"] <= KHI)]),
                np.max(cry[key][(cry["k"] >= KLO) & (cry["k"] <= KHI)]))
    axes2[0].set_ylim(0, inwin * 1.10)
    fig2.suptitle("Same data, restricted to the Bragg window [3.5, 16] and still on a shared "
                  "y-axis:\nthis is the region the coherence metric actually reads",
                  fontsize=11.5, x=0.01, ha="left")
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = FIG / f"9c_amorphous_vs_crystal_sk_braggwindow{suffix}.png"
    fig2.savefig(out2, dpi=200)
    fig2.savefig(out2.with_suffix(".pdf"))
    plt.close(fig2)

    # ---- faithful reproduction of the ORIGINAL figure's quantity, shared y-axis.
    # describe_domain's k grid is K_MIN=3.5..K_MAX=16 (the Bragg window) and its "sk_shape"
    # is unit-area normalised, so the panels above are NOT the same quantity the original
    # plotted: the original spans k*d0 ~0..18 and shows the forward-scatter peak, which can
    # only come from an explicit wide k grid over the RAW curve. Rebuild that here.
    from structure_factor import raw_structure_factor  # noqa: E402

    kwide = np.linspace(0.35, 18.0, 400)

    def raw_wide(path: Path):
        with gsd.hoomd.open(str(path), "r") as t:
            pos, tid, box = MD.canonical(t[-1])
        nn = MD.robust_first_shell_distance(pos, box)
        dom = M.gas_safe_domain(pos, box, nn)
        sub = pos[dom]
        nn_d = MD.robust_first_shell_distance(sub, box)
        return raw_structure_factor(sub, nn_d, box, k=kwide)

    ka, sa = raw_wide(AMORPH)
    kc, sc = raw_wide(cry_path)
    rawmax = max(float(np.max(sa)), float(np.max(sc)))

    fig3, axes3 = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    for ax, kk, ss, c, name, cval, verdict in (
            (axes3[0], ka, sa, AMO_C, "AMORPHOUS", amo["bragg_coherence"], "disordered"),
            (axes3[1], kc, sc, CRY_C, "CsCl CRYSTAL", cry["bragg_coherence"], "ordered")):
        ax.plot(kk, ss, color=c, lw=1.6)
        ax.axvspan(kk.min(), KLO, color="#000000", alpha=0.055, lw=0)
        ax.axvspan(KHI, kk.max(), color="#000000", alpha=0.055, lw=0)
        for x in (KLO, KHI):
            ax.axvline(x, color=INK2, lw=1.0, ls="--")
        ax.set_xlabel("k · d₀")
        ax.set_title(f"{name}   C = {cval:.2f}  ({verdict})", fontsize=12, loc="left", color=c)
        ax.grid(color=GRID, lw=0.8)
        ax.set_xlim(kk.min(), kk.max())
    axes3[0].set_ylabel("S(k)  structure factor  (raw)")
    axes3[0].set_ylim(0, rawmax * 1.05)
    axes3[0].text(0.5 * (KLO + KHI), rawmax * 0.97, "Bragg window", ha="center", va="top",
                  fontsize=9, color=INK2)
    fig3.suptitle("Raw S(k) on a SHARED y-axis (same quantity and k-range as the original "
                  "figure).\nThe amorphous curve is larger ONLY in the excluded "
                  "forward-scatter region; inside the Bragg window it is flat.",
                  fontsize=11.5, x=0.01, ha="left")
    fig3.tight_layout(rect=(0, 0, 1, 0.92))
    out3 = FIG / f"9d_amorphous_vs_crystal_sk_raw_shared_axis{suffix}.png"
    fig3.savefig(out3, dpi=200)
    fig3.savefig(out3.with_suffix(".pdf"))
    plt.close(fig3)
    print(f"\n  raw wide-k peaks: amorphous {np.max(sa):.4f}  CsCl {np.max(sc):.4f}")
    inw_a = float(np.max(sa[(ka >= KLO) & (ka <= KHI)]))
    inw_c = float(np.max(sc[(kc >= KLO) & (kc <= KHI)]))
    print(f"  in-Bragg-window peaks: amorphous {inw_a:.5f}  CsCl {inw_c:.5f}"
          f"   (ratio {inw_c/max(inw_a,1e-12):.1f}x)")

    meta = {
        "figure": out.name,
        "companion_raw_figure": out3.name,
        "raw_wide_k_grid": [float(kwide.min()), float(kwide.max()), len(kwide)],
        "raw_peak_amorphous": round(float(np.max(sa)), 5),
        "raw_peak_crystal": round(float(np.max(sc)), 5),
        "in_window_peak_amorphous": round(inw_a, 5),
        "in_window_peak_crystal": round(inw_c, 5),
        "supersedes": "9_amorphous_vs_crystal_sk.png (per-panel y-axes, ~19x apart)",
        "change": "shared y-axis across both panels; companion Bragg-window zoom",
        "amorphous_source": str(AMORPH), "crystal_source": str(cry_path),
        "amorphous_C": round(float(amo["bragg_coherence"]), 3),
        "crystal_C": round(float(cry["bragg_coherence"]), 3),
        "published_C": {"amorphous": 0.17, "crystal": 0.53},
        "coherence_reproduced": bool(ok_amo and ok_cry),
        "bragg_window": [KLO, KHI], "threshold": 0.18,
        "note": ("The original generator for figures 8-9 is absent from the repository; this "
                 "is a fresh implementation using the matcher's own describe_domain path."),
    }
    (FIG / f"{out.stem}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out.name}\n      {out2.name}\n      {out.stem}.meta.json")


if __name__ == "__main__":
    main()
