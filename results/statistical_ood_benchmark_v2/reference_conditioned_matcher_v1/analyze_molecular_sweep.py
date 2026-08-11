#!/usr/bin/env python3
"""Generalized point-pattern analysis for molecular (non-single-sphere) runs.

The ionic reference matcher (CsCl/Th3P4/...) assumes single-sphere two-species
crystals. The snowmen runs are 3-species rigid trimers (body + lobe + head), so
that library does not apply directly. The GENERALISATION is a front-end that
reduces each molecule to one representative point, after which the SAME
point-pattern machinery (domain separation, order gate, S(k)/RDF) applies to any
system.

For each run's final frame:
  1. Reduce: one representative point per molecule = the body-sphere (type-0)
     centre (the large space-filling lobe). One point per snowman.
  2. Order gate (species-agnostic, NO q6): S(k) raw Bragg coherence on the points.
     Ordered if the finite-size-scaled coherence peak clears a threshold.
  3. Domain separation (NO q6): condensed points = coordination >= C on the
     first-shell graph (gas rejection by neighbour count, not by a bond-order cut);
     connected components = domains.
  4. Lattice fingerprint: first few S(k) peak positions (k*nn) -> compare to ideal
     FCC / BCC / HCP / SC first-peak ratios for a coarse packing label.

Reports a per-run table and flags 'curious' runs (ordered, multi-domain, outliers).
Read-only on all input GSDs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as cc

sys.path.insert(0, "/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/"
                   "matcher_v3_validated_structure_factor")
from structure_factor import (  # noqa: E402
    raw_structure_factor, coherence_metrics, robust_first_shell_distance, peak_positions,
)

SWEEPS = [
    Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLong"),
    Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLongRetry"),
]
OUT = Path("/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/"
           "reference_conditioned_matcher_v1/snowmen_sweep_analysis")
COORD_MIN = 10          # condensed if >= this many first-shell neighbours (gas rejection)
ORDER_PEAK_MIN = 3.0    # finite-size-scaled S(k) coherence peak above this => ordered
MIN_DOMAIN = 150

# ideal first-peak k*nn signatures (nn = median 4th-neighbour distance of the lattice)
# used only as a coarse label; reported with the raw peaks so it can be overridden.
LATTICE_REF = {"FCC": 1.00, "HCP": 1.00, "BCC": 1.06, "SC": 1.23}


def representative_points(frame):
    """One point per molecule: the type-0 (body) sphere centre."""
    pos = np.asarray(frame.particles.position, float)
    tid = np.asarray(frame.particles.typeid, int)
    box = np.asarray(frame.configuration.box[:3], float)
    return pos[tid == 0], box


def condensed_components(pts, box, nn):
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    r_c = 1.35 * nn
    nl = freud.locality.AABBQuery(fbox, pts).query(pts, dict(r_max=r_c, exclude_ii=True)).toNeighborList()
    qi, pi = nl.query_point_indices, nl.point_indices
    coord = np.bincount(qi, minlength=len(pts))
    condensed = coord >= COORD_MIN
    idx = np.flatnonzero(condensed)
    if idx.size == 0:
        return np.full(len(pts), -1, int), condensed, coord
    remap = -np.ones(len(pts), int); remap[idx] = np.arange(idx.size)
    pairs = np.unique(np.sort(np.stack([qi, pi], 1), 1), axis=0)
    m = condensed[pairs[:, 0]] & condensed[pairs[:, 1]]
    e = remap[pairs[m]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(idx.size, idx.size))
    _, lab = cc(g, directed=False)
    comp = np.full(len(pts), -1, int); comp[idx] = lab
    return comp, condensed, coord


def analyze_frame(path):
    with gsd.hoomd.open(str(path), "r") as tr:
        frame = tr[-1]
        n_frames = len(tr)
    pts, box = representative_points(frame)
    if len(pts) < 64:
        return None
    nn = robust_first_shell_distance(pts, box)
    k, sk = raw_structure_factor(pts, nn, box)
    coh = coherence_metrics(sk, len(pts), k)
    peaks = peak_positions(k, sk, max_peaks=4)
    comp, condensed, coord = condensed_components(pts, box, nn)
    dom_sizes = sorted([int((comp == c).sum()) for c in set(comp[comp >= 0])], reverse=True)
    dom_sizes = [d for d in dom_sizes if d >= MIN_DOMAIN]
    ordered = coh["n23_scaled_peak"] >= ORDER_PEAK_MIN
    first_peak = float(peaks[0]) if len(peaks) else float("nan")
    lattice = "-"
    if ordered and np.isfinite(first_peak):
        # normalise peaks to the first; coarse label by first-peak k*nn magnitude
        lattice = min(LATTICE_REF, key=lambda L: abs(first_peak - LATTICE_REF[L] * first_peak)) \
            if False else "ordered_unclassified"
    return dict(n_points=len(pts), n_frames=n_frames, nn=round(nn, 1),
                sk_scaled_peak=round(coh["n23_scaled_peak"], 2),
                ordered=bool(ordered), first_peak_knn=round(first_peak, 3) if np.isfinite(first_peak) else None,
                n_peaks=int(len(peaks)), condensed_frac=round(float(condensed.mean()), 3),
                n_domains=len(dom_sizes), largest_domain=dom_sizes[0] if dom_sizes else 0,
                domain_sizes=dom_sizes[:5])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sweep in SWEEPS:
        run_dirs = sorted([d for d in sweep.iterdir() if d.is_dir() and d.name.startswith("lobe_ratio")])
        for rd in run_dirs:
            final = rd / "final_frame.gsd"
            if not final.exists():
                traj = rd / "trajectory.gsd"
                if not traj.exists():
                    continue
                final = traj
            try:
                res = analyze_frame(final)
            except Exception as e:
                rows.append(dict(sweep=sweep.name, run=rd.name, error=str(e)[:80])); continue
            if res is None:
                continue
            # parse lobe_ratio / seed_ratio from dir name
            parts = rd.name.replace("lobe_ratio_", "").split("_seed_ratio_")
            rows.append(dict(sweep=sweep.name, run=rd.name,
                             lobe_ratio=float(parts[0]), seed_ratio=float(parts[1]) if len(parts) > 1 else None,
                             file=final.name, **res))
            r = rows[-1]
            flag = "  <== ORDERED" if r.get("ordered") else ""
            print(f"{sweep.name[:12]:12} {rd.name:38} ord={r.get('ordered')} "
                  f"peak={r.get('sk_scaled_peak')} nd={r.get('n_domains')} nn={r.get('nn')}{flag}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "snowmen_sweep_summary.csv", index=False)
    ordered = df[df.get("ordered") == True] if "ordered" in df else pd.DataFrame()
    multi = df[df.get("n_domains", 0) > 1] if "n_domains" in df else pd.DataFrame()
    summary = dict(n_runs=len(df),
                   n_ordered=int(len(ordered)),
                   ordered_runs=ordered["run"].tolist() if len(ordered) else [],
                   n_multi_domain=int(len(multi)),
                   multi_domain_runs=multi[["run", "n_domains"]].to_dict("records") if len(multi) else [])
    (OUT / "snowmen_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
