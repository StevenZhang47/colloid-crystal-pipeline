#!/usr/bin/env python3
"""Pick, for each thermalised anchor, one frame that is relaxed but still crystalline.

WHY THIS EXISTS
---------------
`generate_crystals.py` stops after `pacsim-create`. That optimises the LATTICE CONSTANTS
at zero temperature; it runs no dynamics. Every `runs/lib_*/first_frame.gsd` is therefore
a mathematically perfect lattice -- on lib_cscl the nearest-neighbour distance has a
standard deviation of 4e-5 nm, i.e. every particle sits exactly on its site. The matcher
compares 298 K candidates against those 0 K anchors, so part of every reported distance
is a temperature gap and not a structural difference.

`thermalize_anchors.sbatch` runs each anchor's own run.yaml -- the same 30 K -> 298 K ramp
over 500k steps that produced the candidates. This script then chooses which frame of that
trajectory becomes the anchor.

THE RULE IS DECLARED HERE, BEFORE ANY MATCHER RESULT IS SEEN
------------------------------------------------------------
An anchor frame chosen to make the matcher look good is not evidence about the matcher. So
the gates below are fixed in advance, are the same for all 42 anchors, and none of them
refers to any matcher distance.

A frame is ADMISSIBLE when all five hold:

  1. frame_index >= RAMP_DONE_FRAME     the temperature ramp has finished, so the frame is
                                        drawn from the equilibrated 298 K segment
  2. edge_retention   >= 0.90           at least 90% of the frame-0 contact edges survive.
                                        This is the melting test: a liquid loses its
                                        neighbour identities, a warm crystal keeps them.
  3. bond_strain_rms  <= 0.15           the Lindemann criterion. RMS fractional change of
                                        the frame-0 contact bond lengths. 0.15 is the
                                        textbook melting value, not a fitted number.
  4. gas_frac         <= 0.02           at most 2% of particles have sublimated away
                                        (frozen gas screen). This is the "exploding" test.
  5. largest_comp_frac>= 0.95           the crystal has not fragmented into pieces.

Among the admissible frames we take the MEDIAN FRAME INDEX -- deliberately not the best
scoring one. Taking the best would be choosing the least thermal frame, which reintroduces
the very 0 K bias this is meant to remove.

An anchor with NO admissible frame did not survive thermalisation. It is reported as
`FAILED` and must be excluded from the library rather than quietly represented by its 0 K
lattice, because the pair potential does not actually hold that structure together at room
temperature -- which is a real result about the prototype, not a nuisance.

Writes anchor_frame_selection.csv (every probed frame) and anchor_frames.json (the choice).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CS = ROOT / "crystal_discovery/crystal_stability"
sys.path.insert(0, str(ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))

import core3_pipeline as C3  # noqa: E402

# ---- the declared rule -------------------------------------------------------------
RAMP_DONE_FRAME = 200      # run.yaml: final_update_step 200000, trajectory_interval 1000
PROBE_STRIDE = 10          # probe every 10th frame from RAMP_DONE_FRAME on
M_CONTACT = 1.35           # frozen contact-graph cutoff, s = d_ij / (R_i + R_j)
MIN_EDGE_RETENTION = 0.90
MAX_BOND_STRAIN = 0.15     # Lindemann
MAX_GAS_FRAC = 0.02
MIN_LARGEST_COMP = 0.95
# ------------------------------------------------------------------------------------


def contact_edges(pos: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs with d_ij < M_CONTACT * (R_i + R_j), and their distances.

    The crystal sits well inside a walled box (extent 3764 nm inside a 6054 nm box on
    lib_cscl), so no contact can wrap the boundary and a plain KD-tree is exact here.
    """
    rmax = M_CONTACT * 2.0 * radii.max()
    tree = cKDTree(pos)
    pairs = np.asarray(sorted(tree.query_pairs(rmax)), dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        return pairs, np.zeros(0)
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    keep = d < M_CONTACT * (radii[pairs[:, 0]] + radii[pairs[:, 1]])
    return pairs[keep], d[keep]


def probe(run_dir: Path) -> list[dict]:
    traj = run_dir / "trajectory.gsd"
    if not traj.is_file():
        return []
    import gsd.hoomd
    with gsd.hoomd.open(str(traj), mode="r") as t:
        n_frames = len(t)

    f0 = C3.read_frame(run_dir / "first_frame.gsd", 0)
    radii = f0.diameters / 2.0
    e0, d0 = contact_edges(f0.positions, radii)
    key0 = set(map(tuple, e0))
    n = len(f0.positions)

    rows = []
    for fi in range(RAMP_DONE_FRAME, n_frames, PROBE_STRIDE):
        f = C3.read_frame(traj, fi)
        et, _ = contact_edges(f.positions, radii)
        retention = len(key0 & set(map(tuple, et))) / max(len(key0), 1)

        dt = np.linalg.norm(f.positions[e0[:, 0]] - f.positions[e0[:, 1]], axis=1)
        # NON-AFFINE bond strain. A warm crystal contracts or expands relative to its 0 K
        # lattice, and that uniform part is elastic relaxation, not disorder -- it is
        # exactly what we WANT the anchor to have done. Measuring |d_t - d_0| directly
        # counted it as melting: six anchors that kept >92% of their contact topology
        # (fluorite 0.9238, antifluorite 0.9247, zangenite 0.9269, nickeline 0.9495) were
        # failed on strain alone. So fit the best uniform scale first and measure only
        # what is left over.
        scale = float(np.dot(dt, d0) / np.dot(d0, d0)) if len(d0) else 1.0
        ref = scale * d0
        strain = float(np.sqrt(np.mean(((dt - ref) / ref) ** 2))) if len(d0) else np.nan

        prep = C3.preprocess(f)
        gas_frac = float(1.0 - prep.retain_mask.mean())
        keep = prep.component_id[prep.retain_mask]
        largest = float(np.bincount(keep[keep >= 0]).max() / n) if len(keep) else 0.0

        rows.append(dict(run=run_dir.name, frame=fi, n=n,
                         edge_retention=round(retention, 4),
                         lattice_scale=round(scale, 4),
                         bond_strain_rms=round(strain, 4),
                         gas_frac=round(gas_frac, 4),
                         largest_comp_frac=round(largest, 4)))
    return rows


def admissible(r: dict) -> bool:
    return (r["frame"] >= RAMP_DONE_FRAME
            and r["edge_retention"] >= MIN_EDGE_RETENTION
            and r["bond_strain_rms"] <= MAX_BOND_STRAIN
            and r["gas_frac"] <= MAX_GAS_FRAC
            and r["largest_comp_frac"] >= MIN_LARGEST_COMP)


def main() -> None:
    all_rows, choice = [], {}
    for run_dir in sorted((CS / "runs").glob("lib_*")):
        rows = probe(run_dir)
        if not rows:
            choice[run_dir.name] = {"status": "NO_TRAJECTORY"}
            print(f"{run_dir.name:44s} NO TRAJECTORY", flush=True)
            continue
        all_rows += rows
        ok = [r for r in rows if admissible(r)]
        if not ok:
            best = max(rows, key=lambda r: r["edge_retention"])
            choice[run_dir.name] = {
                "status": "FAILED", "reason": "no frame passed the declared gates",
                "best_edge_retention": best["edge_retention"],
                "best_bond_strain_rms": best["bond_strain_rms"],
                "best_gas_frac": best["gas_frac"]}
            print(f"{run_dir.name:44s} FAILED  best retention {best['edge_retention']:.3f} "
                  f"strain {best['bond_strain_rms']:.3f} gas {best['gas_frac']:.3f}", flush=True)
            continue
        pick = ok[len(ok) // 2]                       # median admissible frame index
        choice[run_dir.name] = {"status": "OK", "frame": pick["frame"],
                                "n_admissible": len(ok), "n_probed": len(rows),
                                **{k: pick[k] for k in
                                   ("edge_retention", "bond_strain_rms", "gas_frac",
                                    "largest_comp_frac")}}
        print(f"{run_dir.name:44s} frame {pick['frame']:4d}  "
              f"retention {pick['edge_retention']:.3f}  strain {pick['bond_strain_rms']:.3f}  "
              f"gas {pick['gas_frac']:.3f}  ({len(ok)}/{len(rows)} admissible)", flush=True)

    pd.DataFrame(all_rows).to_csv(CS / "anchor_frame_selection.csv", index=False)
    (CS / "anchor_frames.json").write_text(json.dumps(
        {"rule": {"ramp_done_frame": RAMP_DONE_FRAME, "probe_stride": PROBE_STRIDE,
                  "min_edge_retention": MIN_EDGE_RETENTION,
                  "max_bond_strain_rms": MAX_BOND_STRAIN,
                  "max_gas_frac": MAX_GAS_FRAC, "min_largest_comp_frac": MIN_LARGEST_COMP,
                  "pick": "median admissible frame index"},
         "anchors": choice}, indent=2) + "\n")

    n_ok = sum(v["status"] == "OK" for v in choice.values())
    print(f"\n{n_ok}/{len(choice)} anchors survived thermalisation")


if __name__ == "__main__":
    main()
