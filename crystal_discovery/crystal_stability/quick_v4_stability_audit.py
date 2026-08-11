#!/usr/bin/env python3
"""Bounded read-only audit of completed periodic-bulk v4 stability runs.

This is the inexpensive companion to ``track_v4_stability.py``.  It sequentially
unwraps every stored frame, but evaluates the expensive q6 and species-aware site
occupancy diagnostics on only ten predeclared frames from the trailing 20% of each
trajectory.  It does not compute RDFs and does not modify any trajectory.
"""
from __future__ import annotations

from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
OUT = HERE / "v4_quick_stability_audit.csv"
THR_LINDEMANN = 0.15
THR_SITE_OCCUPANCY = 0.90
THR_Q6_RETENTION = 0.80


def canonical_typeid(frame) -> np.ndarray:
    typeid = np.asarray(frame.particles.typeid, int)
    diameter = np.asarray(frame.particles.diameter, float)
    means = {int(t): float(diameter[typeid == t].mean()) for t in np.unique(typeid)}
    remap = {old: new for new, old in enumerate(sorted(means, key=lambda t: -means[t]))}
    return np.asarray([remap[int(t)] for t in typeid], int)


def median_fourth_neighbor(position: np.ndarray, box: np.ndarray) -> float:
    wrapped = np.mod(position + box / 2.0, box)
    distances, _ = cKDTree(wrapped, boxsize=box).query(wrapped, k=5)
    return float(np.median(distances[:, 4]))


def q6_mean(position: np.ndarray, box: np.ndarray, d0: float) -> float:
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    order = freud.order.Steinhardt(l=6)
    order.compute((fbox, position), neighbors=dict(r_max=1.4 * d0, exclude_ii=True))
    return float(np.nan_to_num(order.particle_order).mean())


def species_aware_occupancy(
    current: np.ndarray, reference: np.ndarray, typeid: np.ndarray,
    box: np.ndarray, cutoff: float,
) -> float:
    current = np.mod(current + box / 2.0, box)
    reference = np.mod(reference + box / 2.0, box)
    occupied = np.zeros(len(reference), bool)
    for species in np.unique(typeid):
        ids = np.flatnonzero(typeid == species)
        tree = cKDTree(current[ids], boxsize=box)
        distance, _ = tree.query(reference[ids], k=1)
        occupied[ids] = distance < cutoff
    return float(occupied.mean())


def audit_run(run_dir: Path) -> dict:
    trajectory_path = run_dir / "trajectory.gsd"
    with gsd.hoomd.open(str(trajectory_path), "r") as trajectory:
        n_frames = len(trajectory)
        first = trajectory[0]
        n_particles = int(first.particles.N)
        if n_frames < 2:
            return {
                "run": run_dir.name, "n_frames": n_frames,
                "n_particles": n_particles, "complete": False,
                "trailing_lindemann_median": np.nan,
                "trailing_site_occupancy": np.nan,
                "trailing_q6_retention": np.nan,
                "pass_lindemann": False, "pass_site_occupancy": False,
                "pass_q6_retention": False, "verdict": "NUMERICAL_FAILURE_OR_INCOMPLETE",
                "reason": "trajectory contains only its initial frame",
            }

        box = np.asarray(first.configuration.box[:3], float)
        reference = np.asarray(first.particles.position, float)
        typeid = canonical_typeid(first)
        d0 = median_fourth_neighbor(reference, box)
        q6_reference = q6_mean(reference, box, d0)
        start = int(np.floor(0.8 * (n_frames - 1)))
        sampled = set(np.unique(np.linspace(start, n_frames - 1, 10).astype(int)).tolist())

        previous = reference.copy()
        unwrapped = reference.copy()
        rows = []
        numerical_ok = True
        for frame_index in range(1, n_frames):
            frame = trajectory[frame_index]
            current = np.asarray(frame.particles.position, float)
            step = current - previous
            step -= box * np.rint(step / box)
            unwrapped += step
            previous = current
            numerical_ok &= bool(np.isfinite(unwrapped).all())
            if frame_index not in sampled:
                continue
            displacement = ((unwrapped - np.median(unwrapped, axis=0))
                            - (reference - np.median(reference, axis=0)))
            lindemann = float(np.median(np.linalg.norm(displacement, axis=1)) / d0)
            occupancy = species_aware_occupancy(current, reference, typeid, box, 0.5 * d0)
            q6_retention = q6_mean(current, box, d0) / max(q6_reference, 1e-12)
            rows.append((frame_index, lindemann, occupancy, q6_retention))

    values = np.asarray(rows, float)
    lindemann = float(np.median(values[:, 1]))
    occupancy = float(np.median(values[:, 2]))
    q6_retention = float(np.median(values[:, 3]))
    passes = (lindemann < THR_LINDEMANN, occupancy > THR_SITE_OCCUPANCY,
              q6_retention > THR_Q6_RETENTION)
    if not numerical_ok:
        verdict = "NUMERICAL_FAILURE"
        reason = "NaN or Inf occurred after sequential periodic unwrapping"
    elif all(passes) and n_frames >= 500:
        verdict = "STABLE"
        reason = "all three trailing criteria passed"
    elif sum(passes) <= 1 and (lindemann > 2 * THR_LINDEMANN or occupancy < 0.5):
        verdict = "MELTED"
        reason = "at most one criterion passed and the lattice-integrity signal collapsed"
    else:
        verdict = "BORDERLINE"
        reason = "conflicting trailing criteria"
    return {
        "run": run_dir.name, "n_frames": n_frames, "n_particles": n_particles,
        "complete": n_frames >= 500, "d0_nm": d0,
        "sampled_trailing_frames": ";".join(str(int(v)) for v in values[:, 0]),
        "trailing_lindemann_median": lindemann,
        "trailing_site_occupancy": occupancy,
        "trailing_q6_retention": q6_retention,
        "pass_lindemann": passes[0], "pass_site_occupancy": passes[1],
        "pass_q6_retention": passes[2], "verdict": verdict, "reason": reason,
    }


def main() -> None:
    rows = []
    for run_dir in sorted((HERE / "runs").glob("*_v4")):
        if not (run_dir / "trajectory.gsd").exists():
            continue
        row = audit_run(run_dir)
        rows.append(row)
        print(
            f"{row['run']:28s} frames={row['n_frames']:3d} "
            f"lind={row['trailing_lindemann_median']!s:>8} "
            f"occ={row['trailing_site_occupancy']!s:>8} "
            f"q6ret={row['trailing_q6_retention']!s:>8} {row['verdict']}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
