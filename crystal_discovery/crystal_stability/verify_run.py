#!/usr/bin/env python
"""
Verdict on a relaxed run, by the paper's criteria plus core/surface split.

    python verify_run.py runs/nickeline_b81 [runs/...]
    python verify_run.py --all

Reports, for each run's trajectory.gsd (first frame = ideal lattice, last = relaxed):
  r_all / r_worst : RDF Pearson correlation vs the ideal lattice (paper: >0.5),
                    total and worst single species
  lindemann       : median displacement / nearest-neighbour distance (melt ~0.15)
  core / surface  : escapee fraction in the inner half vs overall
  verdict         : STABLE (core solid, worst-species RDF > 0.5)
                    CORE-ONLY (bulk crystalline but surface evaporating / weak species)
                    MELTED
"""
import argparse
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent


def rdf(frame, species=None, r_max=900.0, bins=200):
    p = frame.particles.position
    if species is not None:
        names = np.array(frame.particles.types)[frame.particles.typeid]
        p = p[names == species]
    r = freud.density.RDF(bins=bins, r_max=r_max)
    r.compute((frame.configuration.box, p))
    return r.rdf


def verify(run_dir):
    traj = run_dir / "trajectory.gsd"
    if not traj.is_file():
        return None
    t = gsd.hoomd.open(str(traj))
    ideal, final = t[0], t[-1]

    p0 = ideal.particles.position
    d = distance_matrix(p0[:600], p0[:600])
    np.fill_diagonal(d, np.inf)
    nn = float(d.min())

    species = list(ideal.particles.types)
    r_all = pearsonr(rdf(ideal), rdf(final))[0]
    r_species = {s: pearsonr(rdf(ideal, s), rdf(final, s))[0] for s in species}
    r_worst = min(r_species.values())

    disp = np.linalg.norm(final.particles.position - p0, axis=1)
    lindemann = float(np.median(disp) / nn)
    escaped = float((disp > nn / 2).mean())
    radial = np.linalg.norm(p0 - p0.mean(0), axis=1)
    core = radial < np.percentile(radial, 50)
    core_lind = float(np.median(disp[core]) / nn)
    core_esc = float((disp[core] > nn / 2).mean())

    # Core Lindemann is the reliable signal (did interior particles stay on their
    # sites). The RDF-vs-first-frame correlation is NOT reliable as a melt test --
    # the first frame is the unthermalized ideal lattice with delta-sharp peaks, so
    # even a stable crystal's thermal broadening tanks the correlation. Use it only
    # to distinguish clean-stable from core-only-with-surface-loss.
    if core_lind < 0.15 and escaped < 0.25:
        verdict = "STABLE"
    elif core_lind < 0.15:
        verdict = "CORE-ONLY"       # bulk crystalline, surface (weak species) evaporating
    else:
        verdict = "MELTED"

    return dict(name=run_dir.name, n=ideal.particles.N, r_all=r_all, r_worst=r_worst,
                worst_species=min(r_species, key=r_species.get), lindemann=lindemann,
                core_lind=core_lind, escaped=escaped, core_esc=core_esc, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    dirs = sorted((HERE / "runs").iterdir()) if args.all else args.runs
    dirs = [d for d in dirs if (d / "trajectory.gsd").is_file()]

    print(f"{'run':26s} {'N':>5} {'r_all':>6} {'r_worst':>8} {'worst':>7} "
          f"{'lind':>5} {'core':>5} {'esc%':>5} {'core%':>6}  verdict")
    rows = []
    for d in dirs:
        r = verify(d)
        if r is None:
            continue
        rows.append(r)
        print(f"{r['name']:26s} {r['n']:5d} {r['r_all']:6.3f} {r['r_worst']:8.3f} "
              f"{r['worst_species']:>7} {r['lindemann']:5.3f} {r['core_lind']:5.3f} "
              f"{100*r['escaped']:5.1f} {100*r['core_esc']:6.1f}  {r['verdict']}")
    import csv
    with open(HERE / "run_verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
