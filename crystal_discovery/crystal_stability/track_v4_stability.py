#!/usr/bin/env python
"""
v4 PERIODIC-BULK stability tracker — incremental and restart-safe.

WHAT THESE TRAJECTORIES ARE
Fully periodic NVT bulk crystals at 298 K. The box is FIXED at the energy-minimised
lattice spacing. There is no barostat: these are NOT constant-pressure runs, so no
verdict here may be phrased in terms of pressure relaxation or volume change. The only
question is whether the LATTICE survives at 298 K.

This design exists because every earlier generation had a boundary artefact:
    v1 small box  -> overpressure       -> crystals looked MORE stable than they are
    v2 large box  -> vacuum sink        -> crystals sublimated regardless of lattice
    v3 + gas      -> wall-launched gas  -> particles ejected to 3e8 nm
A periodic bulk crystal has no surface, no walls and no vacuum, so none of those apply.

METHOD NOTES THAT MATTER
* Coordinates are UNWRAPPED SEQUENTIALLY frame-to-frame (minimum image between
  consecutive frames, accumulated). Unwrapping against frame 0 alone is wrong once a
  particle has moved more than half a box.
* Global drift is removed AFTER unwrapping, using a MEDIAN (robust) rather than a mean:
  a mean COM is dragged by any runaway particle, which would mask real melting.
* Site occupancy is species-aware and uses PERIODIC nearest-neighbour search: a site
  counts as occupied only if a particle OF THE CORRECT SPECIES sits within 0.5*nn.
  Identity retention (same particle on its own site) is reported separately, because a
  crystal can stay perfectly ordered while species swap sites.
* RDF uses the CORRECTED v2 partial_rdf ONLY (shell-volume + centre-count + density
  normalised, with a finite-domain shape correction). The v1 pair-count histogram is
  invalid and is never used.

INCREMENTAL / RESTART-SAFE
* Per-run state is cached in .v4_cache/<run>.json with (n_frames, mtime, size). A run is
  re-analysed only if its trajectory grew or changed.
* Partial trajectories are analysed as far as they go; verdicts on incomplete runs are
  reported as BORDERLINE_OR_STILL_EVOLVING unless already clearly melted.
* All writes are ATOMIC (temp file + os.replace), so an interrupted run cannot corrupt
  the CSVs.
* Trajectory files are opened read-only and never modified. Nothing here touches the
  matcher outputs or interferes with running SLURM jobs.

PRE-REGISTERED VERDICTS (fixed before looking at any result)
Computed on the TRAILING 20% of available frames — never on a single final frame.

STABLE   trailing median Lindemann          < 0.15
     AND trailing median species-aware site occupancy > 0.90
     AND trailing median q6 retention       > 0.80
     AND no sustained degradation (trailing slopes not significantly worsening)
     AND no numerical failure (no NaN/Inf, no absurd displacement)

MELTED   sustained failure of >= 1 structural criterion WITH clear degradation
         (trailing slope worsening, or metric far past threshold)

BORDERLINE_OR_STILL_EVOLVING
         near-threshold, conflicting, or non-plateaued metrics; or run incomplete
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, wasserstein_distance

HERE = Path(__file__).resolve().parent
RDF_LIB = Path("/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/"
               "matcher_v2_corrected_rdf")
sys.path.insert(0, str(RDF_LIB))
from partial_rdf import partial_rdf  # noqa: E402  (CORRECTED v2 implementation ONLY)

OUT_FRAMES = HERE / "v4_per_frame_metrics.csv"
OUT_SUMMARY = HERE / "v4_run_summary.csv"
OUT_DIAG = HERE / "v4_diagnostics"
CACHE = HERE / ".v4_cache"

# ---- pre-registered thresholds (frozen before inspecting results) ----
THR_LINDEMANN = 0.15          # Lindemann melting criterion
THR_SITE_OCC = 0.90           # species-aware site occupancy
THR_Q6_RET = 0.80             # q6 retention vs frame 0
THR_SLOPE_LIND = 0.05         # per-unit-trailing-window worsening that counts as degradation
THR_SLOPE_OCC = -0.05
TRAILING_FRAC = 0.20          # last 20% of frames
MAX_FRAMES_ANALYSED = 120     # sample evenly if a trajectory has more (cost control)
RDF_RMAX_NN = 4.5


# ------------------------------------------------------------------ helpers
def atomic_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def canonical_types(fr):
    """typeid 0 = LARGE, 1 = SMALL (partial_rdf asserts this)."""
    diam = np.asarray(fr.particles.diameter, float)
    tid = np.asarray(fr.particles.typeid, int)
    md = {int(s): float(diam[tid == s].mean()) for s in np.unique(tid)}
    order = sorted(md, key=lambda s: -md[s])
    remap = {o: n for n, o in enumerate(order)}
    return np.array([remap[int(s)] for s in tid], int), diam


def periodic_nn(pos, box):
    w = np.mod(pos + box / 2.0, box)
    d, _ = cKDTree(w, boxsize=box).query(w, k=2)
    return float(np.median(d[:, 1]))


def q6_per_particle(pos, box, nn):
    fb = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    st = freud.order.Steinhardt(l=6)
    st.compute((fb, pos), neighbors=dict(r_max=1.4 * nn, exclude_ii=True))
    return np.nan_to_num(st.particle_order)


def solid_fraction(pos, box, nn, q6):
    """Largest solid-like connected component, as a fraction of all particles."""
    fb = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    nq = dict(r_max=1.4 * nn, exclude_ii=True)
    nl = freud.locality.AABBQuery(fb, pos).query(pos, nq).toNeighborList()
    coord = np.bincount(nl.query_point_indices, minlength=len(pos))
    solid = (q6 > 0.28) & (coord >= 6)
    if solid.sum() < 8:
        return 0.0
    cl = freud.cluster.Cluster()
    cl.compute((fb, pos[solid]), neighbors=nq)
    return float(np.max(np.bincount(cl.cluster_idx)) / len(pos))


def curve_emd(x, a, b):
    a = np.clip(a, 0, None); b = np.clip(b, 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.nan
    return float(wasserstein_distance(x, x, a / a.sum(), b / b.sum()))


def safe_pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b)[0])


def slope(y):
    """Least-squares slope over a normalised x in [0,1]; robust to short series."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return 0.0
    x = np.linspace(0.0, 1.0, len(y))
    return float(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------------ core
def analyse_run(run_dir: Path):
    traj = run_dir / "trajectory.gsd"
    t = gsd.hoomd.open(str(traj), "r")          # read-only; never modified
    n_frames = len(t)
    if n_frames < 2:
        return None, None

    f0 = t[0]
    box = np.asarray(f0.configuration.box[:3], float)
    tid, diam = canonical_types(f0)
    p0_raw = np.asarray(f0.particles.position, float)
    N = len(p0_raw)
    nn0 = periodic_nn(p0_raw, box)
    species = {0: "LARGE", 1: "SMALL"}

    q6_0_pp = q6_per_particle(p0_raw, box, nn0)
    q6_0 = float(q6_0_pp.mean())
    dom = np.ones(N, bool)                      # periodic bulk: the whole box IS the crystal
    r0, gr0 = partial_rdf(p0_raw, tid, dom, nn0, r_max_nn=RDF_RMAX_NN)

    # state_data (energy / temperature); may lag the trajectory
    pe = temp = None
    sd = run_dir / "state_data.csv"
    if sd.exists():
        try:
            s = pd.read_csv(sd)
            pe = s["Potential Energy (kJ/mole)"].to_numpy(float) / N
            temp = s["Temperature (K)"].to_numpy(float)
        except Exception:
            pe = temp = None

    # frames to analyse (evenly sampled if the trajectory is long)
    idx = (np.unique(np.linspace(0, n_frames - 1, MAX_FRAMES_ANALYSED).astype(int))
           if n_frames > MAX_FRAMES_ANALYSED else np.arange(n_frames))

    # --- SEQUENTIAL periodic unwrapping (must walk every frame, not just sampled ones) ---
    rows = []
    prev = p0_raw.copy()
    unwrapped = p0_raw.copy()
    ref0 = p0_raw.copy()
    prev_unwrapped_sampled = None

    for fi in range(n_frames):
        cur = np.asarray(t[fi].particles.position, float)
        if fi > 0:
            d = cur - prev
            d -= box * np.round(d / box)        # minimum image between CONSECUTIVE frames
            unwrapped = unwrapped + d
            prev = cur
        if fi not in idx:
            continue

        finite = np.isfinite(unwrapped).all()
        # robust global-drift removal (MEDIAN, not mean: a mean is dragged by runaways)
        disp = (unwrapped - np.median(unwrapped, axis=0)) - (ref0 - np.median(ref0, axis=0))
        dmag = np.linalg.norm(disp, axis=1)

        lind_all = float(np.median(dmag) / nn0)
        row = dict(
            frame=int(fi), n_frames_available=int(n_frames), n_particles=int(N),
            nn0_nm=round(nn0, 3), finite=bool(finite),
            lindemann_median=round(lind_all, 5),
            lindemann_rms=round(float(np.sqrt(np.mean(dmag ** 2)) / nn0), 5),
            disp_p90_over_nn=round(float(np.percentile(dmag, 90) / nn0), 5),
            disp_p95_over_nn=round(float(np.percentile(dmag, 95) / nn0), 5),
            disp_max_over_nn=round(float(dmag.max() / nn0), 5),
        )
        for s in (0, 1):
            m = tid == s
            row[f"lindemann_median_{species[s]}"] = round(float(np.median(dmag[m]) / nn0), 5)
            row[f"lindemann_rms_{species[s]}"] = round(
                float(np.sqrt(np.mean(dmag[m] ** 2)) / nn0), 5)

        # frame-to-frame max displacement (numerical-health signal)
        if prev_unwrapped_sampled is not None:
            step = np.linalg.norm(unwrapped - prev_unwrapped_sampled, axis=1)
            row["max_frame_to_frame_disp_over_nn"] = round(float(step.max() / nn0), 5)
        else:
            row["max_frame_to_frame_disp_over_nn"] = 0.0
        prev_unwrapped_sampled = unwrapped.copy()

        # --- lattice integrity, two distinct measures ---
        # (a) identity retention: this particle still on ITS OWN site
        row["identity_retention"] = round(float((dmag < 0.5 * nn0).mean()), 5)
        for s in (0, 1):
            m = tid == s
            row[f"identity_retention_{species[s]}"] = round(
                float((dmag[m] < 0.5 * nn0).mean()), 5)

        # (b) species-aware SITE OCCUPANCY: original site has a CORRECT-SPECIES particle
        #     within 0.5*nn, using PERIODIC search on the wrapped current coords.
        cur_w = np.mod(cur + box / 2.0, box)
        site_w = np.mod(ref0 + box / 2.0, box)
        occ = np.zeros(N, bool)
        for s in (0, 1):
            sites = np.where(tid == s)[0]
            parts = np.where(tid == s)[0]
            if len(sites) == 0 or len(parts) == 0:
                continue
            tree = cKDTree(cur_w[parts], boxsize=box)
            dd, _ = tree.query(site_w[sites], k=1)
            occ[sites] = dd < 0.5 * nn0
        row["site_occupancy_species_aware"] = round(float(occ.mean()), 5)
        for s in (0, 1):
            m = tid == s
            row[f"site_occupancy_{species[s]}"] = round(float(occ[m].mean()), 5)

        # --- bond-orientational order ---
        q6pp = q6_per_particle(cur, box, nn0)
        row["q6_mean"] = round(float(q6pp.mean()), 5)
        row["q6_median"] = round(float(np.median(q6pp)), 5)
        row["q6_retention"] = round(float(q6pp.mean() / max(q6_0, 1e-9)), 5)
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = q6pp >= 0.8 * q6_0_pp
        row["frac_particles_q6_ge80pct"] = round(float(np.nan_to_num(keep).mean()), 5)
        for s in (0, 1):
            m = tid == s
            row[f"q6_mean_{species[s]}"] = round(float(q6pp[m].mean()), 5)
            row[f"q6_retention_{species[s]}"] = round(
                float(q6pp[m].mean() / max(q6_0_pp[m].mean(), 1e-9)), 5)
        row["largest_solid_component_frac"] = round(solid_fraction(cur, box, nn0, q6pp), 5)

        # --- corrected partial RDFs (LL / LS / SS), r <= 4.5*nn ---
        _, gr = partial_rdf(cur, tid, dom, nn0, r_max_nn=RDF_RMAX_NN)
        for ch in ("LL", "LS", "SS", "total"):
            row[f"rdf_{ch}_pearson_vs_f0"] = round(safe_pearson(gr0[ch], gr[ch]), 5)
            row[f"rdf_{ch}_emd_vs_f0"] = round(curve_emd(r0, gr0[ch], gr[ch]), 5)
            row[f"rdf_{ch}_absdiff_vs_f0"] = round(
                float(np.trapz(np.abs(gr[ch] - gr0[ch]), r0)), 5)

        # --- thermodynamics (NVT: no barostat, so no pressure is reported) ---
        if pe is not None and len(pe):
            j = min(int(fi / max(n_frames - 1, 1) * (len(pe) - 1)), len(pe) - 1)
            row["potential_energy_per_particle"] = round(float(pe[j]), 5)
            row["temperature_K"] = round(float(temp[j]), 3) if temp is not None else np.nan
        else:
            row["potential_energy_per_particle"] = np.nan
            row["temperature_K"] = np.nan
        row["pressure"] = np.nan          # NVT, fixed volume: not available and not meaningful here
        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------- trailing-20% pre-registered verdict ----------------
    k = max(3, int(np.ceil(TRAILING_FRAC * len(df))))
    tail = df.tail(k)
    tl_lind = float(tail.lindemann_median.median())
    tl_occ = float(tail.site_occupancy_species_aware.median())
    tl_q6 = float(tail.q6_retention.median())
    sl_lind = slope(tail.lindemann_median)
    sl_occ = slope(tail.site_occupancy_species_aware)
    sl_q6 = slope(tail.q6_retention)
    sl_pe = slope(tail.potential_energy_per_particle)

    numerical_ok = bool(df.finite.all() and df.disp_max_over_nn.max() < 100.0)
    complete = bool(n_frames >= 500)
    degrading = bool(sl_lind > THR_SLOPE_LIND or sl_occ < THR_SLOPE_OCC)

    pass_lind = tl_lind < THR_LINDEMANN
    pass_occ = tl_occ > THR_SITE_OCC
    pass_q6 = tl_q6 > THR_Q6_RET
    n_pass = int(pass_lind) + int(pass_occ) + int(pass_q6)

    reasons = []
    if not numerical_ok:
        verdict = "MELTED"
        reasons.append("numerical failure (NaN/Inf or absurd displacement)")
    elif n_pass == 3 and not degrading and complete:
        verdict = "STABLE"
        reasons.append(f"all 3 trailing criteria met (lind {tl_lind:.3f}<{THR_LINDEMANN}, "
                       f"occ {tl_occ:.3f}>{THR_SITE_OCC}, q6ret {tl_q6:.3f}>{THR_Q6_RET}); "
                       f"no degradation")
    elif n_pass == 0 and degrading:
        verdict = "MELTED"
        reasons.append("all structural criteria failed with sustained degradation")
    elif n_pass <= 1 and (tl_lind > 2 * THR_LINDEMANN or tl_occ < 0.5):
        verdict = "MELTED"
        reasons.append(f"structural collapse (lind {tl_lind:.3f}, occ {tl_occ:.3f})")
    else:
        verdict = "BORDERLINE_OR_STILL_EVOLVING"
        if not complete:
            reasons.append(f"run incomplete ({n_frames}/500 frames)")
        if degrading:
            reasons.append(f"still degrading (slope lind {sl_lind:+.3f}, occ {sl_occ:+.3f})")
        if n_pass in (1, 2):
            failed = [nm for nm, ok in (("lindemann", pass_lind), ("site_occupancy", pass_occ),
                                        ("q6_retention", pass_q6)) if not ok]
            reasons.append(f"conflicting criteria; failed: {','.join(failed)}")

    summary = dict(
        run=run_dir.name, n_frames=int(n_frames), n_frames_analysed=int(len(df)),
        complete=complete, n_particles=int(N), nn0_nm=round(nn0, 3),
        box_nm=f"{box[0]:.0f}x{box[1]:.0f}x{box[2]:.0f}",
        ensemble="NVT_periodic_bulk_fixed_volume",
        trailing_frames_used=int(k),
        trailing_lindemann_median=round(tl_lind, 5),
        trailing_site_occupancy=round(tl_occ, 5),
        trailing_q6_retention=round(tl_q6, 5),
        trailing_lindemann_LARGE=round(float(tail.lindemann_median_LARGE.median()), 5),
        trailing_lindemann_SMALL=round(float(tail.lindemann_median_SMALL.median()), 5),
        trailing_site_occupancy_LARGE=round(float(tail.site_occupancy_LARGE.median()), 5),
        trailing_site_occupancy_SMALL=round(float(tail.site_occupancy_SMALL.median()), 5),
        trailing_q6_retention_LARGE=round(float(tail.q6_retention_LARGE.median()), 5),
        trailing_q6_retention_SMALL=round(float(tail.q6_retention_SMALL.median()), 5),
        trailing_identity_retention=round(float(tail.identity_retention.median()), 5),
        trailing_solid_component_frac=round(float(tail.largest_solid_component_frac.median()), 5),
        trailing_rdf_LS_pearson=round(float(tail.rdf_LS_pearson_vs_f0.median()), 5),
        trailing_rdf_LL_pearson=round(float(tail.rdf_LL_pearson_vs_f0.median()), 5),
        trailing_rdf_SS_pearson=round(float(tail.rdf_SS_pearson_vs_f0.median()), 5),
        trailing_rdf_LS_emd=round(float(tail.rdf_LS_emd_vs_f0.median()), 5),
        slope_lindemann=round(sl_lind, 5), slope_site_occupancy=round(sl_occ, 5),
        slope_q6_retention=round(sl_q6, 5), slope_potential_energy=round(sl_pe, 5),
        degrading=degrading, numerical_ok=numerical_ok,
        pass_lindemann=bool(pass_lind), pass_site_occupancy=bool(pass_occ),
        pass_q6_retention=bool(pass_q6), n_criteria_passed=n_pass,
        thresholds=f"lind<{THR_LINDEMANN};occ>{THR_SITE_OCC};q6ret>{THR_Q6_RET}",
        verdict=verdict, verdict_reason="; ".join(reasons),
    )
    return df, summary


def cache_key(traj: Path):
    st = traj.stat()
    return dict(size=st.st_size, mtime=int(st.st_mtime))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-analyse even if unchanged")
    ap.add_argument("--pattern", default="*_v4", help="run-directory glob")
    args = ap.parse_args()

    CACHE.mkdir(exist_ok=True)
    OUT_DIAG.mkdir(exist_ok=True)

    prev_frames = pd.read_csv(OUT_FRAMES) if OUT_FRAMES.exists() else pd.DataFrame()
    prev_sum = pd.read_csv(OUT_SUMMARY) if OUT_SUMMARY.exists() else pd.DataFrame()
    frames_by_run, sum_by_run = {}, {}
    if len(prev_frames):
        for r, g in prev_frames.groupby("run"):
            frames_by_run[r] = g
    if len(prev_sum):
        for _, r in prev_sum.iterrows():
            sum_by_run[r["run"]] = r.to_dict()

    runs = sorted((HERE / "runs").glob(args.pattern))
    print(f"{'run':26s} {'frames':>7} {'lind':>7} {'occ':>7} {'q6ret':>7} {'rdfLS':>7}  verdict")
    print("-" * 88)
    n_new = n_skip = 0
    for d in runs:
        traj = d / "trajectory.gsd"
        if not traj.exists():
            continue
        ck = CACHE / f"{d.name}.json"
        key = cache_key(traj)
        if ck.exists() and not args.force:
            try:
                if json.loads(ck.read_text()) == key and d.name in sum_by_run:
                    n_skip += 1
                    s = sum_by_run[d.name]
                    print(f"{d.name:26s} {int(s['n_frames']):7d} "
                          f"{s['trailing_lindemann_median']:7.3f} {s['trailing_site_occupancy']:7.3f} "
                          f"{s['trailing_q6_retention']:7.3f} {s['trailing_rdf_LS_pearson']:7.3f}  "
                          f"{s['verdict']} (cached)")
                    continue
            except Exception:
                pass
        try:
            df, summary = analyse_run(d)
        except Exception as e:
            print(f"{d.name:26s}  ERROR: {str(e)[:50]}")
            continue
        if df is None:
            print(f"{d.name:26s}  <2 frames, skipped")
            continue
        df.insert(0, "run", d.name)
        frames_by_run[d.name] = df
        sum_by_run[d.name] = summary
        atomic_write_csv(df, OUT_DIAG / f"{d.name}_per_frame.csv")
        ck.write_text(json.dumps(key))
        n_new += 1
        print(f"{d.name:26s} {summary['n_frames']:7d} "
              f"{summary['trailing_lindemann_median']:7.3f} "
              f"{summary['trailing_site_occupancy']:7.3f} "
              f"{summary['trailing_q6_retention']:7.3f} "
              f"{summary['trailing_rdf_LS_pearson']:7.3f}  {summary['verdict']}")

    if frames_by_run:
        atomic_write_csv(pd.concat(frames_by_run.values(), ignore_index=True), OUT_FRAMES)
    if sum_by_run:
        atomic_write_csv(pd.DataFrame(list(sum_by_run.values())), OUT_SUMMARY)

    print(f"\nanalysed {n_new}, skipped {n_skip} unchanged")
    if sum_by_run:
        v = pd.DataFrame(list(sum_by_run.values())).verdict.value_counts().to_dict()
        print("verdicts:", v)
    print(f"per-frame  -> {OUT_FRAMES}")
    print(f"summary    -> {OUT_SUMMARY}")
    print(f"per-run    -> {OUT_DIAG}/<run>_per_frame.csv")


if __name__ == "__main__":
    main()
