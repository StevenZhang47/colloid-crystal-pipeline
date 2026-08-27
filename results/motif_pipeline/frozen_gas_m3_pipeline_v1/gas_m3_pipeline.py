#!/usr/bin/env python3
"""FROZEN continuous preprocessing pipeline: gas removal + M3 surface-center filter.

One call turns a raw frame (positions, diameters, box) into the set of SOAP pooling centers:

    raw frame
      -> [1] frozen radius-normalized r4 gas screen        (drop diffuse "gas" particles)
      -> [2] physical connected components (s<1.62)          (phase-blind, from the gas screen)
      -> [3] per-particle exposure + contact coordination    (over same-component s<1.35 neighbours)
      -> [4] M3 surface-center mask: exclude_center := exposure > 0.14 AND same_coord < 13
      -> SOAP centers = retained non-gas particles that are NOT excluded

Everything here is PHASE-BLIND: no crystal phase, no construction domain, no GAT/Nicole label, no
BULK_COORD[phase,species] is ever used. Frozen constants are module-level and must not be retuned.

The M3 mask is a REMOVABLE pooling mask: downstream you still compute and KEEP SOAP for every non-gas
particle (surface particles remain neighbourhood context); `is_soap_center` only selects which particles
are POOLED/averaged as centers. Nothing is destroyed, so an over-aggressive cut is always recoverable.

Provenance: gas screen = frozen `radius_normalized_coord_r4_gas_screen_v1`; M3 = frozen
`phase_blind_coord_exposure_v1` (exposure>0.14 AND same_coord<13), selected in
`phase_blind_surface_filter_v1` (see that dir's FINAL_SURFACE_FILTER_DECISION.md). Deterministic; no RNG.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# ---- frozen gas screen (imported, not reimplemented) ----
# Resolved RELATIVE to this file so a clone works from any location. It used to be the
# absolute cluster path, which meant `crystal-discovery analyze` died with
# ModuleNotFoundError on every machine that was not the cluster. The old path is kept as
# a fallback for trees where this module sits outside the repository layout.
# Path resolution only -- no frozen constant is touched.
_REPO = Path(__file__).resolve().parents[3]
_GS = _REPO / ("crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
               "/scripts/gas_screen_v1")
if not _GS.is_dir():
    _GS = Path("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/"
               "preprocessing_pipeline_final/unified_pipeline/scripts/gas_screen_v1")
sys.path.insert(0, str(_GS))
import radius_normalized_coord_r4_gas_screen as GAS   # frozen constants live inside

# ---- FROZEN pipeline constants (do not retune) ----
M_GRAPH = 1.35     # physical contact-graph cutoff, radius-normalized s = d/(R_i+R_j)
TAU_E = 0.14       # M3 exposure threshold      (exclude center if exposure  > TAU_E ...)
TAU_C = 13         # M3 coordination threshold  (... AND same_coord < TAU_C)


@dataclass
class PipelineResult:
    n: int
    retain_mask: np.ndarray          # bool[n]  True = non-gas (survives gas screen)
    component_id: np.ndarray         # int[n]   physical connected component (-1 for gas)
    same_coord: np.ndarray           # float[n] contacts at s<1.35 within same component
    exposure: np.ndarray             # float[n] ||sum unit(r_j-r_i)|| / same_coord  (1.0 if isolated)
    excluded_as_surface: np.ndarray  # bool[n]  M3 surface-center prediction (removed from pooling)
    is_soap_center: np.ndarray       # bool[n]  retain & ~excluded  -> pool these
    canonical_species_id: np.ndarray # int[n]   0 = LARGE, 1 = SMALL (by physical diameter)
    gas_result: object               # the raw GasScreenResult (roles, reason_code, mean_r4*, ...)

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(dict(
            particle_index=np.arange(self.n), retain_non_gas=self.retain_mask,
            component_id=self.component_id, canonical_species_id=self.canonical_species_id,
            same_coord=self.same_coord, exposure=self.exposure,
            excluded_as_surface=self.excluded_as_surface, is_soap_center=self.is_soap_center))

    def summary(self):
        nret = int(self.retain_mask.sum()); nrem = int(self.excluded_as_surface.sum())
        return dict(n=self.n, n_gas=int((~self.retain_mask).sum()), n_non_gas=nret,
                    n_excluded_surface=nrem, n_soap_centers=int(self.is_soap_center.sum()),
                    surface_frac_of_non_gas=round(nrem / max(nret, 1), 4),
                    n_components=int(len({int(c) for c in self.component_id if c >= 0})))


# ------------------------------------------------------------------ helpers
def _canonical_species(diameters, type_ids, n):
    """0 = LARGE, 1 = SMALL, by physical DIAMETER (median per type). Phase-blind size class."""
    if type_ids is None:
        d = np.asarray(diameters, float)
        return (d < np.median(d)).astype(int) if len(np.unique(np.round(d, 6))) > 1 else np.zeros(n, int)
    tid = np.asarray(type_ids).astype(int)
    med = {int(t): float(np.median(np.asarray(diameters, float)[tid == t])) for t in np.unique(tid)}
    order = sorted(med, key=lambda t: -med[t])
    remap = {t: (0 if i == 0 else 1) for i, t in enumerate(order)}
    return np.array([remap[int(t)] for t in tid], int)


def _min_image(delta, box):
    Lx, Ly, Lz, xy, xz, yz = box
    delta = np.asarray(delta, float).copy()
    z = np.round(delta[:, 2] / Lz); delta[:, 0] -= z * xz * Lz; delta[:, 1] -= z * yz * Lz; delta[:, 2] -= z * Lz
    y = np.round(delta[:, 1] / Ly); delta[:, 0] -= y * xy * Ly; delta[:, 1] -= y * Ly
    x = np.round(delta[:, 0] / Lx); delta[:, 0] -= x * Lx
    return delta


def _is_nonperiodic(box):
    return box is None or np.allclose(np.asarray(box, float), 0)


def _exposure_and_coord(pos, R, box, retain, comp):
    """Per-particle same-component contact coordination (s<1.35) and exposure one-sidedness."""
    n = len(pos)
    same = np.zeros(n); acc = np.zeros((n, 3))
    idx = np.flatnonzero(retain)
    if len(idx) < 2:
        return same, np.ones(n)
    sub = pos[idx]
    if _is_nonperiodic(box):
        tree = cKDTree(sub); use_mi = False; L = None
        b6 = None
    else:
        b6 = np.asarray(box, float)
        if b6.size == 3:
            b6 = np.array([b6[0], b6[1], b6[2], 0, 0, 0], float)
        L = np.array([b6[0], b6[1], b6[2]])
        ortho = (b6[3] == 0 and b6[4] == 0 and b6[5] == 0)
        tree = cKDTree(np.mod(sub, L), boxsize=L) if ortho else cKDTree(sub)
        use_mi = not ortho
    rmax = M_GRAPH * 2 * R[idx].max()
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    for a, b in pairs:
        ia, ib = idx[a], idx[b]
        if comp[ia] != comp[ib] or comp[ia] < 0:
            continue
        d = pos[ib] - pos[ia]
        if use_mi:
            d = _min_image(d[None], b6)[0]
        elif not _is_nonperiodic(box):
            d = d - L * np.round(d / L)
        dist = float(np.linalg.norm(d))
        if dist <= 1e-12:
            continue
        if dist / (R[ia] + R[ib]) < M_GRAPH:
            u = d / dist
            same[ia] += 1; same[ib] += 1; acc[ia] += u; acc[ib] -= u
    expo = np.ones(n); nz = same > 0
    expo[nz] = np.linalg.norm(acc[nz], axis=1) / same[nz]
    return same, expo


# ------------------------------------------------------------------ the pipeline
def run(positions, diameters, box=None, type_ids=None):
    """Continuous gas + M3 pipeline. positions (n,3); diameters (n,); box length-3 / length-6 / 3x3 / None
    (None or all-zero = non-periodic isolated cluster). Returns PipelineResult."""
    pos = np.asarray(positions, float); n = len(pos)
    diam = np.asarray(diameters, float); R = diam / 2.0

    # [1] frozen gas screen  (feed a large enclosing box when non-periodic so no wrap occurs)
    gs_box = box
    if _is_nonperiodic(box):
        ext = (pos.max(0) - pos.min(0)).max() + 4 * R.max()
        gs_box = np.array([ext * 10, ext * 10, ext * 10], float)
    gr = GAS.radius_normalized_coord_r4_gas_screen(positions=pos, diameters=diam, box=gs_box,
                                                   type_ids=type_ids, compute_components=True)
    retain = gr.retain_mask.astype(bool)
    comp = np.asarray(gr.component_id).astype(int)          # [2] physical components (s<1.62), -1 = gas

    # [3] exposure + contact coordination over same-component s<1.35 neighbours
    same, expo = _exposure_and_coord(pos, R, box, retain, comp)

    # [4] M3 surface-center mask (phase-blind, removable)
    excluded = retain & (expo > TAU_E) & (same < TAU_C)
    center = retain & (~excluded)
    sp = _canonical_species(diam, type_ids, n)
    return PipelineResult(n=n, retain_mask=retain, component_id=comp, same_coord=same, exposure=expo,
                          excluded_as_surface=excluded, is_soap_center=center,
                          canonical_species_id=sp, gas_result=gr)


# ------------------------------------------------------------------ self-test
def _selftest():
    """Reproduce the validated M3 removed-counts on benchmark crystallites (isolated clusters)."""
    MP = _REPO / "results/motif_pipeline"
    if not MP.is_dir():
        MP = Path("/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline")
    sys.path.insert(0, str(MP / "domain_surface_method_benchmark_v1" / "scripts"))
    import io_frames as IO
    import pandas as pd
    expect = {"single_Zangenite_sphere": 699, "single_CsCl_sphere": 437, "single_CsCl_block": 255,
              "single_Th3P4_sphere": 395, "single_fluorite_sphere": 306, "single_Ni3Sn_block": 308}
    rows = []
    ok = True
    for nm, exp in expect.items():
        f = IO.load_frame(nm)
        r = run(f["pos"], 2 * f["R"], box=f["box"], type_ids=f["tid"])
        s = r.summary()
        match = (s["n_excluded_surface"] == exp)
        ok &= match
        rows.append(dict(frame=nm, non_gas=s["n_non_gas"], excluded=s["n_excluded_surface"],
                         expected=exp, centers=s["n_soap_centers"], match=match))
        print(f"  {nm:26s} non_gas={s['n_non_gas']:4d} excluded={s['n_excluded_surface']:4d} "
              f"(expected {exp}) centers={s['n_soap_centers']:4d}  {'OK' if match else 'MISMATCH'}")
    pd.DataFrame(rows).to_csv(Path(__file__).parent / "pipeline_selftest.csv", index=False)
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
