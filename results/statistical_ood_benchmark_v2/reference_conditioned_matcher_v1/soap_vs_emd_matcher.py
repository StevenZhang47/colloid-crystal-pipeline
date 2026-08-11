#!/usr/bin/env python3
"""Can SOAP tell the library prototypes apart where the EMD channels cannot?

The EMD matcher's three channels are partial g(r), total g(r) and powder S(k) -- all
RADIAL and all orientationally averaged. That is why a Cu3Au crystal matches Al3Ti
(D0_22) at D = 0.359, inside the 0.40 acceptance cutoff: D0_22 is L1_2 with an antiphase
boundary every second cell, so the radial statistics nearly coincide and the difference
lives in the angular arrangement.

SOAP already carries exactly what is missing -- it is species-resolved AND angular -- and
this project already has a frozen SOAP pipeline. So test it directly rather than building
a new bond-order channel.

Method, using the FROZEN descriptor path unchanged
(`soap_atlas_core3_phase2_v1/scripts/core3_pipeline.py`, GLOBAL_D0 representation):

  * frozen gas screen + M3 reliable-centre selection
  * SOAP with n_max=8, l_max=6, r_cut=3.0*d0, sigma=0.10*d0, species [11,17], 952-D,
    d0 = median nearest-neighbour distance in the physical component (so scale-free)
  * per-particle vectors L2-normalised, then averaged SEPARATELY over the LARGE and the
    SMALL sublattice and concatenated -> a 1904-D species-resolved domain fingerprint
  * distance = Euclidean between L2-normalised fingerprints, i.e. sqrt(2 - 2 cos)

Same candidates and anchors as the EMD run, so the two are directly comparable:
  ARM 1  self-match       -- does each family rank its own prototype first?
  ARM 2  leave-own-out    -- with every anchor of the true prototype deleted, does it
                             abstain, or does it confidently name the wrong prototype?
                             This is the arm the EMD channels fail (Cu3Au 0/6, CsCl 1/6).

No threshold is proposed here. SOAP distances live on their own scale; what is comparable
across the two methods is the SEPARATION RATIO (nearest wrong prototype / nearest right
one), which is scale-free.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
CS = ROOT / "crystal_discovery/crystal_stability"
LIB = ROOT / "crystal_discovery/cif_library"
SOAPDIR = ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"
sys.path.insert(0, str(SOAPDIR))

import core3_pipeline as C3  # noqa: E402
import gas_m3_pipeline as M3  # noqa: E402

TEST_FAMILIES = {"CsCl": "cscl", "Th3P4": "th3p4", "Ni3Sn": "ni3sn_d019", "Cu3Au": "cu3au_d6"}
CANDIDATE_FRAMES = [250, 300, 350, 400, 450, 500]
MIN_CENTERS_PER_SPECIES = 25

# Same prototype grouping the corrected EMD scoring used. li3fe is Ni3Sn (D0_19): the
# role-relabelled StructureMatcher fits them at rms 0.045 while correctly separating the
# negative controls (Cu3Au/ReO3, Ni3Sn/I3P, CsCl/NaTl, Laves C14/C36).
PROTOTYPE_OF = {
    "cscl": "CsCl", "cscl_b2_anchor": "CsCl",
    "th3p4__8edd1f72": "Th3P4", "th3p4__72406369": "Th3P4",
    "th3p4_mp-1347_primitive": "Th3P4", "th3p4_anchor": "Th3P4",
    "cu3au": "Cu3Au", "cu3au_mp-2258_conventional_standard": "Cu3Au", "cu3au_l12": "Cu3Au",
    "zangenite__471e7cda": "zangenite", "zangenite__98bb84a4": "zangenite",
    "ni3sn_d019": "Ni3Sn", "li3fe": "Ni3Sn",
    "cr3si_a15": "Cr3Si_A15", "rocksalt_b1": "rocksalt_B1",
    "fluorite_c1": "fluorite", "antifluorite_c1b": "fluorite",   # identical by role
}


def prototype_of(name: str) -> str:
    base = name[len("orig_"):] if name.startswith("orig_") else name
    return PROTOTYPE_OF.get(base.lower(), base)


def anchor_prep(frame):
    """A prep that retains EVERY particle, for the CIF-ideal anchors.

    The anchors are freshly built lattices: there is no gas in them and no thermal
    disorder, so the gas screen has nothing to find and every particle it removes is a
    false positive. It removes a lot. The frozen screen deletes particle i when the mean
    over its FOUR physical-nearest neighbours of d/(R_i+R_j) exceeds 1.2698, which
    presumes at least four neighbours near contact -- true for close-packed prototypes,
    false for open ones. On ReO3 (corner-sharing octahedra, majority site coordination 2)
    it deleted the ENTIRE majority sublattice: 5324 particles in, 1330 out, exactly the
    LARGE count. Nine of 37 cohesive anchors were lost this way, including zangenite.

    Running it here also made the head-to-head unfair: the EMD channels read anchors with
    extract_domain=False, i.e. every particle. Screening only the SOAP side compared two
    different things.

    Thermal CANDIDATES still go through the full frozen path -- they contain real gas and
    a real surface, which is what the screen is for.
    """
    n = len(frame.positions)
    return M3.PipelineResult(
        n=n,
        retain_mask=np.ones(n, bool),
        component_id=np.zeros(n, int),
        same_coord=np.zeros(n, float),
        exposure=np.zeros(n, float),
        excluded_as_surface=np.zeros(n, bool),
        is_soap_center=np.ones(n, bool),
        canonical_species_id=M3._canonical_species(frame.diameters, frame.type_ids, n),
        gas_result=None,
    )


def fingerprint(gsd_path: Path, frame_index: int, is_anchor: bool = False) -> np.ndarray | None:
    """Species-resolved mean SOAP.

    Anchors: every particle, no screening (see anchor_prep).
    Candidates: frozen gas screen + M3 reliable centres, largest physical component.
    """
    frame = C3.read_frame(gsd_path, frame_index)
    if is_anchor:
        prep = anchor_prep(frame)
        centers = np.arange(len(frame.positions))
    else:
        prep = C3.preprocess(frame)
        centers = np.flatnonzero(prep.is_soap_center)
        if len(centers) == 0:
            return None
        # Largest physical component, matching the matcher's domain rule (largest
        # connected condensed cluster) rather than pooling stray fragments.
        comps, counts = np.unique(prep.component_id[centers], return_counts=True)
        keep = comps[np.argmax(counts)]
        centers = centers[prep.component_id[centers] == keep]

    res = C3.soap_global_d0(frame, prep, centers=centers)
    v = res.vectors
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.clip(norms, 1e-12, None)

    species = prep.canonical_species_id[res.particle_indices]   # 0 = LARGE, 1 = SMALL
    parts = []
    for s in (0, 1):
        sel = species == s
        if sel.sum() < MIN_CENTERS_PER_SPECIES:
            return None
        parts.append(v[sel].mean(axis=0))
    fp = np.concatenate(parts)
    return fp / max(np.linalg.norm(fp), 1e-12)


def cohesive_anchor_runs() -> set:
    admissible = set()
    for f in sorted((LIB / "generated" / "records").glob("*.json")):
        rec = json.loads(f.read_text())
        if rec.get("verdict") == "cohesive":
            admissible.add(rec.get("run") or f"lib_{rec['slug'].lower()}")
    return admissible


def main() -> None:
    t0 = time.perf_counter()
    admissible = cohesive_anchor_runs()

    print("=== SOAP fingerprints: anchors ===", flush=True)
    anchors = {}
    for run_dir in sorted((CS / "runs").glob("lib_*")):
        if admissible and run_dir.name not in admissible:
            continue
        gsd = run_dir / "first_frame.gsd"
        if not gsd.is_file():
            continue
        name = run_dir.name[len("lib_"):]
        try:
            fp = fingerprint(gsd, 0, is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}", flush=True); continue
        if fp is None:
            print(f"  [skip] {name}: too few reliable centres", flush=True); continue
        anchors[name] = fp
        print(f"  {name:42s} {prototype_of(name)}", flush=True)
    print(f"{len(anchors)} anchors\n", flush=True)

    rows = []
    for fam, run in TEST_FAMILIES.items():
        for fi in CANDIDATE_FRAMES:
            try:
                cand = fingerprint(CS / "runs" / run / "trajectory.gsd", fi)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {fam} f{fi}: {type(exc).__name__}: {exc}", flush=True); continue
            if cand is None:
                continue
            d = {a: float(np.linalg.norm(cand - anchors[a])) for a in anchors}
            order = sorted(d, key=lambda a: d[a])
            best = order[0]
            # ARM 1: nearest overall, and nearest anchor of a DIFFERENT prototype
            other = next(a for a in order if prototype_of(a) != prototype_of(best))
            # ARM 2: every anchor of the TRUE prototype deleted
            survivors = [a for a in order if prototype_of(a) != fam]
            s1, s2 = survivors[0], survivors[1]
            rows.append(dict(
                family=fam, frame=fi,
                nearest=best, nearest_prototype=prototype_of(best),
                correct=prototype_of(best) == fam, d_self=round(d[best], 4),
                nearest_other=prototype_of(other), d_other=round(d[other], 4),
                separation=round(d[other] / max(d[best], 1e-12), 3),
                holdout_nearest=prototype_of(s1), holdout_d=round(d[s1], 4),
                holdout_runner_up=prototype_of(s2),
                holdout_margin=round(d[s2] / max(d[s1], 1e-12), 3),
            ))
            r = rows[-1]
            print(f"  {fam} f{fi}: -> {r['nearest']} ({r['nearest_prototype']}) "
                  f"d={r['d_self']} | next prototype {r['nearest_other']} d={r['d_other']} "
                  f"sep={r['separation']}x | holdout -> {r['holdout_nearest']} "
                  f"d={r['holdout_d']} margin={r['holdout_margin']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "soap_vs_emd_results.csv", index=False)

    print("\n=== SUMMARY (SOAP) ===")
    summary = {}
    for fam, g in df.groupby("family"):
        summary[fam] = {
            "self_match_correct": f"{int(g.correct.sum())}/{len(g)}",
            "d_self_range": [float(g.d_self.min()), float(g.d_self.max())],
            "d_nearest_other_prototype": [float(g.d_other.min()), float(g.d_other.max())],
            "separation_ratio_min": float(g.separation.min()),
            "holdout_nearest_prototype": sorted(set(g.holdout_nearest)),
            "holdout_margin_range": [float(g.holdout_margin.min()), float(g.holdout_margin.max())],
        }
        print(f"{fam:7s} correct {int(g.correct.sum())}/{len(g)}   "
              f"d_self {g.d_self.min():.4f}-{g.d_self.max():.4f}   "
              f"nearest other prototype {g.d_other.min():.4f}   "
              f"separation {g.separation.min():.2f}x")
    (HERE / "soap_vs_emd_summary.json").write_text(
        json.dumps({"runtime_seconds": round(time.perf_counter() - t0, 1),
                    "n_anchors": len(anchors), "per_family": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
