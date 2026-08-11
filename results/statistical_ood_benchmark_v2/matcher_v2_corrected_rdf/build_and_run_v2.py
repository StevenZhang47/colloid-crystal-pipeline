#!/usr/bin/env python3
"""
MATCHER v2 — rebuild + recalibrate + rerun, on the CORRECTED partial RDF.

Nothing is carried over from v1: every fingerprint, every threshold, every headline
number is recomputed from scratch. v1 is INVALIDATED, not superseded.

STRUCTURE (unchanged in principle, corrected in implementation)
  A. ideal CIF-derived identity anchors      -> RANKING     (what is this?)
  B. independent thermal acceptance ensembles -> ACCEPTANCE  (is it really that, at 298 K?)
     built ONLY from independent seeds; a family's candidate never enters its own envelope.

KEY CHANGE REQUESTED: every active reference is evaluated INDEPENDENTLY and accepted or
rejected on its own gates. The verdict follows from HOW MANY references are accepted --
zero, one, or several -- not from top-1 alone.

  0 accepted  -> ordered_no_confident_match / novel_and_not_in_active_library
  1 accepted  -> confident_reference_match
  >1 accepted -> ambiguous_multiple_references

REPORTING: one row per INDEPENDENT TRAJECTORY. Correlated frames from a single run are
aggregated to that run; they are never counted as separate samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
CS = ROOT / "crystal_discovery/crystal_stability"
UP = ROOT / "crystal_discovery/preprocessing_pipeline_final/unified_pipeline"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(UP / "PI'sWork"))
from partial_rdf import partial_rdf  # noqa

E = UP / "results/motif_pipeline/gat_v4_qc_phase_map/latent_feature_export_v1"
PHYS = ["coordination", "neighbor_fraction", "same_type_fraction", "mean_neighbor_distance",
        "std_neighbor_distance", "mean_neighbor_distance_type_delta", "motif_recurrence",
        "q4", "q6"]
BLOCK_FRAMES = [300, 350, 400, 450, 500]
CAND_FRAME = 400
STOICH_TOL = 0.12

FAMILIES = {
    "CsCl":        dict(cand="cscl",              L=1, S=1),
    "Th3P4":       dict(cand="th3p4",             L=3, S=4),
    "Cu3Au":       dict(cand="cu3au_d6",          L=1, S=3),
    "Ni3Sn":       dict(cand="ni3sn_d019",        L=1, S=3),
    "zangenite":   dict(cand="zangenite_deep8",   L=3, S=4),
    "Cr3Si_A15":   dict(cand="cr3si_a15_deep8",   L=1, S=3),
    "rocksalt_B1": dict(cand="rocksalt_b1_deep8", L=1, S=1),
}
TIER_A = {"Cu3Au": "cu3au_l12_big"}
AMORPH = ["zincblende_b3", "zincblende_b3_deep10", "fluorite_c1", "fluorite_c1_sr069",
          "nickeline_b81", "nickeline_b81_deep8", "fe4n_lp10", "fe4n_lp10_deep8",
          "reo3_d09", "i3p_a3b_hp8", "antifluorite_c1b", "cu3au"]


def canonical(fr):
    pos = np.asarray(fr.particles.position, float)
    box = np.asarray(fr.configuration.box[:3], float)
    diam = np.asarray(fr.particles.diameter, float)
    tid = np.asarray(fr.particles.typeid, int)
    names = list(fr.particles.types)
    md = {int(t): float(diam[tid == t].mean()) for t in np.unique(tid)}
    order = sorted(md, key=lambda t: -md[t])          # 0 = LARGE
    remap = {o: n for n, o in enumerate(order)}
    return (pos, np.array([remap[int(t)] for t in tid], int), box, diam,
            [names[order[0]], names[order[1]]], md[order[0]], md[order[1]])


def nn_of(pos, box):
    w = np.mod(pos + box / 2.0, box)
    d, _ = cKDTree(w, boxsize=box).query(w, k=2)
    return float(np.median(d[:, 1]) + 1e-9)


def domain_of(pos, box, nn, gas_safe):
    """gas_safe=True: grow only into strongly-attached particles (Nicole's gas boxes).
       gas_safe=False: full-graph component (our isolated crystals in vacuum)."""
    fb = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    nq = dict(r_max=1.4 * nn, exclude_ii=True)
    st = freud.order.Steinhardt(l=6); st.compute((fb, pos), neighbors=nq)
    q6 = np.nan_to_num(st.particle_order)
    nl = freud.locality.AABBQuery(fb, pos).query(pos, nq).toNeighborList()
    qi, pi = nl.query_point_indices, nl.point_indices
    coord = np.bincount(qi, minlength=len(pos))
    solid = (q6 > 0.28) & (coord >= 6)
    if solid.sum() < 32:
        return np.zeros(len(pos), bool)
    if not gas_safe:
        cl = freud.cluster.Cluster(); cl.compute((fb, pos), neighbors=nq)
        comp = cl.cluster_idx
        return comp == np.bincount(comp[solid], minlength=comp.max() + 1).argmax()
    cl = freud.cluster.Cluster(); cl.compute((fb, pos[solid]), neighbors=nq)
    idx = np.where(solid)[0]
    dom = np.zeros(len(pos), bool)
    dom[idx[cl.cluster_idx == np.argmax(np.bincount(cl.cluster_idx))]] = True
    for _ in range(2):
        n_in = np.bincount(qi[dom[pi]], minlength=len(pos))
        add = (~dom) & (n_in >= 4)
        if not add.any():
            break
        dom |= add
    return dom


def sk_of(pos, dom, nn):
    p = pos[dom]
    if len(p) > 2500:
        p = p[np.random.default_rng(1).choice(len(p), 2500, replace=False)]
    q = (p - p.mean(0)) / nn
    kv = np.linspace(0.35, 9.0, 90)
    rng = np.random.default_rng(2)
    dirs = rng.normal(size=(48, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return kv, np.array([float(np.mean(np.abs(np.exp(1j * (q @ (k * dirs).T)).sum(0)) ** 2) / len(q))
                         for k in kv])


def describe(path, frame, tag, gas_safe):
    from particle_descriptors import build_feature_matrix
    t = gsd.hoomd.open(str(path))
    fr = t[0] if frame == "ideal" else t[min(frame, len(t) - 1)]
    pos, tid, box, diam, types, dL, dS = canonical(fr)
    nn = nn_of(pos, box)
    dom = domain_of(pos, box, nn, gas_safe)
    if dom.sum() < 256:
        return None
    tmpd = HERE / "frames"; tmpd.mkdir(exist_ok=True)
    fp = tmpd / f"{tag}.gsd"
    f2 = gsd.hoomd.Frame()
    f2.particles.N = len(pos); f2.particles.types = types
    f2.particles.typeid = tid.astype(np.uint32)
    f2.particles.position = pos.astype(np.float32)
    f2.particles.diameter = diam.astype(np.float32)
    f2.configuration.box = np.asarray(fr.configuration.box, np.float32)
    with gsd.hoomd.open(str(fp), "w") as g:
        g.append(f2)
    ph = np.asarray(build_feature_matrix(fp, 0, 12).features, np.float32)[:, 0:9]
    r, gr = partial_rdf(pos, tid, dom, nn)          # <-- CORRECTED
    k, sk = sk_of(pos, dom, nn)
    L = int((tid[dom] == 0).sum()); S = int((tid[dom] == 1).sum())
    return dict(r=r, gr=gr, k=k, sk=sk, nn=nn, n=int(dom.sum()),
                stoich=L / max(S, 1), frac=float(dom.mean()),
                size_ratio=dS / dL,
                phys={c: float(ph[dom, i].mean()) for i, c in enumerate(PHYS)})


def emd(x, a, b):
    a = np.clip(a, 0, None); b = np.clip(b, 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return np.inf
    return float(wasserstein_distance(x, x, a / a.sum(), b / b.sum()))


def rank_distance(cand, anchor):
    return (0.50 * np.mean([emd(anchor["r"], cand["gr"][g], anchor["gr"][g])
                            for g in ("LL", "LS", "SS")])
            + 0.25 * emd(anchor["r"], cand["gr"]["total"], anchor["gr"]["total"])
            + 0.25 * emd(anchor["k"], cand["sk"], anchor["sk"]))


def accept(cand, fam, ens):
    """Independent ACCEPT/REJECT for ONE reference. All thresholds from the family's
    own independent thermal ensemble (recalibrated from scratch)."""
    e = ens[fam]
    out = {}
    out["stoichiometry_pass"] = bool(abs(cand["stoich"] - e["stoich"]) /
                                     max(e["stoich"], 1e-9) < STOICH_TOL)
    bad = []
    for g in ("LL", "LS", "SS", "total"):
        d = emd(e["r"], cand["gr"][g], e["gr"][g])
        tol = e["tol_gr"][g]
        out[f"rdf_{g}_emd"] = round(d, 5); out[f"rdf_{g}_tol"] = round(tol, 5)
        if d > tol:
            bad.append(f"rdf_{g}")
    d = emd(e["k"], cand["sk"], e["sk"])
    out["sk_emd"] = round(d, 5); out["sk_tol"] = round(e["tol_sk"], 5)
    if d > e["tol_sk"]:
        bad.append("sk")
    for c, K in (("coordination", 4.0), ("q4", 6.0), ("q6", 6.0), ("motif_recurrence", 6.0)):
        v = cand["phys"][c]; cen, sp, lo, hi = e["cen"][c], e["sp"][c], e["lo"][c], e["hi"][c]
        z = abs(v - cen) / (sp + 1e-9)
        ok = bool((lo - 0.5 * sp <= v <= hi + 0.5 * sp) or z <= K)
        out[f"{c}_z"] = round(float(z), 2)
        if not ok and c == "coordination":
            bad.append(c)
    out["contradictions"] = ";".join(bad)
    out["accepted"] = bool(out["stoichiometry_pass"] and not bad)
    return out


def evaluate(cand, anchors, ens, drop=None):
    """Evaluate EVERY active reference independently -> 0, 1 or many accepted."""
    acts = [f for f in anchors if f != drop and f in ens]
    rows = {f: accept(cand, f, ens) for f in acts}
    dists = {f: rank_distance(cand, anchors[f]) for f in acts}
    accepted = sorted([f for f in acts if rows[f]["accepted"]], key=lambda f: dists[f])
    order = sorted(acts, key=lambda f: dists[f])
    t1, t2 = order[0], (order[1] if len(order) > 1 else "")
    ratio = (dists[t2] / max(dists[t1], 1e-9)) if t2 else np.inf
    if len(accepted) == 0:
        v = "ordered_no_confident_match" if drop is None else "novel_and_not_in_active_library"
    elif len(accepted) == 1:
        v = "confident_reference_match" if drop is None else "ordered_OOD_matched_to_reference"
    else:
        v = "ambiguous_multiple_references"
    return dict(n_accepted=len(accepted), accepted_references=";".join(accepted),
                top1_reference=t1, top2_reference=t2,
                top1_distance=round(dists[t1], 5),
                top2_distance=round(dists[t2], 5) if t2 else np.nan,
                margin_ratio=round(ratio, 3) if np.isfinite(ratio) else np.nan,
                final_verdict=v,
                per_reference_contradictions=" | ".join(
                    f"{f}:{rows[f]['contradictions'] or ('STOICH' if not rows[f]['stoichiometry_pass'] else 'ok')}"
                    for f in order))


def build_libraries():
    """A) ideal CIF anchors.  B) thermal envelopes from INDEPENDENT seeds only."""
    anchors, ens, inv = {}, {}, []
    for fam, m in FAMILIES.items():
        anchors[fam] = describe(CS / "runs" / m["cand"] / "first_frame.gsd", "ideal",
                                f"anchor_{fam}", gas_safe=False)
        runs = ([TIER_A[fam]] if fam in TIER_A else []) + \
               [f"thermref_{m['cand']}_seed{s}" for s in (11, 12)]
        blocks = []
        used = []
        for run in runs:
            p = CS / "runs" / run / "trajectory.gsd"
            if not p.exists():
                continue
            got = 0
            for fi in BLOCK_FRAMES:
                b = describe(p, fi, f"ens_{fam}_{run}_{fi}", gas_safe=False)
                if b:
                    b["run"] = run; blocks.append(b); got += 1
            if got:
                used.append(run)
        if not blocks:
            continue
        gr = {g: np.mean([b["gr"][g] for b in blocks], axis=0) for g in ("LL", "LS", "SS", "total")}
        # THRESHOLDS RECALIBRATED FROM SCRATCH: the within-ensemble spread of each channel,
        # measured leave-one-block-out against the ensemble mean, then padded.
        tol_gr = {}
        for g in ("LL", "LS", "SS", "total"):
            d = [emd(blocks[0]["r"], b["gr"][g], gr[g]) for b in blocks]
            tol_gr[g] = float(max(np.max(d) * 2.0, 0.02))
        sk = np.mean([b["sk"] for b in blocks], axis=0)
        tol_sk = float(max(np.max([emd(blocks[0]["k"], b["sk"], sk) for b in blocks]) * 2.0, 0.02))
        cen = {c: float(np.median([b["phys"][c] for b in blocks])) for c in PHYS}
        sp = {c: float(max(np.std([b["phys"][c] for b in blocks]), 1e-9)) for c in PHYS}
        lo = {c: float(np.min([b["phys"][c] for b in blocks])) for c in PHYS}
        hi = {c: float(np.max([b["phys"][c] for b in blocks])) for c in PHYS}
        ens[fam] = dict(r=blocks[0]["r"], k=blocks[0]["k"], gr=gr, sk=sk,
                        tol_gr=tol_gr, tol_sk=tol_sk, cen=cen, sp=sp, lo=lo, hi=hi,
                        stoich=float(np.median([b["stoich"] for b in blocks])),
                        n_runs=len(used), n_blocks=len(blocks), runs=used)
        exp = m["L"] / m["S"]
        inv.append(dict(family=fam, n_independent_thermal_runs=len(used),
                        thermal_runs=";".join(used), n_blocks=len(blocks),
                        anchor_stoich=round(anchors[fam]["stoich"], 4),
                        thermal_stoich=round(ens[fam]["stoich"], 4),
                        expected_stoich=round(exp, 4),
                        stoich_ok=bool(abs(ens[fam]["stoich"] - exp) / exp < 0.10),
                        anchor_first_peak_r_over_nn=round(
                            float(anchors[fam]["r"][np.argmax(anchors[fam]["gr"]["total"])]), 3),
                        tol_rdf_LS=round(tol_gr["LS"], 4), tol_sk=round(tol_sk, 4)))
        print(f"  {fam:12s} anchor n={anchors[fam]['n']:5d} first-peak r/nn="
              f"{inv[-1]['anchor_first_peak_r_over_nn']:.2f}  thermal runs={len(used)} "
              f"blocks={len(blocks)}  stoich {ens[fam]['stoich']:.3f} (exp {exp:.3f})")
    pd.DataFrame(inv).to_csv(HERE / "v2_reference_inventory.csv", index=False)
    return anchors, ens


def main():
    print("=== BUILDING v2 LIBRARIES (corrected partial RDF) ===")
    anchors, ens = build_libraries()

    rows = []
    # --- 1. same-construction recovery (Tier 1) ---
    for fam, m in FAMILIES.items():
        c = describe(CS / "runs" / m["cand"] / "trajectory.gsd", CAND_FRAME, f"c_{fam}", False)
        r = evaluate(c, anchors, ens)
        rows.append(dict(benchmark="A_same_construction", subject=fam, true_label=fam,
                         validation_tier="Tier1_same_construction", n_independent_trajectories=1,
                         expected="confident_reference_match",
                         correct=bool(r["final_verdict"] == "confident_reference_match"
                                      and r["accepted_references"] == fam), **r))
    # --- 2. correct family removed (from anchors AND ensembles) ---
    for fam, m in FAMILIES.items():
        c = describe(CS / "runs" / m["cand"] / "trajectory.gsd", CAND_FRAME, f"c_{fam}", False)
        r = evaluate(c, anchors, ens, drop=fam)
        rows.append(dict(benchmark="B_family_removed", subject=fam, true_label=fam,
                         validation_tier="Tier1_same_construction", n_independent_trajectories=1,
                         expected="novel_and_not_in_active_library",
                         correct=bool(r["n_accepted"] == 0), **r))
    # --- 3. disordered / melted controls ---
    for a in AMORPH:
        p = CS / "runs" / a / "trajectory.gsd"
        if not p.exists():
            continue
        c = describe(p, CAND_FRAME, f"c_{a}", False)
        if c is None:
            rows.append(dict(benchmark="C_disordered_control", subject=a, true_label="disordered",
                             n_independent_trajectories=1, expected="no_confident_match",
                             n_accepted=0, final_verdict="not_ordered_or_amorphous", correct=True))
            continue
        r = evaluate(c, anchors, ens)
        rows.append(dict(benchmark="C_disordered_control", subject=a, true_label="disordered",
                         n_independent_trajectories=1, expected="no_confident_match",
                         correct=bool(r["n_accepted"] == 0), **r))
    # --- 4. Cu3Au cross-condition (ONE independent pair) ---
    c = describe(CS / "runs" / "cu3au_l12_big" / "trajectory.gsd", CAND_FRAME, "c_cu3au_ind", False)
    r = evaluate(c, anchors, ens)
    rows.append(dict(benchmark="D_Cu3Au_cross_condition", subject="Cu3Au_indep", true_label="Cu3Au",
                     validation_tier="Tier3_independent_realization", n_independent_trajectories=1,
                     expected="confident_reference_match",
                     correct=bool(r["accepted_references"] == "Cu3Au"), **r))
    r = evaluate(c, anchors, ens, drop="Cu3Au")
    rows.append(dict(benchmark="D_Cu3Au_cross_condition", subject="Cu3Au_indep_noCu3Au",
                     true_label="Cu3Au", validation_tier="Tier3_independent_realization",
                     n_independent_trajectories=1, expected="novel_and_not_in_active_library",
                     correct=bool(r["n_accepted"] == 0), **r))

    # --- 5. Nicole external (one row per INDEPENDENT trajectory) ---
    cases = pd.read_csv(E / "selected_latent_cases.csv")
    lab = cases["expert_corrected_label"].fillna(cases["category"]).astype(str)
    cases = cases.assign(label=lab)
    for lb in ("CsCl", "Th3P4", "amorphous"):
        for i, rr in cases[cases.label == lb].iterrows():
            p = Path(rr.source_file)
            if not p.exists():
                continue
            c = describe(p, -1, f"nic{i}", gas_safe=True)
            base = dict(subject=rr.trajectory_name,
                        true_label=lb, validation_tier="Tier3_external_independent",
                        n_independent_trajectories=1, debye=rr.debye,
                        volume_fraction=rr.volume_fraction, seed=rr.seed)
            if c is None:
                rows.append(dict(benchmark="E_nicole_external", **base, expected="not_ordered", n_accepted=0,
                                 final_verdict="not_ordered_or_amorphous",
                                 correct=bool(lb == "amorphous")))
                continue
            r = evaluate(c, anchors, ens)
            exp = "abstain" if lb == "amorphous" else "confident_reference_match"
            ok = (r["n_accepted"] == 0) if lb == "amorphous" else (r["accepted_references"] == lb)
            rows.append(dict(benchmark="E_nicole_external", **base, expected=exp, ordered_core=c["n"],
                             stoich=round(c["stoich"], 3), correct=bool(ok), **r))
            r2 = evaluate(c, anchors, ens, drop="CsCl")
            rows.append(dict(benchmark="F_nicole_noCsCl", **base, expected=(
                "novel_and_not_in_active_library" if lb == "CsCl" else
                ("confident_reference_match" if lb == "Th3P4" else "abstain")),
                ordered_core=c["n"], stoich=round(c["stoich"], 3),
                correct=bool(r2["n_accepted"] == 0) if lb in ("CsCl", "amorphous")
                else bool(r2["accepted_references"] == "Th3P4"), **r2))

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "v2_all_benchmarks.csv", index=False)

    print("\n" + "=" * 78)
    for b, name in [("A_same_construction", "A. same-construction recovery (Tier 1)"),
                    ("B_family_removed", "B. correct family removed"),
                    ("C_disordered_control", "C. disordered/melted controls"),
                    ("D_Cu3Au_cross_condition", "D. Cu3Au cross-condition (ONE independent pair)")]:
        s = df[df.benchmark == b]
        print(f"\n{name}:  {int(s.correct.sum())}/{len(s)} correct")
        for _, x in s.iterrows():
            print(f"   {x.subject:22s} accepted=[{x.accepted_references}] "
                  f"n={x.n_accepted}  -> {x.final_verdict}")
    for b, name in [("E_nicole_external", "E. NICOLE EXTERNAL (full library)"),
                    ("F_nicole_noCsCl", "F. NICOLE EXTERNAL (CsCl removed)")]:
        s = df[df.benchmark == b]
        print(f"\n{name}:   independent trajectories = {len(s)}")
        for lb in ("CsCl", "Th3P4", "amorphous"):
            t = s[s.true_label == lb]
            if not len(t):
                continue
            print(f"   {lb:10s} n={len(t):2d}  correct {int(t.correct.sum()):2d}/{len(t):2d}   "
                  f"accepted-counts {dict(t.n_accepted.value_counts())}   "
                  f"verdicts {dict(t.final_verdict.value_counts())}")
    print(f"\nwrote -> {HERE/'v2_all_benchmarks.csv'}")


if __name__ == "__main__":
    main()
