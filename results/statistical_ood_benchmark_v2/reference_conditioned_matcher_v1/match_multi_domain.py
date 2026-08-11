#!/usr/bin/env python3
"""Multi-domain reference-conditioned matcher.

Answers: does the matcher work when a frame contains MORE THAN ONE crystal --
separated in space, or attached to each other?

The single-domain matcher (build_and_run_matcher.describe) extracts only the
LARGEST ordered cluster, so it would silently ignore every crystal but one. This
script wires the coexistence segmentation (diagnose_mixed_domains_v1) to the
reference-conditioned matcher:

  1. Order mask: q6 > 0.28 AND coordination >= 6 (same as the gas-safe extractor,
     but we keep ALL ordered particles, not just the largest cluster).
  2. Connected-component segmentation of the ordered particles on the adaptive
     first-shell graph (r_c = 1.35 * median nn). Two fused crystals separate here
     because their shared interface is disordered and drops out of the order mask.
  3. RE-FEATURISE EACH DOMAIN IN ISOLATION: partial-RDF + S(k) computed on that
     domain's own particles with that domain's own nn scale. This is load-bearing --
     frame-global features leak across domains (documented in
     mixed_domain_validation_report.md).
  4. Match each domain against the 7-family library with the anchor-scaled D and
     the open-set decision from build_and_run_matcher.
  5. Classify each domain pair separated vs attached (same connected component of
     the condensed / non-gas particles).

Library anchors + tau are loaded from build_and_run_matcher's cache; scales are the
global inter-anchor medians. Read-only on all references. Frozen v2/v3 channels.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as cc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_and_run_matcher as M  # noqa: E402
from partial_rdf import partial_rdf  # noqa: E402
from structure_factor import (  # noqa: E402
    raw_structure_factor, positive_shape, robust_first_shell_distance, coherence_metrics,
)

MIN_DOMAIN = 200


# ---------- segmentation ----------
def order_mask(pos, box, nn, coord_min=4):
    """Condensed mask (species-agnostic, NO q6): gas rejection by coordination only.
    Crystal-vs-amorphous is left to the matcher's S(k)/RDF, not a per-particle q6 cut,
    and composition is preserved because no species-biased threshold is applied."""
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    q = dict(r_max=1.4 * nn, exclude_ii=True)
    nl = freud.locality.AABBQuery(fbox, pos).query(pos, q).toNeighborList()
    coord = np.bincount(nl.query_point_indices, minlength=len(pos))
    return coord >= coord_min


def first_shell_pairs(pos, box, r_c):
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    nl = freud.locality.AABBQuery(fbox, pos).query(pos, dict(r_max=r_c, exclude_ii=True)).toNeighborList()
    return np.unique(np.sort(np.stack([nl.query_point_indices, nl.point_indices], 1), 1), axis=0)


def components(n, pairs, keep):
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return np.full(n, -1, int)
    remap = -np.ones(n, int); remap[idx] = np.arange(idx.size)
    m = keep[pairs[:, 0]] & keep[pairs[:, 1]]
    e = remap[pairs[m]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(idx.size, idx.size))
    _, lab = cc(g, directed=False)
    out = np.full(n, -1, int); out[idx] = lab
    return out


def orientational_grains(pos, box, mask, nn, min_size, coh_thr=0.6):
    """Separate same-structure grains by crystal ORIENTATION.

    Two grains of the same crystal (e.g. two CsCl crystallites) form one connected
    condensed component and have identical bond-order MAGNITUDES, so a magnitude
    fingerprint cannot tell them apart. Their orientations differ, though: we cut
    first-shell bonds whose complex q6m orientational coherence
    Re(<q6m_i, q6m_j*>)/(|q6m_i||q6m_j|) falls below coh_thr, then take connected
    components. Grain boundaries are low-coherence and get cut; grain interiors stay.
    Returns spatially-coherent grain masks >= min_size (>=2 only for real grains;
    a single crystal stays one grain, a low-order domain just fragments away and is
    handled by the whole-domain fallback).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc
    idx = np.flatnonzero(mask)
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    st = freud.order.Steinhardt(l=6)
    st.compute((fbox, pos), neighbors=dict(r_max=1.4 * nn, exclude_ii=True))
    qlm = np.nan_to_num(st.particle_harmonics)
    qn = qlm / (np.linalg.norm(qlm, axis=1, keepdims=True) + 1e-12)
    nl = freud.locality.AABBQuery(fbox, pos).query(
        pos, dict(r_max=1.35 * nn, exclude_ii=True)).toNeighborList()
    a, b = nl.query_point_indices, nl.point_indices
    s = np.real(np.sum(qn[a] * np.conj(qn[b]), axis=1))
    keep = (s > coh_thr) & mask[a] & mask[b]
    remap = -np.ones(len(pos), int); remap[idx] = np.arange(len(idx))
    e = remap[np.stack([a[keep], b[keep]], 1)]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(idx), len(idx)))
    _, lab = _cc(g, directed=False)
    comp = np.full(len(pos), -1, int); comp[idx] = lab
    return [comp == c for c in sorted(set(comp[comp >= 0])) if int((comp == c).sum()) >= min_size]


def subsegment(pos, box, mask, nn, pairs, min_size, k=2):
    """Split a merged domain into structurally-distinct subdomains, LABEL-FREE.

    Two mechanisms, tried in order:
      1. ORIENTATIONAL grains (same-structure crystallites in contact) -- if this
         finds >=2 spatially-coherent grains, use them.
      2. Bond-order MAGNITUDE clustering (different structures fused, e.g. CsCl+Th3P4)
         -- fallback when orientation gives a single grain.
    Returns a list of subdomain masks; [mask] if neither splits it.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    idx = np.flatnonzero(mask)
    if len(idx) < 2 * min_size:
        return [mask]

    grains = orientational_grains(pos, box, mask, nn, min_size)
    if len(grains) >= 2:
        return grains

    p = pos[idx]
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    q = dict(r_max=1.4 * nn, exclude_ii=True)
    fp = []
    for l in (4, 6, 8):
        st = freud.order.Steinhardt(l=l, average=True)
        st.compute((fbox, p), neighbors=q)
        fp.append(np.nan_to_num(st.particle_order))
    X = StandardScaler().fit_transform(np.stack(fp, 1))
    lab = KMeans(k, n_init=5, random_state=0).fit_predict(X)
    subs = []
    for c in range(k):
        sm = np.zeros(len(pos), bool); sm[idx[lab == c]] = True
        comp = components(len(pos), pairs, sm)
        for cc_id in set(comp[comp >= 0]):
            cmask = comp == cc_id
            if int(cmask.sum()) >= min_size:
                subs.append(cmask)
    return subs if len(subs) >= 2 else [mask]


def min_image(a, b, box):
    best = np.inf
    for i in range(0, len(a), 256):
        d = a[i:i + 256, None, :] - b[None, :, :]
        d -= box * np.round(d / box)
        best = min(best, float(np.sqrt((d ** 2).sum(-1)).min()))
    return best


# ---------- per-domain descriptors (in isolation) ----------
def canonical(frame):
    pos = np.asarray(frame.particles.position, np.float64)
    tid = np.asarray(frame.particles.typeid, int)
    dia = np.asarray(frame.particles.diameter, np.float64)
    box = np.asarray(frame.configuration.box[:3], np.float64)
    md = {int(t): float(dia[tid == t].mean()) for t in np.unique(tid)}
    order = sorted(md, key=lambda t: -md[t])
    remap = {old: new for new, old in enumerate(order)}
    return pos, np.array([remap[int(t)] for t in tid], int), box


def describe_domain(pos, tid, box, mask):
    """Descriptors for ONE domain, computed on that domain's own particles + nn."""
    nn = robust_first_shell_distance(pos[mask], box)
    r, gr = partial_rdf(pos, tid, mask, nn)
    k, sk = raw_structure_factor(pos[mask], nn, box)
    large = int((tid[mask] == 0).sum()); small = int((tid[mask] == 1).sum())
    coh = coherence_metrics(sk, int(mask.sum()), k)
    return {"r": r, "gr": gr, "k": k, "sk_shape": positive_shape(sk, k), "nn": nn,
            "n": int(mask.sum()), "ls": large / max(small, 1),
            "bragg_coherence": round(float(coh["n23_scaled_peak"]), 2)}


# ---------- driver ----------
def match_frame(path, anchors, tau, scales, frame_index=-1):
    with gsd.hoomd.open(str(path), "r") as tr:
        pos, tid, box = canonical(tr[frame_index])
    out = match_points(pos, tid, box, anchors, tau, scales)
    out["frame"] = str(path)
    return out


def match_points(pos, tid, box, anchors, tau, scales):
    """Core: segment a 2-species point set into domains (with sub-segmentation for
    attached crystals) and match each against the library. `tid` must be canonical
    (0=large, 1=small). Works on any 2-species points -- ionic crystals, or a
    molecular system reduced to its two charged lobes."""
    nn = robust_first_shell_distance(pos, box)
    r_c = 1.35 * nn
    ordered = order_mask(pos, box, nn)
    pairs = first_shell_pairs(pos, box, r_c)
    comp = components(len(pos), pairs, ordered)

    def score_mask(name, mask):
        desc = describe_domain(pos, tid, box, mask)
        res = M.score(desc, anchors, tau, scales, list(M.FAMILIES))
        return dict(domain=name, n=desc["n"], ls=round(desc["ls"], 3),
                    bragg_coherence=desc["bragg_coherence"],
                    centroid=[round(float(x), 0) for x in pos[mask].mean(0)], mask=mask,
                    matched=res["matched"], nearest=res["nearest"], verdict=res["verdict"],
                    D_best=round(res["D_best"], 4), margin=round(res["margin"], 3),
                    ranking={f: round(res["fam_scores"][f]["D"], 3) for f in M.FAMILIES})

    dom_ids = [c for c in sorted(set(comp[comp >= 0])) if (comp == c).sum() >= MIN_DOMAIN]
    domains = []
    for c in dom_ids:
        whole = score_mask(f"d{c}", comp == c)
        # Attempt sub-segmentation of any component (orientational grains, then
        # magnitude split). Keep the split only if it yields >=2 subdomains that each
        # MATCH CONFIDENTLY -- this covers two same-family grains in contact AND two
        # different structures fused, while a single crystal (which does not split, or
        # splits into pieces that do not both match) keeps its whole-domain result.
        subs = subsegment(pos, box, comp == c, nn, pairs, MIN_DOMAIN)
        if len(subs) >= 2:
            sub_scored = [score_mask(f"d{c}.{i}", m) for i, m in enumerate(subs)]
            matched = [s for s in sub_scored if s["verdict"] == "matched"]
            if len(matched) >= 2:                    # >=2 confidently-crystalline pieces
                domains.extend(sub_scored)
                continue
        domains.append(whole)

    # separated vs attached, via condensed (non-gas ordered-or-not) connectivity is
    # approximated here by the ordered graph: two ordered cores are 'attached' if the
    # minimum inter-core distance is within one first shell.
    rels = []
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            dmin = min_image(pos[domains[i]["mask"]], pos[domains[j]["mask"]], box)
            rels.append(dict(a=domains[i]["domain"], b=domains[j]["domain"],
                             min_distance=round(dmin, 1),
                             relation="attached" if dmin <= r_c else "separated"))
    for d in domains:
        d.pop("mask")
    return dict(n_particles=int(len(pos)), nn=round(nn, 1),
                n_ordered=int(ordered.sum()), n_domains=len(domains),
                domains=domains, relations=rels)


def _print_frame(tag, out):
    rel = out["relations"][0]["relation"] if out["relations"] else "-"
    print(f"\n=== {tag}: {out['n_domains']} domain(s), {out['n_ordered']}/{out['n_particles']} ordered, rel={rel} ===")
    for d in out["domains"]:
        print(f"   {d['domain']}: n={d['n']:4d} L/S={d['ls']:.3f} centroid={d['centroid']}  ->  "
              f"{d['verdict']:18s} matched={d['matched']} (D={d['D_best']}, margin={d['margin']})")
        print(f"       ranking: " + ", ".join(f"{f}={v}" for f, v in sorted(d['ranking'].items(), key=lambda x: x[1])))
    for r in out["relations"]:
        print(f"   relation {r['a']}-{r['b']}: {r['relation']} (min_dist={r['min_distance']})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsd", type=Path, default=None, help="score one file (else the synthetic suite)")
    ap.add_argument("--frame", type=int, default=-1)
    args = ap.parse_args()

    anchors, tau = pickle.load(open(HERE / "library_cache.pkl", "rb"))
    scales = M.anchor_channel_scales(anchors)

    if args.gsd is not None:
        out = match_frame(args.gsd, anchors, tau, scales, frame_index=args.frame)
        _print_frame(f"{args.gsd.name} frame {args.frame}", out)
        stem = args.gsd.stem.replace(" ", "_")
        (HERE / f"multi_domain_{stem}_f{args.frame}.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
        print(f"\nwrote multi_domain_{stem}_f{args.frame}.json")
        return

    base = Path("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/"
                "unified_pipeline/results/motif_pipeline/mixed_domain_diagnostics_v1/test_gsds")
    cases = ["single_cscl", "single_th3p4", "separated_cscl_th3p4", "attached_cscl_th3p4", "gas_plus_cscl"]
    all_out = {}
    rows = []
    for case in cases:
        out = match_frame(base / f"{case}.gsd", anchors, tau, scales)
        all_out[case] = out
        _print_frame(case, out)
        rel = out["relations"][0]["relation"] if out["relations"] else "-"
        for d in out["domains"]:
            rows.append(dict(case=case, domain=d["domain"], n=d["n"], ls=d["ls"],
                             matched=d["matched"], verdict=d["verdict"], D_best=d["D_best"],
                             margin=d["margin"], relation=rel))
    (HERE / "multi_domain_results.json").write_text(json.dumps(all_out, indent=2, default=str) + "\n")
    pd.DataFrame(rows).to_csv(HERE / "multi_domain_results.csv", index=False)
    print(f"\nwrote multi_domain_results.{{json,csv}}")


if __name__ == "__main__":
    main()
