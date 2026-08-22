#!/usr/bin/env python3
"""Re-judge the 16 "novel_ordered" snowmen: both matchers, 37 anchors, no circular gate.

THREE defects in the original verdicts, which compound:

1. SEVEN anchors. The library now holds 37. "Unmatched" only ever means "unlike the
   things you compared it to".

2. The PRE-FIX zangenite anchor (uniform scale, c/a 0.4804) rather than the
   anisotropically relaxed one (c/a 0.5383) from hockyg's develop-crystal-TI fix. Nine
   of the sixteen had zangenite as their nearest family.

3. A CIRCULAR SEGMENTATION GATE. match_points keeps a sub-segmentation only if >=2 of the
   pieces CONFIDENTLY MATCH the library:

       matched = [s for s in sub_scored if s["verdict"] == "matched"]
       if len(matched) >= 2: ... else: score the MERGED blob

   So when two crystals are fused and the library cannot name at least two of the pieces,
   a CORRECT split is discarded and the merged blob is scored instead -- which of course
   matches nothing, and lands in novel_ordered. Segmentation quality becomes a function of
   library coverage. With 7 anchors that gate was very tight.

This script removes the gate: it reports the whole domain AND every sub-domain the
segmenter finds, labelled, so a merged-blob artifact is visible instead of hidden. The
segmenter itself is unchanged and is already the right one -- orientational grains via
COMPLEX q6m coherence (which separates two same-structure grains at different
orientations, something rotationally-invariant q4/q6/q8 cannot do), falling back to
bond-order magnitude clustering for two different structures fused.

Both matchers score every piece:
  EMD   partial g(r) + total g(r) + powder S(k), 1-D Wasserstein, anchor-scaled
  SOAP  1904-D species-resolved [LARGE mean | SMALL mean], Euclidean
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
CS = ROOT / "crystal_discovery/crystal_stability"
sys.path.insert(0, str(HERE))

import build_and_run_matcher as M          # noqa: E402
import match_multi_domain as MD            # noqa: E402
import soap_vs_emd_matcher as S            # noqa: E402
from snowmen_sweep_match import body_lobe  # noqa: E402

SWEEPS = {"SweepLong": Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLong"),
          "SweepLongRetry": Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLongRetry")}
SOAP_D_TAU, SOAP_M_TAU = 0.2404109239578247, 1.117297887802124


def load_anchors():
    ok = S.cohesive_anchor_runs()
    emd, soap, proto = {}, {}, {}
    for d in sorted((CS / "runs").glob("lib_*")):
        if ok and d.name not in ok:
            continue
        g = d / "first_frame.gsd"
        if not g.is_file():
            continue
        name = d.name[len("lib_"):]
        try:
            e = M.describe(g, 0, extract_domain=False)
            fp = S.fingerprint(g, 0, is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip anchor] {name}: {exc}", flush=True); continue
        if fp is None:
            continue
        emd[name], soap[name], proto[name] = e, fp, S.prototype_of(name)
    return emd, soap, proto


def collapse(dist, proto):
    """One row per PROTOTYPE -- several cell settings of one structure sit at ~0 from
    each other and would break any runner-up test for an irrelevant reason."""
    best = {}
    for a, v in dist.items():
        p = proto[a]
        if p not in best or v < best[p][0]:
            best[p] = (float(v), a)
    o = sorted(best.items(), key=lambda kv: kv[1][0])
    d0 = o[0][1][0]
    d1 = o[1][1][0] if len(o) > 1 else np.inf
    return o[0][0], d0, d1 / max(d0, 1e-12), [(p, round(v, 4)) for p, (v, _) in o[:4]]


def score_piece(pos, tid, box, mask, emd_a, soap_a, proto, scales, tmpdir, tag):
    out = {"n": int(mask.sum())}
    d = MD.describe_domain(pos, tid, box, mask)
    de = {a: float(np.mean([M.channel_distances(d, emd_a[a])[c] / max(scales[c], 1e-12)
                            for c in M.CHANNELS])) for a in emd_a}
    b, dd, mg, top = collapse(de, proto)
    out.update(emd_nearest=b, emd_D=round(dd, 4), emd_margin=round(mg, 3),
               emd_verdict="matched" if dd <= M.ACCEPT_D and mg >= M.MARGIN_MIN else "unmatched",
               emd_top4=str(top), bragg=d.get("bragg_coherence"), ls=round(d["ls"], 3))

    p = Path(tmpdir) / f"{tag}.gsd"
    import gsd.hoomd
    f = gsd.hoomd.Frame()
    f.particles.N = int(mask.sum())
    f.particles.position = np.asarray(pos[mask], np.float32)
    f.particles.typeid = np.asarray(tid[mask], np.uint32)
    f.particles.types = ["LARGE", "SMALL"]
    # canonical species downstream is assigned BY DIAMETER, so these must differ
    f.particles.diameter = np.where(tid[mask] == 0, 256.0, 102.0).astype(np.float32)
    f.configuration.box = [float(box[0]), float(box[1]), float(box[2]), 0, 0, 0]
    with gsd.hoomd.open(str(p), "w") as t:
        t.append(f)
    try:
        # is_anchor=True == no gas screen, no M3. Both are WRONG for a piece that has
        # already been segmented here, and M3 is actively destructive on snowmen:
        # it excludes a particle when exposure > 0.14 AND coordination < 13, and a
        # snowman lobe is a bump on the outside of its molecule, so it always points
        # outward with low coordination. Measured on lobe_ratio_1.0_seed_ratio_0.4:
        # 6750 particles in, gas screen keeps 6712, M3 then keeps 2445 -- ALL of them
        # LARGE, zero SMALL. The entire lobe sublattice is annihilated, which is why
        # 9 of 16 domains returned "insufficient centres".
        #
        # M3 assumes the two species are geometrically interchangeable inside the
        # crystal. In a molecular system where one species is permanently exterior that
        # is false. So the segmented piece is described directly, which ALSO makes the
        # comparison like-for-like: EMD's describe_domain uses the mask as-is, and
        # AUDIT.md 4.2 requires both representations to see the same preprocessed domain.
        fp = S.fingerprint(p, 0, is_anchor=True)
    except Exception as exc:  # noqa: BLE001
        out["soap_verdict"] = f"error: {type(exc).__name__}"; return out
    if fp is None:
        out["soap_verdict"] = "insufficient centres"; return out
    ds = {a: float(np.linalg.norm(soap_a[a] - fp)) for a in soap_a}
    b, dd, mg, top = collapse(ds, proto)
    out.update(soap_nearest=b, soap_D=round(dd, 4), soap_margin=round(mg, 3),
               soap_verdict="matched" if dd <= SOAP_D_TAU and mg >= SOAP_M_TAU else "unmatched",
               soap_top4=str(top))
    return out


def main():
    t0 = time.perf_counter()
    tg = pd.read_csv(HERE / "snowmen_sweep_analysis/snowmen_sweep_results.csv")
    tg = tg[tg.bucket == "novel_ordered"]
    emd_a, soap_a, proto = load_anchors()
    scales = M.anchor_channel_scales(emd_a)
    print(f"{len(emd_a)} anchors, {len(set(proto.values()))} prototypes; "
          f"re-judging {len(tg)} domains\n", flush=True)

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for r in tg.itertuples(index=False):
            traj = SWEEPS[r.sweep] / r.run / "trajectory.gsd"
            base = {"sweep": r.sweep, "run": r.run, "old_domain": r.domain,
                    "old_nearest": r.nearest, "old_D": r.D_best, "path": str(traj)}
            try:
                import gsd.hoomd
                with gsd.hoomd.open(str(traj), "r") as t:
                    pos, tid, box = body_lobe(t[-1])
                nn = MD.robust_first_shell_distance(pos, box)
                ordered = MD.order_mask(pos, box, nn)
                pairs = MD.first_shell_pairs(pos, box, 1.35 * nn)
                comp = MD.components(len(pos), pairs, ordered)
                ids = [c for c in sorted(set(comp[comp >= 0]))
                       if (comp == c).sum() >= MD.MIN_DOMAIN]
                for c in ids:
                    mask = comp == c
                    rows.append({**base, "piece": f"d{c}", "kind": "whole",
                                 **score_piece(pos, tid, box, mask, emd_a, soap_a, proto,
                                               scales, td, f"{r.run}_d{c}")})
                    subs = MD.subsegment(pos, box, mask, nn, pairs, MD.MIN_DOMAIN)
                    if len(subs) >= 2:                 # report ALL pieces, no match gate
                        for i, sm in enumerate(subs):
                            rows.append({**base, "piece": f"d{c}.{i}", "kind": "subdomain",
                                         **score_piece(pos, tid, box, sm, emd_a, soap_a,
                                                       proto, scales, td, f"{r.run}_d{c}_{i}")})
            except Exception as exc:  # noqa: BLE001
                rows.append({**base, "piece": "-", "kind": "error",
                             "error": f"{type(exc).__name__}: {exc}"})
            last = rows[-1]
            print(f"  {r.run:36s} {last.get('piece','-'):8s} {last.get('kind',''):10s} "
                  f"EMD {str(last.get('emd_nearest')):>18s} {str(last.get('emd_verdict')):>10s} | "
                  f"SOAP {str(last.get('soap_nearest')):>18s} {str(last.get('soap_verdict')):>10s}",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "snowmen_dual_matcher.csv", index=False)
    if "emd_nearest" in df:
        df["agree"] = (df.emd_nearest == df.soap_nearest) & (df.emd_verdict == df.soap_verdict)
        print(f"\npieces {len(df)}   agree {int(df.agree.sum())}   "
              f"split into subdomains {int((df.kind=='subdomain').sum())}")
    print(f"({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
