#!/usr/bin/env python3
"""Self-match specificity of the EMD matcher against the EXPANDED reference library.

Question asked: with the library grown from 7 families to every prototype in
cif_library, does a CsCl / Th3P4 / Ni3Sn / Cu3Au crystal still match ONLY itself?

Design
------
CANDIDATES are thermal frames from each family's own crystal_stability trajectory
(frames 250..500), i.e. real 298 K realisations -- not the CIF-ideal lattice.

ANCHORS are the CIF-ideal `first_frame.gsd` of every library prototype that built
successfully (`crystal_stability/runs/lib_<slug>/`), plus the seven original matcher
anchors. A candidate's own family anchor is a DIFFERENT build from its trajectory
(re-designed colloid parameters, anisotropically relaxed lattice), so "matches
itself" is not a tautology.

The channel code, the distance D and the acceptance rule are imported unchanged from
build_and_run_matcher.py -- nothing about the matcher is reimplemented here.

Two normalisations are reported, because they are not interchangeable:

  D_orig  channel scales from the ORIGINAL 7 anchors. ACCEPT_D = 0.40 was calibrated
          on this scale, so this is the column the 0.40 cutoff may be applied to.
  D_exp   channel scales from the EXPANDED anchor set. Adding ~35 anchors changes the
          median inter-anchor distance, so every D moves; the 0.40 cutoff is NOT
          calibrated for this column and is not applied to it.

Near-duplicate anchors are expected and are reported, not hidden: the library holds
three Cu3Au cell settings, three Th3P4, two CsCl and two zangenite. Two anchors of the
same prototype sit at D ~ 0 from each other, which necessarily breaks the
MARGIN_MIN = 1.15 runner-up test. So each candidate gets two verdicts:

  strict     the matcher's own verdict over all anchors (duplicates included)
  prototype  correct iff the nearest anchor belongs to the right PROTOTYPE, with the
             margin computed against the best anchor of any OTHER prototype

Usage:  python expanded_library_selfmatch.py
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
sys.path.insert(0, str(HERE))

import inspect  # noqa: E402

from build_and_run_matcher import (  # noqa: E402
    CHANNELS, FAMILIES, MARGIN_MIN, anchor_channel_scales, channel_distances, describe, score,
)

# ACCEPT_D lives as a default argument of score() rather than a module constant, so
# take it from the signature instead of restating 0.40 and letting the two drift.
ACCEPT_D = float(inspect.signature(score).parameters["accept_D"].default)

# The four families the specificity test was asked for, and the run whose trajectory
# supplies the candidates.
TEST_FAMILIES = {"CsCl": "cscl", "Th3P4": "th3p4", "Ni3Sn": "ni3sn_d019", "Cu3Au": "cu3au_d6"}
CANDIDATE_FRAMES = [250, 300, 350, 400, 450, 500]

# Which library slug belongs to which prototype, for the prototype-level verdict.
# Only entries that need grouping are listed; anything else is its own prototype.
PROTOTYPE_OF = {
    "CsCl": "CsCl", "cscl_B2_anchor": "CsCl",
    "Th3P4__8edd1f72": "Th3P4", "Th3P4__72406369": "Th3P4",
    "Th3P4_mp-1347_primitive": "Th3P4", "th3p4_anchor": "Th3P4",
    "Cu3Au": "Cu3Au", "Cu3Au_mp-2258_conventional_standard": "Cu3Au", "cu3au_L12": "Cu3Au",
    "zangenite__471e7cda": "zangenite", "zangenite__98bb84a4": "zangenite",
    "ni3sn_D019": "Ni3Sn",
    # Li3Fe IS Ni3Sn (D0_19) under another name: the role-relabelled StructureMatcher
    # (minority -> A, majority -> B, then element-aware fit) puts them at rms 0.045 while
    # correctly separating the negative controls Cu3Au/ReO3, Ni3Sn/I3P, CsCl/NaTl and
    # Laves C14/C36. Omitting this makes a correct Ni3Sn match read as a wrong one, and
    # makes the leave-own-prototype-out arm fail to delete the prototype it means to.
    "Li3Fe": "Ni3Sn",
    # Same framework AND same sublattice roles; they differ only in which chemical
    # element sits where, which is not a distinction a colloid system can express.
    "fluorite_C1": "fluorite", "antifluorite_C1b": "fluorite",
    "cr3si_A15": "Cr3Si_A15", "rocksalt_B1": "rocksalt_B1",
}


_PROTOTYPE_LOWER = {k.lower(): v for k, v in PROTOTYPE_OF.items()}


def prototype_of(anchor_name: str) -> str:
    """Anchor name -> prototype, case-insensitively, with the orig_ prefix stripped.

    Anchor names come from directory names (`runs/lib_<slug>`, slug lowercased), so they
    do NOT preserve the catalog's casing: the CsCl anchor arrives as `cscl`, and
    Th3P4__8edd1f72 as `th3p4__8edd1f72`. A case-sensitive lookup silently returned the
    anchor name unchanged, so `correct_prototype` compared "cscl" against "CsCl" and
    reported False for every candidate -- the test would have read as a total
    specificity failure while the matcher was in fact matching correctly.

    The orig_ prefix marks the seven pre-existing matcher anchors; orig_CsCl and the
    library's cscl are the same prototype built twice, and must group together or every
    candidate is trivially "ambiguous" against its own duplicate.
    """
    name = anchor_name[len("orig_"):] if anchor_name.startswith("orig_") else anchor_name
    return _PROTOTYPE_LOWER.get(name.lower(), name)


def cohesive_runs() -> tuple[set, dict]:
    """Which lib_<slug> builds are admissible as references.

    A prototype whose colloid realisation came out DILUTE is not a reference for that
    prototype -- its RDF and S(k) describe a near-gas, not the structure. Admitting one
    would put a junk anchor in the library that candidates can be attracted to, and the
    specificity question being asked here is precisely about spurious attraction. So the
    library is filtered on the generation record's cohesion verdict rather than on
    "a first_frame.gsd exists".

    Known exclusions at the time of writing: AlB2 and the three Laves phases (C14/C15/
    C36) come out at nn/contact 1.84-1.91 and NaTl at 2.05, because those structures need
    a larger radius ratio than the fixed 102/120 nm pair provides (Laves ideal is 1.225,
    this pair gives 1.176). Deepening the pair well does not help -- it lengthens the
    Debye screening and pushes the lattice further apart still.
    """
    records = LIB / "generated" / "records"
    admissible, rejected = set(), {}
    for f in sorted(records.glob("*.json")):
        rec = json.loads(f.read_text())
        run = rec.get("run") or f"lib_{rec['slug'].lower()}"
        if rec.get("verdict") == "cohesive":
            admissible.add(run)
        else:
            rejected[run] = (rec.get("verdict") or rec.get("status", "?"))
    return admissible, rejected


def collect_anchors() -> dict:
    """CIF-ideal anchors: every COHESIVE lib_<slug> build, plus the original seven."""
    anchors, sources = {}, {}
    admissible, rejected = cohesive_runs()
    if rejected:
        print("  excluded as references (not a cohesive crystal):", flush=True)
        for run, why in sorted(rejected.items()):
            print(f"    {run:44s} {why}", flush=True)
    for run_dir in sorted((CS / "runs").glob("lib_*")):
        gsd = run_dir / "first_frame.gsd"
        if not gsd.is_file():
            continue
        if admissible and run_dir.name not in admissible:
            continue
        name = run_dir.name[len("lib_"):]
        try:
            anchors[name] = describe(gsd, 0, extract_domain=False)
            sources[name] = str(gsd.relative_to(ROOT))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip anchor] {name}: {exc}", flush=True)
    for fam, run in FAMILIES.items():
        key = f"orig_{fam}"
        gsd = CS / "runs" / run / "first_frame.gsd"
        if gsd.is_file():
            anchors[key] = describe(gsd, 0, extract_domain=False)
            sources[key] = str(gsd.relative_to(ROOT))
    return anchors, sources


def verdicts(cand, anchors, scales, true_prototype, accept_D):
    D = {}
    for a in anchors:
        d = channel_distances(cand, anchors[a])
        D[a] = float(np.mean([d[c] / max(scales[c], 1e-12) for c in CHANNELS]))
    order = sorted(D, key=lambda a: D[a])
    best = order[0]
    second = order[1] if len(order) > 1 else None

    # strict: the matcher's own rule over every anchor, duplicates included
    margin = (D[second] / max(D[best], 1e-9)) if second else np.inf
    if D[best] > accept_D:
        strict = "library_unmatched"
    elif margin < MARGIN_MIN:
        strict = "ambiguous"
    else:
        strict = "matched"

    # prototype-level: runner-up is the best anchor of a DIFFERENT prototype
    p_best = prototype_of(best)
    other = [a for a in order if prototype_of(a) != p_best]
    p_margin = (D[other[0]] / max(D[best], 1e-9)) if other else np.inf
    if D[best] > accept_D:
        proto_verdict = "library_unmatched"
    elif p_margin < MARGIN_MIN:
        proto_verdict = "ambiguous"
    else:
        proto_verdict = "matched"

    # anything OTHER than the true prototype that would also be accepted on distance
    false_accepts = [a for a in order
                     if prototype_of(a) != true_prototype and D[a] <= accept_D]
    return dict(
        nearest=best, nearest_prototype=p_best, D_best=D[best],
        runner_up=second, D_second=(D[second] if second else np.nan), margin=margin,
        strict_verdict=strict,
        prototype_runner_up=(other[0] if other else None),
        D_prototype_runner_up=(D[other[0]] if other else np.nan),
        prototype_margin=p_margin, prototype_verdict=proto_verdict,
        correct_prototype=(p_best == true_prototype),
        n_false_accept=len(false_accepts),
        false_accepts=";".join(false_accepts[:6]),
        all_D=D,
    )


def main() -> None:
    t0 = time.perf_counter()
    print("=== anchors ===", flush=True)
    anchors, sources = collect_anchors()
    print(f"{len(anchors)} anchors: {', '.join(sorted(anchors))}\n", flush=True)

    orig = {k: v for k, v in anchors.items() if k.startswith("orig_")}
    scales_orig = anchor_channel_scales(orig)
    scales_exp = anchor_channel_scales(anchors)
    print("channel scales (original 7):  " + json.dumps({k: round(v, 4) for k, v in scales_orig.items()}))
    print("channel scales (expanded):    " + json.dumps({k: round(v, 4) for k, v in scales_exp.items()}) + "\n",
          flush=True)

    rows, ranking_rows = [], []
    for fam, run in TEST_FAMILIES.items():
        traj = CS / "runs" / run / "trajectory.gsd"
        for fi in CANDIDATE_FRAMES:
            try:
                cand = describe(traj, fi, extract_domain=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {fam} frame {fi}: {exc}", flush=True)
                continue
            for tag, scales, accept in (("orig", scales_orig, ACCEPT_D),
                                        ("expanded", scales_exp, None)):
                res = verdicts(cand, anchors, scales, fam,
                               accept if accept is not None else np.inf)
                allD = res.pop("all_D")
                rows.append(dict(family=fam, frame=fi, scale_set=tag,
                                 accept_D=(accept if accept is not None else "not applied"),
                                 **{k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in res.items()}))
                if tag == "orig":
                    for a, d in sorted(allD.items(), key=lambda kv: kv[1]):
                        ranking_rows.append(dict(family=fam, frame=fi, anchor=a,
                                                 prototype=prototype_of(a), D_orig=round(d, 4)))
            r = rows[-2]
            print(f"  {fam} f{fi}: nearest={r['nearest']} ({r['nearest_prototype']}) "
                  f"D={r['D_best']:.3f} margin={r['margin']:.2f} "
                  f"strict={r['strict_verdict']} proto={r['prototype_verdict']} "
                  f"false_accepts={r['n_false_accept']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "expanded_selfmatch_results.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(HERE / "expanded_selfmatch_full_ranking.csv", index=False)
    pd.DataFrame([{"anchor": a, "prototype": prototype_of(a), "source": s}
                  for a, s in sorted(sources.items())]).to_csv(
        HERE / "expanded_selfmatch_anchors.csv", index=False)

    o = df.query("scale_set == 'orig'")
    summary = {
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "n_anchors": len(anchors),
        "n_candidate_frames": int(len(o)),
        "accept_D": ACCEPT_D,
        "per_family": {
            fam: {
                "nearest_is_correct_prototype":
                    f"{int(g.correct_prototype.sum())}/{len(g)}",
                "prototype_level_matched":
                    f"{int((g.prototype_verdict == 'matched').sum())}/{len(g)}",
                "strict_matched": f"{int((g.strict_verdict == 'matched').sum())}/{len(g)}",
                "max_other_prototype_accepted": int(g.n_false_accept.max()),
                "D_best_range": [round(float(g.D_best.min()), 3), round(float(g.D_best.max()), 3)],
                "prototype_margin_min": round(float(g.prototype_margin.min()), 3),
            } for fam, g in o.groupby("family")
        },
    }
    (HERE / "expanded_selfmatch_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== SUMMARY ===\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
