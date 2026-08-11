#!/usr/bin/env python3
"""Full sweep: run the domain-separate + sub-segmentation + reference matcher on
every snowman run's final frame.

Each snowman is a charged ionic pair (body = +50 mV large sphere, lobe = -50 mV
medium sphere) plus a small head. We treat the body+lobe as a two-species ionic
system (drop the head) so the ionic reference matcher applies, then:

  reduce -> segment domains (coordination-based, no q6) -> sub-segment attached
  crystals (bond-order fingerprint) -> match each domain against the 7-family
  library -> record verdict + Bragg coherence per domain.

Output: one row per (run, domain) with a classification bucket:
  matched_<family>          confident library match
  novel_ordered             ordered (Bragg coherence high) but no library match -> curious
  ambiguous                 top-2 families too close, or borderline
  disordered                little/no Bragg coherence
  no_domain                 no condensed cluster found

Flags the confidently-classified vs the unsure ones for review. Read-only inputs.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import gsd.hoomd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_and_run_matcher as M          # noqa: E402
import match_multi_domain as MD            # noqa: E402

SWEEPS = [
    Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLong"),
    Path("/scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLongRetry"),
]
OUT = HERE / "snowmen_sweep_analysis"
# Calibrated to the ideal reference crystals: CsCl anchor coherence = 0.52,
# Th3P4 anchor = 0.23. A domain above ~0.18 has crystal-level Bragg order.
ORDER_MIN = 0.18       # Bragg coherence (n23-scaled S(k) peak) above this => ordered


def safe_float(s):
    import re
    m = re.match(r"[-+]?[0-9]*\.?[0-9]+", str(s))
    return float(m.group()) if m else float("nan")


def body_lobe(frame):
    """Two-species ionic subset: body (type0,+,large) + lobe (type1,-,small); drop head."""
    pos = np.asarray(frame.particles.position, float)
    tid = np.asarray(frame.particles.typeid, int)
    dia = np.asarray(frame.particles.diameter, float)
    box = np.asarray(frame.configuration.box[:3], float)
    keep = tid < 2                      # 0=body, 1=lobe; 2=head dropped
    p, t, d = pos[keep], tid[keep], dia[keep]
    # canonical: 0 = larger diameter. body(256) > lobe(102), so already canonical, but assert.
    md = {int(u): d[t == u].mean() for u in np.unique(t)}
    order = sorted(md, key=lambda u: -md[u])
    remap = {old: new for new, old in enumerate(order)}
    return p, np.array([remap[int(x)] for x in t], int), box


def bucket(dom):
    if dom["verdict"] == "matched":
        return f"matched_{dom['matched']}"
    coh = dom.get("bragg_coherence", 0) or 0
    if dom["verdict"] == "ambiguous":
        return "ambiguous"
    # library_unmatched: distinguish ordered-but-novel from disordered
    return "novel_ordered" if coh >= ORDER_MIN else "disordered"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anchors, tau = pickle.load(open(HERE / "library_cache.pkl", "rb"))
    scales = M.anchor_channel_scales(anchors)

    rows = []
    t0 = time.time()
    n_done = 0
    for sweep in SWEEPS:
        run_dirs = sorted([d for d in sweep.iterdir() if d.is_dir() and d.name.startswith("lobe_ratio")])
        for rd in run_dirs:
            final = rd / "final_frame.gsd"
            if not final.exists():
                final = rd / "trajectory.gsd"
            if not final.exists():
                continue
            parts = rd.name.replace("lobe_ratio_", "").split("_seed_ratio_")
            lobe = safe_float(parts[0]); seed = safe_float(parts[1]) if len(parts) > 1 else np.nan
            try:
                with gsd.hoomd.open(str(final), "r") as tr:
                    pos, tid, box = body_lobe(tr[-1])
                out = MD.match_points(pos, tid, box, anchors, tau, scales)
            except Exception as e:
                rows.append(dict(sweep=sweep.name, run=rd.name, lobe_ratio=lobe, seed_ratio=seed,
                                 error=str(e)[:90]))
                print(f"  [err] {rd.name}: {str(e)[:70]}", flush=True); continue
            rel = out["relations"][0]["relation"] if out["relations"] else "-"
            if not out["domains"]:
                rows.append(dict(sweep=sweep.name, run=rd.name, lobe_ratio=lobe, seed_ratio=seed,
                                 domain="-", bucket="no_domain", n_domains=0))
            for d in out["domains"]:
                rows.append(dict(sweep=sweep.name, run=rd.name, lobe_ratio=lobe, seed_ratio=seed,
                                 domain=d["domain"], n_domains=out["n_domains"],
                                 n_particles=d["n"], ls=d["ls"], bragg_coherence=d["bragg_coherence"],
                                 matched=d["matched"], nearest=d["nearest"], verdict=d["verdict"],
                                 D_best=d["D_best"], margin=d["margin"], relation=rel,
                                 bucket=bucket(d)))
            n_done += 1
            best = ", ".join(f"{d['domain']}:{bucket(d)}" for d in out["domains"]) or "no_domain"
            print(f"  {rd.name:40} ndom={out['n_domains']} rel={rel:9} -> {best}", flush=True)
            if n_done % 20 == 0:
                pd.DataFrame(rows).to_csv(OUT / "snowmen_sweep_results.csv", index=False)
                print(f"  ... {n_done} runs, {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "snowmen_sweep_results.csv", index=False)

    dom = df[df.get("bucket").notna() & (df.get("domain") != "-")] if "bucket" in df else df
    bc = df["bucket"].value_counts().to_dict() if "bucket" in df else {}
    classified = df[df.get("bucket", "").astype(str).str.startswith("matched")] if "bucket" in df else pd.DataFrame()
    curious = df[df.get("bucket") == "novel_ordered"] if "bucket" in df else pd.DataFrame()
    summary = dict(
        runtime_s=round(time.time() - t0, 1),
        n_runs=int(df["run"].nunique()),
        n_domain_rows=int(len(df)),
        bucket_counts=bc,
        classified_matches=classified[["run", "domain", "matched", "D_best"]].to_dict("records") if len(classified) else [],
        curious_novel_ordered=curious[["run", "domain", "nearest", "D_best", "bragg_coherence"]].to_dict("records") if len(curious) else [],
    )
    (OUT / "snowmen_sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n=== SUMMARY ===\n" + json.dumps({k: v for k, v in summary.items()
          if k in ("n_runs", "n_domain_rows", "bucket_counts")}, indent=2))
    print(f"classified matches: {len(classified)}   curious novel_ordered: {len(curious)}")
    print(f"wrote {OUT}/snowmen_sweep_results.csv and summary.json")


if __name__ == "__main__":
    main()
