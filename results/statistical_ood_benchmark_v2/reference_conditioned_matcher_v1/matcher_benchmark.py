#!/usr/bin/env python3
"""Head-to-head of candidate matchers on the two things that actually matter.

THE SCORING CRITERION
---------------------
Only two questions decide whether a matcher is usable:

  RANK-1   With the true prototype in the library, is it ranked FIRST? Ranks 2..n are
           not scored -- being "second best" is not a partial credit worth optimising.

  HOLDOUT  With every anchor of the true prototype DELETED, the candidate is a structure
           the library does not contain. It must then match nothing, or barely. The number
           reported is how far the best surviving distance sits above the worst true match.

Those two combine into a single decisive statistic:

    gap = min(holdout distance over all cases) / max(true-match distance over all cases)

  gap > 1  a single global acceptance threshold separates every true match from every
           unknown-structure case. Any value in the window is a valid ACCEPT_D, and the
           pipeline is finished.
  gap < 1  the populations overlap. No threshold works, and `n_false_accepts` counts the
           holdout cases that slip in at the tightest threshold that still keeps all true
           matches.

METHODS COMPARED
----------------
  emd            frozen baseline: partial g(r) + total g(r) + powder S(k), 1-D Wasserstein,
                 scaled by the median inter-anchor distance per channel. Radial only.
  soap_mean      per-particle SOAP averaged within each species, concatenated, Euclidean.
                 Compares the AVERAGE local environment.
  soap_meanstd   mean and per-dimension standard deviation concatenated. Adds the WIDTH of
                 the environment distribution, which thermal disorder broadens.
  soap_mmd       Gaussian-kernel maximum mean discrepancy between the full per-particle
                 SOAP distributions, per species. Uses every moment, not just the first;
                 soap_mean is its linear-kernel special case.
  soap_chamfer   mean over candidate particles of the distance to the NEAREST anchor
                 environment. Asks "does every environment in the candidate exist somewhere
                 in the anchor", which is one-sided on purpose: an anchor may contain sites
                 the thermal candidate does not sample, but not the reverse.

Kernel bandwidth for MMD is the median heuristic computed ONCE over the pooled anchor
population and then held fixed, so it is a constant of the method and not refitted per
comparison.

  python matcher_benchmark.py --stage cache    # SOAP per-particle vectors -> npz
  python matcher_benchmark.py --stage eval     # all methods from the cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
HERE = ROOT / "results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1"
CS = ROOT / "crystal_discovery/crystal_stability"
CACHE = HERE / "benchmark_cache"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"))

import core3_pipeline as C3  # noqa: E402
from soap_vs_emd_matcher import anchor_prep, prototype_of  # noqa: E402

# Every family for which a thermal MD trajectory exists. The earlier study used four of
# these; the rest were already on disk and simply had not been wired into the benchmark.
CANDIDATE_RUNS = {
    "antifluorite_c1b": "fluorite",   "cr3si_a15": "Cr3Si_A15",
    "cscl": "CsCl",                   "cu3au_d6": "Cu3Au",
    "fe4n_lp10": "fe4n_lp10",         "fluorite_c1": "fluorite",
    "i3p_a3b_hp8": "i3p_a3b_hp8",     "ni3sn_d019": "Ni3Sn",
    "nickeline_b81": "nickeline_b81", "reo3_d09": "reo3_d09",
    "rocksalt_b1": "rocksalt_B1",     "th3p4": "Th3P4",
    "zangenite": "zangenite",         "zincblende_b3": "zincblende_b3",
}
CANDIDATE_FRAMES = [250, 300, 350, 400, 450, 500]
MIN_CENTERS = 25
SUBSAMPLE = 600          # particles per species for the pairwise methods
RNG_SEED = 0


# ------------------------------------------------------------------ stage 1: SOAP cache
def per_particle_soap(gsd_path: Path, frame_index: int, is_anchor: bool):
    """L2-normalised per-particle SOAP, split by species. Returns (LARGE, SMALL) or None."""
    frame = C3.read_frame(gsd_path, frame_index)
    if is_anchor:
        prep = anchor_prep(frame)
        centers = np.arange(len(frame.positions))
    else:
        prep = C3.preprocess(frame)
        centers = np.flatnonzero(prep.is_soap_center)
        if len(centers) == 0:
            return None
        comps, counts = np.unique(prep.component_id[centers], return_counts=True)
        centers = centers[prep.component_id[centers] == comps[np.argmax(counts)]]

    res = C3.soap_global_d0(frame, prep, centers=centers)
    v = res.vectors
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    sp = prep.canonical_species_id[res.particle_indices]
    out = [v[sp == s] for s in (0, 1)]
    if any(len(x) < MIN_CENTERS for x in out):
        return None
    return out


def anchor_sources(mode: str) -> dict[str, tuple[Path, int]]:
    """Which frame of each anchor to use. `warm` = the thermalised pick, `cold` = the 0 K
    lattice the library shipped with, kept so the two can be compared directly."""
    src = {}
    if mode == "cold":
        for d in sorted((CS / "runs").glob("lib_*")):
            if (d / "first_frame.gsd").is_file():
                src[d.name[4:]] = (d / "first_frame.gsd", 0)
        return src
    sel = json.loads((CS / "anchor_frames.json").read_text())["anchors"]
    for run, info in sel.items():
        if info.get("status") == "OK":
            src[run[4:]] = (CS / "runs" / run / "trajectory.gsd", info["frame"])
    return src


def stage_cache(mode: str) -> None:
    CACHE.mkdir(exist_ok=True)
    store, t0 = {}, time.perf_counter()

    for name, (path, fi) in anchor_sources(mode).items():
        try:
            r = per_particle_soap(path, fi, is_anchor=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] anchor {name}: {type(exc).__name__}: {exc}", flush=True); continue
        if r is None:
            print(f"  [skip] anchor {name}: too few centres", flush=True); continue
        store[f"anchor::{name}::L"], store[f"anchor::{name}::S"] = r
        print(f"  anchor {name:42s} frame {fi:4d}  {len(r[0])}L {len(r[1])}S", flush=True)

    for run, proto in CANDIDATE_RUNS.items():
        traj = CS / "runs" / run / "trajectory.gsd"
        if not traj.is_file():
            print(f"  [missing] candidate {run}", flush=True); continue
        for fi in CANDIDATE_FRAMES:
            try:
                r = per_particle_soap(traj, fi, is_anchor=False)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {run} f{fi}: {type(exc).__name__}: {exc}", flush=True); continue
            if r is None:
                print(f"  [skip] {run} f{fi}: too few centres", flush=True); continue
            store[f"cand::{run}::{fi}::L"], store[f"cand::{run}::{fi}::S"] = r
            print(f"  cand {run:24s} f{fi}  {len(r[0])}L {len(r[1])}S", flush=True)

    np.savez_compressed(CACHE / f"soap_{mode}.npz", **store)
    print(f"\ncached {len(store)//2} objects in {time.perf_counter()-t0:.0f}s -> "
          f"{CACHE / f'soap_{mode}.npz'}")


# ------------------------------------------------------------------ stage 2: the methods
def _sub(x, rng):
    return x if len(x) <= SUBSAMPLE else x[rng.choice(len(x), SUBSAMPLE, replace=False)]


def m_soap_mean(c, a):
    f = lambda z: np.concatenate([z[0].mean(0), z[1].mean(0)])  # noqa: E731
    u, v = f(c), f(a)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return float(np.linalg.norm(u - v))


def m_soap_meanstd(c, a):
    f = lambda z: np.concatenate([z[0].mean(0), z[0].std(0), z[1].mean(0), z[1].std(0)])  # noqa: E731
    u, v = f(c), f(a)
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return float(np.linalg.norm(u - v))


def _self_k(z, gamma, memo, key):
    """k(x,x') averaged within one set. Depends only on that set, so it is computed once
    per anchor and once per candidate rather than once per (candidate, anchor) pair."""
    if key not in memo:
        memo[key] = [float(np.exp(-gamma * _sqd(z[s], z[s])).mean()) for s in (0, 1)]
    return memo[key]


def m_soap_mmd(c, a, gamma, memo, ckey, akey):
    kxx, kyy = _self_k(c, gamma, memo, ckey), _self_k(a, gamma, memo, akey)
    tot = 0.0
    for s in (0, 1):
        kxy = np.exp(-gamma * _sqd(c[s], a[s])).mean()
        tot += max(kxx[s] + kyy[s] - 2 * kxy, 0.0)
    return float(np.sqrt(tot / 2))


N_PROJ = 256             # sliced-Wasserstein projections; a convergence knob, not a tuning one


def m_soap_sw(c, a, proj):
    """Sliced 2-Wasserstein between the per-particle SOAP DISTRIBUTIONS, per species.

    This is idea (1): stop averaging. Cu3Au (L1_2) puts its majority species on ONE
    Wyckoff site (3c); Al3Ti (D0_22) splits its majority across TWO (1b, 2d, ratio 1:2).
    Both are 1:3 stoichiometry with a single minority site, so the mean of D0_22's two
    environments lands next to L1_2's one and `soap_mean` cannot separate them by
    construction -- no distance applied to a mean can.

    Optimal transport between the full empirical distributions keeps the modes: mass
    sitting at two separated environments cannot be moved onto one cheaply. Exact OT in
    952-D needs a solver we do not have installed, so this uses the sliced form -- project
    onto random directions, take the exact 1-D Wasserstein (a sort) along each, average.
    It is a true metric, needs no regularisation constant and no cluster count, and
    reduces to the mean-difference only when both distributions are single points.
    """
    tot = 0.0
    for s_ in (0, 1):
        x = np.sort(c[s_] @ proj, axis=0)
        y = np.sort(a[s_] @ proj, axis=0)
        if len(x) != len(y):                       # equal-size quantile grid
            q = np.linspace(0, 1, min(len(x), len(y)))
            x = np.quantile(x, q, axis=0); y = np.quantile(y, q, axis=0)
        tot += float(np.sqrt(np.mean((x - y) ** 2)))
    return tot / 2


def m_soap_chamfer(c, a):
    tot = 0.0
    for s in (0, 1):
        tot += np.sqrt(np.maximum(_sqd(c[s], a[s]), 0)).min(axis=1).mean()
    return float(tot / 2)


def _sqd(X, Y):
    return np.maximum(
        (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0.0)


def stage_eval(mode: str) -> None:
    z = np.load(CACHE / f"soap_{mode}.npz")
    rng = np.random.default_rng(RNG_SEED)

    anchors, cands = {}, {}
    for k in z.files:
        p = k.split("::")
        if p[0] == "anchor" and p[2] == "L":
            anchors[p[1]] = [_sub(z[k], rng), _sub(z[f"anchor::{p[1]}::S"], rng)]
        elif p[0] == "cand" and p[3] == "L":
            cands[(p[1], int(p[2]))] = [_sub(z[k], rng), _sub(z[f"cand::{p[1]}::{p[2]}::S"], rng)]
    print(f"{len(anchors)} anchors, {len(cands)} candidate frames [{mode}]\n", flush=True)

    # Median heuristic, fixed once over the pooled anchor population.
    pool = np.concatenate([a[s] for a in anchors.values() for s in (0, 1)])
    pool = pool[rng.choice(len(pool), min(3000, len(pool)), replace=False)]
    med = np.median(np.sqrt(_sqd(pool, pool))[np.triu_indices(len(pool), 1)])
    gamma = 1.0 / (2 * med ** 2)
    print(f"MMD bandwidth: median pairwise SOAP distance {med:.4f} -> gamma {gamma:.4f}\n")

    memo: dict = {}
    proj = rng.normal(size=(next(iter(anchors.values()))[0].shape[1], N_PROJ))
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    methods = {
        "soap_mean": lambda c, a, ck, ak: m_soap_mean(c, a),
        "soap_meanstd": lambda c, a, ck, ak: m_soap_meanstd(c, a),
        "soap_mmd": lambda c, a, ck, ak: m_soap_mmd(c, a, gamma, memo, ck, ak),
        "soap_chamfer": lambda c, a, ck, ak: m_soap_chamfer(c, a),
        "soap_sw": lambda c, a, ck, ak: m_soap_sw(c, a, proj),
    }

    # Idea (2): the SELF-REFERENCED scale. An absolute distance is not transferable --
    # clean MD true matches land at 0.09-0.26 while the external candidates needed
    # 0.35-0.37, which is noise level and not structure, and is why ACCEPT_D kept
    # breaking. Dividing by the candidate's own frame-to-frame distance under the SAME
    # method gives z = "how many candidate-noise-units away is this anchor", which is
    # dimensionless and should carry a threshold across noise levels.
    #
    # The denominator uses other frames of the same trajectory, so it measures THERMAL
    # spread at fixed defect realisation. Independent seeds would also fold in defect
    # variation; those runs exist (*_v2/_v3/_v4) and are the obvious next refinement.
    self_scale: dict = {}
    for (run, fi), c in sorted(cands.items()):
        others = [cands[(r, f)] for (r, f) in cands if r == run and f != fi]
        if not others:
            continue
        ck = ("c", run, fi)
        for mname, fn in methods.items():
            vals = [fn(c, o, ck, ("c", run, j))
                    for o, (r, j) in zip(others, [(r, f) for (r, f) in cands
                                                  if r == run and f != fi])]
            self_scale[(run, fi, mname)] = float(np.median(vals))
    print(f"self-scales computed for {len(self_scale)//len(methods)} candidate frames\n",
          flush=True)

    rows = []
    for (run, fi), c in sorted(cands.items()):
        truth, ck = CANDIDATE_RUNS[run], ("c", run, fi)
        for mname, fn in methods.items():
            D = {a: fn(c, anchors[a], ck, ("a", a)) for a in anchors}
            rows.append(_score(mname, run, fi, truth, D))
            ss = self_scale.get((run, fi, mname))
            if ss and ss > 0:
                rows.append(_score(mname + "_z", run, fi, truth,
                                   {a: v / ss for a, v in D.items()}))
        print(f"  scored {run} f{fi}", flush=True)

    _anchor_dispersion(anchors, mode)

    rows += _emd_rows(mode, sorted(cands))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / f"matcher_benchmark_{mode}.csv", index=False)
    _report(df, mode)


def _score(method, run, fi, truth, D):
    by_proto = {}
    for a, v in D.items():
        p = prototype_of(a)
        by_proto[p] = min(by_proto.get(p, np.inf), v)
    order = sorted(by_proto, key=by_proto.get)
    rank1 = order[0]
    hold = [p for p in order if p != truth]
    return dict(method=method, run=run, frame=fi, truth=truth,
                rank1=rank1, rank1_correct=rank1 == truth,
                D_true=round(by_proto.get(truth, np.nan), 5),
                D_rank1=round(by_proto[rank1], 5),
                holdout_best=hold[0] if hold else None,
                D_holdout=round(by_proto[hold[0]], 5) if hold else np.nan,
                holdout_ratio=round(by_proto[hold[0]] / max(by_proto.get(truth, np.nan), 1e-9), 3)
                if hold and truth in by_proto else np.nan)


def _emd_rows(mode, cand_keys):
    from build_and_run_matcher import CHANNELS, anchor_channel_scales, channel_distances, describe
    src = anchor_sources(mode)
    refs = {}
    for name, (path, fi) in src.items():
        try:
            refs[name] = describe(path, fi, extract_domain=False)
        except Exception as exc:  # noqa: BLE001
            print(f"  [emd skip anchor] {name}: {exc}", flush=True)
    scales = anchor_channel_scales(refs)

    def scaled(a, b):
        d = channel_distances(a, b)
        return float(np.mean([d[ch] / max(scales[ch], 1e-12) for ch in CHANNELS]))

    cands = {}
    for run, fi in cand_keys:
        try:
            cands[(run, fi)] = describe(CS / "runs" / run / "trajectory.gsd", fi,
                                        extract_domain=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [emd skip] {run} f{fi}: {exc}", flush=True)

    rows = []
    for (run, fi), c in sorted(cands.items()):
        D = {a: scaled(c, ref) for a, ref in refs.items()}
        rows.append(_score("emd", run, fi, CANDIDATE_RUNS[run], D))
        # same self-referenced scale as the SOAP methods
        others = [o for (r, f), o in cands.items() if r == run and f != fi]
        if others:
            ss = float(np.median([scaled(c, o) for o in others]))
            if ss > 0:
                rows.append(_score("emd_z", run, fi, CANDIDATE_RUNS[run],
                                   {a: v / ss for a, v in D.items()}))
        print(f"  scored emd {run} f{fi}", flush=True)
    return rows


def _anchor_dispersion(anchors, mode):
    """How spread out each anchor's environments are, per species.

    A prototype whose majority species occupies ONE Wyckoff site has a tight, unimodal
    SOAP cloud; one that splits across two sites has a wide, bimodal one. This is the
    Cu3Au/Al3Ti difference made visible, with no cluster count and no threshold -- just
    the mean distance from each environment to the species mean.
    """
    rows = []
    for name, z in sorted(anchors.items()):
        r = {"anchor": name, "prototype": prototype_of(name)}
        for s_, lab in ((0, "LARGE"), (1, "SMALL")):
            m = z[s_].mean(0)
            r[f"n_{lab}"] = len(z[s_])
            r[f"dispersion_{lab}"] = round(float(np.linalg.norm(z[s_] - m, axis=1).mean()), 5)
        rows.append(r)
    pd.DataFrame(rows).to_csv(HERE / f"anchor_dispersion_{mode}.csv", index=False)


def _report(df, mode):
    print(f"\n{'='*84}\nBENCHMARK [{mode} anchors]\n{'='*84}")
    summary = {}
    hdr = f"{'method':14s} {'rank1':>9s}  {'max D_true':>10s} {'min D_hold':>10s} {'gap':>6s} {'false':>6s}"
    print(hdr); print("-" * len(hdr))
    for m, g in df.groupby("method"):
        ok = g.dropna(subset=["D_true"])
        dt, dh = ok.D_true.max(), g.D_holdout.min()
        gap = dh / dt if dt > 0 else np.nan
        false_acc = int((g.D_holdout <= dt).sum())
        summary[m] = {
            "rank1_correct": f"{int(g.rank1_correct.sum())}/{len(g)}",
            "rank1_accuracy": round(float(g.rank1_correct.mean()), 4),
            "max_D_true": round(float(dt), 5), "min_D_holdout": round(float(dh), 5),
            "gap_ratio": round(float(gap), 3),
            "n_false_accepts_at_tightest_threshold": false_acc,
            "threshold_window": [round(float(dt), 5), round(float(dh), 5)] if gap > 1 else None,
            "families_rank1_wrong": sorted(set(g[~g.rank1_correct].run)),
        }
        print(f"{m:14s} {int(g.rank1_correct.sum()):4d}/{len(g):<4d}  "
              f"{dt:10.4f} {dh:10.4f} {gap:6.2f} {false_acc:6d}")
    (HERE / f"matcher_benchmark_{mode}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote matcher_benchmark_{mode}.csv / .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "eval"], required=True)
    ap.add_argument("--anchors", choices=["warm", "cold"], default="warm")
    a = ap.parse_args()
    (stage_cache if a.stage == "cache" else stage_eval)(a.anchors)
