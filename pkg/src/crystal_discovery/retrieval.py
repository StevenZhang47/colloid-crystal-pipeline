"""Stage 4: which known prototype is this closest to?

Separate from detection on purpose. The detector answers "is this in the atlas"; it never
names a family. Retrieval answers "which CIF prototype does this most resemble", against a
bank of ideal lattices, and it runs whether the domain was called KNOWN or OOD -- a novel
domain still has a nearest neighbour, and that is often the useful information.

THE DISTANCE: one-sided chamfer, median reduction
-------------------------------------------------
For every centre in the candidate, find the nearest environment in the anchor, and take
the MEDIAN of those distances (per species, then averaged over the two species).

It is not a distance between summaries. The earlier matcher averaged each sublattice to a
single 1904-D vector and compared centroids; that is what let amorphous domains pass, since
a diffuse cloud whose CENTRE sits near a prototype's centre reads as that prototype no
matter how spread out it is. Chamfer never forms a centroid, so a candidate is charged for
every particle whose environment no prototype can account for.

Two design choices, both measured rather than assumed (chamfer_direction_test.py,
acceptance_test.py, on 50 held-out validation trajectories against 41 anchors):

  ONE-SIDED, candidate -> anchor.  The reverse question ("are the anchor's sites present
  in the candidate?") is satisfied trivially by a diffuse cloud -- spread widely enough, it
  has some particle near almost any environment -- so being MORE disordered makes it pass
  more easily. Measured: forward separates crystal from amorphous by 1.44x, reverse by
  1.02x, and the symmetric textbook Chamfer distance by 1.25x because it averages the good
  signal with the bad one.

  MEDIAN, not mean.  A clean CsCl crystal has median nearest-anchor distance 0.117 but
  p99 = 0.726: about 1% of centres -- defects, and surface particles M3 did not catch --
  sit far from every anchor environment and inflate the mean. Amorphous is uniformly far
  and has no such tail to lose, so the median tightens only the crystal side: gap 1.44 ->
  1.50. A 90% trimmed mean scores identically, so the gain is robustness in general rather
  than any particular trim, and the median is the version with no free parameter.

TWO INDEPENDENT GATES
---------------------
Both were fitted on the same held-out split and both separate the populations completely:

  distance  crystals <= 0.1294, amorphous >= 0.1943   -> threshold 0.1619, gap 1.50x
  margin    crystals >= 3.309,  amorphous <= 1.590    -> threshold 2.45,   gap 2.08x

The margin is the stronger of the two. A domain must clear both.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_REFS = Path(__file__).resolve().parents[2] / "references"
CHAMFER_BANK = _REFS / "chamfer_bank.npz"
# User additions live in a SEPARATE file layered on at load time, so the shipped artifact
# keeps its checksum and an addition can be undone by deleting one file.
USER_BANK = _REFS / "user_bank.npz"
MEAN_BANK = _REFS / "prototype_fingerprints.npz"
MIN_CENTRES_PER_SPECIES = 25


@dataclass
class Match:
    prototype: str
    anchor: str
    distance: float


@dataclass
class RetrievalResult:
    ranking: list            # list[Match], nearest first, one row per PROTOTYPE
    retrieved: str | None    # best prototype, or None if it failed a gate
    margin: float            # runner-up distance / best distance
    verification: str        # PASS / FAIL(reason)
    method: str = "chamfer_median"


@lru_cache(maxsize=1)
def load_bank(path: str | None = None) -> dict:
    p = Path(path) if path else CHAMFER_BANK
    if not p.is_file():
        raise FileNotFoundError(
            f"chamfer bank not found: {p}\nBuild it with pkg/build_chamfer_bank.py")
    envs: dict[str, list] = {}
    proto: dict[str, str] = {}
    meta = {"anchors": []}
    sources = [p] + ([USER_BANK] if (path is None and USER_BANK.is_file()) else [])
    for src in sources:
        z = np.load(src, allow_pickle=False)
        m = json.loads(bytes(z["__meta__"]).decode())
        meta["anchors"] += [{**a, "bank": src.name} for a in m["anchors"]]
        for k in z.files:
            if k == "__meta__":
                continue
            name, si = k.rsplit("::", 1)
            envs.setdefault(name, [None, None])[int(si)] = z[k].astype(np.float32)
        proto.update({a["anchor"]: a["prototype"] for a in m["anchors"]})

    # Stack every anchor's environments into ONE matrix per species, remembering where
    # each anchor's block starts. Scoring is then two large matrix products instead of
    # 2 x 41 small ones -- same arithmetic, but BLAS is far better used on one big GEMM.
    # Measured on a 525-centre domain: 6.2 s -> well under 1 s.
    names = sorted(envs)
    cat, spans = {}, {}
    for s in (0, 1):
        blocks, off, sp = [], 0, {}
        for n in names:
            b = envs[n][s]
            sp[n] = (off, off + len(b))
            off += len(b)
            blocks.append(b)
        cat[s] = np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
        spans[s] = sp
    sq = {s: np.einsum("ij,ij->i", cat[s], cat[s]) for s in (0, 1)}
    return {"envs": envs, "prototype_of": proto, "meta": meta,
            "names": names, "cat": cat, "spans": spans, "cat_sq": sq}


def _chamfer_median(cand: list, anchor: list) -> float:
    """Median nearest-anchor-environment distance, per species, averaged over species."""
    out = []
    for s in (0, 1):
        X, Y = cand[s], anchor[s]
        d2 = (np.einsum("ij,ij->i", X, X)[:, None]
              + np.einsum("ij,ij->i", Y, Y)[None, :] - 2.0 * X @ Y.T)
        out.append(float(np.median(np.sqrt(np.maximum(d2, 0.0)).min(axis=1))))
    return float(np.mean(out))


def _split_by_species(domain) -> list:
    parts = []
    for s in (0, 1):
        q = domain.species == s
        if q.sum() < MIN_CENTRES_PER_SPECIES:
            raise ValueError(f"too few centres on sublattice {s}: {int(q.sum())}")
        parts.append(np.ascontiguousarray(domain.soap[q], dtype=np.float32))
    return parts


def retrieve(domain, cfg, bank: dict | None = None, top_k: int = 3) -> RetrievalResult:
    b = bank or load_bank()
    cand = _split_by_species(domain)

    # Squared distances from every candidate centre to every environment in the library,
    # one GEMM per species. sqrt is deferred: the nearest environment by squared distance
    # is the nearest by distance, so the root is taken once per centre after the min
    # rather than once per (centre, environment) pair.
    nearest = {}
    for s in (0, 1):
        X, C = cand[s], b["cat"][s]
        d2 = (np.einsum("ij,ij->i", X, X)[:, None] + b["cat_sq"][s][None, :]
              - 2.0 * (X @ C.T))
        nearest[s] = d2

    # Collapse to one row per PROTOTYPE. The library holds several cell settings of the
    # same structure (four Th3P4, two each of CsCl, Cu3Au, Ni3Sn, fluorite); two settings
    # sit at ~0 from each other, which would break the runner-up test for a reason having
    # nothing to do with the candidate.
    best_per: dict[str, tuple[float, str]] = {}
    for name in b["names"]:
        per_species = []
        for s in (0, 1):
            lo, hi = b["spans"][s][name]
            m = nearest[s][:, lo:hi].min(axis=1)
            per_species.append(float(np.median(np.sqrt(np.maximum(m, 0.0)))))
        v = float(np.mean(per_species))
        proto = b["prototype_of"].get(name, name)
        if proto not in best_per or v < best_per[proto][0]:
            best_per[proto] = (v, name)
    ranking = [Match(prototype=p, anchor=n, distance=round(v, 4))
               for p, (v, n) in sorted(best_per.items(), key=lambda kv: kv[1][0])]

    best = ranking[0]
    second = ranking[1].distance if len(ranking) > 1 else float("inf")
    margin = second / max(best.distance, 1e-12)

    d_tau = float(cfg.retrieval["distance_threshold"])
    m_tau = float(cfg.retrieval["margin_threshold"])
    fails = []
    if best.distance > d_tau:
        fails.append(f"distance {best.distance:.4f} > {d_tau:.4f}")
    if margin < m_tau:
        fails.append(f"margin {margin:.3f} < {m_tau:.3f}")
    verdict = "PASS" if not fails else "FAIL(" + "; ".join(fails) + ")"
    return RetrievalResult(ranking=ranking[:top_k],
                           retrieved=None if fails else best.prototype,
                           margin=round(margin, 3), verification=verdict)
