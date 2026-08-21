"""Stage 4: which known prototype is this closest to?

Separate from detection on purpose. The detector answers "is this in the atlas"; it never
names a family. Retrieval answers "which CIF prototype does this most resemble", against
a bank of ideal lattices, and it runs whether the domain was called KNOWN or OOD -- a
novel domain still has a nearest neighbour, and that is often the useful information.

The fingerprint is the mean SOAP of each sublattice concatenated (1904-D), L2-normalised,
compared by Euclidean distance. Keeping the two sublattices separate is what preserves
species information: a pooled average discards it, and measurement on this library showed
the discriminating signal lives almost entirely in the minority sublattice.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

BANK = Path(__file__).resolve().parents[2] / "references/prototype_fingerprints.npz"


@dataclass
class Match:
    prototype: str
    anchor: str
    distance: float


@dataclass
class RetrievalResult:
    ranking: list            # list[Match], nearest first, one row per PROTOTYPE
    retrieved: str | None    # best prototype, or None if it failed the checks
    margin: float            # runner-up distance / best distance
    verification: str        # PASS / FAIL(reason)


def fingerprint(domain) -> np.ndarray:
    parts = []
    for s in (0, 1):
        q = domain.species == s
        if q.sum() < 25:
            raise ValueError(f"too few centres on sublattice {s}: {int(q.sum())}")
        parts.append(domain.soap[q].mean(axis=0))
    fp = np.concatenate(parts)
    return fp / max(np.linalg.norm(fp), 1e-12)


@lru_cache(maxsize=1)
def load_bank(path: str | None = None) -> dict:
    p = Path(path) if path else BANK
    if not p.is_file():
        raise FileNotFoundError(
            f"prototype bank not found: {p}\nBuild it with pkg/build_reference_bank.py")
    z = np.load(p, allow_pickle=False)
    return {"names": z["names"], "prototypes": z["prototypes"], "fingerprints": z["fingerprints"]}


def retrieve(domain, cfg, bank: dict | None = None, top_k: int = 3) -> RetrievalResult:
    b = bank or load_bank()
    fp = fingerprint(domain)
    d = np.linalg.norm(b["fingerprints"] - fp, axis=1)

    # Collapse to one row per PROTOTYPE. The library holds several cell settings of the
    # same structure (three Cu3Au, four Th3P4, two CsCl), and two settings of one
    # prototype sit at distance ~0 from each other -- which would break the runner-up
    # test for a reason that has nothing to do with the candidate.
    best_per: dict[str, tuple[float, str]] = {}
    for dist, proto, name in zip(d, b["prototypes"], b["names"]):
        if proto not in best_per or dist < best_per[proto][0]:
            best_per[proto] = (float(dist), str(name))
    ranking = [Match(prototype=p, anchor=n, distance=round(v, 4))
               for p, (v, n) in sorted(best_per.items(), key=lambda kv: kv[1][0])]

    best = ranking[0]
    second = ranking[1].distance if len(ranking) > 1 else float("inf")
    margin = second / max(best.distance, 1e-12)

    d_tau = float(cfg.retrieval["distance_threshold"])
    m_tau = float(cfg.retrieval["margin_threshold"])
    if best.distance > d_tau:
        verdict, retrieved = f"FAIL(distance {best.distance:.3f} > {d_tau:.3f})", None
    elif margin < m_tau:
        verdict, retrieved = f"FAIL(margin {margin:.3f} < {m_tau:.3f})", None
    else:
        verdict, retrieved = "PASS", best.prototype
    return RetrievalResult(ranking=ranking[:top_k], retrieved=retrieved,
                           margin=round(margin, 3), verification=verdict)
