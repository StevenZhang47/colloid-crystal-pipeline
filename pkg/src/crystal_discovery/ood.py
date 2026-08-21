"""Stage 3: is this domain in the atlas, or is it new?

Loads the frozen detector -- species standardisation, PCA16, Ledoit-Wolf location and
precision, and tau -- and scores a domain. Nothing is fitted here; the artifact is the
model. `rho = mahalanobis / tau`, and rho > 1 means "not in the atlas".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODEL = REPO / "results/motif_pipeline/core3_atlas_v3/frozen/ood_model_v3A_p99.npz"
FREEZE = REPO / "results/motif_pipeline/core3_atlas_v3/frozen/FREEZE.json"


@dataclass
class OODResult:
    score: float
    tau: float
    rho: float
    status: str               # "KNOWN" or "OOD"


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> dict:
    p = Path(path) if path else MODEL
    if not p.is_file():
        raise FileNotFoundError(
            f"frozen OOD model not found: {p}\n"
            "Regenerate with core3_atlas_v3/scripts/v3_13_freeze.py")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def freeze_record() -> dict:
    return json.loads(FREEZE.read_text()) if FREEZE.is_file() else {}


def summarise(domain, model: dict) -> np.ndarray:
    """32-D [LARGE mean | SMALL mean] in PCA space, using the frozen transform.

    Standardisation is species-conditioned -- LARGE and SMALL have their own mean and
    scale -- because the two sublattices occupy different regions of SOAP space and a
    shared scaling would let the majority species dominate.
    """
    x, sp = domain.soap, domain.species
    z = np.empty_like(x)
    for sid in (0, 1):
        q = sp == sid
        if q.any():
            z[q] = (x[q] - model["species_mean"][sid]) / model["species_scale"][sid]
    p = (z - model["pca_mean"]) @ model["pca_components"].T
    if not (sp == 0).any() or not (sp == 1).any():
        raise ValueError("domain lacks one of the two species; cannot form a fingerprint")
    return np.concatenate([p[sp == 0].mean(0), p[sp == 1].mean(0)])


def score_domain(domain, model: dict | None = None) -> OODResult:
    m = model or load_model()
    h = summarise(domain, m)
    d = h - m["location"]
    s = float(np.sqrt(max(float(d @ m["precision"] @ d), 0.0)))
    tau = float(m["tau"][0])
    rho = s / tau
    return OODResult(score=round(s, 3), tau=round(tau, 4), rho=round(rho, 3),
                     status="OOD" if rho > 1 else "KNOWN")
