"""Stages 1-2: preprocessing and the SOAP descriptor.

This module deliberately does NOT reimplement either. It delegates to the frozen code
that produced the frozen model, because a second implementation is a second thing to
drift. Everything downstream of this file therefore sees exactly what the detector was
calibrated on.

  gas screen + M3   results/motif_pipeline/frozen_gas_m3_pipeline_v1/gas_m3_pipeline.py
  SOAP GLOBAL_D0    results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts/core3_pipeline.py

Both live in this repository, so the delegation is self-contained.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
_FROZEN = REPO / "results/motif_pipeline/frozen_gas_m3_pipeline_v1"
_SOAP = REPO / "results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts"


@lru_cache(maxsize=1)
def _core():
    for p in (str(_SOAP), str(_FROZEN)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import core3_pipeline as P
    return P


@dataclass
class Domain:
    index: int
    n_particles: int          # particles in the condensed component
    n_centres: int            # M3 reliable centres actually described
    n_large: int
    n_small: int
    ls_ratio: float
    soap: np.ndarray          # (n_centres, 952) L2-normalised
    species: np.ndarray       # 0 = LARGE, 1 = SMALL


def find_domains(gsd_path, frame_index: int = -1, min_centres: int = 64) -> list[Domain]:
    """Every condensed component large enough to describe, biggest first.

    A frame can hold more than one crystal. Scoring them together averages away the
    thing we are looking for, so each physical component is analysed separately.
    """
    P = _core()
    frame = P.read_frame(Path(gsd_path), frame_index)
    prep = P.preprocess(frame)
    centres_all = np.flatnonzero(prep.is_soap_center)
    if len(centres_all) == 0:
        return []

    out = []
    comps, counts = np.unique(prep.component_id[centres_all], return_counts=True)
    for comp in comps[np.argsort(-counts)]:
        if comp < 0:
            continue                                   # -1 is gas
        centres = centres_all[prep.component_id[centres_all] == comp]
        if len(centres) < min_centres:
            continue
        res = P.soap_global_d0(frame, prep, centers=centres)
        v = res.vectors
        v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        sp = prep.canonical_species_id[res.particle_indices]
        n_l, n_s = int((sp == 0).sum()), int((sp == 1).sum())
        out.append(Domain(
            index=len(out),
            n_particles=int((prep.component_id == comp).sum()),
            n_centres=len(centres), n_large=n_l, n_small=n_s,
            ls_ratio=round(n_l / max(n_s, 1), 3), soap=v, species=sp))
    return out
