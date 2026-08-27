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


def anchor_environments(gsd_path, frame_index: int = 0) -> list:
    """Per-particle SOAP of a CIF-ideal anchor, split by species. No preprocessing.

    Anchors are built lattices: there is no gas in them and no thermal disorder, so the
    gas screen has nothing to find and every particle it removes is a false positive. It
    removes a lot -- on an ideal ReO3 anchor the frozen screen deleted the entire majority
    sublattice (5324 particles in, 1330 out), because it assumes ~4 near contacts.

    Candidates ARE screened, and that asymmetry was tested rather than assumed: removing
    candidate preprocessing made every matcher worse, taking the mean-SOAP method from 0
    to 26 of 26 amorphous domains leaking through.

    Returns [LARGE (n_l, 952), SMALL (n_s, 952)], each L2-normalised.
    """
    P = _core()
    frame = P.read_frame(Path(gsd_path), frame_index)
    n = len(frame.positions)
    M3 = sys.modules["gas_m3_pipeline"]
    prep = M3.PipelineResult(
        n=n, retain_mask=np.ones(n, bool), component_id=np.zeros(n, int),
        same_coord=np.zeros(n, float), exposure=np.zeros(n, float),
        excluded_as_surface=np.zeros(n, bool), is_soap_center=np.ones(n, bool),
        canonical_species_id=M3._canonical_species(frame.diameters, frame.type_ids, n),
        gas_result=None)
    res = P.soap_global_d0(frame, prep, centers=np.arange(n))
    v = res.vectors
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    sp = prep.canonical_species_id[res.particle_indices]
    return [v[sp == 0], v[sp == 1]]
