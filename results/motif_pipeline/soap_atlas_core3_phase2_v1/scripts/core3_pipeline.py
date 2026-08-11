#!/usr/bin/env python3
"""Label-free preprocessing and normalized per-particle SOAP for Phase 2.

This module deliberately has no phase/label argument. Every descriptor call is
raw geometry -> frozen gas+M3 -> SOAP for every non-gas particle. M3 controls
only the stored pooling flag; excluded particles remain centers in this atlas
and remain neighborhood context.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP
from scipy.spatial import cKDTree

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
FROZEN = ROOT / "results/motif_pipeline/frozen_gas_m3_pipeline_v1"
sys.path.insert(0, str(FROZEN))
import gas_m3_pipeline as M3

Z_LARGE = 11
Z_SMALL = 17
SPECIES = [Z_LARGE, Z_SMALL]
N_MAX = 8
L_MAX = 6
R_CUT_NORM = 3.0
SIGMA_NORM = 0.10
ABSOLUTE_D0_REFERENCE = 207.9312728721067
SOAP_DIMENSION = 952
REPRESENTATIONS = ("PAIR_CONTACT", "GLOBAL_D0", "ABSOLUTE_DISTANCE_CONTROL")


@dataclass
class FrameData:
    positions: np.ndarray
    diameters: np.ndarray
    type_ids: np.ndarray
    type_names: list[str]
    box: np.ndarray
    frame_index: int
    brush_lengths: np.ndarray
    brush_length_missing: np.ndarray


@dataclass
class SoapResult:
    particle_indices: np.ndarray
    vectors: np.ndarray
    d0_by_component: dict[int, float]
    neighbor_counts: np.ndarray


def read_brush_length(gsd_path: str | Path) -> tuple[float, bool]:
    """Read the scalar brush length from run.yaml without invoking custom YAML tags."""
    run_yaml = Path(gsd_path).parent / "run.yaml"
    if not run_yaml.exists():
        return np.nan, True
    lines = run_yaml.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("brush_length:"):
            for child in lines[i + 1:i + 7]:
                # Stop when the next top-level YAML key begins.
                if child and not child[0].isspace():
                    break
                match = re.match(r"\s*value:\s*([-+0-9.eE]+)\s*$", child)
                if match:
                    value = float(match.group(1))
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid brush length {value} in {run_yaml}")
                    return value, False
            break
    return np.nan, True


def read_frame(gsd_path: str | Path, frame_index: int) -> FrameData:
    import gsd.hoomd

    with gsd.hoomd.open(str(gsd_path), mode="r") as trajectory:
        snapshot = trajectory[int(frame_index)]
        positions = np.asarray(snapshot.particles.position, dtype=float)
        diameters = np.asarray(snapshot.particles.diameter, dtype=float)
        type_ids = np.asarray(snapshot.particles.typeid, dtype=int)
        type_names = list(snapshot.particles.types)
        box = np.asarray(snapshot.configuration.box, dtype=float)
    brush, missing = read_brush_length(gsd_path)
    brushes = np.full(len(positions), brush, dtype=float)
    missing_flags = np.full(len(positions), missing, dtype=bool)
    return FrameData(positions, diameters, type_ids, type_names, box, int(frame_index), brushes, missing_flags)


def preprocess(frame: FrameData):
    """Frozen geometry-only gas/components/coordination/exposure/M3 calculation."""
    return M3.run(frame.positions, frame.diameters, frame.box, type_ids=frame.type_ids)


def ase_cell(box: np.ndarray) -> np.ndarray:
    Lx, Ly, Lz, xy, xz, yz = np.asarray(box, float)
    return np.asarray([[Lx, 0.0, 0.0], [xy * Ly, Ly, 0.0], [xz * Lz, yz * Lz, Lz]], float)


def min_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return M3._min_image(np.asarray(delta, float), np.asarray(box, float))


def _soap(periodic: bool, r_cut: float, sigma: float) -> SOAP:
    soap = SOAP(
        species=SPECIES, periodic=periodic, r_cut=float(r_cut),
        n_max=N_MAX, l_max=L_MAX, sigma=float(sigma), sparse=False,
    )
    if soap.get_number_of_features() != SOAP_DIMENSION:
        raise RuntimeError(f"SOAP dimension changed: {soap.get_number_of_features()} != {SOAP_DIMENSION}")
    return soap


def _numbers(species_id: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(species_id) == 0, Z_LARGE, Z_SMALL)


def component_d0(frame: FrameData, prep, component_id: int) -> float:
    """Frozen phase-independent rule: median nearest-neighbor distance within the component."""
    idx = np.flatnonzero(prep.retain_mask & (prep.component_id == int(component_id)))
    if len(idx) < 2:
        return float(np.median(frame.diameters[idx])) if len(idx) else 1.0
    p = frame.positions[idx]
    box = frame.box
    if np.allclose(box[3:], 0):
        lengths = box[:3]
        tree = cKDTree(np.mod(p, lengths), boxsize=lengths)
        dist, _ = tree.query(np.mod(p, lengths), k=2)
        nearest = dist[:, 1]
    else:
        # Cohort boxes are orthorhombic; retain a correct triclinic fallback.
        delta = p[:, None, :] - p[None, :, :]
        flat = min_image(delta.reshape(-1, 3), box).reshape(len(p), len(p), 3)
        distances = np.linalg.norm(flat, axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.min(axis=1)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(nearest)) if len(nearest) else float(np.median(frame.diameters[idx]))


def soap_global_d0(frame: FrameData, prep, centers: np.ndarray | None = None) -> SoapResult:
    """Component-normalized SOAP; all non-gas particles are context for every center."""
    context = np.flatnonzero(prep.retain_mask)
    if centers is None:
        centers = context.copy()
    centers = np.asarray(centers, dtype=int)
    centers = centers[prep.retain_mask[centers]]
    vectors = np.empty((len(centers), SOAP_DIMENSION), dtype=float)
    neighbor_counts = np.zeros(len(centers), dtype=int)
    d0s = {}
    context_map = {int(g): i for i, g in enumerate(context)}
    numbers = _numbers(prep.canonical_species_id[context])
    for component in np.unique(prep.component_id[centers]):
        take = np.flatnonzero(prep.component_id[centers] == component)
        d0 = component_d0(frame, prep, int(component))
        d0s[int(component)] = d0
        atoms = Atoms(
            numbers=numbers, positions=frame.positions[context] / d0,
            cell=ase_cell(frame.box) / d0, pbc=True,
        )
        local_centers = [context_map[int(g)] for g in centers[take]]
        vectors[take] = np.atleast_2d(_soap(True, R_CUT_NORM, SIGMA_NORM).create(atoms, centers=local_centers))
        # Explicit audit count uses all non-gas context and the same normalized radius.
        for out_i, global_i in zip(take, centers[take]):
            delta = min_image(frame.positions[context] - frame.positions[global_i], frame.box)
            neighbor_counts[out_i] = int((np.linalg.norm(delta, axis=1) / d0 < R_CUT_NORM).sum() - 1)
    return SoapResult(centers, vectors, d0s, neighbor_counts)


def soap_pair_contact(frame: FrameData, prep, centers: np.ndarray | None = None) -> SoapResult:
    """Center-local contact-normalized SOAP with brush-aware pair distances."""
    context = np.flatnonzero(prep.retain_mask)
    if centers is None:
        centers = context.copy()
    centers = np.asarray(centers, dtype=int)
    centers = centers[prep.retain_mask[centers]]
    radii = frame.diameters / 2.0
    numbers_all = _numbers(prep.canonical_species_id)
    soap = _soap(False, R_CUT_NORM, SIGMA_NORM)
    vectors = np.empty((len(centers), SOAP_DIMENSION), dtype=float)
    counts = np.zeros(len(centers), dtype=int)
    for row, center in enumerate(centers):
        delta = min_image(frame.positions[context] - frame.positions[center], frame.box)
        dist = np.linalg.norm(delta, axis=1)
        brush_i = 0.0 if frame.brush_length_missing[center] else frame.brush_lengths[center]
        brush_j = np.where(frame.brush_length_missing[context], 0.0, frame.brush_lengths[context])
        contact = radii[center] + radii[context] + brush_i + brush_j
        q = dist / contact
        keep = (q < R_CUT_NORM) & (context != center)
        vecs = np.vstack((np.zeros((1, 3)), delta[keep] / contact[keep, None]))
        nums = np.concatenate(([numbers_all[center]], numbers_all[context[keep]]))
        atoms = Atoms(numbers=nums, positions=vecs, pbc=False)
        vectors[row] = np.asarray(soap.create(atoms, centers=[0])).reshape(-1)
        counts[row] = int(keep.sum())
    return SoapResult(centers, vectors, {}, counts)


def soap_absolute(frame: FrameData, prep, centers: np.ndarray | None = None) -> SoapResult:
    """Ordinary fixed-distance SOAP negative control; all non-gas context."""
    context = np.flatnonzero(prep.retain_mask)
    if centers is None:
        centers = context.copy()
    centers = np.asarray(centers, dtype=int)
    centers = centers[prep.retain_mask[centers]]
    mapping = {int(g): i for i, g in enumerate(context)}
    atoms = Atoms(
        numbers=_numbers(prep.canonical_species_id[context]), positions=frame.positions[context],
        cell=ase_cell(frame.box), pbc=True,
    )
    rcut = R_CUT_NORM * ABSOLUTE_D0_REFERENCE
    sigma = SIGMA_NORM * ABSOLUTE_D0_REFERENCE
    vectors = np.atleast_2d(_soap(True, rcut, sigma).create(atoms, centers=[mapping[int(x)] for x in centers]))
    counts = np.zeros(len(centers), dtype=int)
    for row, center in enumerate(centers):
        delta = min_image(frame.positions[context] - frame.positions[center], frame.box)
        counts[row] = int((np.linalg.norm(delta, axis=1) < rcut).sum() - 1)
    return SoapResult(centers, vectors, {}, counts)


def compute_soap(representation: str, frame: FrameData, prep, centers: np.ndarray | None = None) -> SoapResult:
    if representation == "GLOBAL_D0":
        return soap_global_d0(frame, prep, centers)
    if representation == "PAIR_CONTACT":
        return soap_pair_contact(frame, prep, centers)
    if representation == "ABSOLUTE_DISTANCE_CONTROL":
        return soap_absolute(frame, prep, centers)
    raise ValueError(representation)


def parameter_manifest(representation: str) -> dict:
    payload = {
        "representation": representation,
        "species": {"LARGE": Z_LARGE, "SMALL": Z_SMALL},
        "r_cut_normalized": R_CUT_NORM,
        "n_max": N_MAX,
        "l_max": L_MAX,
        "sigma_normalized": SIGMA_NORM,
        "average": "off",
        "periodic": representation != "PAIR_CONTACT",
        "normalization": (
            "center-component median nearest-neighbor d0; all frame non-gas context"
            if representation == "GLOBAL_D0" else
            "pair contact Ri+Rj+Li+Lj; missing brush falls back to Ri+Rj"
            if representation == "PAIR_CONTACT" else
            f"absolute distance; d0_ref={ABSOLUTE_D0_REFERENCE}"
        ),
        "output_dimension": SOAP_DIMENSION,
        "sparse": False,
        "dtype_compute": "float64",
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    payload["parameter_hash"] = hashlib.sha256(encoded).hexdigest()
    return payload
