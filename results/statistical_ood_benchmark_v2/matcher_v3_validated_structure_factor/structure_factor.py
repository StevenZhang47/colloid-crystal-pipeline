#!/usr/bin/env python3
"""Deterministic, particle-count-auditable powder structure factor.

This module is deliberately separate from matcher v2.  It never subsamples
particles.  Pair distances are accumulated in bounded-memory blocks and, when a
box is supplied, use the minimum-image convention so equivalent periodic
wrappings produce identical fingerprints.

The returned raw curve is the pair-normalized coherent term

    S_hat(k) = (S(k) - 1) / (N - 1)
             = <sinc(k r_ij)>_(i != j).

Raw amplitude and normalized shape have different roles:

* raw/scaled coherence is Bragg evidence and must not be discarded;
* positive-area-normalized shape is used only to compare peak locations and
  relative intensities after an ordered/Bragg gate passes.

Normalizing every nonzero curve to unit maximum would amplify amorphous noise;
this module never does that implicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


K_MIN = 3.5
K_MAX = 16.0
N_K = 160
DR = 0.015
PAIR_BLOCK = 384


def k_grid(k_min: float = K_MIN, k_max: float = K_MAX, n_k: int = N_K) -> np.ndarray:
    return np.linspace(float(k_min), float(k_max), int(n_k))


def _box3(box: np.ndarray | None, nn: float) -> np.ndarray | None:
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)[:3] / float(nn)
    if len(b) != 3 or not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError(f"invalid orthorhombic box: {box}")
    return b


def robust_first_shell_distance(
    positions: np.ndarray,
    box: np.ndarray | None = None,
    neighbor_rank: int = 4,
) -> float:
    """Robust local length scale from the median k-th-neighbor distance.

    The median *nearest* neighbor is biased downward after thermal broadening: it
    selects the smallest fluctuation among several equivalent first-shell bonds.
    The fourth neighbor remains inside the first shell for every active matcher
    family, including the low-coordination Th3P4 sublattice, and is much less
    sensitive to that order-statistic bias.
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) <= neighbor_rank:
        raise ValueError(f"need more than {neighbor_rank} particles, got {p.shape}")
    if box is None:
        tree = cKDTree(p)
        query = p
    else:
        b = np.asarray(box, dtype=np.float64).reshape(-1)[:3]
        if len(b) != 3 or not np.isfinite(b).all() or np.any(b <= 0):
            raise ValueError(f"invalid orthorhombic box: {box}")
        query = np.mod(p, b)
        tree = cKDTree(query, boxsize=b)
    distances, _ = tree.query(query, k=neighbor_rank + 1)
    value = float(np.median(distances[:, neighbor_rank]))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid first-shell distance: {value}")
    return value


def pair_distance_histogram(
    positions: np.ndarray,
    nn: float,
    box: np.ndarray | None = None,
    dr: float = DR,
    block: int = PAIR_BLOCK,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Histogram every unordered pair distance, without particle subsampling."""
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise ValueError(f"positions must have shape (N>=2, 3), got {p.shape}")
    if not np.isfinite(p).all() or not np.isfinite(nn) or nn <= 0:
        raise ValueError("positions and nn must be finite, with nn > 0")
    q = p / float(nn)
    b = _box3(box, nn)
    if b is None:
        r_max = float(np.linalg.norm(q.max(axis=0) - q.min(axis=0))) + dr
    else:
        r_max = float(np.linalg.norm(0.5 * b)) + dr
    edges = np.arange(0.0, r_max + dr, dr, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    n = len(q)
    pair_total = 0
    all_j = np.arange(n)[None, :]
    for lo in range(0, n, int(block)):
        hi = min(lo + int(block), n)
        d2 = np.zeros((hi - lo, n), dtype=np.float64)
        for axis in range(3):
            delta = q[lo:hi, axis, None] - q[None, :, axis]
            if b is not None:
                delta -= b[axis] * np.rint(delta / b[axis])
            d2 += delta * delta
        keep = all_j > np.arange(lo, hi)[:, None]
        dist = np.sqrt(d2[keep])
        counts += np.histogram(dist, bins=edges)[0]
        pair_total += int(len(dist))
    expected = n * (n - 1) // 2
    if pair_total != expected or int(counts.sum()) != expected:
        raise RuntimeError(
            f"pair accounting failed: histogram={counts.sum()}, visited={pair_total}, expected={expected}"
        )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, pair_total


def raw_structure_factor(
    positions: np.ndarray,
    nn: float,
    box: np.ndarray | None = None,
    k: np.ndarray | None = None,
    lorch: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the unnormalized pair-coherent Debye powder curve."""
    p = np.asarray(positions, dtype=np.float64)
    r, counts, _ = pair_distance_histogram(p, nn=nn, box=box)
    weights = counts.astype(np.float64)
    if lorch:
        occupied = np.flatnonzero(counts)
        r_cut = r[occupied[-1]] if len(occupied) else r[-1]
        x = np.pi * r / max(r_cut, 1e-12)
        window = np.ones_like(x)
        nz = x > 1e-12
        window[nz] = np.sin(x[nz]) / x[nz]
        weights *= window
    kv = k_grid() if k is None else np.asarray(k, dtype=np.float64)
    kr = np.outer(kv, r)
    sinc = np.ones_like(kr)
    nz = np.abs(kr) > 1e-12
    sinc[nz] = np.sin(kr[nz]) / kr[nz]
    n = len(p)
    curve = 2.0 * (sinc @ weights) / (n * (n - 1.0))
    return kv, curve


def positive_shape(curve: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Unit-area positive Bragg-window shape; never use as an order gate."""
    y = np.clip(np.asarray(curve, dtype=np.float64), 0.0, None)
    area = float(np.trapezoid(y, np.asarray(k, dtype=np.float64)))
    return y / area if area > 1e-15 else np.zeros_like(y)


def coherence_metrics(curve: np.ndarray, n_particles: int, k: np.ndarray) -> dict[str, float]:
    """Raw and finite-size-scaled coherence diagnostics kept separate from shape."""
    y = np.asarray(curve, dtype=np.float64)
    positive = np.clip(y, 0.0, None)
    raw_peak = float(positive.max(initial=0.0))
    return {
        "raw_positive_peak": raw_peak,
        "raw_positive_area": float(np.trapezoid(positive, np.asarray(k, dtype=np.float64))),
        # A finite 3-D crystallite's pair-normalized powder peak scales approximately
        # as N^(-2/3). Keep this diagnostic explicit; it is not a universal threshold.
        "n23_scaled_peak": float(raw_peak * float(n_particles) ** (2.0 / 3.0)),
        "raw_rms": float(np.sqrt(np.mean(y * y))),
    }


def peak_positions(
    k: np.ndarray,
    curve: np.ndarray,
    n_peaks: int = 6,
    min_prominence_fraction: float = 0.04,
) -> np.ndarray:
    """Return actual local maxima, strongest first, then sorted by k."""
    k = np.asarray(k, dtype=np.float64)
    y = np.clip(np.asarray(curve, dtype=np.float64), 0.0, None)
    if not np.any(y > 0):
        return np.array([], dtype=np.float64)
    dk = float(np.median(np.diff(k)))
    distance = max(1, int(round(0.40 / max(dk, 1e-12))))
    peaks, props = find_peaks(
        y,
        prominence=max(float(y.max()) * min_prominence_fraction, 1e-15),
        distance=distance,
    )
    if not len(peaks):
        return np.array([], dtype=np.float64)
    order = peaks[np.argsort(props["prominences"])[::-1][: int(n_peaks)]]
    return np.sort(k[order])


def relative_peak_intensities(k: np.ndarray, curve: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=np.float64)
    if not len(peaks):
        return np.array([], dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    y = np.clip(np.asarray(curve, dtype=np.float64), 0.0, None)
    heights = np.asarray([y[np.argmin(np.abs(k - p))] for p in peaks])
    maximum = float(heights.max(initial=0.0))
    return heights / maximum if maximum > 0 else heights
