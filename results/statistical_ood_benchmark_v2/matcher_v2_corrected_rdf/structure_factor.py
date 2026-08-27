#!/usr/bin/env python3
"""
SIZE-INVARIANT COHERENT STRUCTURE FACTOR.

Replaces `sk_of()` in the removed build_and_run_v2.py, which carried an N-dependent size artifact of
exactly the same class as the original raw-pair-count "RDF" defect.

THREE SEPARATE BUGS WERE FOUND, NOT ONE
---------------------------------------
(1) RANDOM PARTICLE SUBSAMPLING  [the one in the diagnosis]
        if len(p) > 2500: p = p[rng.choice(len(p), 2500)]
    Random subsampling of a lattice destroys coherent Bragg scattering, and the surviving
    amplitude depends on the retained fraction, i.e. on N. Same CsCl lattice:
    max S(k) = 117 (all 6750) / 45 (sub 2500) / 18 (sub 1004). Nicole's 1004-particle
    domains sat below the threshold, were never subsampled, kept full coherence, and so
    always landed nearest the smallest-N anchor.
    FIX: the particle sum is now exact. Never subsampled. Ever.

(2) THE k-WINDOW WAS MOSTLY FORWARD SCATTERING  [found by the regression tests]
    The old grid ran k = 0.35 .. 9.0 (1/nn). As k -> 0 every phase aligns and S(k) -> N
    for ANY arrangement, so the low-k region measures the Fourier transform of the
    crystal's OUTER ENVELOPE -- its size and shape -- not its lattice. The three strongest
    "peaks" of an ideal CsCl crystal in the old window were at k = 0.35, 0.84, 1.32:
    all of them envelope, none of them Bragg. The first true CsCl Bragg peak is at
    k = 2*pi*nn/a = 5.44/nn and was being drowned.
    FIX: K_MIN = 3.5 sits above the envelope for every domain we handle (>= ~500
    particles); the window now spans the first four Bragg families.

(3) DIRECTION SAMPLING CANNOT POWDER-AVERAGE A LATTICE
    The old code sampled 48 random k-directions. A Bragg spot occupies a solid angle that
    SHRINKS as the crystal grows, so a fixed set of directions increasingly MISSES the
    peaks -- another silent N-dependence, and a seed dependence on top.
    FIX: no direction sampling at all. The Debye scattering equation is the EXACT
    orientational average:

        S(k) = 1 + (1/N) * sum_{i != j} sinc(k * r_ij)

    so the pair-normalised form the repair plan calls for,

        S_hat(k) = ( S(k) - 1 ) / ( N - 1 ) = < sinc(k * r_ij) >_{i != j}

    is literally the mean of sinc(k*r) over all ordered pairs. It is bounded in [-1, 1],
    deterministic, seed-free, and exactly angle-averaged. Cost is controlled by histogramming
    the pair distances ONCE (chunked, O(N^2) but vectorised) and then transforming the
    histogram, never by throwing particles away.

WHAT IS, AND IS NOT, SIZE-INVARIANT  (measured -- see test_structure_factor.py)
------------------------------------------------------------------------------
INVARIANT      peak POSITIONS      -- set by the lattice, not by N.
INVARIANT      RELATIVE peak intensities, and the normalised curve SHAPE.
NOT INVARIANT  the absolute amplitude of S_hat. It cannot be: a powder-averaged Bragg peak
               of a finite crystal has height ~N^(1/3) while the pair normalisation divides
               by N, so S_hat at a Bragg vector decays like N^(-2/3). No formulation removes
               this. Peak WIDTH likewise narrows as ~2*pi/L -- that is real finite-size
               broadening, not a bug.
THEREFORE the fingerprint used for matching is the NORMALISED curve within the Bragg window
(`structure_factor(..., normalize=True)`), which is what the EMD comparison already sees --
it divides by the sum. Absolute amplitude is deliberately discarded, because absolute
amplitude is precisely the channel through which particle count leaked in.
"""
from __future__ import annotations

import numpy as np

# k is conjugate to r/nn, so these are dimensionless (units of 1/nn).
# CsCl (SC sublattice, a = 2nn/sqrt(3)) has Bragg families at k = 5.44 (100), 7.70 (110),
# 9.43 (111), 10.88 (200). Rocksalt/Cu3Au (fcc, a = sqrt(2)*nn) at k = 7.70 (111), 8.89 (200).
# The window below spans them all and starts safely above the forward-scattering envelope.
K_MIN = 3.5
K_MAX = 16.0
N_K = 120
DR = 0.02             # pair-histogram resolution in units of nn (>= 20 points per period at K_MAX)
CHUNK = 2048          # rows of the pair-distance matrix computed at once


def k_grid(k_min: float = K_MIN, k_max: float = K_MAX, n_k: int = N_K) -> np.ndarray:
    return np.linspace(k_min, k_max, n_k)


def pair_histogram(q: np.ndarray, dr: float = DR) -> tuple[np.ndarray, np.ndarray]:
    """Histogram of ALL i<j pair distances, in reduced units. Chunked, never subsampled."""
    n = len(q)
    r_max = float(np.linalg.norm(q.max(0) - q.min(0))) + dr
    edges = np.arange(0.0, r_max + dr, dr)
    h = np.zeros(len(edges) - 1, np.float64)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        d = np.linalg.norm(q[a:b, None, :] - q[None, :, :], axis=-1)
        # keep only j > i so each unordered pair is counted exactly once
        i = np.arange(a, b)[:, None]
        j = np.arange(n)[None, :]
        h += np.histogram(d[j > i], bins=edges)[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, h


def structure_factor(pos: np.ndarray,
                     nn: float,
                     k: np.ndarray | None = None,
                     normalize: bool = True,
                     lorch: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Pair-normalised, EXACTLY powder-averaged coherent structure factor.

    S_hat(k) = (S(k) - 1) / (N - 1) = mean of sinc(k * r_ij) over all ordered pairs.

    Parameters
    ----------
    pos        (N,3) positions of the ORDERED DOMAIN ONLY, in nm. Never subsampled.
    nn         nearest-neighbour distance in nm. Lengths are reduced by it, so two crystals
               with the same structure but different lattice constants agree.
    normalize  return the curve scaled to unit maximum. THIS IS THE MATCHING FINGERPRINT:
               absolute amplitude is N-dependent by construction (see module docstring) and
               is deliberately discarded. Pass False only for diagnostics.
    lorch      apply a Lorch window to damp the ripple caused by the pair histogram
               terminating at the crystal's own diameter (a real truncation).
    """
    p = np.asarray(pos, dtype=np.float64)
    n = len(p)
    if n < 2:
        raise ValueError(f"structure factor needs >= 2 particles, got {n}")
    if not np.isfinite(nn) or nn <= 0:
        raise ValueError(f"bad nearest-neighbour distance: {nn}")

    q = (p - p.mean(0)) / nn
    r, h = pair_histogram(q)
    kv = k_grid() if k is None else np.asarray(k, float)

    w = h.copy()
    if lorch:
        r_cut = r[-1]
        x = np.pi * r / r_cut
        w = w * np.where(x > 0, np.sin(x) / np.where(x > 0, x, 1.0), 1.0)

    # S_hat(k) = <sinc(k r_ij)>_{i!=j};  h counts i<j pairs, so 2*h / (N*(N-1)).
    kr = np.outer(kv, r)
    sinc = np.where(kr > 1e-12, np.sin(kr) / np.where(kr > 1e-12, kr, 1.0), 1.0)
    s_hat = (2.0 * (sinc @ w)) / (n * (n - 1.0))

    if normalize:
        m = np.abs(s_hat).max()
        if m > 0:
            s_hat = s_hat / m
    return kv, s_hat


def peak_positions(k: np.ndarray, s: np.ndarray, n_peaks: int = 6,
                   min_sep: float = 0.5) -> np.ndarray:
    """Bragg-peak positions: greedy strongest-first with a minimum separation.

    Positions are the size-INVARIANT part of the signal and are tested separately from the
    intensities, which retain finite-size broadening.
    """
    order = np.argsort(s)[::-1]
    out: list[float] = []
    for i in order:
        if s[i] <= 0:
            break
        if all(abs(k[i] - o) >= min_sep for o in out):
            out.append(float(k[i]))
        if len(out) >= n_peaks:
            break
    return np.array(sorted(out))


def relative_intensities(k: np.ndarray, s: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Peak heights relative to the strongest peak -> scale-free, comparable across N."""
    if len(peaks) == 0:
        return np.array([])
    h = np.array([s[np.argmin(np.abs(k - pk))] for pk in peaks], float)
    m = h.max()
    return h / m if m > 0 else h
