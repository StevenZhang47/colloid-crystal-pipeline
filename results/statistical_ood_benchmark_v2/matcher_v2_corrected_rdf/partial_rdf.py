#!/usr/bin/env python3
"""
CORRECTED partial radial distribution function — matcher v2.

WHAT WAS WRONG IN v1
The v1 "RDF" was a raw pair-distance histogram normalised to max = 1:

    h, _ = np.histogram(r, bins=edges); h /= h.max()

It never divided by the shell volume 4*pi*r^2*dr, never divided by the number of
centre particles, and never divided by the neighbour-species number density. So it
was not a g(r) at all: it grew like r^2 and was dominated by the crystal's overall
SIZE and SHAPE. Its first "peak" for ideal CsCl sat at r/nn = 11.8 instead of 1.0.

Every internal test was blind to this because reference and candidate always came from
the same construction and therefore had identical particle counts and crystal shapes --
the size term cancelled exactly. Nicole's external crystals (~1000 particles vs our
4,000-14,000) destroyed that cancellation and every trajectory collapsed onto one family.

WHAT THIS DOES
A true partial RDF for a FINITE DOMAIN embedded in a gas/vacuum box:

    g_ab(r) = N_ab(r) / [ N_a^centres * rho_b * 4*pi*r^2*dr * S_a(r) ]

  N_ab(r)     pairs with centre of species a, neighbour of species b, in bin r
  N_a^centres number of species-a particles USED AS CENTRES
  rho_b       number density of species b inside the domain (estimated locally, so it
              does not depend on the domain's shape or on a box volume)
  S_a(r)      SHAPE / TRUNCATION CORRECTION: the fraction of the spherical shell of
              radius r around a centre that actually lies INSIDE the domain. Without
              this, surface particles see empty space beyond the domain edge and the
              RDF is biased downward at large r -- exactly the bias that matters for
              Nicole's finite crystals sitting in a gas box.

Centres are drawn preferentially from the INTERIOR of the domain (their shells are
best contained), while NEIGHBOURS may come from the whole extracted domain.

Same-species pair counting: an LL or SS pair is found twice by a symmetric pair search
(i->j and j->i). Both orderings are counted, and the normalisation uses the number of
species-a centres, so g_aa is correct without an extra factor of 2.

CANONICAL CONVENTION (unchanged, and asserted): typeid 0 = LARGE, typeid 1 = SMALL.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

R_MAX_NN = 4.0          # local-structure window, in units of nn
N_BINS = 100
R_MIN_NN = 0.5
N_SHAPE_DIRS = 26       # directions for the shell-containment estimate
N_SHAPE_BINS = 24       # coarse radial grid for the shape correction, then interpolated
MAX_CENTRES = 2000


def _unit_dirs(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def interior_mask(pos_dom, nn, quantile=0.6):
    """Particles well inside the domain: high domain-neighbour count."""
    tree = cKDTree(pos_dom)
    cnt = np.array([len(x) for x in tree.query_ball_point(pos_dom, r=2.0 * nn)])
    thr = np.quantile(cnt, quantile)
    m = cnt >= thr
    return m if m.sum() >= 32 else np.ones(len(pos_dom), bool)


def local_density(pos_dom, nn):
    """Number density inside the domain, estimated LOCALLY (shape independent)."""
    tree = cKDTree(pos_dom)
    inner = interior_mask(pos_dom, nn, quantile=0.75)
    R = 2.0 * nn
    cnt = np.array([len(x) - 1 for x in tree.query_ball_point(pos_dom[inner], r=R)])
    return float(np.median(cnt) / (4.0 / 3.0 * np.pi * R ** 3))


def shape_correction(centres, pos_dom, nn, r_grid, inside_cut=0.75):
    """
    Fraction of the shell at radius r around each centre that lies inside the domain.
    A sampled point counts as inside if it is within `inside_cut * nn` of some domain
    particle (i.e. it sits in the occupied region rather than out in the gas/vacuum).
    Returned averaged over the supplied centres.
    """
    tree = cKDTree(pos_dom)
    dirs = _unit_dirs(N_SHAPE_DIRS)
    S = np.empty(len(r_grid))
    for i, r in enumerate(r_grid):
        pts = (centres[:, None, :] + r * dirs[None, :, :]).reshape(-1, 3)
        d, _ = tree.query(pts, k=1)
        S[i] = float((d < inside_cut * nn).mean())
    return np.clip(S, 1e-3, 1.0)


def partial_rdf(pos, tid, domain_mask, nn, r_max_nn=R_MAX_NN, n_bins=N_BINS, seed=0):
    """
    Returns (r_over_nn, {'LL','LS','SS','total'} -> g(r)) for the extracted domain.
    g -> 1 at large r when the normalisation is correct.
    """
    assert set(np.unique(tid)) <= {0, 1}, "typeid must be canonicalised to {0=LARGE,1=SMALL}"
    p = np.asarray(pos, float)[domain_mask]
    t = np.asarray(tid, int)[domain_mask]
    if len(p) < 64:
        r = np.linspace(R_MIN_NN, r_max_nn, n_bins)
        return r, {k: np.zeros(n_bins) for k in ("LL", "LS", "SS", "total")}

    rho = local_density(p, nn)
    n_b = {b: int((t == b).sum()) for b in (0, 1)}
    rho_b = {b: rho * (n_b[b] / len(p)) for b in (0, 1)}

    edges = np.linspace(R_MIN_NN * nn, r_max_nn * nn, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dr = np.diff(edges)
    shell_vol = 4.0 * np.pi * centers ** 2 * dr

    inner = interior_mask(p, nn)
    rng = np.random.default_rng(seed)
    tree = cKDTree(p)

    out = {}
    r_grid = np.linspace(R_MIN_NN * nn, r_max_nn * nn, N_SHAPE_BINS)
    for a in (0, 1):
        cand = np.where(inner & (t == a))[0]
        if len(cand) < 8:                               # fall back to all species-a
            cand = np.where(t == a)[0]
        if len(cand) == 0:
            continue
        if len(cand) > MAX_CENTRES:
            cand = rng.choice(cand, MAX_CENTRES, replace=False)
        C = p[cand]
        S = np.interp(centers, r_grid, shape_correction(C, p, nn, r_grid))
        nb = tree.query_ball_point(C, r=edges[-1])
        for b in (0, 1):
            h = np.zeros(n_bins)
            for ci, js in zip(cand, nb):
                js = np.asarray(js)
                js = js[(js != ci) & (t[js] == b)]
                if len(js):
                    d = np.linalg.norm(p[js] - p[ci], axis=1)
                    h += np.histogram(d, bins=edges)[0]
            denom = len(cand) * rho_b[b] * shell_vol * S
            g = np.divide(h, denom, out=np.zeros_like(h), where=denom > 0)
            key = ("LL" if (a, b) == (0, 0) else "SS" if (a, b) == (1, 1) else "LS")
            out[key] = out.get(key, 0) + (g if key != "LS" else 0.5 * g)   # LS averages a->b and b->a
    # total: density-weighted combination
    x0 = n_b[0] / len(p); x1 = n_b[1] / len(p)
    out["total"] = (x0 * x0 * out.get("LL", 0) + 2 * x0 * x1 * out.get("LS", 0)
                    + x1 * x1 * out.get("SS", 0)) / (x0 * x0 + 2 * x0 * x1 + x1 * x1)
    for k in ("LL", "LS", "SS", "total"):
        out.setdefault(k, np.zeros(n_bins))
    return centers / nn, out
