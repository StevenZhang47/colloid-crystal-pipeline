#!/usr/bin/env python3
"""radius_normalized_coord_r4_gas_screen_v1 — the standardized default gas-removal method for
crystal-pipeline work.

Frozen provisional rule (do NOT recalibrate here):

    s_ij = d_ij / (R_i + R_j)                      # pair-radius-normalized, dimensionless
    CN_i = #{ j != i : s_ij < 1.62 }               # all AA/AB/BB contacts count equally
    mean_r4*(i) = mean over the 4 PHYSICAL-nearest neighbours of d(i,j_k)/(R_i+R_{j_k})

    if   CN_i < 2 :               remove   obvious_gas
    elif CN_i >= 4:               retain   supported_material
    else (CN_i in {2,3}):
         if mean_r4*(i) > 1.2698: remove   gas_like_ambiguous
         else:                    retain   compact_ambiguous          # strict '>' ; tie -> retain

This function performs GAS REMOVAL ONLY. It does not identify surfaces, cores, grain boundaries,
structural domains, or phase identity (section 9). Optional retain-flags are diagnostics only and
never cause removal (section 8). Connected components, if computed, are returned in full and never
used to filter (section 10).

PBC: orthorhombic boxes use a periodic scipy cKDTree (boxsize); triclinic/tilted HOOMD boxes use
freud with the complete box matrix (tilt factors preserved). The neighbour structure is only an
accelerator; the scientific criterion is the explicit pair-radius-normalized distance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import cKDTree

# ── frozen constants (provisional; do not change in this module) ──────────────────────────────
METHOD_NAME = "radius_normalized_coord_r4_gas_screen_v1"
CONTACT_MULTIPLIER = 1.62
AMBIGUOUS_CN = (2, 3)
SUPPORTED_MATERIAL_CN_MIN = 4
MEAN_R4_STAR_THRESHOLD = 1.2698
REMOVAL_COMPARISON = ">"

# ── role / reason-code vocabulary ─────────────────────────────────────────────────────────────
ROLE_OBVIOUS_GAS = "obvious_gas"
ROLE_GAS_LIKE_AMBIGUOUS = "gas_like_ambiguous"
ROLE_COMPACT_AMBIGUOUS = "compact_ambiguous"
ROLE_SUPPORTED_MATERIAL = "supported_material"
ROLE_INSUFFICIENT = "insufficient_neighbors"
ROLE_INVALID_RADIUS = "invalid_radius"


@dataclass
class GasScreenResult:
    particle_id: np.ndarray
    radius: np.ndarray
    type_id: np.ndarray
    coordination_1p62: np.ndarray          # int per particle (all-contact CN at m=1.62)
    r1: np.ndarray; r2: np.ndarray; r3: np.ndarray; r4: np.ndarray            # physical distances
    r1_star: np.ndarray; r2_star: np.ndarray; r3_star: np.ndarray; r4_star: np.ndarray
    mean_r4_star: np.ndarray
    roles: np.ndarray                      # object array of role strings
    remove_mask: np.ndarray                # bool
    retain_mask: np.ndarray                # bool  (== ~remove_mask & valid)
    reason_code: np.ndarray                # object array
    # optional diagnostic retain-flags (never cause removal)
    retained_compact_component: np.ndarray
    retained_near_surface_uncertain: np.ndarray
    retained_small_nucleus_candidate: np.ndarray
    component_id: np.ndarray               # connected-component id on RETAINED particles (-1 removed)
    metadata: dict = field(default_factory=dict)

    @property
    def coordination(self):
        return self.coordination_1p62

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame({
            "particle_id": self.particle_id, "type_id": self.type_id, "radius": self.radius,
            "coordination_1p62": self.coordination_1p62,
            "r1": self.r1, "r2": self.r2, "r3": self.r3, "r4": self.r4,
            "r1_star": self.r1_star, "r2_star": self.r2_star, "r3_star": self.r3_star,
            "r4_star": self.r4_star, "mean_r4_star": self.mean_r4_star,
            "gas_screen_role": self.roles, "remove_mask": self.remove_mask,
            "retain_mask": self.retain_mask, "reason_code": self.reason_code,
            "component_id": self.component_id,
            "retained_compact_component": self.retained_compact_component,
            "retained_near_surface_uncertain": self.retained_near_surface_uncertain,
            "retained_small_nucleus_candidate": self.retained_small_nucleus_candidate,
        })


# ── box parsing ───────────────────────────────────────────────────────────────────────────────
def _parse_box(box):
    """Return (lengths(3), tilts(3)=[xy,xz,yz], is_triclinic, freud_box_args). Accepts a length-3
    [Lx,Ly,Lz], a length-6 HOOMD [Lx,Ly,Lz,xy,xz,yz], a 3x3 matrix, or a gsd box array."""
    b = np.asarray(box, dtype=float).ravel()
    if b.size == 3:
        lengths, tilts = b.copy(), np.zeros(3)
    elif b.size == 6:
        lengths, tilts = b[:3].copy(), b[3:6].copy()
    elif np.asarray(box).shape == (3, 3):
        m = np.asarray(box, float)
        Lx = m[0, 0]; xy = m[0, 1] / m[1, 1] if m[1, 1] else 0.0
        Ly = m[1, 1]; xz = m[0, 2] / m[2, 2] if m[2, 2] else 0.0
        yz = m[1, 2] / m[2, 2] if m[2, 2] else 0.0
        Lz = m[2, 2]
        lengths, tilts = np.array([Lx, Ly, Lz]), np.array([xy, xz, yz])
    else:
        raise ValueError(f"Unrecognized box shape {np.asarray(box).shape}; expect 3, 6, or 3x3.")
    if not np.all(lengths > 0):
        raise ValueError(f"Non-positive box lengths: {lengths}")
    is_tri = bool(np.any(np.abs(tilts) > 1e-12))
    return lengths, tilts, is_tri


# ── neighbour queries: return (cn[int], nn_idx[n,4], nn_dist[n,4]) with nan/-1 padding ──────────
def _neighbors_ortho(pos, R, lengths):
    n = len(pos)
    w = np.mod(pos, lengths)
    w = np.minimum(w, np.nextafter(lengths, 0))          # strictly inside box for cKDTree
    tree = cKDTree(w, boxsize=lengths)
    # coordination: all pairs within the widest possible contact radius, then explicit s<1.62
    r_ball = CONTACT_MULTIPLIER * 2.0 * float(np.max(R))
    cn = np.zeros(n, dtype=np.int64)
    pairs = tree.query_pairs(r_ball, output_type="ndarray")
    if len(pairs):
        delta = w[pairs[:, 0]] - w[pairs[:, 1]]
        delta -= lengths * np.round(delta / lengths)     # explicit minimum image
        d = np.linalg.norm(delta, axis=1)
        s = d / (R[pairs[:, 0]] + R[pairs[:, 1]])
        ok = s < CONTACT_MULTIPLIER
        np.add.at(cn, pairs[ok, 0], 1); np.add.at(cn, pairs[ok, 1], 1)
    # four physical-nearest neighbours
    k = min(5, n)
    dist, idx = tree.query(w, k=k)
    dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
    nn_idx = np.full((n, 4), -1, dtype=np.int64)
    nn_dist = np.full((n, 4), np.nan)
    for col in range(1, min(5, dist.shape[1])):          # skip self (col 0)
        nn_idx[:, col - 1] = idx[:, col]
        nn_dist[:, col - 1] = dist[:, col]
    return cn, nn_idx, nn_dist


def _neighbors_triclinic(pos, R, lengths, tilts):
    import freud
    n = len(pos)
    fbox = freud.box.Box(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2],
                         xy=tilts[0], xz=tilts[1], yz=tilts[2])
    pts = fbox.wrap(np.asarray(pos, dtype=np.float32))
    aq = freud.locality.AABBQuery(fbox, pts)
    r_ball = CONTACT_MULTIPLIER * 2.0 * float(np.max(R))
    cn = np.zeros(n, dtype=np.int64)
    nl = aq.query(pts, dict(r_max=r_ball, exclude_ii=True)).toNeighborList()
    qi = np.asarray(nl.query_point_indices); pj = np.asarray(nl.point_indices)
    dd = np.asarray(nl.distances)
    s = dd / (R[qi] + R[pj])
    ok = s < CONTACT_MULTIPLIER
    np.add.at(cn, qi[ok], 1)
    # four physical-nearest
    nl2 = aq.query(pts, dict(num_neighbors=4, exclude_ii=True)).toNeighborList()
    q2 = np.asarray(nl2.query_point_indices); p2 = np.asarray(nl2.point_indices)
    d2 = np.asarray(nl2.distances)
    nn_idx = np.full((n, 4), -1, dtype=np.int64)
    nn_dist = np.full((n, 4), np.nan)
    counts = np.zeros(n, dtype=int)
    order = np.lexsort((d2, q2))                          # sort by (query, distance)
    for pos_row in order:
        i = q2[pos_row]; c = counts[i]
        if c < 4:
            nn_idx[i, c] = p2[pos_row]; nn_dist[i, c] = d2[pos_row]; counts[i] = c + 1
    return cn, nn_idx, nn_dist


# ── connected components on retained particles (diagnostic only; never filters) ─────────────────
def _retained_components(pos, R, lengths, tilts, is_tri, retain):
    import scipy.sparse as sp_
    from scipy.sparse.csgraph import connected_components
    n = len(pos)
    comp = np.full(n, -1, dtype=np.int64)
    ridx = np.flatnonzero(retain)
    if len(ridx) < 1:
        return comp, np.zeros(n, dtype=np.int64)
    rpos, rR = pos[ridx], R[ridx]
    r_ball = CONTACT_MULTIPLIER * 2.0 * float(np.max(R))
    if is_tri:
        import freud
        fbox = freud.box.Box(Lx=lengths[0], Ly=lengths[1], Lz=lengths[2],
                             xy=tilts[0], xz=tilts[1], yz=tilts[2])
        pts = fbox.wrap(np.asarray(rpos, np.float32))
        nl = freud.locality.AABBQuery(fbox, pts).query(
            pts, dict(r_max=r_ball, exclude_ii=True)).toNeighborList()
        a, bb = np.asarray(nl.query_point_indices), np.asarray(nl.point_indices)
        s = np.asarray(nl.distances) / (rR[a] + rR[bb]); keep = s < CONTACT_MULTIPLIER
        e0, e1 = a[keep], bb[keep]
    else:
        w = np.mod(rpos, lengths); w = np.minimum(w, np.nextafter(lengths, 0))
        pr = cKDTree(w, boxsize=lengths).query_pairs(r_ball, output_type="ndarray")
        if len(pr):
            delta = w[pr[:, 0]] - w[pr[:, 1]]; delta -= lengths * np.round(delta / lengths)
            s = np.linalg.norm(delta, axis=1) / (rR[pr[:, 0]] + rR[pr[:, 1]])
            keep = s < CONTACT_MULTIPLIER
            e0, e1 = pr[keep, 0], pr[keep, 1]
        else:
            e0 = e1 = np.array([], dtype=int)
    m = len(ridx)
    g = sp_.coo_matrix((np.ones(len(e0)), (e0, e1)), shape=(m, m))
    _, lab = connected_components(g + g.T, directed=False)
    comp[ridx] = lab
    sizes_full = np.zeros(n, dtype=np.int64)
    csz = np.bincount(lab)
    sizes_full[ridx] = csz[lab]
    return comp, sizes_full


# ── main entry point ────────────────────────────────────────────────────────────────────────
def radius_normalized_coord_r4_gas_screen(
    positions,
    radii=None,
    box=None,
    diameters=None,
    type_ids=None,
    particle_ids=None,
    contact_multiplier: float = CONTACT_MULTIPLIER,
    mean_r4_threshold: float = MEAN_R4_STAR_THRESHOLD,
    compact_component_max: int = 60,
    compute_components: bool = True,
    frame_metadata: dict | None = None,
) -> GasScreenResult:
    """Apply the frozen radius-normalized coordination + mean_r4* gas screen to one frame.

    The frozen constants are the module defaults; the arguments exist to make them explicit, not to
    invite recalibration. box may be length-3 (ortho), length-6 (HOOMD tilt) or a 3x3 matrix.
    """
    if contact_multiplier != CONTACT_MULTIPLIER or mean_r4_threshold != MEAN_R4_STAR_THRESHOLD:
        raise ValueError("Frozen constants must not be changed in this task "
                         f"(m={CONTACT_MULTIPLIER}, thr={MEAN_R4_STAR_THRESHOLD}).")
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if diameters is not None and radii is None:
        radii = np.asarray(diameters, dtype=float) / 2.0
    if radii is None:
        raise ValueError("Provide radii or diameters (R_i = diameter_i/2).")
    R = np.asarray(radii, dtype=float).copy()
    type_id = (np.asarray(type_ids).astype(np.int64) if type_ids is not None
               else np.zeros(n, dtype=np.int64))
    pid = (np.asarray(particle_ids) if particle_ids is not None else np.arange(n, dtype=np.int64))
    if box is None:
        raise ValueError("The complete periodic box is required.")
    lengths, tilts, is_tri = _parse_box(box)

    invalid_R = ~(np.isfinite(R) & (R > 0))
    R_safe = R.copy()
    R_safe[invalid_R] = np.nanmedian(R[~invalid_R]) if (~invalid_R).any() else 1.0

    # neighbour structure (accelerator only; criterion is explicit s_ij < 1.62)
    if is_tri:
        cn, nn_idx, nn_dist = _neighbors_triclinic(pos, R_safe, lengths, tilts)
    else:
        cn, nn_idx, nn_dist = _neighbors_ortho(pos, R_safe, lengths)

    # normalized selected-neighbour distances (each by its OWN pair-radius sum)
    star = np.full((n, 4), np.nan)
    valid_nb = nn_idx >= 0
    for c in range(4):
        j = nn_idx[:, c]
        good = valid_nb[:, c]
        star[good, c] = nn_dist[good, c] / (R_safe[good] + R_safe[j[good]])
    n_valid_nb = valid_nb.sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean_r4 = np.where(n_valid_nb >= 4, np.nanmean(star, axis=1), np.nan)

    # ── frozen decision ──────────────────────────────────────────────────────────────────────
    roles = np.empty(n, dtype=object)
    reason = np.empty(n, dtype=object)
    remove = np.zeros(n, dtype=bool)
    have4 = n_valid_nb >= 4

    obvious = cn < 2
    supported = cn >= SUPPORTED_MATERIAL_CN_MIN
    ambiguous = (~obvious) & (~supported)                 # CN in {2,3}
    amb_gas = ambiguous & have4 & (mean_r4 > mean_r4_threshold)     # strict '>'
    amb_keep = ambiguous & have4 & ~(mean_r4 > mean_r4_threshold)
    amb_insuff = ambiguous & ~have4

    roles[obvious] = ROLE_OBVIOUS_GAS;              reason[obvious] = "CN<2"
    roles[supported] = ROLE_SUPPORTED_MATERIAL;     reason[supported] = "CN>=4"
    roles[amb_gas] = ROLE_GAS_LIKE_AMBIGUOUS;       reason[amb_gas] = "CN in {2,3} & mean_r4*>1.2698"
    roles[amb_keep] = ROLE_COMPACT_AMBIGUOUS;       reason[amb_keep] = "CN in {2,3} & mean_r4*<=1.2698"
    roles[amb_insuff] = ROLE_INSUFFICIENT;          reason[amb_insuff] = "CN in {2,3} & <4 neighbours"
    # invalid radius overrides everything (flag, never removed)
    roles[invalid_R] = ROLE_INVALID_RADIUS;         reason[invalid_R] = "radius<=0 or non-finite"

    remove[obvious] = True
    remove[amb_gas] = True
    remove[invalid_R] = False                              # never remove on invalid radius
    retain = ~remove

    # ── optional diagnostic retain-flags (never cause removal) ────────────────────────────────
    comp_id = np.full(n, -1, dtype=np.int64)
    comp_size = np.zeros(n, dtype=np.int64)
    if compute_components:
        comp_id, comp_size = _retained_components(pos, R_safe, lengths, tilts, is_tri, retain)
    retained_compact = retain & (comp_size > 0) & (comp_size < compact_component_max)
    retained_near_surface = (roles == ROLE_COMPACT_AMBIGUOUS)
    retained_nucleus = retain & supported & (comp_size > 0) & (comp_size < compact_component_max)

    # ── frame-level diagnostics ───────────────────────────────────────────────────────────────
    coord_hist = {int(c): int(v) for c, v in zip(*np.unique(cn, return_counts=True))}
    def _dist_summary(sel):
        v = mean_r4[sel & np.isfinite(mean_r4)]
        if v.size == 0:
            return {}
        return {"n": int(v.size), "min": float(v.min()), "p25": float(np.percentile(v, 25)),
                "median": float(np.median(v)), "p75": float(np.percentile(v, 75)),
                "max": float(v.max()), "mean": float(v.mean())}
    meta = {
        "method": METHOD_NAME,
        "contact_multiplier": CONTACT_MULTIPLIER,
        "mean_r4_star_threshold": MEAN_R4_STAR_THRESHOLD,
        "removal_comparison": REMOVAL_COMPARISON,
        "evidence_status": "provisional",
        "pbc_handling": "triclinic_freud" if is_tri else "orthorhombic_ckdtree",
        "box_lengths": lengths.tolist(), "box_tilts": tilts.tolist(),
        "total_particles": int(n),
        "n_removed": int(remove.sum()),
        "fraction_removed": float(remove.mean()) if n else 0.0,
        "n_removed_obvious_gas_CN_lt_2": int(obvious.sum()),
        "n_removed_mean_r4_tiebreaker": int(amb_gas.sum()),
        "n_retain_compact_ambiguous": int(amb_keep.sum()),
        "n_supported_material": int(supported.sum()),
        "n_insufficient_neighbors": int(amb_insuff.sum()),
        "n_invalid_radius": int(invalid_R.sum()),
        "coordination_histogram": coord_hist,
        "mean_r4_star_CN2": _dist_summary(cn == 2),
        "mean_r4_star_CN3": _dist_summary(cn == 3),
        "counts_by_type": {int(t): int(v) for t, v in zip(*np.unique(type_id, return_counts=True))},
        "removed_by_type": {int(t): int((remove & (type_id == t)).sum())
                            for t in np.unique(type_id)},
        "n_retained_compact_component": int(retained_compact.sum()),
        "n_retained_small_nucleus_candidate": int(retained_nucleus.sum()),
    }
    if frame_metadata:
        meta["frame_metadata"] = dict(frame_metadata)

    return GasScreenResult(
        particle_id=pid, radius=R, type_id=type_id, coordination_1p62=cn,
        r1=nn_dist[:, 0], r2=nn_dist[:, 1], r3=nn_dist[:, 2], r4=nn_dist[:, 3],
        r1_star=star[:, 0], r2_star=star[:, 1], r3_star=star[:, 2], r4_star=star[:, 3],
        mean_r4_star=mean_r4, roles=roles, remove_mask=remove, retain_mask=retain,
        reason_code=reason, retained_compact_component=retained_compact,
        retained_near_surface_uncertain=retained_near_surface,
        retained_small_nucleus_candidate=retained_nucleus, component_id=comp_id, metadata=meta)


# convenience alias matching the versioned method name
radius_normalized_coord_r4_gas_screen_v1 = radius_normalized_coord_r4_gas_screen
