"""`crystal-discovery add-reference` -- put a new structure into the matcher library.

WHAT THIS CAN AND CANNOT DO
---------------------------
It extends the MATCHER's reference library, so the pipeline can name a structure it
previously could only call "nothing in the library". It does NOT retrain the detector.
Those are different objects and only one of them is cheap to change:

  matcher library   a set of reference geometries. Adding one is local -- it introduces a
                    new thing to be compared against and changes nothing already there.
                    That is this command.

  detector atlas    a fitted Gaussian (location, covariance, PCA basis, tau) over 380
                    trajectories in 103 condition groups, frozen with a recorded
                    false-positive rate and a sealed single-use test set. Adding data
                    means refitting, which invalidates tau, the freeze record and every
                    validation number quoted in the README. There is deliberately no
                    command for it. See docs at the bottom of this file.

THE FROZEN BANK IS NEVER MODIFIED
---------------------------------
New anchors go to a SEPARATE user bank that layers on top at load time. The shipped
41-anchor artifact keeps its checksum, so `verify_acceptance.py` still reproduces the
published numbers and a user addition can always be dropped by deleting one file.

THE GATES ARE NOT AUTOMATICALLY STILL VALID
-------------------------------------------
`distance <= 0.1619` and `margin >= 2.45` were calibrated against the shipped 41 anchors.
The distance gate is per-prototype and survives an addition unchanged. The MARGIN gate
does not: it is a ratio between rank 1 and rank 2, so inserting a structural neighbour of
an existing prototype lowers rank 2 for everything near it and can push a previously
confident match below 2.45. This is not hypothetical -- it is exactly how Al3Ti (D0_22)
broke the original Cu3Au acceptance when the library was expanded.

So this command scores the new anchor against the existing library and refuses, or warns,
when the addition would collide.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .descriptors import find_domains
from .retrieval import USER_BANK, load_bank

COVER_R = 0.02          # matches the shipped bank
MIN_ENV = 4


def greedy_cover(X: np.ndarray, r: float) -> np.ndarray:
    keep = [0]
    d = np.linalg.norm(X - X[0], axis=1)
    while True:
        i = int(np.argmax(d))
        if d[i] <= r:
            break
        keep.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.asarray(keep, dtype=int)


def _environments(gsd_path: Path, frame: int, min_centres: int) -> list[np.ndarray]:
    doms = find_domains(gsd_path, frame, min_centres=min_centres)
    if not doms:
        raise ValueError("no domain large enough to describe in that file")
    d = max(doms, key=lambda x: x.n_centres)
    out = []
    for s in (0, 1):
        q = d.species == s
        if q.sum() < 25:
            raise ValueError(f"too few centres on sublattice {s}: {int(q.sum())}")
        out.append(np.ascontiguousarray(d.soap[q], dtype=np.float32))
    return out


def add_reference(gsd_path: Path, name: str, prototype: str | None = None,
                  frame: int = 0, cover_r: float = COVER_R, min_centres: int = 64,
                  bank_path: Path | None = None, force: bool = False) -> dict:
    from .config import Config
    from .retrieval import _chamfer_median

    prototype = prototype or name
    target = Path(bank_path) if bank_path else USER_BANK
    cfg = Config.load()
    d_tau = float(cfg.retrieval["distance_threshold"])
    m_tau = float(cfg.retrieval["margin_threshold"])

    envs = _environments(Path(gsd_path), frame, min_centres)
    kept = [greedy_cover(e, cover_r) for e in envs]
    new = [envs[s][kept[s]] for s in (0, 1)]
    print(f"  {len(envs[0])}/{len(envs[1])} centres -> "
          f"{len(new[0])}/{len(new[1])} distinct environments")
    if min(len(new[0]), len(new[1])) < MIN_ENV:
        print(f"  [warn] only {len(new[0])}/{len(new[1])} environments; a structure this "
              f"uniform may be a poor reference")

    # --- collision check against everything already in the library --------------------
    existing = load_bank()
    scores = {}
    for an, e in existing["envs"].items():
        p = existing["prototype_of"].get(an, an)
        v = _chamfer_median(envs, [x.astype(np.float32) for x in e])
        scores[p] = min(scores.get(p, np.inf), v)
    order = sorted(scores, key=scores.get)
    nearest, d_near = order[0], scores[order[0]]
    runner = scores[order[1]] if len(order) > 1 else np.inf

    print(f"  nearest existing prototype: {nearest} at D={d_near:.4f} "
          f"(gate is {d_tau:.4f})")
    report = {"name": name, "prototype": prototype, "source": str(gsd_path),
              "frame": frame, "cover_r": cover_r,
              "n_environments": [int(len(new[0])), int(len(new[1]))],
              "nearest_existing": nearest, "distance_to_nearest": round(float(d_near), 4),
              "would_break_margin": []}

    if d_near <= d_tau:
        if nearest == prototype:
            # Legitimate: another cell setting of a prototype already present. Retrieval
            # collapses to one row per prototype before the margin test, so settings of
            # the same structure never compete with each other. Th3P4 ships with four.
            print(f"  matches '{prototype}' at D={d_near:.4f} -- adding as another cell "
                  f"setting of it, which is fine: retrieval collapses settings of one "
                  f"prototype before the margin test.")
        else:
            msg = (f"the new structure MATCHES the existing prototype '{nearest}' at "
                   f"D={d_near:.4f} <= {d_tau:.4f}, but you called it '{prototype}'. By "
                   f"this pipeline's own criterion they are the same structure, so adding "
                   f"it creates two prototypes that cannot be told apart and every future "
                   f"match against either fails the margin gate. If they really are the "
                   f"same, pass --prototype {nearest} to add it as another cell setting.")
            if not force:
                raise ValueError(msg)
            print(f"  [FORCED] {msg}")

    # A new anchor sitting between two existing prototypes depresses rank 2 for both.
    at_risk = [p for p in order[:5] if scores[p] / max(d_near, 1e-9) < m_tau]
    if at_risk and d_near > d_tau:
        report["would_break_margin"] = at_risk
        print(f"  [warn] within a factor {m_tau} of: {', '.join(at_risk)}")
        print(f"         a candidate that is really one of those may now fail the margin "
              f"gate.\n         Re-run `python pkg/verify_acceptance.py` before trusting "
              f"results.")

    # --- write the user bank ----------------------------------------------------------
    store, meta = {}, []
    if target.is_file():
        z = np.load(target, allow_pickle=False)
        meta = json.loads(bytes(z["__meta__"]).decode())["anchors"]
        for k in z.files:
            if k != "__meta__":
                store[k] = z[k]
        if any(m["anchor"] == name for m in meta):
            raise ValueError(f"'{name}' is already in {target}; pick another name")
    for s in (0, 1):
        store[f"{name}::{s}"] = new[s].astype(np.float16)
    meta.append({"anchor": name, "prototype": prototype,
                 "n_particles": [int(len(envs[0])), int(len(envs[1]))],
                 "n_environments": report["n_environments"],
                 "source": str(gsd_path), "frame": frame,
                 "nearest_existing": nearest,
                 "distance_to_nearest": report["distance_to_nearest"]})
    store["__meta__"] = np.frombuffer(json.dumps(
        {"cover_r": cover_r, "source": "user-added references", "anchors": meta}
    ).encode(), dtype=np.uint8)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **store)
    load_bank.cache_clear()
    print(f"  added '{name}' as prototype '{prototype}' -> {target} "
          f"({len(meta)} user anchor{'s' if len(meta) != 1 else ''})")
    return report


# ---------------------------------------------------------------------------------
# Adding to the DETECTOR atlas, which this command deliberately does not do.
#
# The detector is a Gaussian fitted to 380 trajectories grouped into 103 condition groups,
# split by GROUP rather than by trajectory because seeds within a group are near-copies
# and splitting by trajectory leaks between roles. Extending it means:
#
#   1. new trajectories with condition-group ids, and labels with recorded provenance
#   2. refitting species scaling, PCA16 and the Ledoit-Wolf covariance on train
#   3. re-deriving tau from the calibration split -- tau = 6.715183 does NOT carry over
#   4. re-measuring the false-positive rate on validation
#   5. a new freeze record and a new sha256; every number in the README becomes stale
#   6. the sealed 24-trajectory test is SINGLE USE. Opening it to check a retrained model
#      spends it, and it cannot be restored.
#
# The scripts are in results/motif_pipeline/core3_atlas_v3/scripts/ and DATA/README.md
# documents the split protocol. It is a deliberate, reviewable operation, not a CLI flag.
# ---------------------------------------------------------------------------------
