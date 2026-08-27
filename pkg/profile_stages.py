#!/usr/bin/env python3
"""Where does the time in `crystal-discovery analyze` actually go?

Times each stage separately on the same frame, so the answer is measured rather than
guessed. Run it on your own trajectory before optimising anything:

  python pkg/profile_stages.py TRAJ.gsd [--frame -1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pkg/src"))

from crystal_discovery.config import Config              # noqa: E402
from crystal_discovery.descriptors import _core          # noqa: E402
from crystal_discovery.ood import load_model, score_domain  # noqa: E402
from crystal_discovery.retrieval import load_bank, retrieve  # noqa: E402
from crystal_discovery.descriptors import Domain         # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", type=Path)
    ap.add_argument("--frame", type=int, default=-1)
    a = ap.parse_args()
    T = {}

    t = time.perf_counter(); P = _core(); T["import frozen modules"] = time.perf_counter() - t
    t = time.perf_counter(); model = load_model(); T["load OOD model"] = time.perf_counter() - t
    t = time.perf_counter(); bank = load_bank(); T["load chamfer bank"] = time.perf_counter() - t

    import gsd.hoomd
    t = time.perf_counter()
    with gsd.hoomd.open(str(a.trajectory), "r") as tr:
        n_frames = len(tr)
    T["open GSD + index"] = time.perf_counter() - t

    t = time.perf_counter(); frame = P.read_frame(a.trajectory, a.frame)
    T["read frame"] = time.perf_counter() - t
    n = len(frame.positions)

    t = time.perf_counter(); prep = P.preprocess(frame)
    T["gas screen + M3"] = time.perf_counter() - t

    centres = np.flatnonzero(prep.is_soap_center)
    comps, counts = np.unique(prep.component_id[centres], return_counts=True)
    centres = centres[prep.component_id[centres] == comps[np.argmax(counts)]]

    t = time.perf_counter(); res = P.soap_global_d0(frame, prep, centers=centres)
    T["SOAP descriptor"] = time.perf_counter() - t

    v = res.vectors / np.clip(np.linalg.norm(res.vectors, axis=1, keepdims=True), 1e-12, None)
    sp = prep.canonical_species_id[res.particle_indices]
    dom = Domain(0, n, len(centres), int((sp == 0).sum()), int((sp == 1).sum()),
                 0.0, v, sp)

    t = time.perf_counter(); score_domain(dom, model); T["OOD scoring"] = time.perf_counter() - t
    t = time.perf_counter(); retrieve(dom, Config.load(), bank, 5)
    T["retrieval (41 anchors)"] = time.perf_counter() - t

    total = sum(T.values())
    print(f"\n{a.trajectory.name}")
    print(f"  {n_frames} frames, {n:,} particles, {len(centres):,} SOAP centres\n")
    print(f"  {'stage':26s} {'seconds':>9s}  {'%':>6s}")
    print("  " + "-" * 45)
    for k, v_ in sorted(T.items(), key=lambda kv: -kv[1]):
        print(f"  {k:26s} {v_:9.2f}  {100*v_/total:5.1f}%")
    print("  " + "-" * 45)
    print(f"  {'TOTAL':26s} {total:9.2f}")


if __name__ == "__main__":
    main()
