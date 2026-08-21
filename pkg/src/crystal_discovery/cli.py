"""`crystal-discovery` command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .ood import freeze_record
from .pipeline import analyze


def _render(rep) -> str:
    L = [f"Detected domains: {len(rep.domains)}"]
    if not rep.domains:
        L.append("\n  No condensed domain large enough to describe was found.")
        L.append("  (Every particle may be gas, or the crystal may be below the size cut.)")
        return "\n".join(L)

    for d in rep.domains:
        L += ["", f"Domain {d.index}", "-" * 9,
              f"Particles: {d.n_particles:,}",
              f"SOAP centres: {d.n_centres:,}   L/S ratio: {d.ls_ratio}"]
        if d.ood is None:
            L.append(f"Status: ERROR ({d.error})")
            continue
        L += [f"OOD score: {d.ood.score}   (tau = {d.ood.tau}, rho = {d.ood.rho})",
              f"Status: {d.ood.status}"]
        if d.retrieval is None:
            L.append(f"Retrieval: unavailable ({d.error})")
            continue
        L.append("")
        L.append("Nearest CIF prototypes:")
        for i, m in enumerate(d.retrieval.ranking, 1):
            L.append(f"{i}. {m.prototype:<18s} distance={m.distance:.3f}")
        L += ["", f"Retrieval: {d.retrieval.retrieved or 'none'}"
                  f"   (margin {d.retrieval.margin})",
              f"Verification: {d.retrieval.verification}"]
    return "\n".join(L)


def _as_dict(rep) -> dict:
    return {"path": rep.path, "frame": rep.frame, "domains": [
        {"index": d.index, "n_particles": d.n_particles, "n_centres": d.n_centres,
         "ls_ratio": d.ls_ratio, "error": d.error,
         "ood": None if d.ood is None else vars(d.ood),
         "retrieval": None if d.retrieval is None else {
             "retrieved": d.retrieval.retrieved, "margin": d.retrieval.margin,
             "verification": d.retrieval.verification,
             "ranking": [vars(m) for m in d.retrieval.ranking]}}
        for d in rep.domains]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="crystal-discovery",
        description="Detect unknown colloidal crystal structures and identify the known ones.")
    ap.add_argument("--version", action="version", version=f"crystal-discovery {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="analyse one trajectory")
    a.add_argument("trajectory", type=Path)
    a.add_argument("--frame", type=int, default=-1, help="frame index (default: last)")
    a.add_argument("--config", type=Path, default=None)
    a.add_argument("--min-centres", type=int, default=64,
                   help="smallest describable domain (default 64)")
    a.add_argument("--top-k", type=int, default=3)
    a.add_argument("--json", action="store_true", help="machine-readable output")

    sub.add_parser("info", help="show the frozen model this build will use")

    args = ap.parse_args(argv)

    if args.command == "info":
        rec = freeze_record()
        if not rec:
            print("No frozen model found.", file=sys.stderr)
            return 1
        for k in ("status", "cohort", "covariance_structure", "threshold_rule", "tau",
                  "known_false_positive_rate", "artifact_sha256", "untouched_test_opened"):
            if k in rec:
                print(f"{k:28s} {rec[k]}")
        print("\nknown limitations:")
        for lim in rec.get("known_limitations", []):
            print(f"  - {lim}")
        return 0

    if not args.trajectory.is_file():
        print(f"no such file: {args.trajectory}", file=sys.stderr)
        return 1
    rep = analyze(args.trajectory, frame=args.frame, config=Config.load(args.config),
                  min_centres=args.min_centres, top_k=args.top_k)
    print(json.dumps(_as_dict(rep), indent=2) if args.json else _render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
