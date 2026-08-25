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


def _cfg_thresholds(config=None):
    c = Config.load(config)
    return float(c.retrieval["distance_threshold"]), float(c.retrieval["margin_threshold"])


def _render(rep, config=None) -> str:
    d_tau, m_tau = _cfg_thresholds(config)
    L = [str(rep.path), f"frame {rep.frame}", "", f"Detected domains: {len(rep.domains)}"]
    if not rep.domains:
        L += ["", "  No condensed domain large enough to describe was found.",
              "  (Every particle may be gas, or the crystal may be below the size cut.)"]
        return "\n".join(L)

    for d in rep.domains:
        L += ["", f"Domain {d.index}", "=" * 62,
              f"  particles {d.n_particles:,}   SOAP centres {d.n_centres:,}   "
              f"L/S ratio {d.ls_ratio}"]
        if d.ood is None:
            L.append(f"  ERROR: {d.error}")
            continue

        known = float(d.ood.rho) <= 1.0
        L += ["",
              f"  SEEN BEFORE      {'YES, in-distribution' if known else 'NO - novel'}",
              f"                   OOD score {d.ood.score}   tau {d.ood.tau}   "
              f"rho {d.ood.rho}   [{d.ood.status}]"]

        if d.retrieval is None:
            L += ["", f"  IDENTIFIED AS    unavailable ({d.error})"]
            continue

        r = d.retrieval
        best = r.ranking[0] if r.ranking else None
        if r.retrieved:
            L += ["", f"  IDENTIFIED AS    {r.retrieved}",
                  f"                   distance {best.distance:.4f}   margin {r.margin}"]
        elif best:
            L += ["", "  IDENTIFIED AS    nothing in the library",
                  f"                   closest was {best.prototype} at "
                  f"{best.distance:.4f} (margin {r.margin})"]
        L.append(f"  VERIFICATION     {r.verification}")

        # The ranked guesses. Rank 1 is the answer; the ranks below it are what it beat,
        # which is what tells you whether to trust it -- a rank 1 sitting just ahead of
        # rank 2 means the library cannot separate those two on this domain, however
        # small the absolute distance happens to be.
        L += ["", "  RANKED GUESSES",
              f"    {'#':>2s}  {'prototype':<20s} {'distance':>9s} {'vs #1':>7s}  verdict"]
        for i, m in enumerate(r.ranking, 1):
            rel = m.distance / max(best.distance, 1e-12)
            if i == 1:
                v = "best match" if r.retrieved else "best match (failed a gate)"
            elif m.distance > d_tau:
                v = "beyond accept range"
            else:
                v = "plausible alternative"
            L.append(f"    {i:>2d}. {m.prototype:<20s} {m.distance:9.4f} {rel:6.2f}x  {v}")
        L.append("        anchors: " + ", ".join(m.anchor for m in r.ranking[:3]))
        L.append(f"        gates: distance <= {d_tau:.4f}, margin >= {m_tau:.2f}")
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
    a.add_argument("--top-k", type=int, default=5,
                   help="how many ranked guesses to show (default 5)")
    a.add_argument("--json", action="store_true", help="machine-readable output")

    sc = sub.add_parser("scan", help="analyse every trajectory under a directory")
    sc.add_argument("directory", type=Path)
    sc.add_argument("-o", "--out", type=Path, default=None, help="CSV to write")
    sc.add_argument("--frame", type=int, default=-1)
    sc.add_argument("--pattern", default="*.gsd")
    sc.add_argument("--no-recursive", action="store_true")
    sc.add_argument("--min-centres", type=int, default=64)
    sc.add_argument("--top-k", type=int, default=5)
    sc.add_argument("--config", type=Path, default=None)
    sc.add_argument("--quiet", action="store_true")

    ar = sub.add_parser("add-reference",
                        help="add a structure to the MATCHER library (not the detector)")
    ar.add_argument("trajectory", type=Path)
    ar.add_argument("--name", required=True, help="anchor name, must be unique")
    ar.add_argument("--prototype", default=None,
                    help="prototype it belongs to (default: same as --name). Give an "
                         "existing prototype to add another cell setting of it.")
    ar.add_argument("--frame", type=int, default=0)
    ar.add_argument("--min-centres", type=int, default=64)
    ar.add_argument("--bank", type=Path, default=None)
    ar.add_argument("--force", action="store_true",
                    help="add even if it matches an existing prototype")

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

    if args.command == "scan":
        from .batch import scan
        if not args.directory.exists():
            print(f"no such directory: {args.directory}", file=sys.stderr)
            return 1
        rows = scan(args.directory, out_csv=args.out, frame=args.frame,
                    config=Config.load(args.config), min_centres=args.min_centres,
                    top_k=args.top_k, pattern=args.pattern,
                    recursive=not args.no_recursive, quiet=args.quiet)
        return 0 if rows else 1

    if args.command == "add-reference":
        from .addref import add_reference
        if not args.trajectory.is_file():
            print(f"no such file: {args.trajectory}", file=sys.stderr)
            return 1
        try:
            add_reference(args.trajectory, name=args.name, prototype=args.prototype,
                          frame=args.frame, min_centres=args.min_centres,
                          bank_path=args.bank, force=args.force)
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        return 0

    if not args.trajectory.is_file():
        print(f"no such file: {args.trajectory}", file=sys.stderr)
        return 1
    rep = analyze(args.trajectory, frame=args.frame, config=Config.load(args.config),
                  min_centres=args.min_centres, top_k=args.top_k)
    print(json.dumps(_as_dict(rep), indent=2) if args.json
          else _render(rep, args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
