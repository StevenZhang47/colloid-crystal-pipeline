"""`crystal-discovery scan DIR` -- analyse every trajectory under a directory.

Exists as its own entry point because of where the time goes. Importing dscribe costs
~30 s and analysing a trajectory costs ~2 s, so invoking the CLI once per file spends
94% of its life importing. One process over a folder pays the import once.

Writes a CSV with one row per DOMAIN, not per file: a frame can hold several separate
crystals and each is scored independently. Files that fail are recorded with the reason
rather than dropped, so the row count always matches what was on disk.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from .config import Config
from .ood import load_model
from .pipeline import analyze

FIELDS = ["path", "file", "frame", "domain", "n_particles", "n_centres", "ls_ratio",
          "ood_score", "tau", "rho", "novel", "ood_status",
          "retrieved", "rank1", "distance", "margin", "verification", "top5", "error"]


def find_trajectories(root: Path, pattern: str = "*.gsd", recursive: bool = True) -> list[Path]:
    if root.is_file():
        return [root]
    it = root.rglob(pattern) if recursive else root.glob(pattern)
    return sorted(p for p in it if p.is_file())


def scan(root: Path, out_csv: Path | None = None, frame: int = -1,
         config: Config | None = None, min_centres: int = 64, top_k: int = 5,
         pattern: str = "*.gsd", recursive: bool = True, quiet: bool = False) -> list[dict]:
    cfg = config or Config.load()
    load_model()          # warm the frozen model once rather than per file
    paths = find_trajectories(root, pattern, recursive)
    if not paths:
        print(f"no files matching {pattern} under {root}", file=sys.stderr)
        return []

    rows, t0 = [], time.perf_counter()
    for i, p in enumerate(paths, 1):
        base = {"path": str(p), "file": p.name, "frame": frame}
        try:
            rep = analyze(p, frame=frame, config=cfg, min_centres=min_centres, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            rows.append({**base, "error": f"{type(exc).__name__}: {exc}"})
            if not quiet:
                print(f"[{i}/{len(paths)}] {p.name}: ERROR {exc}", flush=True)
            continue
        if not rep.domains:
            rows.append({**base, "error": "no domain large enough to describe"})
            if not quiet:
                print(f"[{i}/{len(paths)}] {p.name}: no domain", flush=True)
            continue
        for d in rep.domains:
            r = {**base, "domain": d.index, "n_particles": d.n_particles,
                 "n_centres": d.n_centres, "ls_ratio": d.ls_ratio, "error": d.error}
            if d.ood is not None:
                r.update(ood_score=d.ood.score, tau=d.ood.tau, rho=d.ood.rho,
                         novel=float(d.ood.rho) > 1.0, ood_status=d.ood.status)
            if d.retrieval is not None:
                q = d.retrieval
                r.update(retrieved=q.retrieved, rank1=q.ranking[0].prototype,
                         distance=q.ranking[0].distance, margin=q.margin,
                         verification=q.verification,
                         top5="|".join(m.prototype for m in q.ranking))
            rows.append(r)
        if not quiet:
            d0 = rep.domains[0]
            tag = (f"{d0.retrieval.retrieved or 'no match'}" if d0.retrieval else "-")
            novel = "NOVEL" if d0.ood and float(d0.ood.rho) > 1.0 else "known"
            print(f"[{i}/{len(paths)}] {p.name[:44]:44s} {len(rep.domains)}dom  "
                  f"{novel:5s}  {tag}", flush=True)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    if not quiet:
        summarise(rows, len(paths), time.perf_counter() - t0, out_csv)
    return rows


def summarise(rows, n_files, secs, out_csv) -> None:
    ok = [r for r in rows if r.get("rank1")]
    novel = [r for r in ok if r.get("novel")]
    matched = [r for r in ok if r.get("retrieved")]
    failed = [r for r in rows if r.get("error")]
    print(f"\n{'='*70}")
    print(f"  files scanned      {n_files}")
    print(f"  domains described  {len(ok)}")
    print(f"  novel (rho > 1)    {len(novel)}")
    print(f"  identified         {len(matched)}")
    print(f"  no match           {len(ok) - len(matched)}")
    if failed:
        print(f"  failed             {len(failed)}")
    if matched:
        counts: dict[str, int] = {}
        for r in matched:
            counts[r["retrieved"]] = counts.get(r["retrieved"], 0) + 1
        print("\n  identified as:")
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<22s} {v}")
    print(f"\n  {secs:.1f} s total, {secs/max(n_files,1):.1f} s/file")
    if out_csv:
        print(f"  wrote {out_csv}")
