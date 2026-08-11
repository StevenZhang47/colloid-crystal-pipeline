#!/usr/bin/env python3
"""Generate a colloidal crystal for every CIF in the library.

One CIF in, one `crystal_stability/runs/lib_<slug>/` out, containing the relaxed
lattice as `first_frame.gsd` -- which is exactly what the EMD matcher reads as a
family ANCHOR.

Three stages per CIF, all of which must succeed:

  1. DESIGN   design.py screens both colloid assignments x the physical charge
              grid, solving lambda_D for the target unlike-pair well depth, and
              ranks on the binding of the WEAKEST species. If nothing reaches the
              target depth in the physical charge range we retry at a deeper
              target (the low-coordination rescue) before giving up.
  2. BUILD    build.py renders configuration.yaml + run.yaml. Every system written
              here carries `anisotropic_energy: true` -- a no-op for cubic cells,
              and required for the other 19 library CIFs whose axial ratios come
              from an ATOMIC crystal and are wrong for colloids of the same
              structure type.
  3. CREATE   pacsim-create relaxes the lattice and writes first_frame.gsd.

Then the result is checked for the exploded-lattice failure mode: nearest-neighbour
spacing over the steric contact distance. A lattice whose energy scan ran away to a
dilute cell produces a "crystal" in which nothing touches anything.

REQUIRES the anisotropic LatticeBuilder (PACSim >= 582a2d0, hockyg/develop-crystal-TI).
The worktree at crystal_discovery/PACSim_aniso is put on PYTHONPATH by the sbatch
wrapper; this script asserts it is the one actually imported rather than trusting it.

    python generate_crystals.py --list                 # what would be built
    python generate_crystals.py --slug laves_C14_MgZn2 # one
    python generate_crystals.py --index 7              # by array index
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/projects/rps/gmh4/hockygroup/sz5808")
LIB = ROOT / "crystal_discovery/cif_library"
CS = ROOT / "crystal_discovery/crystal_stability"
PROTO = LIB / "prototypes"

# Ladder of unlike-pair well depths (kT). -5 is where CsCl and Cu3Au held; deeper
# binds harder and is the documented rescue for low-coordination lattices, at the
# cost of approaching the irreversible-aggregation regime. We only go deeper when
# the shallower target admits no candidate at all.
DEPTH_LADDER = (-5.0, -8.0, -12.0)

# A design is only accepted if the screen says the WEAKEST species is actually held.
# Stepping the ladder solely on "design.py wrote no file" was too weak a test: for the
# Laves phases and AlB2 it happily wrote a system whose weakest species was held at
# 0.0 kT -- nothing binding at all -- and that built a dilute lattice (nn/contact 1.84)
# which is the exploded-lattice failure mode under a different name. CsCl, which
# survives MD, holds both species near -25 kT, so -1.0 is a very permissive floor.
WEAKEST_KT_MAX = -1.0

CONTACT_COHESIVE = 1.25      # nn / steric contact distance; > this is suspect
CONTACT_EXPLODED = 2.0


def cif_preflight(cif: Path) -> dict:
    """Reject a CIF the builder cannot use, BEFORE spending a design run on it.

    The builder parses with site_tolerance=0.0, frac_tolerance=0.0, so it does NOT merge
    coincident sites the way a default CifParser does. A symmetry-expanded file whose
    expansion duplicated its sites therefore reaches the builder with pairs at distance
    zero, the geometric no-overlap scale divides by zero, and every downstream coordinate
    is NaN -- producing a first_frame.gsd with a NaN box rather than an error.

    This is exactly how runs/cu3au/Cu3Au.cif got into the library: the catalog read it as
    4 sites (default tolerances) while the builder reads 96 with 40176 coincident pairs.
    Parse here the way the BUILDER does, not the way the catalog does.
    """
    from pymatgen.io.cif import CifParser
    from scipy.spatial import distance_matrix
    s = CifParser(str(cif), site_tolerance=0.0, frac_tolerance=0.0
                  ).parse_structures(check_occu=False, primitive=False)[0]
    t = s.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
    d = distance_matrix(t.cart_coords, t.cart_coords)
    np.fill_diagonal(d, np.inf)
    coincident = int((d < 1e-8).sum() // 2)
    return {"n_sites_builder": len(s), "min_site_distance": round(float(d.min()), 6),
            "coincident_pairs": coincident,
            "ok": coincident == 0 and np.isfinite(d.min()) and d.min() > 1e-8}


def catalog_slugs() -> list[str]:
    cat = pd.read_csv(LIB / "catalog.csv")
    return sorted(cat["slug"].astype(str).unique())


def assert_anisotropic_builder() -> None:
    from colloids.colloids_create.configuration_generators.lattice_builder import LatticeBuilder
    import colloids
    if not hasattr(LatticeBuilder, "_axis_groups"):
        raise SystemExit(
            f"LatticeBuilder without anisotropic scaling was imported from "
            f"{colloids.__file__}. Put PACSim_aniso on PYTHONPATH.")


def run(cmd: list[str], cwd: Path, log: Path) -> None:
    with open(log, "ab") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n".encode())
        fh.flush()
        subprocess.run(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT, check=True)


def contact_validity(gsd_path: Path) -> dict:
    """nn spacing vs steric contact distance. The exploded-lattice detector."""
    import gsd.hoomd
    with gsd.hoomd.open(str(gsd_path), "r") as tr:
        fr = tr[0]
        pos = np.asarray(fr.particles.position, np.float64)
        tid = np.asarray(fr.particles.typeid, int)
        dia = np.asarray(fr.particles.diameter, np.float64)
        box = np.asarray(fr.configuration.box[:3], np.float64)
    import freud
    fbox = freud.box.Box(Lx=box[0], Ly=box[1], Lz=box[2])
    aq = freud.locality.AABBQuery(fbox, pos)
    nl = aq.query(pos, dict(num_neighbors=1, exclude_ii=True)).toNeighborList()
    d = np.asarray(nl.distances, float)
    nn = float(np.median(d))
    # steric contact for the nearest pair type present: use the smallest sum of radii,
    # which is the most permissive reading (hardest to call exploded).
    radii = {int(t): float(dia[tid == t].mean()) / 2 for t in np.unique(tid)}
    contact = min(radii[a] + radii[b] for a in radii for b in radii)
    if not np.isfinite(nn) or not np.isfinite(contact) or contact <= 0:
        # A NaN box means the lattice scan diverged; report it rather than dividing.
        return {"n_particles": int(len(pos)), "nn": float(nn), "contact": float(contact),
                "nn_over_contact": float("nan"), "verdict": "NON_FINITE",
                "box": [float(b) for b in box]}
    ratio = nn / contact
    verdict = ("cohesive" if ratio <= CONTACT_COHESIVE
               else "suspect_dilute" if ratio <= CONTACT_EXPLODED else "EXPLODED")
    return {"n_particles": int(len(pos)), "nn": round(nn, 3),
            "contact": round(contact, 3), "nn_over_contact": round(ratio, 4),
            "verdict": verdict,
            "box": [round(float(b), 1) for b in box]}


def generate(slug: str, force: bool = False) -> dict:
    cif = PROTO / f"{slug}.cif"
    if not cif.is_file():
        return {"slug": slug, "status": "ERROR::no such CIF"}
    name = f"lib_{slug.lower()}"
    sysyaml = CS / "systems" / f"{name}.yaml"
    run_dir = CS / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "generate.log"
    rec = {"slug": slug, "run": name, "cif": str(cif.relative_to(ROOT))}
    t0 = time.perf_counter()

    if (run_dir / "first_frame.gsd").is_file() and not force:
        rec.update(contact_validity(run_dir / "first_frame.gsd"))
        rec["status"] = "ALREADY_BUILT"
        return rec

    pre = cif_preflight(cif)
    rec.update({k: v for k, v in pre.items() if k != "ok"})
    if not pre["ok"]:
        rec["status"] = (f"ERROR::degenerate CIF: {pre['coincident_pairs']} coincident site "
                         f"pairs, min distance {pre['min_site_distance']} A")
        return rec

    # --- 1. design -----------------------------------------------------------
    if not sysyaml.is_file() or force:
        import yaml as _yaml
        attempts, best = [], None
        for depth in DEPTH_LADDER:
            run([sys.executable, "design.py", str(cif), "--write",
                 "--target-depth", str(depth), "--tag", "lib"], CS, log)
            # design.py names the written file <cifstem.lower()>_<tag>.yaml
            written = CS / "systems" / f"{cif.stem.lower()}_lib.yaml"
            if not written.is_file():
                attempts.append({"target_depth_kT": depth, "weakest_kT": None})
                continue
            cand = _yaml.safe_load(written.read_text())
            weakest = cand.get("weakest_kT")
            attempts.append({"target_depth_kT": depth, "weakest_kT": weakest})
            if best is None or (weakest is not None
                                and (best[1] is None or weakest < best[1])):
                best = (cand, weakest, depth)
            written.unlink()
            if weakest is not None and weakest <= WEAKEST_KT_MAX:
                break
        rec["depth_ladder"] = attempts
        if best is None:
            rec["status"] = "ERROR::no candidate reached any target depth"
            return rec
        cand, weakest, depth = best
        rec["target_depth_kT"] = depth
        rec["weakest_kT"] = weakest
        # Keep the deepest-binding design even when it never clears the floor, so the
        # crystal is still built and the failure is visible in the cohesion check
        # rather than being silently absent from the library.
        rec["binding_screen"] = ("pass" if weakest is not None and weakest <= WEAKEST_KT_MAX
                                 else "FAIL::weakest species not bound at any target depth")
        sysyaml.write_text(_yaml.safe_dump(cand, sort_keys=False))
    if not sysyaml.is_file():
        rec["status"] = "ERROR::design wrote no system"
        return rec

    import yaml
    system = yaml.safe_load(sysyaml.read_text())
    system["name"] = name
    if not system.get("anisotropic_energy"):
        rec["status"] = "ERROR::system yaml is missing anisotropic_energy"
        return rec
    sysyaml.write_text(yaml.safe_dump(system, sort_keys=False))
    rec["lattice_repeats"] = system["lattice_repeats"]
    rec["debye_length_nm"] = system["debye_length_nm"]

    # --- 2. build ------------------------------------------------------------
    run([sys.executable, "build.py", str(sysyaml.relative_to(CS))], CS, log)
    cfg = (run_dir / "configuration.yaml").read_text()
    if "anisotropic_energy: true" not in cfg:
        rec["status"] = "ERROR::configuration.yaml lacks anisotropic_energy"
        return rec

    # --- 3. create -----------------------------------------------------------
    run(["pacsim-create", "configuration.yaml", "first_frame.gsd"], run_dir, log)
    rec.update(contact_validity(run_dir / "first_frame.gsd"))
    rec["status"] = "OK"
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--index", type=int, help="0-based index into the sorted slug list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    slugs = catalog_slugs()
    if args.list:
        for i, s in enumerate(slugs):
            print(f"{i:3d}  {s}")
        print(f"\n{len(slugs)} CIFs")
        return

    assert_anisotropic_builder()
    if args.index is not None:
        targets = [slugs[args.index]]
    elif args.slug:
        targets = [args.slug]
    else:
        targets = slugs

    rows = []
    for s in targets:
        try:
            rec = generate(s, force=args.force)
        except subprocess.CalledProcessError as exc:
            rec = {"slug": s, "status": f"ERROR::{exc.cmd[1] if len(exc.cmd) > 1 else exc.cmd[0]} "
                                        f"exited {exc.returncode}"}
        except Exception as exc:  # noqa: BLE001
            rec = {"slug": s, "status": f"ERROR::{type(exc).__name__}: {exc}"}
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    out = LIB / "generated" / "records"
    out.mkdir(parents=True, exist_ok=True)
    for rec in rows:
        (out / f"{rec['slug']}.json").write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
