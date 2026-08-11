#!/usr/bin/env python
"""
Given a CIF, find surface potentials and a Debye length that make it stable --
then write the systems/<name>.yaml you can submit.

    python design.py structures/ni3sn_D019.cif
    python design.py --all                      # every CIF in structures/
    python design.py structures/ni3sn_D019.cif --write   # emit systems/*.yaml

THE RECIPE, from what CsCl and Cu3Au taught us:

1. The COLLOIDS ARE FIXED. The experiment has one pair of particles:
   small = 102 nm, positive; large = 120 nm, negative. You do not get to choose
   radii -- you choose which sublattice each one sits on, their charges, and the
   salt. So for each prototype there are exactly two assignments to try:
   the big colloid on the majority site, or on the minority site.

2. PAIR WELL DEPTH SETS THE SALT. The unlike-pair minimum has to land near
   -5 kT: that is where CsCl (-5.6) and Cu3Au (-5.4) both held. Deeper is the
   irreversible-aggregation/gel regime (at -28 kT the integrator blew up with
   Energy is NaN); shallower is the gas regime. So for any charge pair we solve
   for the lambda_D that hits the target depth, rather than guessing it.

3. THE WEAKEST SPECIES DECIDES STABILITY. Not the average cohesive energy --
   that metric rates a structure whose majority species is held by 0 kT as the
   most bound one available. Rank on the per-site binding of the worst-held
   species (energy to pluck one interior particle out of the relaxed lattice).
   CsCl, which survives, holds both species at about -25 kT.

The screen is seconds per candidate. Only the survivors are worth GPU time.
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from openmm import unit
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from scipy.optimize import brentq

from tune import HERE, MILLIVOLT, NANOMETER, evaluate, pair_depth, run_parameters_for
from colloids.units import energy_unit

# The experimental colloid pair (van Kesteren et al.): 204 nm (+) and 240 nm (-).
# radius_nm can be overridden with --small-radius / --large-radius for the
# size-ratio rescue experiments (which mean *different building blocks*, off the
# paper's fixed pair -- flagged as such in the output).
SMALL = dict(radius_nm=102.0, mass_dalton=1.884998982291879)
LARGE = dict(radius_nm=120.0, mass_dalton=1.0)

# Default target pair well. -5 kT is where CsCl (-5.6) and Cu3Au (-5.4) held.
# Deeper wells bind harder (rescue for low-coordination lattices) but approach the
# irreversible-aggregation/gel regime -- a -28 kT well NaN'd the integrator.
TARGET_PAIR_DEPTH_KT = -5.0
LAMBDA_BRACKET_NM = (1.5, 30.0)

# Bare colloids are about +35 / -45 mV; surfactant only ATTENUATES (SDS lowers the
# positive, CTAB lowers the negative). So these are the physically reachable ranges.
PSI_POSITIVE_GRID = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
PSI_NEGATIVE_GRID = [-45.0, -35.0, -25.0, -15.0]


def structure_atom_count(cif_path):
    """Atoms in the conventional cell -- used to size the supercell to ~6000 particles."""
    s = CifParser(str(cif_path), site_tolerance=0.0, frac_tolerance=0.0
                  ).parse_structures(check_occu=False, primitive=False)[0]
    return s.atomic_numbers


def base_system(name, cif_name):
    """A system dict with everything except the two type entries and debye."""
    return dict(
        name=name, cif=cif_name, lattice_repeats=4,
        types={}, debye_length_nm=5.0, potential_temperature_K=298.0,
        brush_length_nm=10.0, brush_density_per_nm2=0.09, dielectric_constant=80.0,
        cutoff_factor=21.0, radii_padding_nm=15.0, lattice_padding_nm=1000.0,
        optimize_energy=True, energy_scale_range=[0.5, 1.5], energy_scale_samples=50,
        start_temperature_K=30.0, ramp_steps=200000, run_steps=500000, n_frames=500,
        timestep_ps=0.3, friction_per_ps=1.0, seed=1, use_gravity=True,
        particle_density_g_per_cm3=1.05, water_density_g_per_cm3=0.998,
        walls=[True, True, True], platform="CUDA",
    )


def solve_lambda(system, positive, negative, radii, masses, target_kT):
    """Find the debye length whose unlike-pair well depth equals target_kT."""
    def deficit(lambda_nm):
        s = dict(system, debye_length_nm=float(lambda_nm))
        rp = run_parameters_for(s)
        kT = (unit.MOLAR_GAS_CONSTANT_R * rp.potential_temperature).value_in_unit(energy_unit)
        potentials = {t: v * MILLIVOLT for t, v in [positive, negative]}
        return pair_depth(positive[0], negative[0], radii, potentials, masses, rp) / kT - target_kT

    try:
        # Deeper well as lambda grows, so deficit is monotone decreasing.
        if deficit(LAMBDA_BRACKET_NM[0]) < 0 or deficit(LAMBDA_BRACKET_NM[1]) > 0:
            return None
        return float(brentq(deficit, *LAMBDA_BRACKET_NM, xtol=0.02))
    except (ValueError, RuntimeError):
        return None


def screen_cif(cif_path, repeats=3, verbose=True, target_depth=TARGET_PAIR_DEPTH_KT,
               small=SMALL, large=LARGE):
    """Try both colloid assignments x the charge grid. Return ranked candidates.

    target_depth: the unlike-pair well depth (kT) that lambda_D is solved for.
                  Make it more negative to bind harder (rescue for low-coordination
                  lattices); too deep -> gel/NaN.
    small, large: the two colloids (radius_nm, mass_dalton). Change radii for the
                  size-ratio rescue -- means different building blocks, not the
                  paper's fixed pair.
    """
    SMALL, LARGE = small, large
    structure = CifParser(str(cif_path), site_tolerance=0.0, frac_tolerance=0.0
                          ).parse_structures(check_occu=False, primitive=False)[0]
    type_map = {z: str(Element.from_Z(z)) for z in np.unique(structure.atomic_numbers)}
    counts = {}
    for z in structure.atomic_numbers:
        counts[type_map[z]] = counts.get(type_map[z], 0) + 1
    if len(counts) != 2:
        raise SystemExit(f"{cif_path.name}: need exactly 2 species, found {counts}")

    majority = max(counts, key=counts.get)
    minority = min(counts, key=counts.get)
    if counts[majority] == counts[minority]:      # 1:1 -- the labels are arbitrary
        majority, minority = sorted(counts)

    name = cif_path.stem.lower()
    results = []
    for assignment in ("big_on_minority", "big_on_majority"):
        big, small = (minority, majority) if assignment == "big_on_minority" else (majority, minority)

        # The large colloid is the negative one, the small colloid the positive one.
        for psi_pos in PSI_POSITIVE_GRID:
            for psi_neg in PSI_NEGATIVE_GRID:
                system = base_system(name, cif_path.name)
                system["types"] = {
                    small: dict(SMALL, surface_potential_mV=psi_pos),
                    big: dict(LARGE, surface_potential_mV=psi_neg),
                }
                lambda_nm = solve_lambda(system, (small, psi_pos), (big, psi_neg),
                                         {small: SMALL["radius_nm"] * NANOMETER,
                                          big: LARGE["radius_nm"] * NANOMETER},
                                         {small: SMALL["mass_dalton"] * unit.dalton,
                                          big: LARGE["mass_dalton"] * unit.dalton},
                                         target_depth)
                if lambda_nm is None:
                    continue

                system["debye_length_nm"] = round(lambda_nm, 2)
                rp = run_parameters_for(system)
                kT = (unit.MOLAR_GAS_CONSTANT_R * rp.potential_temperature).value_in_unit(energy_unit)
                radii = {t: v["radius_nm"] * NANOMETER for t, v in system["types"].items()}
                masses = {t: v["mass_dalton"] * unit.dalton for t, v in system["types"].items()}
                potentials = {t: v["surface_potential_mV"] * MILLIVOLT for t, v in system["types"].items()}

                out = evaluate(structure, type_map, radii, potentials, masses, rp,
                               system["radii_padding_nm"], repeats)
                if not np.isfinite(out["e_lat"]):
                    continue
                sites = {t: v / kT for t, v in out["sites"].items()}
                weakest = max(sites.values())          # least-bound species
                results.append(dict(
                    prototype=name, assignment=assignment, big=big, small=small,
                    psi_small=psi_pos, psi_large=psi_neg, lambda_nm=round(lambda_nm, 2),
                    weakest_kT=weakest, weakest_species=max(sites, key=sites.get),
                    sites={t: round(v, 1) for t, v in sites.items()},
                    nn=out["nn"], system=system,
                ))

    results.sort(key=lambda r: r["weakest_kT"])
    if verbose:
        print(f"\n{cif_path.name}   stoichiometry {counts}   (majority = {majority})")
        print(f"{'assignment':17s} {'psi_s':>6} {'psi_l':>6} {'lambda':>7} "
              f"{'weakest':>8} {'species':>8}  {'nn':>7}  per-site")
        for r in results[:6]:
            print(f"{r['assignment']:17s} {r['psi_small']:6.0f} {r['psi_large']:6.0f} "
                  f"{r['lambda_nm']:7.2f} {r['weakest_kT']:8.1f} {r['weakest_species']:>8}  "
                  f"{r['nn']:7.1f}  {r['sites']}")
        if not results:
            print("  no candidate reached the target pair depth in the physical charge range")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cifs", type=Path, nargs="*", help="CIF(s) in structures/")
    parser.add_argument("--all", action="store_true", help="every CIF in structures/")
    parser.add_argument("--repeats", type=int, default=3, help="supercell for the lattice sum")
    parser.add_argument("--write", action="store_true", help="write systems/<name>.yaml for the best candidate")
    parser.add_argument("--target-depth", type=float, default=TARGET_PAIR_DEPTH_KT,
                        help="unlike-pair well depth in kT to solve lambda_D for (default -5; "
                             "more negative binds harder, for low-coordination rescue)")
    parser.add_argument("--small-radius", type=float, default=SMALL["radius_nm"],
                        help="positive colloid radius nm (default 102; change for size-ratio rescue)")
    parser.add_argument("--large-radius", type=float, default=LARGE["radius_nm"],
                        help="negative colloid radius nm (default 120)")
    parser.add_argument("--tag", default=None, help="suffix for the written yaml name")
    args = parser.parse_args()

    small = dict(SMALL, radius_nm=args.small_radius)
    large = dict(LARGE, radius_nm=args.large_radius)

    paths = sorted((HERE / "structures").glob("*.cif")) if args.all else args.cifs
    if not paths:
        raise SystemExit("give a CIF or --all")

    best_rows = []
    for path in paths:
        try:
            results = screen_cif(path, repeats=args.repeats, target_depth=args.target_depth,
                                 small=small, large=large)
        except Exception as exc:
            print(f"\n{path.name}: SKIPPED ({type(exc).__name__}: {exc})")
            continue
        if not results:
            continue
        best = results[0]
        best_rows.append(best)
        if args.write:
            system = best["system"]
            # Size the supercell to reach ~6000 particles. A fixed repeats=8 gives
            # only 2048 particles for a 4-atom cell (Cu3Au, ReO3, nickeline), and a
            # crystal that small is mostly surface: Cu3Au at repeats=8 reads as
            # MELTED (core Lindemann 0.183) purely from surface evaporation, while
            # the SAME parameters at repeats=12 are STABLE (0.075).
            atoms_per_cell = len(structure_atom_count(path))
            system["lattice_repeats"] = max(6, min(16, round((6000 / max(atoms_per_cell, 1)) ** (1 / 3))))
            # Record where the CIF came from, so a system designed against the
            # deduplicated cif_library does not silently pick up a same-named file in
            # structures/ (Th3P4.cif exists in both and the two are byte-different).
            system["cif_dir"] = str(path.resolve().parent)
            # Record the screen's verdict IN the system, so a downstream driver can tell
            # "designed and binding" from "designed but held at 0.0 kT" without
            # re-parsing the header comment.
            system["weakest_kT"] = round(float(best["weakest_kT"]), 2)
            system["weakest_species"] = str(best["weakest_species"])
            system["target_depth_kT"] = float(args.target_depth)
            # Always on for new designs: no-op for cubic, required for the rest.
            system["anisotropic_energy"] = True
            stem = best["prototype"] if args.tag is None else f"{best['prototype']}_{args.tag}"
            system["name"] = stem
            out = HERE / "systems" / f"{stem}.yaml"
            header = (
                f"# {stem}: designed by design.py"
                f"{'' if args.tag is None else ' (rescue experiment: '+args.tag+')'}.\n"
                f"# {best['small']} = small (+, r={args.small_radius:.0f}nm) on the "
                f"{'majority' if best['assignment']=='big_on_minority' else 'minority'} site;"
                f" {best['big']} = large (-, r={args.large_radius:.0f}nm).\n"
                f"# lambda_D solved so the unlike-pair well is {args.target_depth} kT.\n"
                f"# weakest species {best['weakest_species']} held at {best['weakest_kT']:.1f} kT "
                f"(CsCl, which survives, holds both at about -25 kT).\n"
            )
            out.write_text(header + yaml.safe_dump(system, sort_keys=False))
            print(f"  wrote {out}")

    if len(best_rows) > 1:
        print("\n\n=== best candidate per prototype (ranked by weakest species) ===")
        print(f"{'prototype':22s} {'assignment':17s} {'psi_s':>6} {'psi_l':>6} {'lambda':>7} {'weakest':>8}")
        for r in sorted(best_rows, key=lambda r: r["weakest_kT"]):
            print(f"{r['prototype']:22s} {r['assignment']:17s} {r['psi_small']:6.0f} "
                  f"{r['psi_large']:6.0f} {r['lambda_nm']:7.2f} {r['weakest_kT']:8.1f}")


if __name__ == "__main__":
    main()
