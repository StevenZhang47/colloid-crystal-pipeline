#!/usr/bin/env python
"""
Find surface potentials that actually hold a crystal together, without running MD.

    python tune.py systems/cu3au.yaml
    python tune.py systems/cu3au.yaml --psi-pos 90 120 150 --ratios 2 3 4 5

WHAT IT MEASURES, and why it is not what you would guess.

The obvious metric -- the average cohesive energy per particle -- is a trap. For
Cu3Au it rates psi = +120/-60 mV as the most bound structure available (-15.7 kT
per particle), but that structure melts instantly: the average is dominated by Au,
which is hugely over-bound, while Cu (75% of all particles) is held by exactly
0 kT. Averages hide the weak species.

So this screens on the PER-SITE binding energy: pluck one interior particle of each
type out of the energy-optimized lattice and measure what it costs. A crystal is
only as stable as its WEAKEST species. For reference, CsCl at +/-50 mV -- which
survives MD -- binds both of its species at about -25 kT.

    E_lat      average cohesive energy per particle   (reported, not trusted)
    E_site(X)  cost to remove one interior X          (this is the one that matters)
    min_site   the weakest species. Rank on this.

Nothing here writes a run directory or runs MD -- it prints a table in seconds.
Paste the winning surface potentials into systems/<name>.yaml and submit.
"""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import openmm
import yaml
from openmm import unit
from pymatgen.core import Element
from pymatgen.io.cif import CifParser
from scipy.optimize import minimize_scalar
from scipy.spatial import distance_matrix

from colloids.colloid_potentials_algebraic import ColloidPotentialsAlgebraic
from colloids.colloid_potentials_parameters import ColloidPotentialsParameters
from colloids.run_parameters import RunParameters
from colloids.units import energy_unit, length_unit, mass_unit

from build import RUN_TEMPLATE, render_fields

HERE = Path(__file__).resolve().parent
MILLIVOLT = unit.milli * unit.volt
NANOMETER = unit.nano * unit.meter


def run_parameters_for(system):
    """Render the system's run.yaml to a temp file and parse it, so the potential
    parameters used here are byte-for-byte the ones the simulation will use."""
    fields = render_fields(system, "tune.py", system["cif"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.yaml"
        path.write_text(RUN_TEMPLATE.format(**fields))
        return RunParameters.from_yaml(str(path))


def make_potentials(run_parameters, periodic=False):
    return ColloidPotentialsAlgebraic(
        colloid_potentials_parameters=ColloidPotentialsParameters(
            brush_density=run_parameters.brush_density, brush_length=run_parameters.brush_length,
            debye_length=run_parameters.debye_length, temperature=run_parameters.potential_temperature,
            dielectric_constant=run_parameters.dielectric_constant),
        use_log=run_parameters.use_log, cutoff_factor=run_parameters.cutoff_factor,
        periodic_boundary_conditions=periodic,
        steric_radius_average=run_parameters.steric_radius_average,
        electrostatic_radius_average=run_parameters.electrostatic_radius_average)


def energy_context(types, positions_nm, radii, surface_potentials, masses, run_parameters):
    """An OpenMM context for an arbitrary subset of particles, in vacuum."""
    system = openmm.System()
    potentials = make_potentials(run_parameters)
    for t in types:
        system.addParticle(masses[t].value_in_unit(mass_unit))
        potentials.add_particle(radius=radii[t], surface_potential=surface_potentials[t])
    for force in potentials.yield_potentials():
        force.setForceGroup(system.getNumForces())
        system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions_nm * length_unit)
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)


def _vacuum_energy(types, radii, surface_potentials, masses, run_parameters):
    """energy_context with the OpenMM system built ONCE and the context reused.

    Same numbers as energy_context, but the anisotropic scan evaluates several hundred
    geometries per candidate and rebuilding the system each time dominates the runtime.
    """
    system = openmm.System()
    potentials = make_potentials(run_parameters)
    for t in types:
        system.addParticle(masses[t].value_in_unit(mass_unit))
        potentials.add_particle(radius=radii[t], surface_potential=surface_potentials[t])
    for force in potentials.yield_potentials():
        force.setForceGroup(system.getNumForces())
        system.addForce(force)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("Reference"))

    def energy_of(positions_nm):
        context.setPositions((positions_nm - positions_nm.mean(axis=0)) * length_unit)
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(energy_unit)

    return energy_of


def _axis_groups_for(structure):
    """Which lattice vectors symmetry requires to scale together.

    Delegates to the builder's own implementation so the screen cannot drift from the
    build. Falls back to a uniform scan if an older PACSim without the anisotropic
    LatticeBuilder is on the path, rather than silently doing something different.
    """
    from colloids.colloids_create.configuration_generators.lattice_builder import LatticeBuilder
    if not hasattr(LatticeBuilder, "_axis_groups"):
        return [[0, 1, 2]]
    return [list(g) for g in LatticeBuilder._axis_groups(structure)]


def pair_depth(type_a, type_b, radii, surface_potentials, masses, run_parameters):
    """Depth of the two-particle potential minimum (kJ/mol). Positive => purely repulsive."""
    contact = (radii[type_a] + radii[type_b]).value_in_unit(length_unit)

    def energy(gap):
        pos = np.array([[0.0, 0.0, 0.0], [contact + gap, 0.0, 0.0]])
        return energy_context([type_a, type_b], pos, radii, surface_potentials, masses, run_parameters)

    result = minimize_scalar(energy, bounds=(1.0, 200.0), method="bounded")
    return float(result.fun)


def evaluate(structure, type_map, radii, surface_potentials, masses, run_parameters,
             radii_padding, repeats, scale_range=(0.5, 1.5), samples=50,
             anisotropic=True):
    """
    Build the finite crystal, find the lattice scale that minimizes total energy
    (exactly as LatticeBuilder does), then report:
      - average cohesive energy per particle
      - per-site binding energy of each type (energy to remove one interior particle)
      - the nearest-neighbour distance and surface gap at the optimum
    All energies in kJ/mol.

    ANISOTROPIC (default): the three lattice vectors are relaxed independently, with
    symmetry-equivalent axes tied together, by calling
    LatticeBuilder._optimize_anisotropic_scale_factors directly. It has to be the
    builder's own optimizer and not an equivalent-looking copy, because the screen
    decides the surface potentials the build is then run with, and coordinate descent
    is not convex -- a copy that seeds differently lands in a different minimum.
    A uniform scan applied to a non-cubic CIF holds the ATOMIC axial ratios fixed, and
    those ratios are wrong for colloids of the same structure type: MgZn2 screens as
    two species held at 0.0 kT (nothing binds) under the uniform scan, which would
    reject the prototype for a reason that is an artefact of the search, not physics.
    Cubic cells have a single axis group, so this reduces to the old uniform scan.
    """
    brush = run_parameters.brush_length.value_in_unit(length_unit)

    # Geometric no-overlap scale factor from a small test cell, as LatticeBuilder does.
    test = structure.make_supercell((3, 3, 3), in_place=False, to_unit_cell=True)
    eff = np.array([radii[type_map[z]].value_in_unit(length_unit) + brush + radii_padding
                    for z in test.atomic_numbers])
    d_test = distance_matrix(test.cart_coords, test.cart_coords)
    np.fill_diagonal(d_test, np.inf)
    geometric_scale = float(np.max((eff[:, None] + eff[None, :]) / d_test))

    full = structure.make_supercell(repeats, in_place=False)
    types = [type_map[z] for z in full.atomic_numbers]
    coords = full.cart_coords
    frac = full.frac_coords
    lattice_matrix = full.lattice.matrix

    # One reusable context instead of rebuilding the OpenMM system per candidate --
    # the anisotropic descent needs several hundred single-point energies.
    energy_of = _vacuum_energy(types, radii, surface_potentials, masses, run_parameters)

    def energy_at(factors):
        return energy_of(frac @ (np.asarray(factors)[:, None] * lattice_matrix))

    groups = ([[0, 1, 2]] if not anisotropic else _axis_groups_for(structure))
    candidates = np.linspace(geometric_scale * scale_range[0],
                             geometric_scale * scale_range[1], samples)

    if anisotropic and len(groups) > 1:
        # Call the BUILDER'S optimizer rather than reimplementing it. Coordinate descent
        # is not convex, so a reimplementation that merely looks equivalent lands in a
        # different local minimum: seeding on the best uniform scale instead of on the
        # geometric scale moved AlB2 from c/a 0.944 (builder) to 0.899. Delegating makes
        # screen and build identical by construction.
        from colloids.colloids_create.configuration_generators.lattice_builder import LatticeBuilder
        factors = LatticeBuilder._optimize_anisotropic_scale_factors(
            frac_coords=frac, lattice_matrix=lattice_matrix, types=types, radii=radii,
            surface_potentials=surface_potentials, masses=masses,
            run_parameters=run_parameters, geometric_scale_factor=geometric_scale,
            scale_range=tuple(scale_range), scale_samples=samples,
            symmetry_structure=structure)
        best_energy = energy_at(factors)
    else:
        # Cubic (one axis group) or explicitly isotropic: the uniform scan, unchanged.
        factors, best_energy = None, np.inf
        for scale in candidates:
            e = energy_at(np.full(3, float(scale)))
            if np.isfinite(e) and e < best_energy:
                factors, best_energy = np.full(3, float(scale)), e
    if factors is None or not np.isfinite(best_energy):
        return dict(e_lat=np.nan, sites={}, nn=np.nan, gap=np.nan, scale_factors=None,
                    n_axis_groups=len(groups))

    positions = frac @ (np.asarray(factors)[:, None] * lattice_matrix)
    positions -= positions.mean(axis=0)

    d = distance_matrix(positions, positions)
    np.fill_diagonal(d, np.inf)
    nn = float(d.min())
    i, j = np.unravel_index(np.argmin(d), d.shape)
    gap = nn - (radii[types[i]].value_in_unit(length_unit) + radii[types[j]].value_in_unit(length_unit))

    # Per-site binding: remove the most interior particle of each type and re-evaluate.
    radial = np.linalg.norm(positions, axis=1)
    sites = {}
    for t in set(types):
        interior = next(i for i in np.argsort(radial) if types[i] == t)
        keep = [i for i in range(len(types)) if i != interior]
        without = energy_context([types[i] for i in keep], positions[keep],
                                 radii, surface_potentials, masses, run_parameters)
        sites[t] = best_energy - without

    return dict(e_lat=best_energy / len(types), sites=sites, nn=nn, gap=gap,
                scale_factors=[float(f) for f in factors],
                n_axis_groups=len(groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", type=Path, help="systems/<name>.yaml")
    parser.add_argument("--psi-pos", type=float, nargs="+", default=[60.0, 90.0, 120.0, 150.0],
                        help="surface potentials (mV) to try for the positive type")
    parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                        help="psi_positive / |psi_negative| ratios to try")
    parser.add_argument("--repeats", type=int, default=4,
                        help="supercell for the lattice sum (default 4; keep small, it is O(N^2))")
    parser.add_argument("--debye", type=float, default=None,
                        help="override debye_length_nm")
    args = parser.parse_args()

    system = yaml.safe_load(args.system.read_text())
    if args.debye is not None:
        system["debye_length_nm"] = args.debye
    run_parameters = run_parameters_for(system)
    kT = (unit.MOLAR_GAS_CONSTANT_R * run_parameters.potential_temperature).value_in_unit(energy_unit)

    types_yaml = system["types"]
    positives = [t for t, v in types_yaml.items() if v["surface_potential_mV"] > 0]
    negatives = [t for t, v in types_yaml.items() if v["surface_potential_mV"] < 0]
    if len(positives) != 1 or len(negatives) != 1:
        raise SystemExit("This screen assumes exactly one positive and one negative type.")
    pos, neg = positives[0], negatives[0]

    structure = CifParser(str(HERE / "structures" / system["cif"]),
                          site_tolerance=0.0, frac_tolerance=0.0
                          ).parse_structures(check_occu=False, primitive=False)[0]
    type_map = {z: str(Element.from_Z(z)) for z in np.unique(structure.atomic_numbers)}
    counts = {t: sum(1 for z in structure.atomic_numbers if type_map[z] == t) for t in types_yaml}

    radii = {t: v["radius_nm"] * NANOMETER for t, v in types_yaml.items()}
    masses = {t: v["mass_dalton"] * unit.dalton for t, v in types_yaml.items()}

    print(f"{system['name']}: {system['cif']}")
    print(f"  stoichiometry {counts},  debye = {run_parameters.debye_length}, "
          f"radii {pos} = {radii[pos]}, {neg} = {radii[neg]}")
    print(f"  lattice sum on a {args.repeats}^3 supercell; energies in kT at "
          f"{run_parameters.potential_temperature}\n")

    header = (f"{'psi_'+pos:>7} {'psi_'+neg:>7} {'ratio':>6} {'E_lat':>7} "
              f"{'E_site('+pos+')':>12} {'E_site('+neg+')':>12} {'weakest':>9} "
              f"{'pair '+pos+neg:>9} {'nn':>7} {'gap':>6}  verdict")
    print(header)
    print("-" * len(header))

    results = []
    for psi_pos in args.psi_pos:
        for ratio in args.ratios:
            psi_neg = -psi_pos / ratio
            surface_potentials = {pos: psi_pos * MILLIVOLT, neg: psi_neg * MILLIVOLT}

            out = evaluate(structure, type_map, radii, surface_potentials, masses,
                           run_parameters, system["radii_padding_nm"], args.repeats)
            if not np.isfinite(out["e_lat"]):
                continue
            sites = {t: v / kT for t, v in out["sites"].items()}
            # The WEAKEST species is the least-bound one, i.e. the largest (least
            # negative) site energy -- max, not min. Getting this backwards
            # recommends structures whose majority species is held by 0 kT.
            weakest = max(sites.values())
            depth_ab = pair_depth(pos, neg, radii, surface_potentials, masses, run_parameters) / kT

            # CsCl, which survives MD, binds both of its species at about -25 kT.
            verdict = ("STABLE?" if weakest < -15.0 else
                       "marginal" if weakest < -8.0 else "MELTS")
            print(f"{psi_pos:7.0f} {psi_neg:7.1f} {ratio:6.1f} {out['e_lat']/kT:7.1f} "
                  f"{sites[pos]:12.1f} {sites[neg]:12.1f} {weakest:9.1f} "
                  f"{depth_ab:9.1f} {out['nn']:7.1f} {out['gap']:6.1f}  {verdict}")
            results.append((weakest, psi_pos, psi_neg, sites))

    if results:
        best = min(results)
        print(f"\nbest (weakest species most strongly held): "
              f"psi_{pos} = {best[1]:.0f} mV, psi_{neg} = {best[2]:.1f} mV, "
              f"min_site = {best[0]:.1f} kT")
        print(f"  per-site: " + ", ".join(f"{t} {v:.1f} kT" for t, v in best[3].items()))
        print("  paste into systems/<name>.yaml under `types:` and submit.")


if __name__ == "__main__":
    main()
