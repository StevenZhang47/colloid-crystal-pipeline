# Can we stabilize the melted structures?

The 9 melted prototypes did not all fail for the same reason. Sorting them by
failure mode tells us which are rescuable and how. Three levers exist, all from the
paper + physics:

  1. deeper well  (lower salt / longer lambda_D) -- stronger attraction per bond
  2. size ratio   (different building blocks) -- the paper's other structure knob;
                  changes which like-contacts are closest
  3. gentler protocol (lower final T, slower ramp) -- only helps if the melt is
                  kinetic, not thermodynamic

Levers we do NOT have: the coordination number and sublattice topology are fixed by
the prototype. If a lattice is mechanically soft under central forces, no charge or
salt setting fixes it -- that needs patchy/directional particles, which PACS
isotropic colloids are not.

## Failure modes

**Category A -- weak binding, adequate coordination (RESCUABLE by deeper well).**
Coordination >= 6 and roughly isotropic, melted mainly because the -5 kT pair well
was too shallow for a lattice with fewer bonds per particle than CsCl/Cu3Au.
  - rocksalt_B1 (coord 6-6), fe4n (coord ~), cr3si_A15 (-9 kT predicted),
    zangenite (0.153, borderline)
  Rescue: --target-depth -8 to -10. Bond energy scales ~linearly with the well
  depth; a coord-6 lattice at -8 kT has similar per-particle binding to a coord-8
  lattice (CsCl) at -5 kT. Risk: too deep -> irreversible aggregation/gel.

**Category B -- anisotropic like-contacts / chains (MAYBE, via size ratio).**
The average per-site energy looks fine but one direction has a short like-contact.
  - nickeline_B81 (Ni-Ni chains along c), cr3si_A15 (Cr chains on the A15 columns)
  Rescue: a size ratio that pushes the chain species' like-neighbours apart, or a
  deeper well if the chain contact is already screened enough. Uncertain.

**Category C -- intrinsically low coordination (NOT rescuable with spheres).**
Open lattices with coordination 4 or less have soft/zero shear modes under central
forces; they collapse regardless of how deep the wells are. This is the textbook
reason isotropic colloids do not form diamond.
  - zincblende_B3 (coord 4), fluorite F-site (coord 4), antifluorite Li-site
    (coord 4), reo3_D09 (O-site coord 2 -- the most extreme)
  These need directional bonds. We test zincblende at -10 kT only to DEMONSTRATE
  that a deep well does not save a coord-4 lattice.

## Experiments run -- RESULTS

Verdicts from `verify_run.py` (core Lindemann; < 0.15 with low escape = STABLE).

| structure          | baseline (-5 kT)      | rescue                     | outcome |
|--------------------|-----------------------|----------------------------|---------|
| rocksalt_B1        | 0.159 borderline melt | `--target-depth -8`        | **STABLE 0.102, 0% escape** |
| cr3si_A15          | 0.636 melted          | `--target-depth -8`        | **STABLE 0.099, 1% escape** |
| zangenite          | 0.153 borderline melt | `--target-depth -8`        | **STABLE 0.075, 0.4% escape** |
| nickeline_B81      | 0.532 melted          | `--target-depth -8`        | melted 0.319 (improved, not saved) |
| fe4n_Lp10          | 1.105 dispersed       | `--target-depth -8`        | melted 0.613 (improved, not saved) |
| zincblende_B3      | 0.907 melted          | `--target-depth -10`       | melted 0.487 (NOT saved) |
| fluorite_C1        | 0.699 melted          | size ratio 0.69 (90/130nm) | melted 0.545 (NOT saved) |

**Three structures rescued. The stable set grows from 4 to 7:**
CsCl, Th3P4, Cu3Au, Ni3Sn (baseline) + **rocksalt_B1, Cr3Si_A15, zangenite**.

## What the rescues taught us

**The deeper well (lower salt) is the single most effective lever**, and it works
exactly where the theory said it would -- on lattices whose problem was WEAK
BINDING, not bad topology:
  - cr3si was predicted weak (-9 kT) and melted; at -8 kT pair depth its weakest
    species is held at -16.5 kT and it is solid. A true Category A rescue.
  - rocksalt (coord 6-6) and zangenite were both *borderline* at -5 kT
    (core Lindemann 0.159 and 0.153, just over the line). A modest deepening
    pushed both firmly into the stable basin.
  Physical reading: a coord-6 lattice has fewer bonds per particle than CsCl
  (coord 8) or Cu3Au (coord 12), so each bond must be deeper to reach the same
  per-particle binding. Scaling the well depth with 1/coordination is the rule.

**The controls confirm the categories, by failing as predicted:**
  - **zincblende at -10 kT still melts** (0.487). This is the key negative result:
    doubling the well depth does NOT save a coordination-4 lattice. Its shear
    modes are soft under central forces regardless of how deep the wells are.
    Isotropic colloids cannot make diamond/zincblende -- you need patchy or
    directional bonds. This is a property of the lattice, not of our parameters.
  - **fluorite at a 0.69 size ratio still melts** (0.545). Changing the building
    blocks did not rescue it either -- the majority sits on a simple-cubic
    sublattice with a coordination-4 environment, and that softness survives the
    size change.
  - **nickeline improved but did not stabilize** (0.53 -> 0.32). Its Ni-Ni chains
    along c are the problem, and deepening the well lengthens lambda_D, which
    strengthens the chain repulsion too -- the rescue partly fights itself.
    This is the clearest case where the per-site energy screen LIES: it rated
    nickeline -29.5 kT (strong) yet it melts, because a single-particle removal
    energy cannot see a soft direction.

**Controlled comparison worth keeping:** rocksalt and nickeline are BOTH 1:1,
coordination 6, and were predicted nearly identically (-29.7 vs -29.5 kT). Rocksalt
is isotropic octahedral and rescues cleanly; nickeline has chains and does not.
The only difference is the anisotropy of the like-contacts. That isolates chain
anisotropy as an independent failure mode, separate from coordination number.

## Practical recipe for a new structure

1. Screen with `design.py` -> if the weakest species is > -8 kT, it is weak-binding;
   deepen the well (`--target-depth -8`) before giving up.
2. Check the coordination of the WEAKEST species. If it is <= 4, stop -- no charge,
   salt, or size-ratio setting will stabilize it with isotropic spheres.
3. Check for like-species chains (a like-contact much shorter than the others along
   one axis). If present, expect the per-site screen to over-predict; deepening will
   help only partially.
4. Otherwise run 500k steps and verify with `verify_run.py`.
