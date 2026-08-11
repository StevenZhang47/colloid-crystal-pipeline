# Colloidal crystal stabilization: the full decision path

This is the reasoning log for `crystal_stability/`: what we tried, why, what
broke, and which structures are worth pursuing. Read top to bottom -- it is
chronological, because each decision came from the previous failure.

---

## 0. The goal

Take a candidate crystal structure (a CIF), build it as a colloidal crystal under
the PACS interaction potential, relax it with Langevin dynamics, and get a
multi-frame GSD -- but only for structures that are actually STABLE. The point is
to tell stable from unstable *before* spending GPU hours, and to know which
surface charges and salt concentration make a given lattice hold together.

The physics reference is van Kesteren, Zang, Hocky & Sacanna, "Structure
Selection in Ionic Colloidal Crystals via Independent Charge Tuning." Its thesis:
**structure selection is governed by the competition between Coulombic attraction
and lattice-specific like-charge repulsion**, tuned by the surface charge of each
species. That one sentence ended up driving every decision below.

---

## 1. CsCl: the easy case that set the template

CsCl is the colloidal analog of table salt: every nearest neighbour is
counter-charged. We built it from `CsCl.cif` with LatticeBuilder (15^3 supercell,
energy-optimized spacing) and relaxed it with a 30 K -> 298 K temperature ramp.

- charges +50 / -50 mV, radii 85 / 105 nm, lambda_D = 5 nm
- **Result: stable.** Median displacement 0.10 a (below the 0.15 a Lindemann
  melting line), potential energy fell, only 1.5% of particles left their sites
  and all of those were surface.

**Decision recorded:** the diagnostics that decide stability are (1) RDF Pearson
correlation of the final frame vs the ideal lattice (paper's own >0.5 criterion,
computed for all / positive / negative particles separately), (2) the Lindemann
ratio, (3) how many particles escaped and whether they are surface or bulk.
The pair well depth of this run was **-5.6 kT** -- remember this number.

CsCl worked on the first guess *because it is electrostatically self-stabilizing*.
That made it a bad teacher: it hid every mistake that Cu3Au was about to expose.

---

## 2. Cu3Au attempt 1: reasoning from stoichiometry -- WRONG

Cu3Au (L1_2) is fcc: 3 majority + 1 minority per cell, and **8 of every majority
particle's 12 nearest neighbours are its own kind** (like-charged, repulsive).

First guess, by analogy to charge-balancing the 3:1 cell:
- psi = +90 / -30 mV (ratio 3:1), radii 100 / 100 nm (equal), lambda_D = 5 nm
- **Result: melted.** 85% of particles left their sites, potential energy *rose*.

Three separate errors, uncovered one at a time:

**Error A -- equal radii (the fatal one).** The PACS potential depends on the
SURFACE separation h, not the centre distance. In fcc all 12 neighbours are
equidistant centre-to-centre, but if the two species differ in size,

    h(small-small, like)   = nn - 2*R_small
    h(small-large, unlike) = nn - R_small - R_large

so the like pairs sit `R_large - R_small` further apart, surface-to-surface, and
`exp(-dh/lambda_D)` suppresses their repulsion for free. Equal radii sets dh = 0
and throws that mechanism away. This is the paper's core point and I had deleted it.

**Error B -- reasoning from stoichiometry, not the neighbour shell.** The 3:1
formula told me to use a 3:1 charge ratio. But a particle feels its 12 neighbours,
split 8-like / 4-unlike. Minimizing a majority particle's site energy
(`-4*psi_maj*psi_min + 8*psi_maj^2`) gives the optimum ratio psi_min/psi_maj = 4,
not 3. The 8 and 4 are neighbour counts; they have nothing to do with 3:1.

**Error C -- Debye length too short.** Attenuated/weaker charges need LONGER
screening to bind. The paper's whole experimental story is that reducing surface
charge shifts the crystallization window to larger lambda_D.

---

## 3. The metric trap: average energy lies

To stop guessing I built `tune.py`. First version ranked candidates by the
**average cohesive energy per particle** -- and confidently recommended
psi = +120/-60 mV for Cu3Au, whose majority species turns out to be held by
**exactly 0 kT**. The average looked great (-15.7 kT/particle) only because the
minority was over-bound at -164 kT, and it is just 25% of the particles.

**Decision recorded:** never rank on the average. Rank on the **per-site binding
of the WEAKEST species** -- pluck one interior particle of each type out of the
relaxed lattice, measure the energy cost, and take the least-bound one. A crystal
is only as stable as the species that is least held. (CsCl holds both species at
about -25 kT; that is the scale that survives.)

Second bug in the same tool: I ranked by `min(site_energies)` when the weakest
species is the `max` (least negative). Fixed. Lesson: the sign of the ranking key
matters as much as the key.

---

## 4. Cu3Au done right: the colleague's files as ground truth

The colleague's `configurationnew.yaml` + `latticenew.lmp` gave the arrangement
that actually assembles, and it was inverted from mine on every axis:

|                    | my guess (melted) | colleague (stable) |
|--------------------|-------------------|--------------------|
| majority (3x) site | negative, r=100   | **positive, r=102, +25 mV** |
| minority (1x) site | positive, r=100   | **negative, r=120, -45 mV** |
| size ratio         | 1.0               | **0.85** |

The size ratio does the heavy lifting: the like-charged majority pairs sit 18 nm
further apart surface-to-surface, which screening turns into a 4-40x suppression
of the repulsion. With that, only modest +25/-45 mV charges are needed.

**But my first "corrected" run still failed** -- `Energy is NaN` at step 230k. I
had set lambda_D = 12 nm (misread from the paper's Fig. 5, which is a
repulsion-comparison condition, not a crystallization one). That gave a **-28 kT**
pair well: five times deeper than the CsCl run that worked, deep in the
irreversible-aggregation regime, so particles fell into the brush-overlap
divergence and the integrator blew up.

**Decision recorded, and this became the central design rule:** set lambda_D by
the **unlike-pair well depth**, anchored to a run you have seen succeed. CsCl held
at -5.6 kT; matching Cu3Au to -5.4 kT meant lambda_D = 6 nm. At 6 nm Cu3Au is
**stable** (all-particle RDF 0.84, core displacement 0.084 a). The ~12% surface
loss is the weak majority evaporating from the surface, exactly as predicted.

---

## 5. The design tool: `design.py`

Everything above collapses into an automatic procedure. For a CIF:

1. The colloid pair is FIXED by the experiment: small = 102 nm positive,
   large = 120 nm negative. The only geometric freedom is **which sublattice each
   sits on** -- so try both assignments (big-on-minority, big-on-majority).
2. For each charge pair, **solve lambda_D so the unlike pair well = -5 kT** (the
   regime CsCl and Cu3Au both held in). This replaces guessing salt.
3. Screen charges over the physically reachable range (bare colloids ~+35/-45 mV,
   surfactant only attenuates), and **rank on the weakest species' per-site
   binding**.

Validation (the reason to trust it): run on CsCl and Cu3Au with no hints, it
independently recovers the colleague's Cu3Au arrangement (big-on-minority) and the
Steven1 charge condition. It is a re-derivation of the known answers, not a fit.

`design.py --all --write` emits a `systems/<name>.yaml` per prototype; `build.py`
+ `submit_gpu.sbatch` turn each into a relaxed GSD.

---

## 6. What the task14 library did and did NOT establish

The `task14_relaxed_reference_library` contains perfect + variant GSDs for the
prototype family and a `reference_family_qc.csv` -- but its `survived_short_md`
column is **NaN for all 60 rows**: the stability check was never run. And the
`short_md_stability` frames are only **2000 steps x 0.0032 ps = 6.3 ps** of MD,
far too short for anything to melt (every one shows ~0.03 Lindemann, 0% escaped).
So the "120/120 PASS" means the files were written, not that the crystals held.
`check_library.py` re-checks them; the conclusion is that the library carries no
stability evidence either way. `zangenite_reference.gsd` is separately unusable
(all diameters = 1.0 -- a bare lattice, radii never applied).

**Decision recorded:** stability must be established here, from real 500k-step
runs, not inherited from that library.

---

## 7. Which structures are fruitful -- the MD ground truth

Ran 500k-step relaxations for 13 prototypes at their designed parameters and
verified with `verify_run.py` (see `run_verdicts.csv`). The reliable signal is the
CORE Lindemann ratio (interior half of the crystal): < 0.15 = solid, plus the
escaped fraction. The RDF-vs-first-frame correlation is NOT reliable (the first
frame is the unthermalized ideal lattice with delta-sharp peaks, so thermal
broadening alone tanks it -- it wrongly flags stable CsCl as melted).

STABLE (core solid to frame 500):
    th3p4       core Lindemann 0.049, 7% surface loss    <- best
    cscl        0.082, 1%
    cu3au (d6)  0.084, 12%
    ni3sn_d019  0.098, 18%

MELTED / dispersed:
    zangenite   0.153 core (borderline, weak-species surface loss)
    antifluorite 0.53   nickeline 0.53   cr3si 0.64   fluorite 0.70
    zincblende  0.91    cu3au(+90/-30, old) 0.66
    reo3 1.41   i3p 1.38    (these two did not melt but DISPERSED into gas)

**The headline result: the four stable structures are exactly the four the paper
reports experimentally** -- Th3P4, CsCl, Cu3Au, and Ni3Sn (= the D0_19 / Li3Fe
prototype). Every other prototype melts or disperses under the same fixed colloid
pair (102 nm +, 120 nm -). The MD independently rediscovers the paper's structure
set. That is the strongest validation available.

**Why the design screen over-predicted some of these.** The per-site binding
energy is NECESSARY but NOT SUFFICIENT. It measures the depth of the well for
pulling one particle straight out; it does not see:
- **Low coordination.** Zincblende (coord 4, diamond-like) and the fluorite
  minority site are open lattices with soft/zero shear modes under central
  colloidal forces. They melt despite deep single-particle wells. This is the
  classic reason isotropic colloids do not form diamond without patchiness.
- **Anisotropic like-contacts.** Nickeline has Ni-Ni chains along c whose
  spacing the average per-site energy does not flag.
- **Weak absolute binding.** ReO3 (-10 kT) and I3P (-5 kT) were correctly
  predicted marginal/weak; at a -5 kT pair well the open frameworks just
  evaporate. These are true-negative predictions, not screen failures.

So the design tool is a good FILTER (cheap, catches the weak-binding cases) but
its "strong" verdict must be confirmed by (a) coordination number >= ~8 and
(b) an actual short MD run. The reliable stability predictor turned out to be:
high coordination + suppressed like-repulsion + pair well near -5 kT.

## 8. The rescue campaign (see RESCUE.md for full detail)

The -5 kT pair well was calibrated on CsCl (coord 8) and Cu3Au (coord 12). A
coord-6 lattice has fewer bonds per particle, so it needs DEEPER bonds to reach the
same per-particle binding. Re-designing the melted structures at a -8 kT well
(`design.py --target-depth -8`, i.e. lower salt) rescued three of them:

    rocksalt_B1   0.159 borderline  -> 0.102  STABLE
    cr3si_A15     0.636 melted      -> 0.099  STABLE
    zangenite     0.153 borderline  -> 0.075  STABLE

**Final stable set: 7 structures.**
    baseline (-5 kT): CsCl, Th3P4, Cu3Au, Ni3Sn   (= the paper's four)
    rescued  (-8 kT): rocksalt_B1, Cr3Si_A15, zangenite

The failures are as informative as the rescues, because the controls failed exactly
as the categories predicted:
  - **zincblende at -10 kT still melts** (0.487). Doubling the well depth does not
    save a coordination-4 lattice -- its shear modes are soft under central forces
    no matter how deep the wells. Isotropic colloids cannot make diamond; that
    needs patchy/directional bonds. This is a property of the lattice, not of our
    parameters.
  - **fluorite at size ratio 0.69 still melts** (0.545) -- changing the building
    blocks does not fix a coordination-4 sublattice either.
  - **nickeline improves but does not stabilize** (0.53 -> 0.32). Deepening the
    well lengthens lambda_D, which strengthens its Ni-Ni chain repulsion too, so
    the rescue partly fights itself.

Controlled comparison worth keeping: rocksalt and nickeline are both 1:1,
coordination 6, and were predicted almost identically (-29.7 vs -29.5 kT). Rocksalt
is isotropic octahedral and rescues cleanly; nickeline has chains and does not. That
isolates **like-contact anisotropy** as a failure mode independent of coordination
number -- and it is precisely the thing the per-site screen cannot see.

## 8b. SYSTEM SIZE IS A CONFOUND -- found while validating the Cu3Au FCC parameters

Asked to confirm that `design.py`'s recommended Cu3Au parameters (psi +5/-45 mV,
lambda_D = 11.59 nm) give a stable crystal from `pacsim-create`, the first run came
back MELTED -- and that verdict was WRONG.

    cu3au_l12      +5/-45, lambda 11.59, repeats=8   N=2048   core 0.183   "melted"
    cu3au_l12_big  +5/-45, lambda 11.59, repeats=12  N=6912   core 0.075   STABLE
    cu3au_d6       +25/-45, lambda 6.0,  repeats=12  N=6912   core 0.084   STABLE

Same parameters, different supercell -> opposite verdict. At repeats=8 a 4-atom cell
gives only 2048 particles, and a crystal that small is mostly SURFACE; surface
evaporation swamps the core signal. The design parameters are in fact stable, and
marginally better than the hand-anchored set.

Two consequences:
1. `design.py` no longer writes a fixed `lattice_repeats: 8`. It now sizes the
   supercell to ~6000 particles from the atoms-per-cell count (CsCl -> 14, Cu3Au
   -> 11, Th3P4 -> 6).
2. **Any borderline verdict at N <= 4096 must be re-run larger before it is
   trusted.** The strongly-melted structures (core Lindemann 0.5-1.4) are safe --
   a surface artifact moved Cu3Au only from 0.075 to 0.183, nowhere near 0.5 -- but
   marginal calls in that size range are not.

Also settled here: lambda_D = 11.59 nm is perfectly safe. So the earlier
`Energy is NaN` crash was caused by the WELL DEPTH (-28 kT), not by the long Debye
length. A long lambda_D with attenuated charges is fine; a deep well is not.

## 9. Practical rule going forward

1. Screen with `design.py`. If the weakest species is held weaker than about -8 kT,
   it is WEAK-BINDING -- deepen the well (`--target-depth -8`) before giving up.
2. Check the coordination of the weakest species. **If it is <= 4, stop.** No charge,
   salt, or size-ratio setting stabilizes it with isotropic spheres.
3. Check for like-species chains (one like-contact much shorter than the rest). If
   present, expect the screen to over-predict and depth to help only partially.
4. Otherwise run 500k steps and confirm with `verify_run.py` (trust core Lindemann,
   not the RDF correlation).

Caveat on the rescued three: they need a deeper well = lower salt than the paper's
conditions, i.e. crystallizing closer to the aggregation boundary. We tested
"stable once formed"; that is necessary but not sufficient for assembly kinetics.

---

## 8. What still needs external data

Prototypes with free internal Wyckoff coordinates (low symmetry: oP20, oP28,
mC28, tP20, most hR10, hP12) cannot be built from the AFLOW label alone -- the
label fixes the site symmetry but not the x,y,z. These need their structure pulled
from Materials Project or AFLOW:

    cr2s3_A2B3_hR10   zr3al2_A2B3_tP20   fe3sn2_A3B2_hR10   pt2ge3_A3B2_oP20
    zr4al3_A3B4_hP7   ni4b3_A3B4_mC28    rh4p3_A3B4_oP28    al4c3_A4B3_hR7
    bi4te3_A4B3_hR7   wal5_A5B_hP12

The acquisition script exists (`scripts/motif_pipeline/acquire_task14_mp_cifs.py`)
and reads `MP_API_KEY` from the environment. It was NOT hardcoded in the task14
folder (only `MP_API_KEY_set: true` is recorded). To fetch these:

    export MP_API_KEY=<your key>
    python .../acquire_task14_mp_cifs.py     # or give me the mp-ids

(th3p4_D73_A4B3 in the list IS Th3P4, which we already have as Th3P4.cif.)
