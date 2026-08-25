# Colloidal crystal structure detection and identification

Open-set detection of unknown colloidal crystal structures, and identification of the ones
that are detected against a reference library of crystallographic prototypes.

The pipeline has four stages:

```
GSD trajectory
   |
   |  1. PREPROCESS   frozen gas screen (s < 1.62) -> physical components
   |                  M3 reliable-centre selection (exposure > 0.14 AND coord < 13)
   v
   |  2. DESCRIBE     per-particle SOAP, 952-D, GLOBAL_D0
   |                  n_max 8, l_max 6, r_cut 3.0*d0, sigma 0.10*d0, species [11,17]
   |                  d0 = median nearest-neighbour distance in the component
   v
   |  3. DETECT       species-conditioned standardisation -> shared PCA16
   |                  -> 32-D domain fingerprint [LARGE | SMALL]
   |                  -> Ledoit-Wolf Mahalanobis, group-balanced
   |                  -> reject if rho = d/tau > 1
   v
   |  4. IDENTIFY     one-sided chamfer against 41 CIF-ideal anchors: for each candidate
   |                  centre, the distance to its NEAREST anchor environment; median over
   |                  centres, per species. Accept on distance <= 0.1619 AND margin >= 2.45
   v
matched prototype, or "not in the library"
```

## Layout

| path | what it is |
|---|---|
| `crystal_discovery/cif_library/` | canonical deduplicated CIF library, its catalog, and the crystal generator |
| `crystal_discovery/crystal_stability/` | colloid parameter design (`design.py`, `tune.py`) and PACSim run construction (`build.py`) |
| `crystal_discovery/preprocessing_pipeline_final/.../gas_screen_v1/` | the frozen gas screen |
| `results/motif_pipeline/frozen_gas_m3_pipeline_v1/` | frozen gas + M3 reliable-centre pipeline |
| `results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts/` | frozen SOAP descriptor path |
| `results/motif_pipeline/core3_atlas_v3/scripts/` | v3 detection atlas: predeclaration, shard build, A/B comparison |
| `results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1/` | the EMD matcher and its audits |
| `results/final_architecture_decision_v1/` | preregistration for the SOAP vs GAT vs self-supervised architecture study |
| `manuscript/` | LaTeX draft |

## Data is not in this repository

Trajectories, model checkpoints and graph caches are excluded — the working tree they come
from is ~7 GB. `.gitignore` is an **allowlist**: everything is ignored by default and code
paths are re-admitted individually, so `git add .` cannot sweep in simulation output. Adding
a new code directory requires un-ignoring each parent directory on the way down.

Also excluded by design: `crystal_stability/library_stability.csv`, the sealed
candidate-to-prototype mapping held out of the analysis.

## Reproducing a crystal

```bash
source crystal_discovery/crystal_stability/env.sh
export PYTHONPATH=<path to PACSim with anisotropic LatticeBuilder>   # >= 582a2d0
cd crystal_discovery/cif_library
python generate_crystals.py --slug CsCl
```

`generate_crystals.py` screens colloid parameters, builds the PACSim inputs with
`anisotropic_energy: true`, relaxes the lattice, and checks the result is a cohesive crystal
rather than a dilute one.

**The anisotropic LatticeBuilder is required.** A uniform lattice scan preserves the CIF's
axial ratios, which describe an *atomic* crystal and are generally wrong for colloids of the
same structure type. `tune.evaluate` calls the builder's own optimiser rather than
reimplementing it, because coordinate descent is not convex and a copy that seeds
differently lands in a different minimum.

## Known limitations

- **The gas screen assumes close packing.** It removes a particle when the mean normalised
  distance over its four nearest neighbours exceeds 1.2698, which presumes four contacts.
  Open frameworks violate this: on an ideal ReO3 anchor it removed the entire majority
  sublattice (5324 particles in, 1330 out). Any open structure is stripped before the
  descriptor sees it.
- **Four prototypes are not realisable with the fixed 102/120 nm colloid pair** — the three
  Laves phases and NaTl. Laves needs a radius ratio near 1.225; this pair gives 1.176.
  Deepening the pair well does not help, it lengthens the Debye screening.
- **Two library CIFs are unusable**: one Cu3Au file expands to 96 sites with coincident
  pairs, and one zangenite file uses non-element species labels.
- **The matcher is validated on CsCl, Th3P4 and amorphous only.** On the 50 held-out
  validation trajectories it scores 13/13, 11/11 and rejects 26/26 amorphous. A wider
  14-family prototype benchmark fails on the seven families that lose most of their
  particles to the gas screen, which is the open-framework limitation above rather than a
  matcher limitation. `pkg/verify_acceptance.py` reproduces the acceptance test.
