# Data provenance

Every dataset this project consumed, where it lives, and how it was split. Regenerate
with `python DATA/build_manifest.py` — every number below is read from the frozen
manifests, never typed by hand, so this record cannot drift from what the code used.

| file | contents |
|---|---|
| `detector_trajectories.csv` | all 410 trajectories: atlas, sealed test, candidate panel — with role, label, cohort, path |
| `reference_library.csv` | 43 CIFs and the crystals built from them |
| `external_datasets.csv` | data from outside the atlas, used only as probes |
| `SUMMARY.json` | the counts below, machine-readable |

---

## 1. Detector atlas — 380 trajectories, 103 condition groups

Source: `crystal_discovery/preprocessing_pipeline/nicoles_dataset/phase_diagram/`

Each trajectory is one (Debye length, volume fraction, seed) simulation. Full paths are
in `detector_trajectories.csv`.

```
label       trajectories
amorphous   171
Th3P4       115
CsCl         94
```

### How it was split

**The split unit is the condition group, never the trajectory.** A condition group is one
(Debye, volume fraction) setting containing several seeds. Seeds within a group are
near-copies — same physics, different random start — so treating them as independent
would overstate the sample size and leak information between roles.

```
role          trajectories   condition groups   purpose
train                 260             68/70     fit mu, Sigma, PCA, standardisation
calibration            70                19     set tau (the acceptance threshold)
validation             50                14     measure the false-positive rate
```

**Verified: zero condition groups appear in more than one role**, in either cohort. A seed
used to fit the model can never also set the threshold or measure it.

Group assignment is deterministic — `sha256` rank of the condition group id, per dominant
family — so the split is reproducible from the ids alone. Fractions: 40% of
expert-pure groups to calibration, 30% to validation.

### The two cohorts

Both share **identical** calibration and validation sets. They differ only in training
content, which is what makes the comparison clean.

```
cohort A   246 regions   126 train   expert-supported labels ONLY
cohort B   380 regions   260 train   expert + 134 machine-generated (GAT-provisional)
```

Provisional labels are confined to training and asserted never to reach calibration or
validation. **Cohort A is the production model** — the extra 134 provisional regions
bought nothing and cost threshold stability (relative width 0.031 vs 0.098).

---

## 2. Untouched test — 24 trajectories, 6 condition groups, NEVER OPENED

Listed in `detector_trajectories.csv` with `role = sealed_never_opened`. Reserved in the
predeclaration (`V3_PREDECLARATION.json`, sha256 `29c1cf5d…`) before any model existed.

`FREEZE.json` records `untouched_test_opened: false`. It is a single-use resource: opening
it after model and threshold are frozen gives one unbiased estimate, and only one.

---

## 3. Candidate novel panel — 6 trajectories

Reserved in the predeclaration, held out of the atlas. The GAT calls all six Th₃P₄, but
their L/S ratios are 2.9–6.3 where true Th₃P₄ is 0.75. The frozen detector rejects all
six (rho 2.55–4.93).

---

## 4. Reference library — 43 CIFs, 37 usable

`reference_library.csv` joins each CIF to the colloidal crystal built from it.

```
origin        21 local tree + 21 Materials Project + 1 from a cookbook branch
built         crystal_stability/runs/lib_<slug>/first_frame.gsd
usable        37 (all 37 are in the matcher's fingerprint bank)
```

Every crystal was built with `optimize_energy: true` **and** `anisotropic_energy: true`
(PACSim ≥ 582a2d0). A uniform lattice scan preserves the CIF's axial ratios, which
describe an *atomic* crystal and are generally wrong for colloids of the same structure
type. Cubic cells are unaffected; the other 19 are not.

The 6 unusable, with reasons, are in the CSV. Briefly: one Cu₃Au CIF has 96 sites with
40176 coincident pairs; one zangenite uses non-element species labels; the three Laves
phases and NaTl come out dilute because they need a radius ratio near 1.225 and the fixed
102/120 nm colloid pair gives 1.176.

---

## 5. External probes — never used in any fit

```
K4C60                  k4c60_inspect/k4c60_final_frame.gsd
                       novelty positive control; rho 22.5 -> OOD, retrieval abstains

external candidate 1   unified_pipeline/data/test_set/steven_example_1.gsd
external candidate 2   unified_pipeline/data/test_set/steven_example_2.gsd

Hoellmer snowmen       /scratch/ph2484/ProjectHocky/NewSnowmenRuns/SweepLong{,Retry}
                       rigid trimers, 156 run dirs, external collaborator data
```

**Caveat on the snowmen.** The pipeline treats them as a two-species ionic crystal by
keeping body+lobe and dropping the head. That reduction is only valid while the head is
small. Above roughly `head/body = 0.35` the system is genuinely ternary and the reduction
discards a real species; at `lobe_ratio = 1.0` the body and lobe are both 256.0 nm, so the
LARGE/SMALL assignment is decided by an exact tie, and at 1.1 the two swap. Any match
reported in that regime is not trustworthy.

---

## Not in this repository

Trajectories, SOAP shards, model checkpoints and graph caches — several GB, all
regenerable from the code and the paths above. `.gitignore` is an allowlist, so data
cannot be committed by accident.

`crystal_stability/library_stability.csv` is **sealed** and deliberately excluded: it maps
external candidates to prototypes and is held out of the analysis.
