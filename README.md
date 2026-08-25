# Colloidal crystal structure detection and identification

Point it at a trajectory. It tells you whether the structure is one it has seen before,
and which crystallographic prototype it is.

```bash
crystal-discovery analyze trajectory.gsd
```

```
Domain 0
==============================================================
  particles 481   SOAP centres 222   L/S ratio 0.345

  SEEN BEFORE      NO - novel
                   OOD score 42.446   tau 6.7152   rho 6.321   [OOD]

  IDENTIFIED AS    Ni3Sn
                   distance 0.0497   margin 4.628
  VERIFICATION     PASS

  RANKED GUESSES
     #  prototype             distance   vs #1  verdict
     1. Ni3Sn                   0.0497   1.00x  best match
     2. al3ti_d022              0.2300   4.63x  beyond accept range
     3. mosi2_c11b              0.2674   5.38x  beyond accept range
     4. Cu3Au                   0.2809   5.65x  beyond accept range
     5. bif3_d03                0.2816   5.67x  beyond accept range
        gates: distance <= 0.1619, margin >= 2.45
```

That output is the whole idea: the detector has **never seen this structure**, and the
matcher says it is **Ni3Sn** anyway. Those are different questions, asked by different
machinery, and they are allowed to disagree. "Novel" means *not in the training atlas*;
"Ni3Sn" means *this is what it looks like*. A structure can be both.

---

## Install

```bash
git clone git@github.com:StevenZhang47/colloid-crystal-pipeline.git
cd colloid-crystal-pipeline
python -m pip install -e pkg
crystal-discovery analyze /path/to/your/trajectory.gsd
```

That is the entire setup. **No environment variables are required and nothing needs
downloading** — the frozen detector and the 17 MB reference bank travel with the
repository. A fresh clone is 37 MB and works immediately; this is tested, not assumed.

### Requirements

Python >= 3.10 and six packages, all installed automatically by the command above:

| package | why |
|---|---|
| `dscribe` | the SOAP descriptor (pulls in `ase`) |
| `freud-analysis` | periodic neighbour queries in the gas screen |
| `gsd` | reading HOOMD trajectories |
| `numpy`, `scipy`, `pyyaml` | everything else |

### Optional environment variables

None are needed. Two are worth knowing:

```bash
export OMP_NUM_THREADS=8      # BLAS threads. Retrieval is two large matrix products,
                              # so this is the one knob that changes analysis speed.
```

`OPENMM_PLUGIN_DIR` and the PACSim environment are **only** for *building* reference
crystals from CIFs (see [Rebuilding](#rebuilding-the-reference-bank)). Analysis never
touches OpenMM.

---

## Running it

```bash
crystal-discovery analyze traj.gsd                 # last frame
crystal-discovery analyze traj.gsd --frame 250     # a specific frame
crystal-discovery analyze traj.gsd --top-k 10      # more ranked guesses
crystal-discovery analyze traj.gsd --json          # machine-readable
crystal-discovery scan DIR -o results.csv          # every .gsd under a folder
crystal-discovery add-reference s.gsd --name X     # extend the matcher library
crystal-discovery info                             # the frozen model this build uses
```

### Scanning a folder

```bash
crystal-discovery scan /data/runs -o results.csv
```

```
[1/3] ideal_cscl.gsd                 1dom  NOVEL  CsCl
[2/3] steven_example_1.gsd           1dom  NOVEL  Ni3Sn
[3/3] steven_example_2.gsd           1dom  NOVEL  Cu3Au

  files scanned      3        identified as:
  novel (rho > 1)    3          CsCl     1
  identified         3          Ni3Sn    1
  no match           0          Cu3Au    1
```

Recursive by default (`--no-recursive` to stay flat, `--pattern` to change the glob). The
CSV carries one row per **domain**, not per file, since a frame can hold several separate
crystals. Files that fail are written with the reason rather than dropped, so the row count
always accounts for everything on disk.

This is a separate entry point for a practical reason: importing dscribe costs ~30 s and
analysing a trajectory costs ~2 s, so calling the CLI once per file would spend most of
its life importing. `scan` pays it once.

A frame containing several separate crystals is split into domains and each is analysed
independently — scoring them together averages away the thing you are looking for.

### Speed

```
cold start           ~30-45 s     dominated by `import dscribe` (pulls in ase); ONCE
per trajectory         ~2 s       read + preprocess + SOAP + detect + rank
```

The startup cost dominates, so for batch work drive the library from one Python process
rather than invoking the CLI once per file:

```python
from crystal_discovery.config import Config
from crystal_discovery.pipeline import analyze

cfg = Config.load()
for path in trajectories:
    rep = analyze(path, frame=-1, config=cfg, min_centres=64, top_k=5)
    dom = max(rep.domains, key=lambda d: d.n_centres)
    print(path, dom.ood.rho, dom.retrieval.retrieved)
```

---

## Examples

Five trajectories, one for each answer the pipeline can give:

```bash
bash pkg/examples/run_examples.sh
```

| case | detector | matcher |
|---|---|---|
| CsCl crystal | in-distribution | **CsCl**, D=0.1227, margin 3.40 |
| Th3P4 crystal | in-distribution | **Th3P4**, D=0.0778, margin 4.01 |
| amorphous | in-distribution | **nothing** — both gates fail |
| `steven_example_1` | **novel**, rho 6.32 | **Ni3Sn**, D=0.0497, margin 4.63 |
| `steven_example_2` | **novel** | **Cu3Au**, D=0.0752, margin 2.67 |

The amorphous case is the one worth running. It reports `SEEN BEFORE: YES` together with
`IDENTIFIED AS: nothing`, which is correct rather than contradictory — amorphous domains
are *in* the training atlas, so the detector recognises them; there is simply no
crystalline prototype that describes one.

```
  IDENTIFIED AS    nothing in the library
                   closest was Th3P4 at 0.2071 (margin 1.548)
  VERIFICATION     FAIL(distance 0.2071 > 0.1619; margin 1.548 < 2.450)
```

Both gates reject it independently, which is the redundancy they are there for.

---

## How it works

```
GSD trajectory
   |
   |  1. PREPROCESS   frozen gas screen: gas if the mean over the 4 nearest neighbours
   |                  of s = d/(R_i+R_j) exceeds 1.2698; components at s < 1.62
   |                  M3 reliable centres: excluded when exposure > 0.14 AND coord < 13
   v
   |  2. DESCRIBE     per-particle SOAP, 952-D, n_max 8, l_max 6, r_cut 3.0*d0,
   |                  sigma 0.10*d0, species [11,17].  d0 = median nearest-neighbour
   |                  distance in the component, so the descriptor is scale-free
   v
   +------------------------------+-------------------------------+
   |                                                              |
   v  3. DETECT                                    v  4. IDENTIFY
   species-conditioned standardisation             one-sided chamfer against 41
   -> shared PCA16 -> 32-D [LARGE|SMALL]           CIF-ideal anchors: for each centre,
   -> Ledoit-Wolf Mahalanobis                      distance to its NEAREST anchor
   rho = d/tau, novel when rho > 1                 environment; MEDIAN over centres
   tau = 6.715183                                  accept if D <= 0.1619 AND margin >= 2.45
```

Detection and identification are deliberately separate. The detector never names a family;
it only says whether the domain resembles anything in the atlas. Retrieval names the
nearest prototype and abstains when nothing is close enough.

### The matcher, and why it is shaped this way

The obvious approach — average each sublattice into one vector and compare centroids —
**cannot reject amorphous domains**, and this is structural rather than a tuning problem.
A diffuse cloud whose *centre* lands near a prototype's centre reads as that prototype no
matter how spread out it is. Measured: with candidate preprocessing removed, the
mean-SOAP matcher let **26 of 26** amorphous domains through.

Chamfer never forms a centroid. For every centre in the candidate it asks *is there any
environment in this crystal that looks like mine?*, and averages the answers. A structure
is charged for every particle whose neighbourhood nothing in the library can account for.

Two design choices, both measured rather than argued:

**One-sided, candidate → anchor.** The reverse question — *are the anchor's sites present
in the candidate?* — is satisfied trivially by a diffuse cloud, since spread widely enough
it has some particle near almost any environment you name. Being *more* disordered makes
it pass more easily. Across 41 anchors:

```
forward (used)   1.44x        reverse   1.02x        symmetric (textbook Chamfer)   1.25x
```

The textbook symmetric distance averages the good signal with the anti-signal, which is
also why MMD and sliced-Wasserstein lost here despite modelling the full distribution —
both are symmetric and inherit the defect.

**Median, not mean.** A clean CsCl crystal has median nearest-anchor distance 0.117 but
p99 = 0.726: about 1% of centres are defects or surface particles M3 missed, and they
inflate the mean. Amorphous is uniformly far and has no such tail to lose, so the median
tightens only the crystal side, 1.44x -> 1.50x. A 90% trimmed mean scores identically, so
the median is simply the version with no free parameter.

### Reading the output

| line | question |
|---|---|
| `SEEN BEFORE` | is this in the training atlas? `rho = d/tau`, novel when `rho > 1` |
| `IDENTIFIED AS` | which prototype, or `nothing in the library` |
| `VERIFICATION` | `PASS`, or exactly which gate failed and by how much |
| `RANKED GUESSES` | the full ranking, each candidate's distance relative to rank 1 |

Read the ranking when a result surprises you. A rank 1 sitting just ahead of rank 2 means
the library cannot separate those two on this domain, however small the absolute distance
— which is why the margin is a gate and not a footnote.

---

## Validation

Both gates were fitted on the **held-out validation split** and each separates the
populations completely:

```
             crystals      amorphous     window             gap
distance     <= 0.1294     >= 0.1943     [0.1294, 0.1943]   1.50x
margin       >= 3.309      <= 1.590      [1.590, 3.309]     2.08x
```

Reproduce with `python pkg/verify_acceptance.py`, which runs through the shipped code path
rather than the research scripts:

```
CsCl       13/13  ->  CsCl                 PASS
Th3P4      11/11  ->  Th3P4                PASS
amorphous  26/26  correctly rejected       PASS
```

Scanning the **entire** atlas, not just the held-out split
(`boundary_scan.py`, 286 trajectories):

```
Th3P4      n=115   D 0.0673 - 0.0935   accepted as Th3P4    115/115
amorphous  n=171   D 0.1921 - 0.2601   accepted as anything   0/171

worst Th3P4 0.0935  |  gate 0.1619  |  best amorphous 0.1921       gap 2.05x
```

The hardest amorphous cases all sit at low volume fraction across several Debye lengths
and all point at Th3P4 — a specific corner of the phase diagram rather than scattered
noise, which is what approaching a phase boundary looks like.

---

## Repository layout

| path | what |
|---|---|
| `pkg/` | the installable CLI, frozen config, reference bank, examples |
| `crystal_discovery/cif_library/` | 43 CIF prototypes, deduplicated, with catalog and generator |
| `crystal_discovery/crystal_stability/` | colloid parameter design and PACSim run construction |
| `results/motif_pipeline/frozen_gas_m3_pipeline_v1/` | the frozen gas + M3 preprocessing |
| `results/motif_pipeline/soap_atlas_core3_phase2_v1/` | the frozen SOAP descriptor path |
| `results/motif_pipeline/core3_atlas_v3/` | detection atlas and the frozen OOD model |
| `results/statistical_ood_benchmark_v2/.../` | matcher benchmarks and the acceptance test |
| `DATA/` | every dataset, path and split, machine-readable |
| `manuscript/` | LaTeX draft |

**Nothing is reimplemented.** `pkg` delegates preprocessing and SOAP to the frozen modules
that produced the frozen model, rather than carrying a second copy. A second
implementation is a second thing to drift.

### Data is not in this repository

Trajectories and model checkpoints are excluded — the working tree they come from is ~7 GB.
`.gitignore` is an **allowlist**: everything is ignored by default and code paths are
re-admitted one at a time, so `git add .` cannot sweep in simulation output. Adding a new
directory requires un-ignoring every parent on the way down.

`crystal_stability/library_stability.csv` — the candidate-to-prototype mapping held out of
the analysis — is excluded by design, and that exclusion is re-asserted as the **last**
rule in the file, since `.gitignore` is last-match-wins.

`DATA/` records every trajectory actually used, its label, its split and its full path.

---

## Rebuilding the reference bank

Not needed to use the tool. The bank ships pre-computed: 41 anchors, 9512 distinct
environments, float16, 17 MB.

It stores *environments*, not particles. An ideal lattice has very few distinct
neighbourhoods — every particle on a Wyckoff site has the same one — so a greedy cover at
radius 0.02 collapses `lib_cscl`'s 2744 particles to 55 representatives.

Rebuilding needs the built colloidal crystals, which are excluded from the repository and
require PACSim and a GPU:

```bash
source crystal_discovery/crystal_stability/env.sh   # sets OPENMM_PLUGIN_DIR
export PYTHONPATH=<PACSim with anisotropic LatticeBuilder>   # >= 582a2d0
python crystal_discovery/cif_library/generate_crystals.py --slug CsCl
python pkg/build_chamfer_bank.py
```

**The anisotropic LatticeBuilder is required.** A uniform lattice scan preserves the CIF's
axial ratios, which describe an *atomic* crystal and are generally wrong for colloids of
the same structure type. `tune.evaluate` calls the builder's own optimiser rather than
reimplementing it, because coordinate descent is not convex and a copy that seeds
differently lands in a different minimum.

---

## Extending it

### Adding a structure to the matcher library

```bash
crystal-discovery add-reference my_structure.gsd --name MyPhase
```

It extracts SOAP environments from the largest domain, dedupes them with the same greedy
cover the shipped bank uses, and appends. No PACSim and no GPU — any GSD of a well-ordered
structure will do.

Additions go to a **separate** `references/user_bank.npz` that layers on at load time. The
shipped 41-anchor artifact keeps its checksum, `verify_acceptance.py` still reproduces the
published numbers, and any addition is undone by deleting one file.

**It refuses collisions**, and this is the point of the command rather than a nicety:

```
refused: the new structure MATCHES the existing prototype 'CsCl' at D=0.0000 <= 0.1619,
but you called it 'my_new_phase'. By this pipeline's own criterion they are the same
structure, so adding it creates two prototypes that cannot be told apart and every future
match against either fails the margin gate. If they really are the same, pass
--prototype CsCl to add it as another cell setting.
```

The margin gate is a *ratio* between rank 1 and rank 2, so inserting a structural
neighbour of an existing prototype depresses rank 2 for everything near it. That is not
hypothetical: when the library was expanded, Al3Ti (D0_22) landed inside the acceptance
cutoff and a Cu3Au crystal was confidently identified as Al3Ti in 6 of 6 frames. The
command also warns, without refusing, when a new anchor lands within a factor 2.45 of
anything already present, and tells you to re-run the acceptance test.

Adding another **cell setting** of a prototype already present is allowed and expected —
Th3P4 ships with four. Pass `--prototype <existing>`; retrieval collapses settings of one
prototype before the margin test, so they never compete with each other.

### Adding to the detector atlas

There is deliberately no command for this. The detector is not a list to append to; it is
a Gaussian fitted over 380 trajectories in 103 condition groups, and extending it means:

1. new trajectories **with condition-group ids** — the split unit is the group, not the
   trajectory, because seeds within a group are near-copies and splitting by trajectory
   leaks between roles
2. refitting species scaling, PCA16 and the Ledoit-Wolf covariance on the train split
3. **re-deriving tau**; `6.715183` does not carry over
4. re-measuring the false-positive rate on validation
5. a new freeze record and sha256 — every number quoted here becomes stale
6. and the sealed 24-trajectory test is **single use**. Opening it to check a retrained
   model spends it, permanently

The scripts are in `results/motif_pipeline/core3_atlas_v3/scripts/` and `DATA/README.md`
documents the split protocol. It is a reviewable procedure, not a flag.

## Limitations

Run `crystal-discovery info` — these are carried in the freeze record, not only here.

- **Open frameworks are not supported.** The gas screen assumes ~4 near contacts, and M3
  protects interior particles with `same_coord < 13`, a clause no open structure ever
  satisfies — so exposure alone deletes most of the crystal. On an ideal ReO3 anchor the
  screen removed the entire majority sublattice (5324 particles in, 1330 out); on a
  zincblende candidate, 2048 centres became 168. A 14-family prototype benchmark failed on
  exactly the seven families that lose most of their particles this way. The validated
  scope is CsCl, Th3P4 and amorphous.
- **The detector is validated on three well-separated families.** The crowded 4-5 family
  regime, where sequential expansion saw known-FPR rise 0.09 -> 0.36, is not covered.
- **Four prototypes are unrealisable with the 102/120 nm colloid pair** — the three Laves
  phases and NaTl. Laves needs a radius ratio near 1.225; this pair gives 1.176.
  Independently confirmed: those four were the only anchors that disintegrated completely
  under thermalisation (contact retention 0.000). They remain in the bank as reference
  geometries and never appear in any ranking, top-5 included.
- **Two library CIFs are unusable**: one Cu3Au file expands to 96 sites with coincident
  pairs, and one zangenite file uses non-element species labels.
