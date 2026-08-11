# build_training_set.py — Documentation & Data Analysis

## What the Script Does

`build_training_set.py` generates a per-particle labeled CSV (`training_set.csv`) suitable for training a supervised classifier to distinguish crystal structures in binary colloid simulations.

Each row is one particle with cheap structural features computed from the particle's local coordination shell, plus provenance columns (`source_file`, `frame_index`, `tier`) that enable sound train/test splitting.

> **Note on the labeled_data XYZ files:** The original task spec stated these files "lack box info" and should be used only for labels. This is incorrect — each XYZ file has the box dimensions on line 2 and full particle coordinates on subsequent lines. These files are complete single-frame particle snapshots and are used as geometry sources in SOURCE 4.

---

## Pipeline Steps

### 1. Source discovery
Four data sources are processed in priority order:

| Source | Path | Tier | Label basis |
|--------|------|------|-------------|
| Ground truth | `ground_truth/` | `reference` | Filename |
| Unknown examples | `unknown_examples/` | `reference` | Specified (8 & 9 = crystal; rest = amorphous) |
| Nicole's GSD dataset | `nicoles_dataset/phase_diagram/` | `broadcast` | Manifest or filename |
| Nicole's XYZ snapshots | `labeled_data (1)/CsCl\|Th3P4\|blob\|gas/` | `broadcast` | Directory name |

### 2. Adaptive first-shell cutoff
For each frame, the median nearest-neighbor distance (`d_ref`) is computed, and `cutoff = 1.35 × d_ref` defines the first-shell radius. This adapts automatically to different volume fractions and box sizes — no hardcoded length scale needed.

### 3. Gas removal and cluster isolation
Particles with coordination < `--gas-coord` (default 4) are classified as gas and removed before clustering. The remaining dense particles are connected-component clustered; only the largest cluster (the crystal or amorphous blob) is kept for labeling.

### 4. Single-core gate
For crystal files, if the second-largest cluster exceeds 25% of the largest (`--warn-frac`), the file is multi-blob and skipped. This prevents mislabeling a coexistence snapshot as a pure crystal.

### 5. Surface exclusion
Within the crystal cluster, only particles with coordination ≥ 80th-percentile coordination (the bulk reference) are labeled as interior. Surface particles, whose truncated coordination shells would blur class boundaries, are excluded entirely.

### 6. Multi-frame sampling
Rather than using just the last frame, the script samples every 5th frame (`--frame-step`) from the final 15% of each trajectory (`--frame-frac-start 0.85`). Each frame is an independent thermal snapshot of the same crystal, multiplying labeled particles by ~16× for ground-truth files and ~61× for unknown-example files.

### 7. Crystal type inference (trajectories 8 & 9)
Unknown trajectories 8 and 9 are known to be crystals but their type must be inferred. The script:
1. Pre-computes interior-particle profiles from `nicoles_dataset/phase_diagram/CsCl-like.gsd` and `Th3P4-like.gsd` (same N=1024 system size as the unknowns).
2. Extracts interior particles from the unknown trajectory.
3. Computes mean `frac_unlike` and mean `coord` and finds the nearest reference in (coord/5, frac_unlike) space.

Result: both trajectories 8 and 9 infer as **CsCl-like** (interior coord≈14.0, frac_unlike≈0.57, matching the CsCl reference profile vs Th3P4 interior coord≈11.3).

### 8. Output
All labeled particles are pooled into `training_set.csv` with columns:

```
source_file, frame_index, particle_index, label, tier,
typeid, coord, n_unlike, n_like, frac_unlike, d_mean, d_std,
coord_minus_bulkref, x, y, z
```

`x, y, z` are retained so richer descriptors (SOAP, bond-order parameters) can be computed and joined on `(source_file, frame_index, particle_index)` later.

---

## Output Summary (current run)

| Label | Particles | Sources |
|-------|-----------|---------|
| amorphous | 607,933 | 6 × unknown_examples (61 frames each) + 289 blob XYZ |
| CsCl-like | 139,060 | GT (3,789) + traj 8 & 9 (55,615) + 200 XYZ + 2 nicoles GSD (1,286) |
| Th3P4-like | 112,254 | GT (3,321) + 272 XYZ + 1 nicoles GSD (492) |
| gas | 79,770 | 85 gas XYZ files |
| **Total** | **939,017** | 858 files, 16 columns, no NaN |

| Tier | Particles |
|------|-----------|
| reference | 403,071 |
| broadcast | 535,946 |

---

## Train/Test Splitting Rule — **Critical**

**Never split by particle.** Particles from the same frame are highly correlated (near-duplicate thermal configurations of the same structure). A random particle split leaks spatial and temporal correlations into the test set and will artificially inflate accuracy by ~10–20%.

## Clean-Th3P4 Leakage Audit — 2026-06-12

The clean-crystal retrain uses one global neighbor rule
(`1.35 * median nearest-neighbor distance`), not a Th3P4-specific cutoff.
Clean Th3P4 labels come from configuration provenance rather than q6
thresholds. Physical crystals are split whole: `small`, `medium`, and
`perfect_reference` train; `slab` validation; `rod` test.

No filename, source path, configuration name, frame/particle index, split, or
label enters XGBoost. There are zero duplicate particle keys, zero exact clean
feature vectors shared by train and test, and no physical configuration crosses
splits. Ablations without `typeid` and without all species-identity/count
features retain 99.9812% held-out perfect-crystal accuracy, ruling out those
fields as the shortcut.

The q6 labeling firewall was removed on 2026-06-12 and the canonical dataset
was rebuilt from source. q6 remains a numeric feature, a diagnostic, and an
onset-sampling signal, but it never keeps, drops, or relabels a particle.
The synthetic perfect-crystal signature still does not transfer to real
Th3P4: the firewall-free model predicts 89.21% of held-out real
Th3P4-labelled particles as amorphous. Full numbers and safeguards are
documented in `README_model.md` and
`phase_label_audit/clean_th3p4_retrain/leakage_audit.json`.

## q6 Firewall Removal — 2026-06-12

The q6 firewall has been removed from the active pipeline:

- `build_dataset.py` assigns every eligible interior particle its
  source/configuration label.
- No q6 threshold keeps, drops, or relabels CsCl, Th3P4, or amorphous rows.
- The destructive standalone `q6_firewall.py` script was deleted.
- q6 remains in Stage-1 so the model can learn its usefulness from training
  data rather than having the answer encoded during label construction.

The canonical dataset was rebuilt with **3,654,367 unique particle keys**:

| Class | Particles |
|---|---:|
| gas | 1,001,643 |
| amorphous | 823,447 |
| Th3P4-like | 898,041 |
| CsCl-like | 931,236 |

All 117/117 broadcast CsCl and 136/136 broadcast Th3P4 trajectories
contributed. A source-label invariance test ran the same coordinates under
CsCl, Th3P4, and amorphous provenance across q6 values from 0.228 to 1.000;
every interior row retained the supplied source label.

After regenerating trajectory-safe audit sets and retraining:

- held-out physical `th3p4_rod`: **99.9859%** correct;
- test amorphous F1: **0.9686**;
- test CsCl F1: **0.9697**;
- real held-out Th3P4-labelled -> amorphous: **89.21%**;
- real amorphous -> Th3P4: **0.00%**.

This removes the circular q6/CsCl labeling path and substantially improves
ordinary held-out CsCl and amorphous performance. It does not resolve the
synthetic-to-real Th3P4 domain gap.

## Test-Case Phase Composition — 2026-06-12

`predict_test_case_composition.py` ran the latest firewall-free clean-Th3P4
XGBoost model over every frame of:

`test_cases/example1_trajectory.gsd`

The trajectory contains 1,267 frames and 27,723 particles per frame
(35,125,041 particle classifications total; every particle is assigned exactly
one of gas, amorphous, Th3P4-like, or CsCl-like). The output diagram uses:

- x-axis: time as trajectory frame index;
- top x-axis: simulation step;
- y-axis: percent of all particles;
- each vertical slice sums to exactly 100%.

Key compositions:

| Time | gas | amorphous | Th3P4 | CsCl |
|---|---:|---:|---:|---:|
| frame 0 | 56.39% | 0.74% | 0.00% | 42.87% |
| frame 633 | 17.17% | 7.06% | 0.06% | 75.71% |
| frame 1266 | 7.57% | 1.04% | 0.04% | 91.35% |
| final 50-frame mean | 7.75% | 1.10% | 0.047% | 91.11% |

The model predicts a strong gas-to-CsCl transition. Amorphous material rises
early and then declines to approximately 1%; Th3P4 remains negligible.

Outputs:

- `test_cases/example1_phase_composition.csv`
- `test_cases/example1_phase_composition.png`

The CSV contains hard-label counts/percentages and mean class probabilities
for every frame. Interior particles use the exact training feature
construction. Coordination-defined gas uses the training gas convention.
Dense surface particles are included so percentages cover all particles; they
retain cheap local features and use the established zero-BOP outside-interior
convention, so surface-resolved fractions should be interpreted with more
caution than bulk-interior predictions.

**Correct splits use whole physical configurations/trajectories:**
- Assign each `source_file` or canonical physical trajectory to exactly one of
  train, validation, or test.
- Keep every frame from that configuration in the same split.
- Never train on early frames and test on late frames from the same crystal;
  those are thermal copies, not independent generalization examples.

The CSV includes `source_file`, `frame_index`, and `particle_index` so rows can
be joined by stable keys. `frame_index` is not permission to frame-split one
physical trajectory across train and test.

---

## Data Analysis & Findings

### Coordination feature separation (strong classifier signal)

| Class | Median coord | Mean coord | frac_unlike |
|-------|-------------|-----------|-------------|
| gas | 1.0 | 1.07 | 0.42 (noisy — isolated particles) |
| amorphous | 11.0 | 10.09 | ~0.63 |
| Th3P4-like | 11.0 | 11.54 | ~0.60 |
| CsCl-like | 14.0 | 13.94 | ~0.57 |

Gas is trivially separable (coord ≤ 3). CsCl vs Th3P4 vs amorphous are well-separated in (coord, frac_unlike) space, giving a strong baseline classifier.

### What worked well

- **Ground truth files** are clean: CsCl.gsd (501 frames, N=500) and th3p4.gsd (501 frames, N=512) produce stable interior-particle yields across all 76 usable frames (last 15%). The single-core gate never triggered, confirming these are pure single-crystal trajectories.

- **Unknown examples (post-redownload)** are now fully readable (50 MB each, 2001 frames, N=1024). Multi-frame sampling gives 61 frames × 6 amorphous files ≈ 340K amorphous particles — the amorphous class is richly represented.

- **Trajectories 8 & 9** are clearly CsCl-like. Their interior coordination (mean=14.0, matching the CsCl reference exactly) and frac_unlike (0.57) are unambiguous. The inference is robust with a distance margin >150× to the next candidate.

- **Coordination distributions are well-separated:**

  | Class | Median coord | Mean coord | frac_unlike |
  |-------|-------------|-----------|-------------|
  | CsCl-like (interior) | 14.0 | 13.87 | ~0.57 |
  | Th3P4-like (interior) | 12.0 | 11.91 | ~0.60 |
  | amorphous (core) | 11.0 | 10.14 | ~0.63 |

  These features alone should give a strong baseline classifier.

### Issues & Warnings

#### ⚠️ Remaining class imbalance (amorphous dominates)
| Class | Count | Share |
|-------|-------|-------|
| amorphous | 607,933 | 64.7% |
| CsCl-like | 139,060 | 14.8% |
| Th3P4-like | 112,254 | 12.0% |
| gas | 79,770 | 8.5% |

Amorphous is still ~5× overrepresented relative to crystals. The XYZ source largely fixed the Th3P4 deficit (was 0.9%, now 12%). Mitigation: use class weights in the classifier; or subsample amorphous broadcast rows (they're the most abundant and noisiest-labeled).

#### ✓ Gas class is now fully populated (79,770 particles)
Gas particles come from 85 labeled XYZ snapshots in `labeled_data (1)/gas/`. They have coord ≤ 3 (mean=1.07) — trivially distinguishable from all other classes.

#### ⚠️ 193 nicoles_dataset trajectory files missing entirely
These files were never downloaded (`cont/trajectory_cont.gsd` continuation runs). Of the 486 manifest entries, 293 have 0-byte stubs and 193 are fully absent.

#### ⚠️ Coordination statistics alone cannot confirm crystallinity
Several unknown_examples trajectories (1, 5, 12) show mean interior coordination that matches CsCl-like or Th3P4-like reference profiles. This is **a density artifact, not a crystal signature**: a dense amorphous blob at the same volume fraction as a crystal produces the same mean coordination number. Confirmed visually in OVITO — only trajectories 8 and 9 show long-range crystalline order; the rest are amorphous blobs.

The script correctly gates crystal-type inference behind `UNKNOWN_CRYSTAL_NUMS`, so amorphous trajectories are never passed through the inference step regardless of their coordination statistics. To properly identify crystallinity, use bond-order parameters (Steinhardt Q4/Q6) or OVITO's polyhedral template matching — not mean coordination alone.

#### ✓ No multi-blob files detected
All processed crystal files passed the single-core gate. No files were skipped for multi-blob reasons.

---

## Running the Script

```bash
# Activate the colloids conda environment (requires gsd, scipy, pandas, numpy)
conda activate colloids

# Default run (outputs training_set.csv in the same directory)
python build_training_set.py

# Custom options
python build_training_set.py \
    --frame-step 10          \  # sparser frame sampling (faster, fewer rows)
    --bulk-percentile 75     \  # slightly looser interior definition
    --surface-margin 1       \  # keep particles 1 below bulk_ref too
    --warn-frac 0.20         \  # stricter multi-blob gate
    --out my_training_set.csv
```

### Key parameters

| Flag | Default | Effect |
|------|---------|--------|
| `--cutoff-factor` | 1.35 | First-shell cutoff multiplier on median NN distance |
| `--gas-coord` | 4 | Particles below this coordination are gas |
| `--bulk-percentile` | 80 | Interior threshold: 80th percentile of core coordination |
| `--surface-margin` | 0 | Relax interior threshold by this many coordination units |
| `--frame-frac-start` | 0.85 | Use frames from this fraction of trajectory length onward |
| `--frame-step` | 5 | Sample every N-th saved frame in the usable window |
| `--warn-frac` | 0.25 | Multi-blob skip threshold (2nd cluster / 1st cluster) |

---

## Dependencies

```
gsd >= 5.0     # GSD trajectory I/O (conda: gsd)
numpy
scipy
pandas
```

The `gsd` package is only available in the `colloids` conda environment on this machine.

---

## GSD-Only Rebuild and Descriptor Benchmark (2026-06-11)

This section records the current pipeline and supersedes the earlier XYZ-inclusive,
within-trajectory split recommendations above. The older notes are retained as
historical context.

### Problem framing

The model predicts one particle-level phase label: `gas`, `amorphous`,
`Th3P4-like`, or `CsCl-like`. Particle predictions can be aggregated within a
frame to estimate the phase composition used in a phase diagram. Crystal/blob
confusion is more costly than gas/interior confusion because it moves a state
point across a phase boundary; therefore macro F1 and per-class recall are
reported instead of accuracy alone.

### New full CSV

`build_dataset.py` was rerun from scratch and produced:

- `labeled_dataset.csv`: 3,850,395 rows, 961 MB, GSD sources only.
- `frame_stats.csv`: per-frame labeling and q6 audit.
- No XYZ snapshots are included.
- Coordinates are retained for descriptor recomputation but are never model
  features.

| Label | Particles |
|---|---:|
| gas | 1,148,900 |
| amorphous | 840,649 |
| Th3P4-like | 941,223 |
| CsCl-like | 919,623 |
| **Total** | **3,850,395** |

### Complete Nicole Th3P4 retention

The manifest contains 136 Nicole Th3P4 GSD files. All 136 are present and
non-empty. The old q6 firewall caused only a small subset to contribute
Th3P4-labeled particles. The new rule is:

> For a Nicole trajectory labeled `Th3P4-like`, retain every eligible interior
> particle as `Th3P4-like`. Do not drop or relabel it using q6. Let the model
> learn how much weight to give q6, SOAP, entropy, and the other descriptors.

The rebuild produced 888,192 broadcast Th3P4 rows from **136/136 files**:

| Window | Files represented | Frames | Th3P4 particles | Mean q6_avg |
|---|---:|---:|---:|---:|
| early | 136 | 544 | 240,104 | 0.327 |
| growth | 136 | 952 | 368,628 | 0.174 |
| tail | 136 | 680 | 279,460 | 0.161 |

No Th3P4 particles were discarded as ambiguous and none were relabeled
amorphous. `build_dataset.py` now raises an error unless every readable
manifest Th3P4 trajectory contributes particles.

### Leakage-safe split

`benchmark_descriptors.py` creates a canonical trajectory ID before splitting.
An original run and its `cont/trajectory_cont.gsd` continuation share one ID,
so they can never land in different splits. Ground-truth and unknown-example
GSD files are also assigned whole. No trajectory is split by frame or particle.

The rebuilt data contains 303 canonical trajectories. Assignment is
deterministic (`seed=42`) and stratified by each trajectory's dominant label.
Descriptor computation uses only the current frame; no future frame, final
state, source filename, coordinates, or split identity is passed to a model.

For a tractable like-for-like SOAP comparison, the benchmark uses 5,000
particles per class per split (60,000 total). Sampling is trajectory-aware:
all 97 canonical trajectories containing Nicole/reference Th3P4 particles
remain represented. The complete 3.85-million-row CSV remains available for
final non-SOAP training.

### Descriptor definitions

- **Cheap:** particle type, coordination, like/unlike neighbor counts,
  unlike fraction, neighbor-distance mean/std, and coordination relative to
  the frame bulk reference.
- **Lechner-Dellago:** averaged `q4`, `q6`, and `q8` already computed by
  `build_dataset.py`.
- **SOAP:** species-resolved local SOAP, `n_max=4`, `l_max=4`, 180 features.
  Each frame is scaled by its median nearest-neighbor distance before SOAP so
  simulations with different coordinate scales are comparable.
- **Entropy fingerprint:** Shannon entropy of normalized radial-shell
  occupancy, species-mixing entropy, effective occupied radial bins, and local
  neighbor count.

All models are class-balanced XGBoost classifiers evaluated on exactly the
same held-out canonical trajectories.

### Held-out test results

| Features | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| cheap | 0.9042 | 0.9042 | **0.9041** |
| cheap + LD | **0.9043** | **0.9043** | 0.9036 |
| cheap + SOAP | 0.9027 | 0.9027 | 0.9027 |
| cheap + entropy | 0.9020 | 0.9020 | 0.9021 |
| cheap + LD + SOAP + entropy | 0.9019 | 0.9019 | 0.9012 |

Per-class test F1:

| Features | gas | amorphous | Th3P4-like | CsCl-like |
|---|---:|---:|---:|---:|
| cheap | 0.998 | **0.809** | **0.843** | 0.966 |
| cheap + LD | **0.999** | 0.808 | 0.842 | 0.966 |
| cheap + SOAP | 0.998 | 0.805 | 0.841 | **0.967** |
| cheap + entropy | 0.998 | 0.805 | 0.840 | 0.965 |
| combined | 0.998 | 0.800 | 0.839 | **0.968** |

![Descriptor comparison](descriptor_benchmark/descriptor_comparison.png)

Confusion matrices:
[cheap](descriptor_benchmark/confusion_cheap.png),
[LD](descriptor_benchmark/confusion_ld.png),
[SOAP](descriptor_benchmark/confusion_soap.png),
[entropy](descriptor_benchmark/confusion_entropy.png), and
[combined](descriptor_benchmark/confusion_combined.png).

### Interpretation

The richer descriptors did not improve macro F1 on this benchmark. The main
remaining error is amorphous versus Th3P4, while gas and CsCl are already easy.
This result does **not** show that SOAP or entropy are generally useless. It
shows that, with the current labels, sample size, and XGBoost settings, the
cheap local density/composition features already capture most of the available
signal and the added dimensions slightly increase variance.

There is also an important label-quality caveat: all Nicole Th3P4 labels are
now intentionally trusted at the trajectory level, including low-q6 growth and
tail frames. That is required to avoid defining Th3P4 circularly from q6, but
it means mislabeled or phase-mixed trajectories can limit every descriptor.
Manual review of the largest amorphous/Th3P4 confusion sources is the
highest-value next data-quality step.

### Reproduction

```bash
conda activate colloids
cd /Users/stevenzhang/MLprojects/preprocessing_pipeline_final

# Rebuild the complete GSD-only CSV and enforce 136/136 Th3P4 coverage.
python build_dataset.py

# Reproduce split audits, descriptors, models, metrics, and plots.
python benchmark_descriptors.py --force
```

Detailed machine-readable results are in
`descriptor_benchmark/metrics.json`; the exact benchmark particles and split
audits are in `descriptor_benchmark/benchmark_rows.csv` and
`descriptor_benchmark/split_audit_*.csv`.

### Deployment and monitoring

For phase-diagram inference, aggregate particle probabilities by frame and
report both the dominant phase and class fractions. Monitor descriptor drift,
class-probability entropy, and the fraction of particles outside the training
distribution. Retraining should require newly reviewed whole trajectories and
must preserve canonical trajectory-level splitting.

---

## Orthogonal Descriptor Full-Data Experiment (2026-06-11)

This section appends a second experiment to the GSD-only benchmark above. It
tests the requested core geometry, thermodynamic entropy, and coordination
context on **every row** of the rebuilt dataset, not only the 60,000-row SOAP
comparison sample.

### Research basis

- Lechner and Dellago, *Accurate determination of crystal structures based on
  averaged local bond order parameters*:
  [arXiv:0806.3345](https://arxiv.org/abs/0806.3345).
  The complex `q_lm` vectors are averaged over a particle and its first-shell
  neighbors before constructing rotational invariants.
- Piaggi and Parrinello, *Entropy based fingerprint for local crystalline
  order*: [arXiv:1707.09892](https://arxiv.org/abs/1707.09892).
  The fingerprint projects the two-body excess-entropy expression onto each
  particle using a smoothed local radial distribution function.

### Descriptor implementation

`build_dataset.py` now writes these columns for every retained CSV row:

| Family | Columns | Implementation |
|---|---|---|
| Core geometry | `q4`, `q6_avg`, `w4_avg`, `w6_avg` | Lechner-Dellago neighbor averaging; normalized third-order `w_l` invariants from `freud.order.Steinhardt` |
| Thermodynamic | `pp_entropy`, `pp_entropy_avg` | Local two-body entropy from a Gaussian-smoothed RDF, plus first-shell neighbor averaging |
| Context | `coord`, `coord_minus_bulkref` | First-shell coordination and offset from the frame's 80th-percentile bulk coordination |

The local entropy is evaluated as

```text
s_i = -2 pi rho integral [g_i(r) ln g_i(r) - g_i(r) + 1] r^2 dr
```

with distances divided by each frame's median nearest-neighbor distance. The
numerical settings are `r_max=2.5`, 48 radial bins, and Gaussian width `0.08`
in these normalized units. This removes arbitrary coordinate-scale differences
between simulations. `pp_entropy_avg` averages `s_i` over the particle and its
first-shell neighbors.

The regenerated `labeled_dataset.csv` has:

- 3,850,395 rows and 25 columns (about 1.1 GB).
- Valid finite descriptor values in every row.
- 888,192 broadcast Th3P4 rows from 136/136 Nicole files.
- No XYZ data.

### Full-data evaluation

All models use the same canonical trajectory split as the previous section:

| Split | Particles | Canonical trajectories |
|---|---:|---:|
| train | 2,633,396 | 213 |
| validation | 622,643 | 45 |
| test | 594,356 | 45 |

The split audit confirms each of the 303 canonical trajectories occurs in
exactly one split.

Feature ablations:

| Model | Features | Macro F1 | Balanced accuracy |
|---|---|---:|---:|
| cheap baseline | existing 8 cheap features, including CN | 0.8445 | 0.8403 |
| orthogonal only | q4, q6, w4, w6, entropy, CN | 0.8259 | 0.8249 |
| cheap + averaged q/w | cheap + q4, q6, w4, w6 | **0.8445** | 0.8400 |
| cheap + entropy | cheap + both entropy values | 0.8299 | 0.8308 |
| all orthogonal | cheap + q4, q6, w4, w6 + entropy | 0.8250 | 0.8261 |
| learned hierarchy, cheap | gas/condensed model, then dense-phase model | **0.8445** | **0.8404** |
| learned hierarchy, all orthogonal | same hierarchy with q/w and entropy | 0.8242 | 0.8232 |

The tiny macro-F1 difference between the cheap baseline (`0.844458`) and
cheap + q/w (`0.844473`) is not practically meaningful. The learned hierarchy
also ties rather than improves the baseline.

![Orthogonal descriptor comparison](orthogonal_benchmark/orthogonal_comparison.png)

### Per-class comparison

| Model | gas F1 | amorphous F1 | Th3P4 F1 | CsCl F1 |
|---|---:|---:|---:|---:|
| cheap baseline | 1.000 | **0.687** | **0.750** | 0.940 |
| cheap + q/w | 1.000 | 0.687 | 0.749 | **0.942** |
| cheap + entropy | 1.000 | 0.672 | 0.708 | 0.940 |
| all orthogonal | 1.000 | 0.663 | 0.696 | 0.942 |

The averaged geometry slightly improves CsCl precision/F1, but loses a similar
amount on amorphous and Th3P4. Entropy shifts predictions toward CsCl and
reduces Th3P4 recall. The remaining bottleneck is still the
amorphous-versus-Th3P4 boundary.

### Interpretation

These descriptors are not useless: XGBoost assigns substantial gain to
`q6_avg`, `q4`, `w4_avg`, `w6_avg`, and `pp_entropy`. However, their patterns
do not transfer better to unseen trajectories than the existing
composition/coordination features.

The likely limitation is label and domain heterogeneity:

- Nicole's full-trajectory Th3P4 labels intentionally include low-order growth
  and tail frames (`mean q6_avg` about 0.17), so many labeled Th3P4 particles
  geometrically overlap the amorphous class.
- Entire files labeled gas can contain locally coordinated particles. A hard
  `coord < 4` gas rule was tested and rejected because it contradicts these
  labels; the reported hierarchy therefore learns gas versus condensed instead.
- The entropy fingerprint is principally a solid-versus-liquid signal. Here
  gas is already perfectly separable and the difficult pair is two dense,
  overlapping classes, limiting its incremental value.

The result therefore does not support deploying the larger feature set yet.
The cheap model remains the best generalizing model, while the enriched CSV is
valuable for auditing mislabeled or phase-mixed trajectories. The most useful
next experiment is trajectory-level label review or frame/particle relabeling
near the amorphous-Th3P4 boundary, followed by this same frozen benchmark.

### Outputs and reproduction

```bash
conda activate colloids
cd /Users/stevenzhang/MLprojects/preprocessing_pipeline_final

# Recalculate qbar/wbar, Piaggi-Parrinello entropy, CN, and all existing
# descriptors for every CSV row.
python build_dataset.py

# Train flat and learned-hierarchy ablations on all 3,850,395 rows.
python benchmark_orthogonal.py
```

Results, split audit, models, and confusion matrices are in
`orthogonal_benchmark/`. Machine-readable metrics are in
`orthogonal_benchmark/metrics.json`.

---

## Phase Label Audit: Clean vs Mixed vs Uncertain Trajectories

### Motivation

The descriptor benchmarks showed that richer local descriptors did not beat
the cheap coordination/composition baseline. This points to label and domain
heterogeneity rather than descriptor shortage, especially among Nicole
trajectories globally labeled `Th3P4-like`.

> Particles are not independent scientific samples. Frames and canonical
> trajectories are the units used for this audit. Original and continuation
> GSD files from the same simulation are assigned one canonical trajectory.

### Data integrity fix

The audit first exposed 161,792 duplicate particle keys in the old CSV. In
all cases, a coordination-1 particle had been emitted once with a phase label
and again as gas. The cause was an all-gas frame with `bulk_thresh=0`, which
allowed `interior_mask` and `gas_mask` to overlap.

`build_dataset.py` now defines:

```text
dense_mask    = coord >= GAS_COORD
gas_mask      = coord < GAS_COORD
interior_mask = dense_mask & (coord >= bulk_thresh)
```

It asserts interior/gas disjointness for every frame, checks per-frame emitted
particle IDs, writes to a temporary CSV, performs a global duplicate-key
check, and only then replaces `labeled_dataset.csv`.

The corrected rebuild contains:

| Integrity measure | Result |
|---|---:|
| Unique particle rows | 3,613,946 |
| Duplicate `(source_file, frame_index, particle_index)` keys | **0** |
| Nicole Th3P4 files represented | **136/136** |
| Broadcast Th3P4 rows | 845,010 |

This was a material correction, not only row deduplication: the overlapping
particles had also affected order parameters and CsCl firewall decisions in
all-gas frames. The corrected 60,000-row benchmark is:

| Features | Macro F1 | Balanced accuracy |
|---|---:|---:|
| cheap | **0.9191** | **0.9194** |
| cheap + LD | 0.9149 | 0.9155 |
| cheap + SOAP | 0.9123 | 0.9128 |
| cheap + entropy | 0.9170 | 0.9173 |
| combined | 0.9114 | 0.9121 |

The scientific conclusion is unchanged: richer descriptors still do not beat
the cheap baseline on unseen trajectories.

### Audit method

`audit_phase_labels.py` summarizes dense particles (`coord >= 4`) within every
frame using coordination, averaged q4/q6/w4/w6, and Piaggi-Parrinello entropy.
Gas is retained as `gas_fraction` but does not dilute dense structural
statistics. Trajectories are ranked with configurable, interpretable q6 rules:

- `clean_crystal`: at least 50% of frames have mean q6 >= 0.35, or at least
  30% have mean q6 >= 0.40.
- `amorphous_like`: median frame mean q6 < 0.20 and fewer than 10% of frames
  have mean q6 >= 0.35.
- `mixed`: substantial low-q6 and high-q6 frame fractions.
- `gas_like`: gas-majority trajectory with high mean gas fraction.
- `uncertain`: none of the above.

Nothing is silently dropped. `frame_audit.csv` and `trajectory_audit.csv`
retain all audited units and the reason for every status.

### Status summary

| Majority label | Clean crystal | Amorphous-like | Uncertain | Gas-like |
|---|---:|---:|---:|---:|
| Th3P4-like | **3** | **77** | **17** | 0 |
| amorphous | 4 | 2 | 106 | 0 |
| CsCl-like | 70 | 0 | 1 | 0 |
| gas | 0 | 0 | 0 | 23 |

Particle counts by audited trajectory status:

| Status | Particles |
|---|---:|
| clean_crystal | 1,119,380 |
| amorphous_like | 1,247,802 |
| uncertain | 949,265 |
| gas_like | 297,499 |

The central finding is severe: only 3 of 97 canonical Th3P4 trajectories meet
the clean-crystal rule, while 77 look amorphous-like.

![Trajectory q6 distributions](phase_label_audit/plots/trajectory_q6_histogram.png)
![Th3P4 status counts](phase_label_audit/plots/th3p4_status_counts.png)
![q6 versus entropy](phase_label_audit/plots/q6_vs_entropy.png)

### Cleanest Th3P4 trajectories

| Trajectory | Status | Mean frame q6 | Frames q6>=0.35 | Clean score | Amorphous score |
|---|---|---:|---:|---:|---:|
| `th3p4.gsd` | clean_crystal | 0.499 | 1.000 | 1.000 | 0.000 |
| `debye5.45_volumefraction.005575710117_seed1` | clean_crystal | 0.471 | 0.812 | 1.000 | 0.000 |
| `debye5.4_volumefraction.005575710117_seed1` | clean_crystal | 0.474 | 1.000 | 1.000 | 0.000 |
| `debye5.55_volumefraction.005575710117_seed1` | uncertain | 0.258 | 0.000 | 0.087 | 0.343 |
| `debye5.35_volumefraction.004294716531_seed3` | uncertain | 0.248 | 0.000 | 0.072 | 0.378 |
| `debye5.25_volumefraction.024130533489_seed2` | uncertain | 0.244 | 0.000 | 0.066 | 0.401 |
| `debye5.3_volumefraction.004294716531_seed4` | uncertain | 0.236 | 0.000 | 0.055 | 0.421 |
| `debye5.3_volumefraction.004294716531_seed3` | uncertain | 0.235 | 0.000 | 0.052 | 0.431 |
| `debye5.35_volumefraction.004294716531_seed2` | uncertain | 0.233 | 0.000 | 0.049 | 0.422 |
| `debye5.3_volumefraction.004294716531_seed1` | uncertain | 0.232 | 0.000 | 0.048 | 0.438 |

### Most amorphous-like Th3P4 trajectories

| Trajectory | Status | Mean frame q6 | Frames q6>=0.35 | Clean score | Amorphous score |
|---|---|---:|---:|---:|---:|
| `debye5.15_volumefraction.010890058823_seed2` | amorphous_like | 0.148 | 0.000 | 0.000 | 1.000 |
| `debye5.25_volumefraction.007274079383_seed1` | amorphous_like | 0.169 | 0.000 | 0.000 | 1.000 |
| `debye5.2_volumefraction.008655804247_seed1` | amorphous_like | 0.147 | 0.000 | 0.000 | 1.000 |
| `debye5.3_volumefraction.024130533489_seed1` | amorphous_like | 0.174 | 0.000 | 0.000 | 1.000 |
| `debye5.15_volumefraction.010890058823_seed1` | amorphous_like | 0.157 | 0.000 | 0.000 | 1.000 |
| `debye5.15_volumefraction.010890058823_seed4` | amorphous_like | 0.163 | 0.000 | 0.000 | 1.000 |
| `debye5.25_volumefraction.010890058823_seed3` | amorphous_like | 0.150 | 0.000 | 0.000 | 1.000 |
| `debye5.35_volumefraction.002704544521_seed1` | amorphous_like | 0.153 | 0.000 | 0.000 | 1.000 |
| `debye5.15_volumefraction.014494668294_seed2` | amorphous_like | 0.150 | 0.000 | 0.000 | 1.000 |
| `debye5.2_volumefraction.008655804247_seed2` | amorphous_like | 0.183 | 0.000 | 0.000 | 1.000 |

### Most crystal-like amorphous trajectories

| Trajectory | Status | Mean frame q6 | Frames q6>=0.35 | Clean score | Amorphous score |
|---|---|---:|---:|---:|---:|
| `trajectory (12).gsd` | clean_crystal | 0.542 | 0.985 | 1.000 | 0.000 |
| `debye5.5_volumefraction.010890058823_seed2` | clean_crystal | 0.387 | 0.500 | 1.000 | 0.000 |
| `debye5.15_volumefraction.008655804247_seed2` | clean_crystal | 0.412 | 0.714 | 1.000 | 0.000 |
| `debye5.55_volumefraction.010890058823_seed2` | clean_crystal | 0.405 | 1.000 | 1.000 | 0.000 |
| `debye5.5_volumefraction.024130533489_seed1` | uncertain | 0.350 | 0.429 | 0.848 | 0.075 |
| `debye5.55_volumefraction.012521039450_seed2` | uncertain | 0.339 | 0.429 | 0.833 | 0.154 |
| `debye5.4_volumefraction.012521039450_seed3` | uncertain | 0.289 | 0.143 | 0.381 | 0.327 |
| `debye5.55_volumefraction.015943113378_seed1` | uncertain | 0.324 | 0.000 | 0.185 | 0.096 |
| `debye5.5_volumefraction.014494668294_seed4` | uncertain | 0.297 | 0.000 | 0.146 | 0.185 |
| `debye5.55_volumefraction.012521039450_seed1` | uncertain | 0.296 | 0.000 | 0.143 | 0.198 |

`trajectory (12).gsd` is the strongest immediate manual-review candidate: it
is labeled amorphous but has a mean frame q6 of 0.542 and 98.5% crystalline
frames under the audit rule.

### Audited dataset variants

`make_audit_training_sets.py` samples trajectories first, then frames, then
particles. Every canonical trajectory is assigned wholly to one split.

| Variant | Sampled rows | Canonical trajectories | Intended use |
|---|---:|---:|---|
| `strict_clean_dataset.csv` | 510,600 | 97 | Crystal recognition using only rows matching clean trajectory labels |
| `transition_holdout_dataset.csv` | 415,912 | 125 | Boundary evaluation; do not use for initial training |
| `noisy_label_dataset.csv` | 600,000 | 303 | Reproduce learning from original Nicole labels with audit metadata |

All three outputs have zero duplicate keys, zero train/validation/test
trajectory overlap, and no frame exceeds the `--max-per-frame=2000` cap.

Important independence warnings:

- Strict Th3P4 has 60,600 available particles but only **3 trajectories**.
- Strict amorphous has 153,133 available particles but only **2 trajectories**;
  both are assigned to train because a meaningful three-way split is impossible.

These warnings are why particle counts alone must not be interpreted as sample
size.

Recommended datasets:

1. **Strict clean-only crystal-recognition dataset** for learning canonical
   crystal structure. Its Th3P4 test should be treated as a three-trajectory
   case study, not a stable population estimate.
2. **Mixed/uncertain transition dataset** as a frozen holdout for interface and
   growth-boundary behavior.
3. **Noisy Nicole-label reproduction dataset** for comparison with historical
   results and robustness methods.

### Audit outputs and reproduction

```bash
conda activate colloids
cd /Users/stevenzhang/MLprojects/preprocessing_pipeline_final

# Rebuild with disjoint masks and global duplicate-key assertion.
python build_dataset.py

# Rank every frame and canonical trajectory without dropping data.
python audit_phase_labels.py

# Create strict, transition, and noisy-label trajectory-safe variants.
python make_audit_training_sets.py
```

Outputs are under `phase_label_audit/`:

- `frame_audit.csv`
- `trajectory_audit.csv`
- `th3p4_ranked_trajectories.csv`
- `amorphous_ranked_trajectories.csv`
- `audit_summary.json`
- `*_dataset.csv` and `*_sampling_report.csv`
- diagnostic plots under `phase_label_audit/plots/`

---

## Semi-Supervised Structure Discovery (label-free clustering) — 2026-06-13

A **new, separate module** under `discovery/` that does NOT touch the supervised
pipeline, `labeled_dataset.csv`, or the trained models. It asks whether amorphous
and Th₃P₄ can be told apart *without* using Nicole's labels, by clustering on
independent structural descriptors plus a new **temporal-persistence** axis, then
naming clusters with clean reference anchors only afterward.

### Method (and the scientific firewalls, enforced in code)

The unit of analysis is a **particle track** `(canonical_trajectory, particle_index)`
followed across its sampled frames — never a single frame, never frame-split. All
joins are by `(source_file, frame_index, particle_index)`; nothing positional.

- **Step 1 — assemble (label-free).** Pool real amorphous, Nicole Th₃P₄, real
  CsCl, gas, and the clean ground-truth CsCl/Th₃P₄ crystals (anchors). Whole tracks
  sampled trajectory-aware (≤150 tracks/trajectory; long reference/gas trajectories
  time-strided). Labels kept as **provenance only**.

  | role | label | #traj | #tracks | #frames |
  |---|---|--:|--:|--:|
  | ref_cscl (anchor) | CsCl-like | 1 | 150 | 8,638 |
  | ref_th3p4 (anchor) | Th3P4-like | 1 | 150 | 7,870 |
  | real_cscl | CsCl-like | 67 | 10,050 | 207,420 |
  | nicole_th3p4 | Th3P4-like | 89 | 13,350 | 178,903 |
  | real_amorphous | amorphous | 114 | 17,100 | 133,265 |
  | gas | gas | 31 | 4,650 | 76,904 |
  | **TOTAL** | | **303** | **45,450** | **613,000** |

- **Step 2 — independent descriptor allowlist (13).** `coord, n_like, n_unlike,
  frac_unlike, d_mean, d_std, q4, q6_avg, w4_avg, w6_avg, q8, pp_entropy,
  pp_entropy_avg` (time-mean per track). A **fail-fast assertion** forbids any
  identifier, split, label, supervised prediction, **RDF**, or **template/motif**
  column from entering the matrix. RDF is excluded because Nicole's labels were
  *derived* from RDF — clustering on it would rediscover her labels (and errors).

- **Step 3 — temporal persistence (4, the new axis).** Per track: `q6_time_std`
  (flicker), `q6_slope` (growth trend), `q6_frac_above` (fraction of frames with
  q6≥0.35), `coord_time_std` (coordination volatility). Intuition: growing Th₃P₄
  should show order that *persists/grows*; thermal-fluctuation amorphous should
  *flicker*. Distributions by role:

  | feature | ref_th3p4 | nicole_th3p4 | real_amorphous | ref_cscl | gas |
  |---|--:|--:|--:|--:|--:|
  | q6_time_mean | 0.376 | **0.123** | 0.205 | 0.556 | 0.071 |
  | q6_frac_above | 0.753 | **0.039** | 0.098 | 0.967 | 0.084 |
  | coord_time_std | 0.264 | **3.424** | 1.624 | 0.007 | 1.752 |

  **The clean Th₃P₄ reference has persistent order (frac_above 0.75); Nicole's
  Th₃P₄-labeled particles do not (0.039) — they are *more* disordered than the
  amorphous class** (q6_time_mean 0.123 < 0.205, Cohen d = −0.93 in the wrong
  direction) and the highest-volatility population (coord_time_std 3.42).

- **Step 4 — clustering: UMAP+HDBSCAN primary, PCA+GMM cross-check.** Standardized
  17-feature matrix. HDBSCAN keeps a noise/**boundary** class for ambiguous
  transition particles. GMM component count by BIC (k=8), KMeans fallback. Run on
  both the instant-only (13) and instant+persistence (17) feature sets.

- **Step 5 — name by anchors + novelty.** Cluster centroid distance (standardized
  feature space) to the CsCl/Th₃P₄/amorphous/gas anchor centroids; nearest anchor
  names the cluster; `novelty = nearest-anchor distance / mean inter-anchor
  distance` (≈1 ⇒ as far from any reference as references are from each other).

- **Step 6 — POST-HOC RDF audit (annotation only).** *After* clustering, a
  normalized total RDF fingerprint (median-NN-distance units, freud) is recomputed
  per cluster and cosine-compared to the clean reference RDFs. Kept strictly
  separate from the discovery result. The excluded `motif_recurrence` template
  signal is also reported here.

### Step 7 — the diagnostic answers (stated plainly)

**Q1 — Do amorphous and Th₃P₄ separate with independent descriptors + temporal
persistence, where single frames don't?** *Partially, and the two methods agree —
but not in the way the hypothesis predicted.* Amorphous-vs-Nicole-Th₃P₄ balanced
accuracy (evaluation only, cluster-majority vote): UMAP+HDBSCAN **0.76**, PCA+GMM
**0.74** with persistence (0.81 / 0.85 with single-frame features). The separation
**replicates across the nonlinear and linear methods and in physical space**
(amorphous clusters: q6 0.217, coord-volatility 0.51; Th₃P₄ clusters: q6 0.130,
coord-volatility **4.44**), so it is not a UMAP artifact. **Adding persistence did
not increase the amorphous/Th₃P₄ separation** (it slightly traded it) — its real
contribution was to raise **method agreement ARI 0.43 → 0.68** and to reveal that
Nicole-Th₃P₄'s signature is *temporal instability* (flickering coordination), not
growing crystalline order. The separation is therefore "two kinds of disorder,"
**not** "amorphous vs crystal."

**Q2 — Where do Nicole-Th₃P₄ tracks land?** Overwhelmingly in disordered clusters:
**59.8 % amorphous-named, 36.9 % gas-named, only 1.7 % in the clean Th₃P₄-reference
cluster** (1.6 % boundary). This independently reconfirms the ~91 % particle-level
contamination of Nicole's Th₃P₄ labels — from clustering that never saw a label or
a q6 threshold. There is also a distinct **dense-but-disordered Nicole-Th₃P₄ pocket**
(cluster 5: 97.6 % Nicole-Th₃P₄, coord 11.6, q6 0.12, but unusually high
`frac_unlike` 0.69) — compositional alternation without orientational order, the
best candidate for "incipient" Th₃P₄ and a high-value manual-review target.

**Q3 — Internally-consistent clusters far from all references (novel candidates)?**

| cluster | size | novelty | ordered? (q6 / frac_above / motif) | dominant | %Nicole-Th₃P₄ | reading |
|--:|--:|--:|---|---|--:|---|
| **10** | 526 | **1.38** | 0.47 / 0.77 / 0.21 (yes) | nicole_th3p4 | 42 % | **genuine candidate**: ordered yet far from *both* GT references; low-φ debye5.4–5.55 trajectories |
| 9 | 978 | 0.74 | 0.48 / 0.90 / 0.38 (yes) | real_cscl | 0.5 % | ordered CsCl from `trajectory (8/9/12).gsd` |
| 11/12/13 | 7,270 | 0.69–0.72 | 0.49–0.57 / 0.88–0.999 / 0.33–0.43 (yes) | real_cscl | ~0 % | **reference gap, NOT novel** |

**Important caveat (no false discovery):** clusters 9/11/12/13 are highly ordered
(high q6, high `motif_recurrence`) and ~100 % *real CsCl*, yet sit far from the
GT-CsCl anchor. This is a **reference-domain gap** — the ground-truth CsCl crystal
does not represent Nicole's CsCl trajectories (a direct parallel to the known
synthetic-Th₃P₄→real-Th₃P₄ gap) — so their "novelty" reflects inadequate anchors,
not a new phase. Only **cluster 10** is a true novel/intermediate-structure
candidate: internally ordered, compact, far from every clean reference, and built
largely from Nicole-Th₃P₄ trajectories.

### Bottom line

Independent descriptors + temporal persistence **cannot rescue Nicole's Th₃P₄ as a
crystal** — label-free clustering sends 97 % of it into disordered/gas territory —
but they **do** resolve it as its own *unstable, compositionally-ordered* flavor of
disorder distinct from real amorphous, and they surface one ordered structure
(cluster 10) that matches **neither** clean reference. Temporal persistence's value
here is diagnostic (it stabilizes the clustering and explains *why* Nicole-Th₃P₄ is
mislabeled), not a new separating descriptor.

### Outputs and reproduction

```bash
conda activate colloids   # has umap-learn + hdbscan installed for this module
cd preprocessing_pipeline_final
python discovery/step1_assemble.py        # label-free analysis set
python discovery/step23_features.py       # allowlist + persistence matrix
python discovery/step4_cluster.py         # UMAP+HDBSCAN primary, PCA+GMM cross-check
python discovery/step5_name_novelty.py    # anchor naming + novelty
python discovery/step6_rdf_audit.py       # POST-HOC RDF audit (annotation only)
python discovery/step7_summary.py         # diagnostic answers + discovery_summary.json
```

Artifacts in `discovery/`: `discovery_table.csv` (per-track provenance + cluster +
reference distances + novelty + post-hoc RDF), `cluster_scatter.png` (UMAP+HDBSCAN
vs PCA+GMM, by cluster and by true role), `persistence_comparison.png`,
`cluster_physical_profile.csv`, `cluster_naming.csv`, `rdf_audit.csv`, and
`discovery_summary.json` (machine-readable Step-7 answers).

---

## 2026-06-15 — Vector q_lm neighbor-coherence descriptor (ten Wolde–Frenkel)

All prior tests used **scalar** q6 (a per-particle magnitude: "how much order"). This
analysis adds a distinct axis: keep the full complex **q_lm vector** per particle and
measure whether *neighboring* cages point the same way (orientational coherence), which
scalar q6 cannot see. Per interior particle, for l=4,6,8, we Lechner-Dellago-average the
complex q_lm vector and compute the normalized vector correlation
`S_ij = Re[Σ_m q̄_lm(i)·q̄_lm*(j)] / (|q̄_l(i)||q̄_l(j)|)` against each first-shell
neighbor, split same-type vs unlike-type (Th₃P₄ has inequivalent sublattice sites).
Same interior-only neighbor list as the rest of the pipeline (1.35×median NN, 80th-pct
coordination filter); joined by `(source_file, frame_index, particle_index)`.

Implementation: `coherence_descriptor.py` (validated: reproduces pipeline scalar q6 to
1e-7; CsCl=0.999, th3p4=0.988). Cache `coherence_features.pkl` (2.66 M interior rows over
7789 frames). Steps: `coh_step2_validate.py`, `coh_step3_clusters.py`,
`coh_step4_classify.py`. Reference scope per constraint: only `CsCl.gsd`, `th3p4.gsd`
(and disordered roles) are treated as clean — the generated `th3p4_{medium,small,
perfect_reference,slab,rod}` crystals are excluded (and are not in `labeled_dataset.csv`).

### (a) Does vector coherence separate clean crystal from amorphous? — **YES, valid.**

| group (interior particle-rows) | scalar q6 | **q6_corr_mean** | frac_S>0.5 | same-type | unlike-type |
|---|---|---|---|---|---|
| clean Th₃P₄ (`th3p4.gsd`) | 0.498 | **0.988 ± 0.005** | 1.000 | 0.987 | 0.989 |
| clean CsCl (`CsCl.gsd`)   | 0.575 | **0.999 ± 0.002** | 1.000 | 0.998 | 0.999 |
| real amorphous (dense)    | 0.248 | **0.710 ± 0.131** | 0.857 | 0.679 | 0.738 |
| gas                       | 0.372 | **0.823 ± 0.196** | 0.892 | 0.810 | 0.837 |

Clean crystals (≈0.99) cleanly exceed dense amorphous (0.71); Δ/pooled-σ = 1.71. The
LD-averaged vector coherence has a high floor (~0.71 even for amorphous), but the
crystal/amorphous separation is unambiguous. Per-group corr(scalar q6, q6_corr_mean) is
0.71 in amorphous and 0.86 in gas — coherence is partly but not fully redundant with
scalar q6.

### (b) Does coherence reveal order in cluster 5 that scalar q6 missed? — **NO.**

| cluster (frame-instances) | scalar q6 | **q6_corr_mean** | same-type | unlike-type | vs amorphous |
|---|---|---|---|---|---|
| **cluster 5** (97.6% Nicole-Th₃P₄, frac_unlike 0.69) | 0.116 | **0.536 ± 0.102** | 0.597 | 0.516 | **Δ=−0.174, Δ/σ=−1.05** |
| cluster 10 (novelty 1.38) | 0.511 | **0.908 ± 0.102** | 0.894 | 0.891 | Δ=+0.198, Δ/σ=+1.19 |

Cluster 5's vector coherence (0.536) is **below** dense amorphous (0.710), not above it.
Only 1.6% of its particles exceed the amorphous p75; **0%** reach the clean-Th₃P₄ range.
The species-aware split does not rescue it — even same-type neighbor coherence (0.597) is
below amorphous. The stronger vector test **confirms cluster 5 is orientationally
disordered**: it has compositional alternation (high frac_unlike) but the cages do not
align. Scalar q6 was right to call it amorphous. (Cluster 10, by contrast, shows genuine
coherence 0.908 ≫ amorphous, consistent with its high scalar q6 and top novelty — a real
ordered structure matching neither clean reference, as the discovery module already found.)

We do **not** claim cluster 5 is crystalline; it does not exceed the amorphous
distribution — it sits below it.

### (c) Does coherence improve the real classifier? — **NO, it plateaus.**

Stage-1 XGBoost, same leakage-safe split (GT frames chronological; broadcast 80/20 by
dominant label), class-balanced. Baseline = CHEAP + scalar BOPs (q4, q6_avg, q8);
+coherence adds the 9 vector features. TEST set (780,653 particles):

| class | F1 base | F1 +coh | ΔF1 | recall base | recall +coh |
|---|---|---|---|---|---|
| gas        | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 |
| amorphous  | 0.718 | 0.718 | −0.000 | 0.604 | 0.604 |
| Th₃P₄-like | 0.355 | 0.355 | −0.000 | 0.999 | 0.998 |
| CsCl-like  | 0.777 | 0.777 | +0.000 | 0.840 | 0.840 |
| macro avg  | 0.713 | 0.713 | −0.000 | | |

The amorphous↔Th₃P₄ boundary is unchanged (amor→Th₃P₄ misclassifications 95,730 →
95,760). The model *does* use the coherence features (q6_corr_mean importance 0.18,
unlike_type 0.14) but they are redundant with the existing cheap+BOP features and move no
class boundary. This is consistent with prior benchmarks where BOPs and SOAP also failed
to separate Nicole-Th₃P₄ from amorphous — the limit is label heterogeneity, not the
descriptor.

### Bottom line

Vector coherence is a **valid, working descriptor** (clean crystals separate sharply from
amorphous) but it **does not rescue cluster 5** — under the stronger orientational test,
cluster 5 is *more* disordered than real amorphous, confirming scalar q6's call and the
discovery module's "unstable compositionally-ordered disorder" verdict. It also **does not
improve the supervised classifier**. A useful negative result: the orientational-coherence
axis carries no hidden Th₃P₄ crystallinity that scalar q6 was missing.

### Outputs and reproduction

```bash
conda activate colloids
cd preprocessing_pipeline_final
python coh_step0_cache.py        # compute & cache vector-coherence over all frames -> coherence_features.pkl
python coh_step2_validate.py     # reference-group distributions + corr vs scalar q6
python coh_step3_clusters.py     # cluster 5 / 10 coherence vs references (decisive test)
python coh_step4_classify.py     # baseline vs +coherence XGBoost (does not touch labeled_dataset.csv)
```

Artifacts: `coherence_features.pkl`, `coh_step2_joined.pkl`, `coh_step4_results.json`.
The supervised pipeline (`train_classifier.py`) and `labeled_dataset.csv` are unmodified.

---

## 2026-06-15 — Interpretable descriptor analysis + translational order + simulation spec

Goal: stop using opaque XGBoost and *understand* what separates the classes and where we
lack clean data, to target simulations precisely. Adds a physically new axis
(translational order), fits interpretable linear models, visualizes the feature space, and
turns the result into a simulation specification. Read-only; new folder `interpretability/`;
all on interior particles joined by `(source_file, frame_index, particle_index)`.

Descriptor set (interpretable; SOAP excluded — cache absent + 5000-dim, not interpretable):
cheap (8) + LD BOPs (6) + entropy (2) + motif (1) + q_lm coherence (9) + **translational (5,
new)** + temporal persistence (4) = 35 features over 2.66 M interior particle-rows.

### (a) Do translational-invariance features separate crystal from amorphous? — YES, but not better than q6.

New per-particle local structure factor / translational order parameter
(`interpretability/translational_descriptor.py`): `psi_i(k)=(1/deg)Σ_j exp(i k·r_ij)` over
neighbors out to 2.5×median-NN (2–3 shells, so it probes *periodicity*, not just the first
shell), scanned over |k|=m·2π/d_ref (m∈[0.7,1.9]) × 28 directions; max over (|k|,direction).
This is the one property the prior six descriptors never directly tested.

| group | scalar q6 | trans_order_excess | trans_Sk_max |
|---|---|---|---|
| clean Th₃P₄ | 0.498 | 0.605 | **25.3** |
| clean CsCl  | 0.575 | 0.604 | **29.1** |
| real amorphous | 0.248 | 0.296 | **6.9** |
| gas | 0.372 | 0.438 | 13.1 |

Translational order clearly separates crystal from amorphous (`trans_order_excess` Δ/σ=+2.06,
`trans_Sk_max` Th₃P₄-vs-amorphous Δ/σ=+1.78) — it is a real, valid axis. But it does **not
beat scalar q6** on the clean references (q6 Δ/σ=+2.99). It is a *distinct* physical axis, not
a stronger one.

### (b) What the linear models say distinguishes the classes (interpretable).

Lasso (L1) and Ridge (L2) multinomial logistic, class-balanced, leakage-safe split,
standardized features (`interpretability/step2_linear.py`).

**Linear ≈ XGBoost on the same features** — macro-F1 Lasso 0.459 / Ridge 0.459 / XGBoost 0.448;
amorphous-F1 0.718 vs 0.690; Th₃P₄-F1 0.342 vs 0.322. The interpretable linear model *matches
or beats* the tree model → **the problem is essentially linear and the plateau is real, not a
model-capacity limit.** (gas-F1≈0: interior gas is rare, 8951 rows.)

Binary amorphous-vs-Th₃P₄ (Lasso, standardized coef; + ⇒ Th₃P₄, − ⇒ amorphous): `coord` −4.19,
`n_unlike` −3.02, `coord_minus_bulkref` +2.79, **`trans_order_max` +2.33**, `frac_unlike` +2.15.
So the boundary is driven by **coordination + composition, with translational order the 4th-
strongest contributor (toward Th₃P₄)**. Tellingly, `q6_avg` points *away* from Th₃P₄ (−0.37):
Nicole-Th₃P₄ has low q6 — a direct fingerprint of the label heterogeneity. Despite all 35
features, Th₃P₄-F1 stays 0.35: the boundary is intrinsically fuzzy.

### (c) Feature space (PCA, interpretable loadings).

`interpretability/step3_pca_viz.py` (→ `step3_pca_class.png`). PC1 (33%) loads positively on
q6, coherence, *and* translational order together = an overall **"order" axis**. PC3 (8.8%) =
composition (`frac_unlike`, `n_unlike`). Class centroids on PC1: CsCl **+3.6** (cleanly
separated), amorphous −1.1, **Th₃P₄ −2.7 (lower order than amorphous)**, gas −1.9 (separated on
PC2 by low coordination). **Only CsCl separates on the order axis; amorphous and Th₃P₄ overlap
completely and Nicole-Th₃P₄ is the *least* ordered class.** Discovery **cluster 10** sits at
PC1=+2.4 (ordered, near CsCl's order level) but PC2=+4.6 (distinct from both CsCl and the
amorphous cloud) — a genuinely separate ordered structure, with high q6 (0.51) but only modest
translational order (`trans_Sk_max` 9.2).

### (d) Descriptor-space gaps → what to simulate (the payoff).

Crystal axis = `trans_Sk_max`; crystalline threshold = amorphous p95 = 15.15
(`interpretability/step4_gap.py`).

**Domain gap (Th₃P₄-labeled):** clean reference (`th3p4.gsd`) mean 25.1, 84% above threshold;
**Nicole-broadcast mean 8.2, only 0.9% above threshold** — 99% of Nicole's Th₃P₄ is below
crystalline translational order. The clean labeled examples live almost entirely in the
moderate/strong-crystal bins (from the two reference GSDs); the **weak-to-moderate crystal
regime (the incipient/nucleating Th₃P₄ that Nicole's data actually occupies) has essentially no
clean labels.**

**The decisive finding:** the two trajectories already identified as the cleanest crystalline
Th₃P₄ — `debye5.4_volumefraction.005575710117_seed1` and `debye5.45_…_seed1` — **exist on disk**
(`nicoles_dataset/phase_diagram/`, full trajectories, in the discovery set) but are **absent
from `labeled_dataset.csv`**. And on the new axis they sit at `trans_Sk_max` ≈ **10–11** (q6
≈0.46–0.54) — crystalline orientationally but only *moderately* translationally ordered, **far
below the idealized `th3p4.gsd` reference (25)**. So the clean Th₃P₄ reference is
**translationally over-idealized**: the supervised model is anchored to a periodicity that real
domain-matched Th₃P₄ crystals never reach — a concrete reason the boundary fails.

**Simulation specification:**
1. **No new bulk-Th₃P₄ simulation is needed first.** Label and add the already-simulated clean
   crystals (`debye5.4/5.45_φ0.0056_seed1`) to the training set — they populate the exact
   weak-to-moderate translational-order regime (`trans_Sk_max` 10–11) that currently has no clean
   labels, and they are domain-matched (real interaction/conditions) unlike `th3p4.gsd`.
2. **Replace/supplement the over-idealized `th3p4.gsd` anchor** with these domain-matched
   crystals so the Th₃P₄ class is defined at the translational order *achievable* in Nicole's
   ensemble, not at an unreachable ideal.
3. **Cluster 10 needs a targeted simulation** — it is ordered (PC1 +2.4, q6 0.51) yet
   translationally sub-crystalline (`trans_Sk_max` 9.2) and distinct on PC2, with **no clean
   reference anywhere in descriptor space**. If it is a real phase, simulate its
   conditions (top trajectories: debye5.4–5.55/φ≈0.0043–0.0056) to obtain a clean anchor.
4. **CsCl is already well-anchored** (separates cleanly on PC1); no new CsCl simulation needed
   for the boundary.

### Verdict

(a) Translational features separate crystal from amorphous (valid new axis) but not better than
q6. (b) Linear models match XGBoost → the problem is linear and the amorphous↔Th₃P₄ plateau is
real, driven by coordination+composition with translational order contributing; `q6_avg` points
*away* from Nicole-Th₃P₄, confirming label heterogeneity. (c) Only CsCl separates on the order
axis; Th₃P₄ overlaps amorphous and is the least-ordered class; cluster 10 is a distinct ordered
structure. (d) The biggest gap is not missing simulations but **unused existing ones** — the
clean domain-matched Th₃P₄ crystals are on disk but absent from training, and the current
`th3p4.gsd` anchor is translationally over-idealized; cluster 10 alone needs a new targeted run.
The planned domain-matched conditions are therefore **sufficient in principle but mis-applied**:
the fix is to label/include them and re-anchor the Th₃P₄ class, not to run more bulk crystals.

### Reproduction

```bash
conda activate colloids
cd preprocessing_pipeline_final
python interpretability/step1_cache_translational.py   # translational_features.pkl (2.66M rows)
python interpretability/step1_report.py                # crystal-vs-amorphous separation
python interpretability/step2_linear.py                # Lasso/Ridge coefficients vs XGBoost
python interpretability/step3_pca_viz.py               # PCA loadings + scatter (step3_pca_class.png)
python interpretability/step4_gap.py                   # gap analysis + simulation spec (step4_gap.png)
```

Artifacts in `interpretability/`: `translational_features.pkl`, `step{2,3,4}_results.json`,
`step3_pca_class.png`, `step4_gap.png`. Read-only: `labeled_dataset.csv`, `train_classifier.py`,
and trained models are unmodified.

---

## 2026-06-15 — Re-anchoring the Th₃P₄ class with domain-matched crystals

Acting on the interpretability finding: the Th₃P₄ reference anchor (`th3p4.gsd`) is
translationally over-idealized (`trans_Sk_max`≈25) versus what real Th₃P₄ in Nicole's domain
reaches (≈10–11), and the two cleanest real Th₃P₄ crystals
(`debye5.4_volumefraction.005575710117_seed1`, `debye5.45_…_seed1`) are on disk but absent
from training. This run adds them as clean Th₃P₄ and re-anchors the class, then tests whether
the amorphous↔Th₃P₄ boundary moves. `labeled_dataset.csv` is unmodified (augmentation is
in-memory); code in `interpretability/reanchor_extract.py` + `reanchor_retrain.py`.

### (a) Are the two new crystals genuinely clean Th₃P₄? — YES.

Extracted with the pipeline's own onset detection + crystallization-window sampling +
`process_frame` (interior-only, identical features), keyed by (source_file, frame_index,
particle_index):

| trajectory | interior Th₃P₄ particles | q6_avg | trans_Sk_max | coord (type0/type1) | n_unlike (type0/type1) |
|---|---|---|---|---|---|
| debye5.4_φ0.0056_seed1  | 3,621 | 0.475 | 11.3 | 13.6 / 13.4 | 7.5 / 7.0 |
| debye5.45_φ0.0056_seed1 | 3,948 | 0.457 | 10.4 | 13.6 / 12.7 | 7.4 / 6.4 |

Both pass the crystallinity gate (q6 0.40–0.60, trans_Sk_max>8): crystalline orientational
order with the asymmetric cross-species coordination expected of Th₃P₄, at the *domain*
translational order (~10–11), not the idealized 25. Total added: **7,569 Th₃P₄ interior
particles**, tier=reference, →train only.

### (b) Did re-anchoring shift the class toward achievable order? — YES.

| Th₃P₄ clean anchor | n | q6_avg | trans_Sk_max |
|---|---|---|---|
| OLD (`th3p4.gsd` ref) | 53,031 | 0.499 | **25.1** |
| NEW (domain crystals) | 7,569 | 0.465 | **~10–11** |

The clean reference moves from the unreachable prototype (25) toward achievable domain order
(10–11), keeping crystalline q6 (~0.46–0.48).

### (c)/(d) Did the boundary improve, and was the old anchor hurting?

Whole-trajectory split (directory-level; frames never cross train/test): 57 broadcast
trajectories held out to test (17 with Th₃P₄ label), identical fixed test set across all three
variants; only the training Th₃P₄ anchor changes. Feature set fixed = cheap + scalar BOPs
(q4, q6_avg, q8) to isolate the label-anchor effect. Class-balanced XGBoost.

| variant | Th₃P₄-F1 | Th₃P₄-P | Th₃P₄-R | amor-F1 | macro-F1 | amor→Th₃P₄ | Th₃P₄→amor |
|---|---|---|---|---|---|---|---|
| baseline (th3p4.gsd anchor) | 0.939 | 1.000 | 0.885 | 0.834 | 0.919 | 5 | 9088 |
| primary (+ domain crystals) | 0.947 | 0.998 | 0.900 | 0.836 | 0.924 | 84 | 8736 |
| drop (domain-only, no th3p4.gsd) | 0.947 | 0.998 | 0.901 | 0.836 | 0.924 | 90 | 8707 |

**The decisive metric — crystalline-minority recovery on held-out real Th₃P₄** (the success
criterion: recognize the genuinely-crystalline minority, NOT force the amorphous majority into
Th₃P₄). Of 102,322 held-out Th₃P₄-labeled particles, only **1.5% (1,530)** are genuinely
crystalline (`trans_Sk_max`>15):

| variant | crystalline minority → Th₃P₄ | (→ amorphous) | amorphous majority → amorphous |
|---|---|---|---|
| baseline | **53.8%** | 7.8% | 8.9% |
| primary  | **67.1%** | 4.6% | 8.6% |
| drop     | **67.9%** | 4.0% | 8.6% |

> ⚠️ **CORRECTED 2026-06-15 (see "Re-anchor CORRECTION" below): the +14-point result here was a
> TRAIN/TEST LEAKAGE ARTIFACT, not a real effect.** debye5.4 was in the test holdout *and*
> re-added to train; 46% of the crystalline test minority was leaked. The leak-free re-run shows
> re-anchoring changes nothing. The table/claim immediately below are retained for the record but
> are SUPERSEDED.

**~~Re-anchoring lifts crystalline-Th₃P₄ recognition by ~+14 points (53.8→67.9%)~~** [SUPERSEDED] —
the genuinely
crystalline minority is now recognized far better, which is exactly the stated goal. Standard
F1 barely moves (0.939→0.947) because the test *labels* are Nicole's heterogeneous labels, so
they can't credit the physically-correct behavior. The amorphous-majority overreach is
**unchanged** (~90% still called Th₃P₄): re-anchoring the clean reference cannot overcome the
~795 k heterogeneously-labeled broadcast Th₃P₄ in training (the ~91% amorphous-but-labeled-Th₃P₄),
which is a *labeling* problem, not an anchor problem.

**Was `th3p4.gsd` actively hurting? — No, only mildly.** drop ≈ primary (crystalline recovery
67.9% vs 67.1%; identical F1). Removing the over-idealized anchor adds only +0.8 pt; essentially
all the gain comes from *adding* the domain-matched crystals (+13 pt baseline→primary), not from
*removing* `th3p4.gsd`. The ideal anchor was a missed-opportunity, not an active poison.

### Verdict

(a) Both new crystals are genuinely clean domain Th₃P₄ (crystalline q6, asymmetric coordination,
domain-level translational order). (b) Re-anchoring shifted the clean Th₃P₄ reference from
`trans_Sk_max`≈25 to ≈10–11. (c) The boundary improved **on the metric that matters** —
crystalline-minority recovery +14 pt (53.8→67.9%) — while correctly *not* forcing the amorphous
majority into Th₃P₄ (that majority's behavior is unchanged and bounded by broadcast label
heterogeneity, the next bottleneck). (d) The old `th3p4.gsd` anchor was not actively hurting
(drop≈primary); the win is adding domain-matched crystals, not deleting the ideal. **Recommended
config: primary** (keep `th3p4.gsd` + add the two domain crystals) — same gain as drop, more
clean examples, no downside. The remaining ceiling is the broadcast Th₃P₄ labels themselves
(~91% amorphous), which a future relabel pass — not a new simulation — should address.

### Reproduction

```bash
conda activate colloids
cd preprocessing_pipeline_final
python interpretability/reanchor_extract.py    # Steps 1-2: characterize + extract clean crystals
python interpretability/reanchor_retrain.py    # Steps 3-5: re-anchor, retrain 3 variants, evaluate
```

Artifacts: `interpretability/reanchor_new_crystals.pkl`, `interpretability/reanchor_results.json`.
Read-only: `labeled_dataset.csv`, `train_classifier.py`, and trained models are unmodified.

### Re-anchor CORRECTION (2026-06-15 audit + leak-free re-run)

An audit found two errors in the run above:

1. **The domain crystals were never "absent" from `labeled_dataset.csv`.** They are present as
   `tier=broadcast, label=Th₃P₄-like` — `debye5.4_φ0.0056_seed1` (3,621 Th₃P₄ + 550 gas) and
   `debye5.45_φ0.0056_seed1` (3,948 + 438), 16 frames each (150–2000). The earlier "absent"
   claim came from a basename search (`trajectory.gsd`) that missed the `debye…` id in the path.
   They do **not** need merging.
2. **The +14-point result was train/test leakage.** The primary/drop variants re-added those
   same particles (identical keys) as `tier=reference` forced to train, while `debye5.4` was also
   in the test holdout. **46% (709/1530) of the crystalline test minority was leaked into train**,
   and that is exactly the population whose "recovery" jumped — the model was memorizing leaked
   particles, not generalizing.

**Leak-free re-run** (`interpretability/reanchor_retrain_leakfree.py`): no duplication; the two
domain crystals forced entirely to **train** (never test); test = held-out broadcast trajectories
excluding them (crystalline minority = 760 particles from 16 *other*, truly-held-out trajectories;
verified debye5.4-in-test = 0; 0 trajectories span train/test; 0 key overlap). Feature set
cheap+BOP, class-balanced. Variants change only the training Th₃P₄ anchor:

| variant | Th₃P₄-F1 | macro-F1 | crystalline minority → Th₃P₄ | amorphous majority → Th₃P₄ (overreach) |
|---|---|---|---|---|
| baseline (idealized + domain anchors present) | 0.953 | 0.929 | 99.6% | 91.0% |
| drop_ideal (remove `th3p4.gsd`) | 0.953 | 0.929 | 99.6% | 91.0% |
| drop_domain (remove domain crystals) | 0.954 | 0.929 | 99.6% | 91.1% |

**Corrected conclusion:** re-anchoring has **no effect** — all three variants are identical.
Removing the over-idealized `th3p4.gsd` anchor changes nothing (it was **not** hurting), and the
domain crystals are **not** specially helping; the broadcast Th₃P₄ training data already lets the
model recognize crystalline Th₃P₄. Note the model calls the crystalline minority Th₃P₄ at 99.6%
*and* the amorphous majority at 91% — i.e. it over-predicts Th₃P₄ across the board; the small
99.6-vs-91 gap is the only discrimination and it is anchor-invariant. **The earlier "+14 pt
re-anchoring win" is retracted.** The amorphous↔Th₃P₄ bottleneck is entirely the heterogeneous
broadcast Th₃P₄ labels (91% overreach), unmovable by re-anchoring the clean reference — confirming
that the only real lever is **re-labeling the broadcast Th₃P₄**, not adding/swapping anchors.
Artifact: `interpretability/reanchor_leakfree_results.json`. `labeled_dataset.csv` unmodified.

---

## 2026-06-15 — Identity-resolved neighbor-dynamics (blob→crystal vs static churn)

Hypothesis: all Th₃P₄ goes blob→crystal, so the discriminator is whether a particle's
neighborhood STABILIZES over time (nucleating) vs CHURNS forever (static amorphous) —
something single-frame descriptors cannot see. This measures WHO a particle's neighbors are
(specific IDs, species, geometry) and how that set evolves. Read-only; new folder
`neighbor_dynamics/`.

### STEP 0 (gate) — is the data adequate? **PASS, with a caveat.**

Particle identity is stable (N constant; neighbor retention ≫ random 12/1024≈0.01 ⇒ no
reindexing — position-index tracks the same physical particle). Save interval: `th3p4.gsd`
10k steps/frame; **real Nicole sims 500k steps/frame**. Neighbor retention between consecutive
*saved* frames: **clean crystal 0.972, amorphous 0.828** — crystal ≫ 0.7 threshold, so the
spacing supports the analysis. Caveat: per-particle displacement is 5–7×d_ref/frame even after
COM removal (large-scale **collective/affine flow**), so only identity/relative descriptors are
valid (neighbor sets, relative geometry, q_lm on bond vectors) — which is the class used.

### STEP 1-2 — battery + validity. **Dynamic descriptors separate crystal from amorphous.**

12-feature battery per tracked interior particle (`dynamics_descriptor.py`): neighbor-identity
persistence (retention mean/trend, core/permanent-cage count), species-resolved retention,
geometric churn (qlm_drift, bond_angle_stability), and order-arc (q6/trans slopes,
frac_frames_ordered, time_to_order). Group means and Cohen's d (crystal vs amorphous):

| feature | clean Th₃P₄ | domain xtal | CsCl | amorphous | gas | Cohen d |
|---|---|---|---|---|---|---|
| frac_frames_ordered | 1.00 | 0.88 | 1.00 | 0.10 | 1.00 | **+5.38** |
| time_to_order | — | — | — | — | — | −2.14 |
| bond_angle_stability | 0.001 | 0.004 | — | 0.007 | — | −1.74 |
| core_neighbor_count (permanent cage) | 11.9 | 12.5 | 12.0 | 7.5 | 13.6 | **+1.47** |
| same_type_retention | 0.99 | 0.92 | 1.00 | 0.75 | 0.99 | +1.38 |
| neighbor_retention_mean | 1.00 | 0.95 | 1.00 | 0.81 | 0.99 | +1.29 |
| qlm_drift | 0.54 | 1.42 | 0.54 | 1.40 | 1.34 | −1.29 |
| neighbor_retention_**trend** | −0.00 | +0.024 | 0.00 | +0.014 | +0.009 | **−0.00** |

Clean crystals separate strongly from amorphous on the **state-based** temporal features
(order fraction, permanent-cage size, retention magnitude, cage-shape churn). The **trend/slope**
features (the "journey" signals) are individually **weak/noisy** — because the reference crystals
are already ordered over the sampled window, and trends are noisy at 500k-step spacing.

### STEP 3 (decisive) — is Nicole-Th₃P₄ / cluster 5 nucleating or static? **STATIC.**

| group | n | retention_mean | retention_TREND | q6_slope | core_count | frac_ordered |
|---|---|---|---|---|---|---|
| real_amorphous (baseline) | 5759 | 0.81 | +0.014 | −0.004 | 7.5 | 0.10 |
| domain_th₃p₄ (forming xtal) | 501 | 0.95 | +0.024 | **+0.031** | 12.5 | 0.88 |
| **nicole_th₃p₄** | 17313 | 0.77 | +0.048 | **−0.017** | **5.6** | 0.018 |
| **cluster 5** | 434 | 0.77 | +0.034 | **−0.018** | **6.5** | 0.003 |

Nicole-Th₃P₄ and cluster 5 are **not on a blob→crystal journey**: their retention trend (+0.048,
+0.034) is barely above amorphous (+0.014) with **negligible Cohen's d ≈ 0.04–0.05**; their q6
slope is **negative** (order not rising); their permanent-cage count (5.6, 6.5) is **below even
generic amorphous** (7.5) and far below crystal (12); frac_frames_ordered ≈ 0.01–0.02. By the
pre-registered rule (need clearly positive trend, d>0.5), this is **static disorder** — confirming
the discovery module's cluster-5 verdict, not contradicting it. The hypothesis "all Th₃P₄ goes
blob→crystal" is **false for Nicole's broadcast Th₃P₄**: only the narrow φ≈0.0056 / debye 5.4–5.45
window actually crystallizes (domain crystals: q6_slope +0.031, core 12.5, frac_ordered 0.88);
the rest churn statically.

### STEP 4 — watching one crystal form (`step4_blob_to_crystal.png`).

The clean domain crystal `debye5.4_φ0.0056_seed1` shows the explicit transition: q6 rises 0→0.52
(crosses 0.35 at ~102 M steps) and neighbor retention rises 0.80→0.97 **together** as the crystal
forms. This validates the battery captures the journey and is the dynamical template for "forming
Th₃P₄" — a template Nicole's broadcast Th₃P₄ does **not** match.

### STEP 5 — classification & agent utility. **No classifier gain; yes a trajectory-level flag.**

Adding the 9 best temporal features to Stage-1 (whole-trajectory split, class-balanced) does
**not** help: Th₃P₄-F1 0.939→0.928, amor→Th₃P₄ confusion 5→168, macro +0.006; temporal feature
importances are near-zero (max frac_frames_ordered 0.079) because `q6_avg`/`coord` already encode
the same order. **Plateau, like all six prior descriptors.** BUT at the **trajectory level**,
`frac_frames_ordered`, `core_neighbor_count`, and `neighbor_retention_mean` **cleanly separate**
forming/formed crystals from static amorphous (clean_sep=YES; forming 0.88/12.5/0.95 vs static
0.10/7.5/0.81) — a usable **crystal-detection flag for the discovery agent**. The pure
identity-dynamics *trend* (`neighbor_retention_trend`, `q6_slope`) does **not** cleanly flag
nucleation (clean_sep=no) — timing the nucleation *moment* would need denser frame output than
500k steps.

### Verdict

(a) **Frame spacing adequate** — identity stable, crystal retention 0.972 at the 500k interval
(caveat: large collective flow ⇒ relative descriptors only). (b) **Dynamic descriptors separate
clean crystal from amorphous** — strongly on state features (frac_ordered d=5.4, permanent-cage
d=1.5, retention d=1.3, qlm_drift d=−1.3); trend features are weak. (c) **Nicole-Th₃P₄ / cluster 5
are STATIC, not nucleating** — retention trend ≈ amorphous (d≈0.04), q6 slope negative, permanent
cage *below* amorphous; the blob→crystal hypothesis holds only for the narrow crystallizing window,
not Nicole's broadcast Th₃P₄. (d) **No per-particle classification gain** (plateau; temporal
≈redundant with static q6/coord), **but a clean trajectory-level crystal/nucleation flag** for the
agent (frac_ordered / core-cage / retention). Net: identity-dynamics confirms — by the strongest
temporal test to date — that most of Nicole's Th₃P₄ is genuinely static disorder, and its value is
a nucleation *flag*, not a fix for the amorphous↔Th₃P₄ per-particle boundary.

### Reproduction

```bash
conda activate colloids
cd preprocessing_pipeline_final
python neighbor_dynamics/dynamics_descriptor.py     # self-test the battery
python neighbor_dynamics/run_battery.py             # Steps 2-3 cache (refs + Nicole + cluster5)
python neighbor_dynamics/analyze_battery.py         # Step 2-3 distributions + Cohen's d + decisive test
python neighbor_dynamics/plot_trajectory.py         # Step 4 blob->crystal plot
python neighbor_dynamics/step5_cache_all.py         # Step 5 per-track dynamics for all labeled trajectories
python neighbor_dynamics/step5_classify.py          # Step 5 classifier + nucleation-signal assessment
```

Artifacts in `neighbor_dynamics/`: `dynamics_battery.pkl`, `dynamics_all_tracks.pkl`,
`step4_blob_to_crystal.png`, `step5_results.json`. Read-only: `labeled_dataset.csv`, pipeline,
and models unmodified.
