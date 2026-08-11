# Crystal Discovery — PACSim Ionic Crystal ML Project

**Owner:** Steven Zhang (sz5808), Hocky Lab, NYU  
**Cluster:** NYU Torch HPC (`torch.hpc.nyu.edu`)  
**Conda env:** `crystal_ml` (Python 3.10.20, installed at `/home/sz5808/.conda/envs/crystal_ml`)  
**SLURM accounts:** `torch_pr_146_general`, `torch_pr_146_chemistry`  
**Last updated:** 2026-06-26

---

## Project goal

Build a 4-class classifier (gas / blob-amorphous / CsCl-like / Th₃P₄-like) for
per-particle phase labels in PACSim ionic-colloid MD simulations. Then apply a novelty
detector to identify crystalline state points in new simulation data — particularly a
new NaCl_K4C60 dataset with a 2.2:1 size ratio — that may represent previously
unseen crystal structures.

The pipeline has two tracks:
1. **Training pipeline** — builds labeled per-particle CSV from labeled GSD trajectories → trains GAT/XGBoost classifier
2. **Structural sweep** — model-free, scans all GSD files for crystalline order via RDF + S(k), produces a ranked crystalline candidate list

---

## Directory map

```
crystal_discovery/
├── README.md                              ← this file
├── recon_new_data_map.md                  ← reconnaissance notes on 5 new PI data dirs
│
├── PACSim/                                ← PACSim simulation framework (git submodule)
│   └── colloids/                          ← HOOMD-blue simulation code
│
├── preprocessing_pipeline/               ← training data pipeline
│   ├── make_labels_csv.py                 ← merges Nicole's labels + cont labels → labels.csv
│   ├── build_training_set.py              ← per-particle feature extraction → training CSV
│   ├── ground_truth/                      ← hand-crafted reference GSD files (CsCl, Th3P4)
│   ├── labeled_data (1)/                  ← Nicole's manifest.csv (193 cont trajectories)
│   └── nicoles_dataset/phase_diagram/     ← Nicole's lattice_classification.csv (293 regular)
│
├── preprocessing_pipeline_final/         ← unified ML pipeline (supersedes scattered scripts)
│   └── unified_pipeline/                  ← ACTIVE — see README inside
│       ├── scripts/01–04_*.py             ← linear pipeline: descriptors → datasets → train
│       ├── data/                          ← parquet datasets + SOAP/q_lm caches
│       ├── models/xgb_4class.ubj          ← trained 4-class XGBoost model
│       └── results/                       ← metrics, Nicole audit, feature importance
│
└── sweep_results/                         ← structural sweep outputs (completed 2026-06-26)
    ├── collect_paths.py                   ← finds all trajectory*.gsd across 3 data roots
    ├── sweep_worker.py                    ← SLURM array worker: RDF + S(k) per file
    ├── submit_sweep.sbatch                ← SLURM job array script (64 tasks)
    ├── merge_results.sh                   ← cats shards → sweep_results_all.csv
    ├── all_paths.txt                      ← 1,826 GSD paths (generated 2026-06-26)
    ├── sweep_results_all.csv              ← merged results, 1,826 rows
    ├── sweep_results_shard_000–063.csv    ← per-task output shards
    ├── clean_crystalline_subset.csv       ← 130-row defensible NaCl_K4C60 crystalline set
    ├── audit_report.md                    ← detailed audit of sweep results (read this first)
    └── logs/sweep_11852803_*.out/.err     ← SLURM logs for completed array job 11852803
```

---

## Data roots (READ-ONLY — never write here)

All source GSD trajectories live under three roots. Never write to these locations.

| Root | Size | Contents |
|------|------|----------|
| `/projects/rps/gmh4/hockygroup/sp6310/ionic_crystal/ionic_crystal/ExperimentalData` | 746 GB | Active simulations — New_simulation2, Closed_box, using_seed, etc. |
| `/projects/rps/gmh4/hockygroup/greene-archive/sanjib_sp6310/ionic_crystal/ExperimentalData` | 125 GB | Archive — Closed_box (72 trajectories), using_seed (23) |
| `/projects/rps/gmh4/hockygroup-archive/PublishedPapers/2025-Zang-NonClassicalCrystal` | 175 GB | Published paper data — 85-105+50-50_bulk (265 trajectories) |

The `all_paths.txt` file holds 1,826 deduplicated `trajectory*.gsd` paths across all three roots.
Breakdown: Root1=1,411 | Root2=141 | Root3=274.

Directories excluded from sweep: `charged_wall`, `short_simulations_for-test`,
`movie_images+50-50`, `movies_*`, `movies_and_figures`, hidden dirs (`.`).

**NaCl_K4C60 new data** (the primary target for novelty detection):
```
ExperimentalData/New_simulation2/corrected_pot2/bulk_simulations/NaCl_K4C60simulations/
```
Contains 194 trajectories across subdirs `+50-50`, `+200-400nm`, `+300-130nm`, `333-140nm`
with particle sizes RN = {65, 70, 80, 200} nm. The PI is most interested in the 2.2:1 ratio.

---

## GSD file conventions (critical for any code touching positions)

GSD stores particle positions in `[-L/2, L/2]` convention per axis.

- **cKDTree** (`scipy.spatial`) requires `[0, L]` → use `wrap_unit_cell(pos, L)`:
  ```python
  p = pos + L/2
  return p - L * np.floor(p/L)
  ```
- **freud** requires `[-L/2, L/2]` → use `wrap_centered(pos, L)`:
  ```python
  return pos - L * np.round(pos/L)
  ```

Mixing these conventions causes `ValueError: Negative input data are outside of the
periodic box` from freud, or silently wrong neighbor distances from cKDTree.

Box array from GSD: `frame.configuration.box` → `[Lx, Ly, Lz, xy, xz, yz]`.
Only `[:3]` is needed for orthogonal boxes (all simulations here are orthogonal).

---

## Phase diagram parameters (filename encoding)

Trajectory filenames encode the state point:
```
trajectory_LS{X}_FP{X}_RN{X}_BL{X}_DL{X}_N{X}.gsd
```

| Tag | Meaning |
|-----|---------|
| `LS` | Lattice spacing (ratio) |
| `FP` | Fraction positive (charge stoichiometry) |
| `RN` | Radius negative particle (nm) |
| `BL` | Bjerrum length (nm) |
| `DL` | Debye length (nm) |
| `N`  | HOOMD N seed (not particle count — actual N is in the GSD frame) |

The 4-class labels are: **CsCl-like**, **Th3P4-like**, **blob**, **gas**.

---

## Training pipeline (preprocessing_pipeline + preprocessing_pipeline_final)

### Label sources

Two separate label sources must be merged before running `build_training_set.py`:

1. `nicoles_dataset/phase_diagram/labeled_data/lattice_classification.csv`
   — Nicole's original labels, 293 regular (non-cont) trajectories

2. `labeled_data (1)/manifest.csv`
   — 193 cont trajectory entries with per-directory labels

`make_labels_csv.py` merges these into `labels.csv` (486 unique rows).
Run once: `python preprocessing_pipeline/make_labels_csv.py --out labels.csv`

### Known bug that was fixed — label key collision

The original `load_labels` used `trajectory_id.split("/")[0]` as the key, which kept
only the first directory component. This caused 15 base directories where regular
(e.g., blob) and cont (e.g., Th3P4-like) trajectories shared the same truncated key,
silently dropping one label. Fixed: use the full normalized path as key; `match_label`
does exact match then suffix match.

### build_training_set.py — four-class labeling

All four classes (crystal + blob + gas) are included. The pipeline treats them differently:

```
crystal labels → interior particles only (filter by coordination ≥ 80th percentile)
blob/gas      → use all core cluster particles (no interior filter, gas_coord=0)
```

The `analyze()` call signature:
```python
is_crystal = label in crystal_labels
gas_coord_eff = args.gas_coord if is_crystal else 0
a = analyze(pos, typeid, box, args.cutoff_factor, gas_coord_eff, args.bulk_percentile, args.surface_margin)
labeled_idx = a["interior"] if is_crystal else a["core"]
```

### Unified pipeline (preprocessing_pipeline_final/unified_pipeline)

The active 4-class XGBoost model is at `models/xgb_4class.ubj`.
12 features used: `coord, n_unlike, n_like, frac_unlike, d_mean, d_std, q4_avg, q6_avg, w4_avg, w6_avg, q8_avg, pp_entropy`

Run order:
```bash
conda activate crystal_ml
cd preprocessing_pipeline_final/unified_pipeline
python scripts/01_compute_descriptors.py
python scripts/02_build_datasets.py
python scripts/03_filter.py
python scripts/04_split_and_train.py
```

Verification gate: `results/q8_avg_auc_benchmark.txt` must show AUC ≥ 0.93.
If it drops, the data join in `02_build_datasets.py` corrupted label-to-feature alignment.

---

## Structural sweep (sweep_results/)

Model-free, size-scaled scan of all 1,826 GSD files. Ran 2026-06-26 as SLURM array
job 11852803 (64 tasks, `cpu_short`, `torch_pr_146_general`).

### What it does (sweep_worker.py)

For each file, loads the **final frame only**, then:

1. **Adaptive NN distance**: cKDTree k=2 query → `d_nn_med = median(1st-NN distances)`
2. **Coordination**: neighbors within `1.35 × d_nn_med`
3. **Gas fast-exit**: `mean_coord < 3.0` → `verdict='gas-like'`, skip RDF/S(k)
4. **RDF**: freud `RDF(bins=150, r_min=0.3×d_nn, r_max=min(5×d_nn, 0.48×L))`
5. **S(k)**: freud `StaticStructureFactorDirect` (see bug note below)
6. **Verdict**:
   - `rdf_n_peaks ≥ 4 AND rdf_peak1_h ≥ 3.0` → `crystalline` (RDF alone, S(k) irrelevant)
   - `sk_n_peaks ≥ 2 OR sk_max_h > 5.0` AND `rdf_n_peaks ≥ 2` → `crystalline` (S(k) required)
   - `rdf_n_peaks ≤ 1 AND sk_n_peaks == 0 AND sk_max_h < 3.0` → `amorphous`
   - else → `unclear`

### Sweep results summary

| Verdict | Count |
|---------|-------|
| crystalline | 1,080 |
| gas-like | 714 |
| unclear | 18 |
| error | 14 |
| **total** | **1,826** |

NaCl_K4C60: **130/194 crystalline**, 64 gas-like, 0 errors, 0 unclears.

### KNOWN BUG in sweep_worker.py — freud S(k) always fails

`freud.diffraction.StaticStructureFactorDirect` requires a `bins` positional argument
that is missing from the current call:

```python
# BROKEN — missing bins argument
sf = freud.diffraction.StaticStructureFactorDirect(
    num_sampled_k_points=2000, k_max=k_max, k_min=k_min)

# FIXED — add bins:
sf = freud.diffraction.StaticStructureFactorDirect(
    bins=100, num_sampled_k_points=2000, k_max=k_max, k_min=k_min)
```

This caused the FFT fallback (`_sk_fft()`) to fire for every non-gas file.
The `sk_n_peaks` and `sk_max_h` columns in the CSV are from the broken FFT estimator
and are unreliable (FFT gives ~1.0–1.5 on a real crystal; freud direct gives 6–40).

**Blast radius: zero** — all 1,080 crystalline verdicts used the RDF-only branch
(`rdf_n_peaks ≥ 4 AND rdf_peak1_h ≥ 3.0`). S(k) was never the deciding factor.
The fix must be applied before the next sweep if S(k) verdicts are needed.

### Errors (14 total)

All 14 are `RuntimeError: Corrupt GSD file`. All are RN=105 files from prod2_v2 or
85-105+50-50_bulk. Zero NaCl_K4C60 files errored. These specific state points may
need re-simulation or recovery from backup — flag to PI.

### Unclears (18 total)

All 18 are FP=0.15, RN=65, prod2_v2. All have rdf_n_peaks=2–3 (just below the 4-peak
threshold) and high first-peak heights. Likely a partially ordered / intermediate-phase
corner. Not NaCl_K4C60. Worth manual visual inspection.

### clean_crystalline_subset.csv

130-row defensible NaCl_K4C60 crystalline state points. All verdicts are RDF-only.
This is the input for the novelty detector. Columns:

```
path, stoichiometry, LS, FP, RN, DL, N, Lx, d_nn_med,
coord_cutoff, rdf_r_min, rdf_r_max, rdf_bin_w,
mean_coord, rdf_n_peaks, rdf_peak1_h,
sk_method, sk_max_h, verdict_basis, notes
```

`sk_method` = `"fft-fallback (freud failed: missing bins arg)"` for all rows.
82/130 rows are flagged `high-coord-std:possible-coexistence` (std_coord > 4.0),
indicating some gas-phase particles coexist with the crystal domain. Crystal order is
confirmed by sharp RDF; the domain may not be space-filling.

Parameter space of clean subset:

| param | values |
|-------|--------|
| RN | 65 (37), 70 (6), 80 (72), 200 (15) |
| LS | 3 – 5.5 |
| DL | 4.8 – 6.1 |
| FP | 0.33 (123), 0.5 (7) |

---

## Re-running the sweep (if new data arrives or S(k) bug is fixed)

```bash
# 1. Regenerate path list (picks up any new files)
cd /projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/sweep_results
python collect_paths.py

# 2. Apply S(k) fix in sweep_worker.py (add bins=100)

# 3. Submit array
sbatch submit_sweep.sbatch

# 4. After completion, merge and rank
bash merge_results.sh
```

---

## Environment notes

```bash
# Activate
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate crystal_ml

# Key packages (freud version matters for API)
python -c "import freud; print(freud.__version__)"  # 3.4.0

# SLURM accounts for sz5808
sacctmgr show user sz5808 withassoc
# Valid: torch_pr_146_general, torch_pr_146_chemistry
```

---

## What's done / what's next

| Status | Task |
|--------|------|
| ✓ Done | Structural sweep — 1,826 files, results in sweep_results_all.csv |
| ✓ Done | Audit — S(k) bug identified, blast radius zero, RDF verdicts confirmed |
| ✓ Done | clean_crystalline_subset.csv — 130 NaCl_K4C60 rows ready for novelty detector |
| ✓ Done | Training pipeline — 4-class XGBoost model trained (unified_pipeline/) |
| Next | Fix freud S(k) bug (`bins=100`) in sweep_worker.py and re-run if S(k) needed |
| Next | Run GAT/novelty detector on clean_crystalline_subset.csv |
| Next | Investigate 18 unclears (FP=0.15, RN=65) — possible partial-order phase |
| Next | Report 14 corrupt GSD files to PI (all RN=105) |
