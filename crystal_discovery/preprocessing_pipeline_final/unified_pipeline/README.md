# Unified Crystal Classifier Pipeline

**Created:** 2026-06-17  
**Location:** `preprocessing_pipeline_final/unified_pipeline/`  
**Status:** Active — supersedes the scattered `benchmark_v2/project_A*` scripts.

---

## What this is

A single, linear, maintainable pipeline that:
1. Reuses all existing SOAP/.npz caches (no recomputation)
2. Assembles two canonical parquet datasets
3. Trains a **4-class** (gas / amorphous / CsCl / Th₃P₄) XGBoost model on clean GT
4. Applies the model to all 486 Nicole trajectories and reports contention

All label curation decisions are encoded in one place (`03_filter.py`).  
Labels are unified at dataset-build time (no string mismatches in model code).

---

## Directory structure

```
unified_pipeline/
  data/
    dataset_clean_gt.parquet     # GT + curated sims + unknowns, all descriptors
    dataset_nicole.parquet       # 486 Nicole trajectories, BOP descriptors only
    dataset_filtered.parquet     # Training-ready subset (after 03_filter.py)
    dataset_holdout.parquet      # Boundary holdout (T320, phi partials)
    caches/                      # Symlinks → benchmark_v2/project_A_clean_gt_extended/*.npz
    build_report.txt             # 02 provenance summary
    filter_report.txt            # 03 curation decisions
  scripts/
    01_compute_descriptors.py    # Idempotent; symlinks existing caches
    02_build_datasets.py         # Assembles both parquets from sources + caches
    03_filter.py                 # ALL label assignment and curation rules
    04_split_and_train.py        # Split + train + Nicole audit
  models/
    xgb_4class.ubj               # Trained XGBoost model (UBJ format)
    feature_medians.json         # Training-set medians (for inference imputation)
  results/
    confusion_val.csv / confusion_test.csv
    per_class_metrics.csv
    feature_importance.csv
    q8_avg_auc_benchmark.txt     # Verification: AUC≥0.93 (Project A accepted)
    nicole_audit.csv             # 486 trajectories ranked by contention
    nicole_audit_report.txt
    train_report.txt
```

---

## Label vocabulary (canonical)

| Canonical label | Nicole raw label | Physical meaning |
|---|---|---|
| `gas` | `gas` | Isolated/low-coordination particles (coord < 4) |
| `amorphous` | `blob` | Dense disordered aggregate |
| `CsCl` | `CsCl-like` | CsCl-type crystal |
| `Th3P4` | `Th3P4-like` | Th₃P₄-type crystal |
| `boundary_melting` | — | T320 borderline (held out, not trained) |
| `crystal_uncertain` | — | unknown_example traj(12); not trained |

**Unification rule applied at dataset level:**
```
blob → amorphous
Th3P4-like → Th3P4
CsCl-like → CsCl
gas → gas
```

---

## Dataset 1 — dataset_clean_gt.parquet

### Sources (q6-verified, 2026-06-16 diagnostics)

| source_dataset | Label assigned | Condition | q6 range | Notes |
|---|---|---|---|---|
| `perfect_th3p4_reference` | Th3P4 | perfect_th3p4_trajectory, Prob_best, also_perfect(12) | 0.48–0.54 | Ground truth |
| `seeded_clean_th3p4` | Th3P4 | debye5.30/5.35/5.40/5.50 (×3 seeds) | 0.51–0.54 | Seeded, domain-matched |
| `seeded_clean_th3p4` | Th3P4 | temperature_fine T305/T310/T315 (×3 seeds) | 0.50–0.55 | q6-verified crystals |
| `seeded_clean_th3p4` | Th3P4 | phi0.50x/0.70x/0.80x/1.00x (×3 seeds) | 0.53–0.54 | q6-verified crystals |
| `stage1_seeded_stable_th3p4` | Th3P4 | stage1_perfect (debye5.40, phi0.0056) | ≈0.55 | Domain-matched |
| `seeded_dense_amorphous_hard_negative` | amorphous | phi0.60x + phi0.90x (×3 seeds) | 0.26 | **VERIFIED dense blob** — NOT crystal |
| `spontaneous_amorphous_gate` | amorphous | unknown_examples (1,2,4,5,10) | — | OVITO-verified amorphous |
| `nicole_curated_blob` | amorphous | Nicole's 188 blob trajectories | — | Nicole ground truth |
| `gt_cscl_reference` | CsCl | ground_truth/CsCl.gsd | 0.575 | Ground truth |
| `nicole_curated_cscl` | CsCl | Nicole's 117 CsCl trajectories | — | Single-domain filtered |
| `nicole_curated_gas` | gas | Nicole's 45 gas trajectories | — | Low-coord dispersed |
| `off_domain_crystal_holdout` | **holdout** | phi partials (boundary diagnostic) | — | NOT training |
| `boundary_melting_holdout` | **holdout** | T320 (×3 seeds) | 0.425 (borderline) | NOT training |

### What is EXCLUDED

| Condition | Reason |
|---|---|
| ground_truth/th3p4.gsd, th3p4_*.gsd (CIF structures) | 3:1 stoich or over-idealized; domain mismatch (q6≈0.26 under pipeline, not 0.48) |
| Perfect_Th3P4crystals/th3p4.gsd | 3:1 stoichiometry duplicate |
| debye5.60, debye5.80, debye6.00 | Crystalline but off-domain (not in Nicole's phase diagram region) |
| temperature T340/T360–T500 | Mostly-gas small clusters (n_dense < 100); not representative |
| phi1.10x–phi1.50x | 6-frame crashed runs; not thermalized |
| boundary_melting label | T320 borderline; held out for boundary evaluation |
| label == crystal_uncertain | unknown_example traj(12); too ambiguous for training |

### Columns

38 base columns from `curated_gt_labeled_particles.csv` + SOAP/q_lm cache summaries:
- **Cheap geometry:** `coord`, `n_unlike`, `n_like`, `frac_unlike`, `d_mean`, `d_std`, `local_density`
- **Steinhardt BOPs:** `q4_raw`, `q4_avg`, `w4_avg`, `q6_raw`, `q6_avg`, `w6_avg`, `q8_raw`, `q8_avg`, `w8_avg`
- **Entropy:** `pp_entropy`, `pp_entropy_avg`
- **Score features:** `th3p4_coord_score`, `cscl_coord_score`, `trans_Sk_max`
- **SOAP summaries** (16 cols each for n4l4, n6l6, n8l8): similarity scores, PCA components, margins
- **q_lm summaries** (6 cols each for l=6,8,10,12 × raw/avg × all/unlike): 96 additional columns
- **Provenance:** `source_dataset`, `provenance_tier` (ground_truth / seeded_sim / unknown / nicole)

---

## Dataset 2 — dataset_nicole.parquet

All 486 Nicole trajectories, 3 frames sampled from second half each.
**SOAP not available** (no cache; DScribe computation not run for Nicole batch).

| Column | Description |
|---|---|
| `sequence`, `trajectory_id` | Nicole manifest identifiers |
| `nicole_raw` | Nicole's original label (blob, Th3P4-like, CsCl-like, gas) |
| `nicole_unified` | Canonical label after unification (amorphous, Th3P4, CsCl, gas) |
| `frame_index`, `particle_index`, `typeid` | Particle identifiers |
| `coord`, `n_unlike`, `n_like`, `frac_unlike`, `d_mean`, `d_std` | Local geometry |
| `q4_avg`, `q6_avg`, `w4_avg`, `w6_avg`, `q8_avg`, `pp_entropy` | Steinhardt BOPs |
| `is_interior`, `is_gas` | Interior/gas flags |

---

## 4-class model training features

These 12 features are used for the deployed model. All are computable from any GSD file:

```python
FEAT_COLS = [
    "coord", "n_unlike", "n_like", "frac_unlike", "d_mean", "d_std",
    "q4_avg", "q6_avg", "w4_avg", "w6_avg", "q8_avg", "pp_entropy",
]
```

Rationale: all 12 are present in both `dataset_clean_gt.parquet` and `dataset_nicole.parquet`, enabling direct model application to Nicole's data without SOAP recomputation.

---

## Caches (do not delete, do not recompute)

Original location: `benchmark_v2/project_A_clean_gt_extended/*.npz`  
Symlinked to: `unified_pipeline/data/caches/*.npz`

| Cache | Size | Description |
|---|---|---|
| `soap_n4_l4_vectors.npz` | 55 MB | (78693, 180) SOAP vectors, n=4 l=4 |
| `soap_n6_l6_vectors.npz` | 166 MB | (78693, 546) SOAP vectors, n=6 l=6 |
| `soap_n8_l8_vectors.npz` | 372 MB | (78693, 1224) SOAP vectors, n=8 l=8 |
| `soap_n*_l*_summary.npz` | 4 MB each | 16-column similarity/PCA summaries |
| `qlm_l{6,8,10,12}_*_vectors.npz` | 5–11 MB each | Complex q_lm vectors |
| `qlm_l{6,8,10,12}_*_summary.npz` | 1.6 MB each | 6-column margin/autocorr summaries |

All caches keyed by `curated_row_id` (join key for augmenting dataset_clean_gt).

---

## How to run

```bash
conda activate /home/sz5808/.conda/envs/crystal_ml
cd preprocessing_pipeline_final/unified_pipeline

# Step 1: Link caches (idempotent, skips if already linked)
python scripts/01_compute_descriptors.py

# Step 2: Build datasets (idempotent; --force-gt / --force-nicole to rebuild)
python scripts/02_build_datasets.py

# Step 3: Apply q6-verified label rules
python scripts/03_filter.py --verify-splits

# Step 4: Train + evaluate + Nicole audit
python scripts/04_split_and_train.py

# Full pipeline (force rebuild):
python scripts/02_build_datasets.py --force-gt --force-nicole
python scripts/03_filter.py
python scripts/04_split_and_train.py --force
```

---

## Verification: q8_avg AUC benchmark

After step 4, `results/q8_avg_auc_benchmark.txt` reports the Th₃P₄-vs-amorphous AUC
using q8_avg alone on the test set. This must be **≥ 0.93** (the accepted Project A result).

If it drops below 0.93, the data join in `02_build_datasets.py` may have corrupted
the label-to-feature alignment. Do not proceed to Nicole audit until this is resolved.

---

## Cache/label coverage note

The .npz caches cover 78,693 rows of the 94,173-row curated GT (those included in
Project A's analysis-row-cap-25000 sample). Rows without cache entries get NaN for
SOAP/q_lm columns. The 4-class model uses only the 12 BOP features (all present for
all 94,173 rows), so the NaN SOAP columns do not affect training or inference.

---

## Connection to prior work

| Prior artifact | Role in unified pipeline |
|---|---|
| `benchmark_v2/curated_gt_labeled_particles.csv` | Primary source for dataset_clean_gt |
| `benchmark_v2/project_A_clean_gt_extended/*.npz` | Caches (symlinked, not moved) |
| `benchmark_v2/project_A_clean_gt_extended/project_B_4class_*.csv` | Superseded by results/nicole_audit.csv |
| `preprocessing_pipeline/nicoles_dataset/phase_diagram/labeled_data/manifest.csv` | Nicole source for dataset_nicole |

**Do not delete old files** — this pipeline reads from them. Archive after verification.
