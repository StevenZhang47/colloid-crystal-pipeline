# Draft staleness — existing LaTeX reports vs current repository

Two LaTeX drafts already exist. The user's instruction: *"we made a lot of changes however
so don't take that report for granted."* They are treated here as **untrusted input**. A
claim is CURRENT only where the supporting code or artifact was located today; anything
otherwise is STALE or UNVERIFIABLE, never given the benefit of the doubt.

- `reports/ood_pipeline/` — 1592 lines, 12 sections, 5 tables
- `reports/gat_pipeline/` — 1367 lines, 15 sections

---

## Verdicts

| # | draft file | claim as written | current repository value | verdict | evidence |
|---|---|---|---|---|---|
| 1 | ood `05_methods_ood` | "projected by PCA", ablation table states **PCA 20** | **PCA 16** in v3 production | **STALE** | `core3_atlas_v3/scripts/v3_common.py` `PCA_DIM`; `data/V3_PREDECLARATION.json` |
| 2 | ood `05_methods_ood` | "$n = 70$ training rows" | v3-A **126** training regions / 68 groups; v3-B **260** / 70 | **STALE** | `core3_atlas_v3/data/v3_region_manifest.csv` |
| 3 | ood `05_methods_ood` | "a five-family atlas" | core-three atlas: **CsCl, Th3P4, amorphous** | **STALE** | `phase5a_common.CLASSES`; v3 region manifest |
| 4 | ood `05_methods_ood` | feature blocks standardised separately, concatenated, standardised again, then PCA | **species-conditioned** standardisation (separate mean/scale for LARGE and SMALL), then shared PCA. No physical block, no double standardisation | **STALE** | `v3_common.fit_transform()` |
| 5 | ood `03_methods_descriptors` | ablation of SOAP+physical (961) vs SOAP-only (952) | physical block is gone; production is SOAP-only 952 | **CURRENT in conclusion, STALE in framing** — the ablation still justifies the choice but describes a configuration that no longer exists | `core3_pipeline.SOAP_DIMENSION` |
| 6 | ood `03_methods_descriptors` | SOAP **952**-D, `rcut_factor = 3.0`, `sigma_factor = 0.1`, L2-normalised before pooling | **CURRENT**. `N_MAX=8`, `L_MAX=6`, `R_CUT_NORM=3.0`, `SIGMA_NORM=0.10`, `SOAP_DIMENSION=952` asserted at runtime | **CURRENT** | `core3_pipeline.py` lines 28–37, `_soap()` |
| 7 | ood `03_methods_descriptors` | exported array shape (76672, 952) | shape claim not re-verified; the 952 axis is confirmed, the row count is not | **UNVERIFIABLE** | — |
| 8 | ood `03_methods_descriptors` | rotation-invariance check: max change 4.8e-8 on a 998-particle CsCl domain | test not located in current tree | **UNVERIFIABLE** | \TODO{VERIFY} |
| 9 | ood `05_methods_ood` | Ledoit–Wolf because "$n=70$, $p=952$ raw features, sample covariance singular" | v3 fits LW on **32-D** region summaries (2×PCA16), not on 952-D. With 272–280 group-balanced rows in 32-D the problem is well-posed; measured shrinkage is **0.0062 (A) / 0.0059 (B)** — near-zero, i.e. essentially the empirical covariance | **STALE — and the rationale inverts** | `v3_common.fit_covariance()`; `v3_threshold_diagnostic.csv` |
| 10 | ood `05_methods_ood` | production selected on "known FPR = 0 under leave-one-trajectory-out while rejecting every held-out positive" | v3 calibration is **leave-one-condition-group-out via a hard split**, not leave-one-trajectory-out. Measured v3 known-FPR is **0.24 (A) / 0.22 (B)**, not 0 | **STALE** | `v3_common.threshold()`; `v3_m2_false_positive.csv` |
| 11 | ood `05_methods_ood` | k-NN, PCA-residual, Isolation Forest compared as alternatives | v2/v3 comparator set is `mahalanobis_particle`, `mmd`, `global_lw_d16/d32`, `hybrid`. The k-NN / IsoForest comparison is not in the current path | **HISTORICAL** | `core3_comparator_benchmark_v2_expanded/scripts/run_v2_strict_lofo.py` |
| 12 | ood `07_methods_matcher` | library holds **17** CIFs in `crystal_stability/structures/` | **42** distinct CIFs now catalogued (21 local + 21 MP-fetched); the 7-family matcher subset is unchanged | **STALE** | `cif_library/catalog.csv` |
| 13 | ood `07_methods_matcher` | matcher consumes 7 runs, does not read CIFs directly, uses `first_frame.gsd` | **CURRENT**, and the family→run table matches exactly | **CURRENT** | `build_and_run_matcher.py` `FAMILIES` |
| 14 | ood `07_methods_matcher` | zangenite CIF "internally `data_La3S4`" | **CURRENT**, and now sharpened: two zangenite CIFs exist with identical geometry differing only in species labels (`L`/`S` vs `La`/`S`); the matcher uses the `La`/`S` variant | **CURRENT (incomplete)** | `cif_library/audit_ambiguity.csv` |
| 15 | ood `08_results`, `tab04`, `tab05` | pooled benchmark and leave-family-out tables | superseded by v3 `v3_m1_lofo.csv` and by the v2→v3 architecture change | **STALE** \TODO{VERIFY: which specific rows survive} | `core3_atlas_v3/data/` |
| 16 | both drafts | anything depending on the zangenite or Ni3Sn reference anchor | Ni3Sn anchor unchanged under the anisotropic fix; **zangenite changes materially** (c/a 0.480→0.596) | **CAVEAT — not stale yet, but conditional** | `SOURCE_OF_TRUTH.md` |
| 17 | gat `08_results`, `11_ablations` | GAT results | not yet audited | **UNVERIFIABLE** \TODO{VERIFY} | — |

---

## What survives

The **descriptor-level Methods in `03_methods_descriptors` are largely reusable**: the SOAP
equations, parameters, dimensionality and the scale-invariance argument all check out
against current code. The physical-block ablation is still a valid justification for the
present representation even though the compared configuration is retired.

The **matcher Methods in `07_methods_matcher` are broadly current** — the library/anchor
architecture, the seven families and the run mapping are unchanged.

## What must be rewritten

Everything describing the **statistical detector**: PCA dimension, training-set size, number
of families, standardisation scheme, covariance rationale, calibration protocol and reported
false-positive rate have all changed. §9 is the sharpest case — the draft justifies
Ledoit–Wolf by singularity at *p*=952, whereas v3 applies it to a well-conditioned 32-D
problem where the fitted shrinkage is ~0.006 and the estimator is nearly the raw empirical
covariance. Carrying that sentence forward would misstate why the method is used.

## Not found

No sequential detect→match→add→recognise expansion experiment exists anywhere in the tree.
The "EXPAND" stage of the conceptual pipeline is **not demonstrated**.
