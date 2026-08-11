# Source of truth — subsystem inventory

Compiled by direct inspection of code and frozen artifacts. Every value below was read
from the file cited, not from an earlier report. Where a value could not be confirmed it
is marked UNVERIFIED rather than filled in.

Status key: **FROZEN** = current production, asserted in code or hashed;
**PROVISIONAL** = current but not yet sealed; **HISTORICAL** = superseded.

---

## Preprocessing

| subsystem | implementation | key values | status |
|---|---|---|---|
| Gas removal | `results/motif_pipeline/frozen_gas_m3_pipeline_v1/gas_m3_pipeline.py` → `radius_normalized_coord_r4_gas_screen` | radius-normalised separation `s = d_ij/(R_i+R_j)`; gas screen and physical components at **s < 1.62**; gas particles get `component_id = -1` | FROZEN |
| Physical contact graph | same file | contact cutoff **M_GRAPH = 1.35** (`s < 1.35`), evaluated over same-component neighbours | FROZEN |
| Reliable-centre (M3) selection | same file, `_exposure_and_coord` | exclude a particle **as a SOAP centre** iff `exposure > TAU_E` AND `same_coord < TAU_C`, with **TAU_E = 0.14**, **TAU_C = 13**. `exposure = ‖Σ_j unit(r_j − r_i)‖ / same_coord`, set to 1.0 for an isolated particle | FROZEN |
| Selector provenance | file header | selector named `phase_blind_coord_exposure_v1` | FROZEN |

**Critical distinction, stated in the code header:** M3 excludes particles from being SOAP
*centres*. Excluded particles remain in the structure and in the SOAP *neighbour context*.
This is not deletion. `core3_pipeline.soap_global_d0` passes **every non-gas particle** as
context (`context = np.flatnonzero(prep.retain_mask)`) while restricting centres.

---

## Representation

| subsystem | implementation | key values | status |
|---|---|---|---|
| SOAP | `results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts/core3_pipeline.py`, `_soap()` via DScribe | species `Z = [11, 17]` (placeholders for LARGE/SMALL); **n_max = 8**, **l_max = 6**; `r_cut = 3.0·d0`, `σ = 0.10·d0` | FROZEN |
| Dimensionality | same, `SOAP_DIMENSION = 952` | **952**, asserted at runtime: `soap.get_number_of_features() != SOAP_DIMENSION` raises. Consistent with DScribe crossover: `(N_s·n_max)(N_s·n_max+1)/2 × (l_max+1) = (16)(17)/2 × 7 = 952` | FROZEN |
| Length scale `d0` | same, `component_d0()` | **median nearest-neighbour distance within the physical component**. So `r_cut = 3.0·d0` is three neighbour shells, not an absolute length | FROZEN |
| Representation variant | same, `REPRESENTATIONS` | production is **`GLOBAL_D0`**; `PAIR_CONTACT` and `ABSOLUTE_DISTANCE_CONTROL` are controls | FROZEN |
| Normalisation | `core3_comparator_benchmark_v1/scripts/development_scoring_common.py`, `load_raw` | `C.l2_rows(x)` — L2 row normalisation applied before standardisation | FROZEN |

---

## Statistical detector — CURRENT (v3)

`results/motif_pipeline/core3_atlas_v3/`

| subsystem | implementation | key values | status |
|---|---|---|---|
| PCA | `scripts/v3_common.py`, `fit_transform()` | **PCA_DIM = 16**, `svd_solver="randomized"`, `iterated_power=5`, seeded | FROZEN (predeclared) |
| Standardisation | same | **species-conditioned**: mean and scale computed separately for LARGE and SMALL, applied per species before PCA | FROZEN |
| PCA fitting sample | same, `stratum_sample()` | every (family, condition-group, species) stratum contributes exactly **CENTER_CAP = 128** centres, quota-split evenly across regions in the stratum | FROZEN |
| Domain fingerprint | same, `summary()` | `h = [mean of LARGE latents ‖ mean of SMALL latents]` → **32-D** (2 × 16) | FROZEN |
| Covariance | same, `fit_covariance()` | Ledoit–Wolf on region summaries; **group-balanced**: every training condition group contributes exactly `GROUP_BALANCE_R = 4` summaries by deterministic cyclic tiling | FROZEN |
| Score | same, `mahal()` | `d(h) = sqrt((h−μ)ᵀ Σ_LW⁻¹ (h−μ))`; `ρ = d/τ`; reject if `ρ > 1` | FROZEN |
| Threshold | same, `threshold()` | `τ` = 95th percentile (`method="higher"`) of **per-condition-group mean** calibration scores; asserted that no non-expert region reaches calibration | FROZEN |
| Architecture | same, `fit_model()` | **hard condition-group split, NO group cross-fitting** | FROZEN |
| Predeclaration | `data/V3_PREDECLARATION.json` | split + metrics hashed **before any model existed**: `29c1cf5d25d758bc147b5992d257b10f7c68ee3d7a9d374992e4a66ac6330bcc` | FROZEN |

## Statistical detector — v2 (HISTORICAL for architecture, still cited for controls)

`results/motif_pipeline/core3_comparator_benchmark_v2_expanded/`
46 regions / 18 condition groups; PCA d16 **and** d32; **group cross-fitting**; methods
`mahalanobis_particle`, `mmd`, `global_lw_d16`, `global_lw_d32`, `hybrid`.
Stage-C robustness on 24 Phase-4 Method-A controls. Frozen global threshold
`9.540582396555283`. Status: **HISTORICAL** as an architecture; its 24-control panel is
still the control set re-scored under v3.

---

## Physical matcher — CURRENT

`results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1/build_and_run_matcher.py`

| subsystem | key values | status |
|---|---|---|
| Library | 7 families → `crystal_stability` runs: CsCl→`cscl`, Th3P4→`th3p4`, Cu3Au→`cu3au_d6`, Ni3Sn→`ni3sn_d019`, zangenite→`zangenite_deep8`, Cr3Si_A15→`cr3si_a15_deep8`, rocksalt_B1→`rocksalt_b1_deep8` | FROZEN |
| Anchor | `first_frame.gsd` at frame 0, `extract_domain=False` — the CIF-ideal realisation | FROZEN |
| Thermal envelope | `trajectory.gsd`, `ENVELOPE_FRAMES = range(250, 501, 25)` → **11 frames per family** | FROZEN |
| Domain extraction | `gas_safe_domain()` — coordination-only gas rejection (`coord ≥ 4`), then largest connected condensed cluster, keeping **every** particle in it. Explicitly **not** q6-thresholded (documented as species-biased) | FROZEN |
| Channels | `CHANNELS = ("rdf_species", "rdf_total", "sk_shape")` | FROZEN |
| Channel scaling | `anchor_channel_scales()` — median pairwise **inter-anchor** distance per channel, a GLOBAL scale, not per-family | FROZEN |
| Ranking statistic | `D = mean_c [ d_c / scale_c ]` — dimensionless; `D ≈ 1` means "as different as two unrelated prototypes" | FROZEN |
| Acceptance | `S_f = max_c ρ_{f,c}` with `ρ_{f,c} = d_c/τ_{f,c}`; accept iff `D < ACCEPT_D` and margin ≥ `MARGIN_MIN` | FROZEN |
| Constants | `ACCEPT_D = 0.40`, `MARGIN_MIN = 1.15`, `TAU_QUANTILE = 0.90`, `TAU_INFLATE = 1.30`, `STOICH_TOL = 0.15` | FROZEN |
| Stoichiometry gate | `FAMILY_STOICH` L/S: CsCl 1.0, rocksalt_B1 1.0, Th3P4 0.75, zangenite 0.75, Cu3Au 1/3, Ni3Sn 1/3, Cr3Si_A15 1/3 | FROZEN |

### Matcher sub-components

| subsystem | implementation | status |
|---|---|---|
| Partial RDF | `matcher_v2_corrected_rdf/partial_rdf.py` — `g_ab(r) = N_ab(r) / [N_a^ctr · ρ_b · 4πr²Δr · S_a(r)]`, with **shell-containment correction** `S_a(r)` sampled over `N_SHAPE_DIRS = 26` directions; `R_MAX_NN = 4.0`, `R_MIN_NN = 0.5`, `N_BINS = 100`, `MAX_CENTRES = 2000`; local density estimated in a `2·nn` ball | FROZEN |
| S(k) | `matcher_v3_validated_structure_factor/structure_factor.py` | FROZEN — \TODO{VERIFY: exact k-vector construction and powder averaging not yet read line-by-line} |
| EMD | `build_and_run_matcher.py`, `emd()` | `scipy.stats.wasserstein_distance(grid, grid, a/Σa, b/Σb)` — both distributions on the **same support**, so this is the 1-D closed form `∫|F_P−F_Q|dx`, NOT a general multidimensional transport solve | FROZEN |

---

## Reference library

| subsystem | implementation | status |
|---|---|---|
| CIF catalog | `crystal_discovery/cif_library/catalog.csv` — **43** distinct CIFs, space group determined **from coordinates**, axis groups from the patched `LatticeBuilder` | PROVISIONAL (rebuilt 2026-08-07 after the slug defect below) |
| Coverage | `audit_coverage.csv` — 31/31 target prototypes present. Matching is by space group + stoichiometry, which cannot separate fluorite/antifluorite, Cu3Au/ReO3, Ni3Sn/Li3Fe, C14/C36 | PROVISIONAL |
| Build validity | `audit_build_validity.csv` — all **122** built runs cohesive, nn/contact 1.02–1.11 | PROVISIONAL |
| Lattice builder | `hockyg/PACSim@582a2d0`, opt-in `anisotropic_energy` | **CONFLICT — see below** |

### Open conflict: anisotropic lattice scaling

The builder that produced every current reference optimised a **single uniform scale
factor**, which preserves the CIF's axial ratios. That is correct for cubic cells (one free
parameter) and wrong for non-cubic. Measured effect of the fix:

- **Ni3Sn** (hexagonal): anchor **unchanged**, bit-identical.
- **Zangenite** (orthorhombic): a −9.2%, b −6.9%, c +12.6%; **c/a 0.480 → 0.596**.

Zangenite is the nearest surviving reference for all 23 deleted-CsCl domains and is the
runner-up that sets the margin for the external candidate, so current matcher numbers that
depend on the zangenite anchor carry a live caveat. 19 of 42 library CIFs are non-cubic.

\TODO{VERIFY: whether the zangenite thermal envelope must also be regenerated — the
envelope frames come from an MD trajectory started from the pre-fix lattice.}

### Defect found and fixed 2026-08-07: the design screen was still isotropic

The anisotropic fix had been applied only to the *builder*. `crystal_stability/tune.py`
`evaluate()` — which chooses the surface potentials and Debye length that every build is
then run with — still scanned a **single uniform scale factor**. So the screen selected
colloid parameters for a lattice the build would not produce, for all 19 non-cubic CIFs.

Measured on the AlB2 case documented in the PACSim commit (200 nm / 120 nm colloids):

| | uniform screen | anisotropic screen |
|---|---|---|
| lattice energy | −1.20 kT/particle | −60.96 kT/particle |
| weakest species | B at **+2.4 kT** (unbound) | B at −170.1 kT |
| c/a | 1.0906 (the atomic ratio) | 0.9436 |

`evaluate()` now calls `LatticeBuilder._optimize_anisotropic_scale_factors` directly
rather than reimplementing the descent. This matters: an equivalent-looking
reimplementation that seeded on the best uniform scale instead of on the geometric scale
converged to c/a 0.899 rather than the builder's 0.944, because coordinate descent is not
convex. Screen and builder now agree bit-for-bit on AlB2, CsCl, MgZn2 and Ni3Sn.

Side effect: the OpenMM system is now built once per screen instead of once per candidate,
taking a single-prototype screen from ~5 min to ~2 s.

### Defect found and fixed 2026-08-07: the catalog held the wrong Th3P4

`build_catalog.py` disambiguated same-stem CIFs by counting `paths[0].stem` — the first
path the tree walk happened to reach — instead of the *canonical* path's stem. The count
therefore missed the Th3P4 collision, and `prototypes/Th3P4.cif` was silently overwritten
by PACSim's **test fixture** Th3P4 rather than the CIF the EMD matcher reads. Now split
into `Th3P4__8edd1f72` (`crystal_stability/structures/Th3P4.cif`, the matcher's) and
`Th3P4__72406369` (the fixture). Any Th3P4 result derived from `prototypes/` before this
date is suspect; results derived from `crystal_stability/structures/` are unaffected.

---

## Firewalls

| firewall | scope |
|---|---|
| Untouched test | six condition groups never opened: `debye5.4_vf.010890058823`, `debye5.3_vf.024130533489`, `debye5.5_vf.002704544521`, `debye5.5_vf.005575710117`, `debye5.5_vf.008655804247`, `debye5.5_vf.017591158913` |
| Sealed mapping | `crystal_discovery/crystal_stability/library_stability.csv` — Steven↔prototype mapping, not read |
| Provisional labels | in v3-B, GAT-provisional labels are confined to **training**; calibration and validation are 100 % expert-labelled, asserted in `v3_common.load_cohort` and `threshold()` |

---

## Not yet located

- GAT baseline implementation and whether any strict-retraining LOFO exists.
  \TODO{VERIFY: `reports/gat_pipeline/` describes a GAT pipeline; the training code path and
  whether the held-out-family comparison retrains or only re-fits statistics is unconfirmed.}
- A sequential detect→match→add→recognise expansion experiment. Not found.
