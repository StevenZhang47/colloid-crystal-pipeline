# radius_normalized_coord_r4_gas_screen_v1 — standardized default gas removal

One shared, importable implementation of the validated **radius-normalized coordination +
mean_r4\*** hybrid. This is the default gas-removal method for all future crystal-pipeline work. It
**replaces** raw-distance coordination thresholds, the `1.35 × median-NN` cutoff, bare `CN < 4`,
SANN / mSANN / SANNR, and "keep only the largest connected component" as a gas substitute.

**Constants are frozen and must not be recalibrated here.**

## The frozen rule

```
s_ij  = d_ij / (R_i + R_j)                          # pair-radius-normalized (dimensionless)
CN_i  = #{ j != i : s_ij < 1.62 }                   # strict '<' ; all AA/AB/BB count equally
mean_r4*(i) = mean over the 4 PHYSICAL-nearest neighbours of  d(i,j_k)/(R_i + R_{j_k})

if   CN_i < 2:                 remove   obvious_gas
elif CN_i >= 4:                retain   supported_material
else  (CN_i in {2,3}):
     if mean_r4*(i) > 1.2698:  remove   gas_like_ambiguous          # strict '>' ; tie retains
     else:                     retain   compact_ambiguous
```

`R_i = diameter_i / 2`. Radii may differ per particle; a valid positive radius is required (else the
particle is flagged `invalid_radius` and never removed). Neighbours are selected by **physical**
distance, then each selected pair is normalized by **its own** `R_i+R_j` (not a global denominator,
not sorted by normalized distance, not the 4th-neighbour alone). Frames with `<4` available
neighbours for a CN∈{2,3} particle return `insufficient_neighbors` rather than inventing neighbours.

## Usage

```python
import sys; sys.path.insert(0, ".../scripts/gas_screen_v1")
from radius_normalized_coord_r4_gas_screen import radius_normalized_coord_r4_gas_screen as gas_screen

res = gas_screen(positions=pos, radii=diam/2, box=box, type_ids=tid)   # box: len-3 / len-6 / 3x3
retain = res.retain_mask                 # bool per particle
res.coordination_1p62, res.mean_r4_star, res.r1_star ... res.r4_star
res.roles, res.reason_code, res.component_id
res.metadata                             # frame-level diagnostics (see below)
res.to_dataframe()                       # per-particle table
```

- **PBC:** orthorhombic boxes → periodic `scipy.cKDTree(boxsize)`; triclinic/tilted HOOMD boxes →
  `freud` with the full box matrix (xy/xz/yz tilt preserved). The neighbour structure is only an
  accelerator; the criterion is the explicit `s_ij < 1.62`. `metadata["pbc_handling"]` records which.
- **Gas removal only.** It does **not** identify surfaces, cores, grains, domains, or phase (a CN=4/5
  particle is `supported_material`, never "surface"). Surface/domain detection is separate code.
- **Connected components** are computed on retained particles for *diagnostic flags and IDs only*
  (`component_id`, `retained_compact_component`, `retained_small_nucleus_candidate`) — never used to
  filter, and never "keep largest only". Gas prediction happens **before** any component logic.
- **Optional retain-flags** (`retained_compact_component`, `retained_near_surface_uncertain`,
  `retained_small_nucleus_candidate`) are advisory; they never cause removal.

## Frame-level diagnostics (`res.metadata`)

total particles; number/fraction removed; number removed via `CN<2`; number removed via the
`mean_r4*` tiebreaker; coordination histogram; `mean_r4*` distribution for CN=2 and CN=3; counts and
removals by type; box lengths/tilts; and whether orthorhombic or triclinic PBC handling was used.

## Config

`radius_normalized_coord_r4_gas_screen_v1.yaml` (contact_multiplier 1.62, ambiguous {2,3}, supported
≥4, threshold 1.2698, comparison `>`, `evidence_status: provisional`, compact/near-surface =
retain_and_flag).

## Validated behaviour (synthetic-exact, frozen from `results/motif_pipeline/sann_gas_screen_v2_final`)

- worst-system gas recall ≈ **0.862**; exact crystal-core removal **0 %**; exact surface removal
  ≈ **0.2 %**; fluorite gas recall ≈ **0.970**; difficult near-surface (s≤1.3) gas recall ≈ **0.479**;
  amorphous dense-region removal ≈ **1 %**; within the CN∈{2,3} subset **2693/2711 ≈ 99.3 %** of gas
  removed with essentially no crystal material lost.

**Known, accepted blind spot:** compact **same-species** clusters of size 6–8 self-coordinate to
CN≥4 and are **retained** — indistinguishable from small legitimate nuclei, which the conservative
objective prioritizes preserving. Retained means "not safely identifiable as gas by this local
screen," **not** "confirmed material."

**Provisional:** exact gas truth is mainly synthetically constructed; the threshold stays provisional
until real particle-level gas labels exist. Do not recalibrate ad hoc.

## Implementation tests — `test_gas_screen.py` (20/20 pass)

| test | result |
|---|---|
| A uniform-scale invariance (×0.5,1,1.5,2) | CN, mean_r4*, r*_star, decisions identical |
| B species-radius sensitivity | changing one species radius changes s/CN (83 particles) |
| C PBC min-image | across-boundary neighbour → r1 = 0.20 (not 9.8), counted in CN |
| D physical-then-normalize ordering | selects 4 by physical distance; r1_star = physical-nearest's own normalization (1.667), not the normalized-nearest (0.8) |
| E boundaries + strict `>` | CN=1→remove, CN=4→retain, CN=3 flips at threshold; ambiguous removal == (mean_r4* > 1.2698) exactly (tie retains) |
| F frozen-reference reproduction vs H3 | rule: **0 mismatches / 63,581** H3-evaluated particles; end-to-end geometry: **0 / 3,180** (CsCl, Th3P4, fluorite finite) |

## Migration report

**Previous gas-removal call sites discovered (all in frozen or completed code):**

| call site | old rule | disposition |
|---|---|---|
| `build_particle_graph_dataset_v6.py::particle_qc` | `coord < 4` gas mask + 1.35×NN cutoff + largest-component | **left unchanged** — produced the frozen GAT v4-QC model; also contains surface detection (forbidden to modify) |
| `build_particle_graph_dataset_v3/v4/v5.py` | same family | **left unchanged** — superseded by v6; historical |
| `run_mahalanobis_verified_ood_controls_v1.py::resolver_core_mask` | `coord≥4` + largest dense component, core=dense&coord≥6 | **left unchanged** — feeds the locked Mahalanobis detector (threshold 8.419550193355859); modifying it changes a frozen threshold (forbidden) |
| `match_multi_domain.py::order_mask` | `coord≥4` | **left unchanged** — structural-domain logic (explicitly forbidden to modify) |
| `example1_gat_free_common_v1.py`, `extract_and_audit_external_zanganite_cores_v2.py`, `diagnose_mixed_domains_v1.py`, `pilot_unsupervised_atlas_v1.py`, `analyze_*`, `resolve_dir2_*` | assorted coord/1.35×/largest-component | **left unchanged** — completed diagnostics / one-offs |
| `results/motif_pipeline/coordination_*`, `sann_gas_screen_v2_final/` | validation studies (this method's own derivation) | **left unchanged** — completed result directories |

**Migrated in place: none.** Every active production site is a frozen artifact whose preprocessing
the task forbids changing (Mahalanobis threshold, GAT model, structural-domain / surface logic). The
standardization is therefore forward-facing: **all new pipeline code, and any future GAT retrain or
atlas re-export, must import this module** rather than re-implement a gas rule. When the atlas is
next re-exported (a separate, explicitly-authorized task), swap its gas mask to `retain_mask` from
this module so training atlas, validation, test candidates, known controls, and OOD structures all
share one gas preprocessing (task §11).

**Not modified by this task:** normalized SOAP definitions; atlas scaler/PCA/covariance; Mahalanobis
thresholds; structural-domain logic; surface-detection logic.
