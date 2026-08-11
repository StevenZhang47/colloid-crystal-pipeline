# Final architecture-decision study: data splits

Status: **preregistered; no study model has been run**  
Study root: `/projects/rps/gmh4/hockygroup/sz5808/results/final_architecture_decision_v1`

## 1. Unit of independence

The independent unit is a source condition/trajectory (or a single ideal prototype), not an extracted frame, center, particle, or region. Correlated late frames from one run may improve numerical estimates inside that run, but they count as one group in selection, uncertainty, and reporting. No result may claim a sample size obtained by counting correlated frames as independent observations.

All fitting and threshold estimates are group-aware. Any fold that holds out a family removes every member of that family from representation training, fitted score models, calibration, reference banks, and early stopping.

## 2. Frozen source map

| Source | Intended role in this study | Independence/provenance rule |
|---|---|---|
| `results/motif_pipeline/core3_atlas_v3/data/v3_region_manifest.csv`, `v3_group_split.csv`, and the v3-A products under the same study | Core-three expert-only development/calibration source | Use the existing v3-A group roles: 126 train regions/68 groups, 70 calibration regions/19 groups, and 50 validation regions/14 groups. Provisional v3-B additions are excluded. |
| `results/statistical_ood_benchmark_v2/expanded_selfmatch_anchors.csv` and its referenced CIF/trajectory sources | Prototype retrieval candidates and geometric anchors | Collapse aliases to one canonical prototype. Do not let the original plus library copy of the same prototype become independent observations. |
| `results/statistical_ood_benchmark_v2/verified_thermal_reference_library_v1/source_inventory.csv` | Independent stable thermal runs for known-family calibration, held-out-family candidates, and incremental updates | Frames 300/350/400/450/500 from a run remain one group. Seed 1, seed 11, and seed 12 are distinct runs. |
| Existing corrected GAT LOFO artifacts under `experiments/gat_vs_soap_ood/strict_lofo_v2/` | Historical supervised baseline and implementation check | Only the corrected three-class, no-held-out-node models are admissible. Original four-output/message-passing-leaky LOFO is audit-only. |
| Four gas-only controls recorded in `results/motif_pipeline/soap_atlas_core3_phase2_v1/data/source_trajectory_audit.csv` | Open preprocessing negatives | Never treated as coherent positive structures or threshold-calibration units. |
| Ideal low-coordination sources under `crystal_discovery/crystal_stability/runs/`: `lib_reo3_d09/first_frame.gsd`, `lib_cuprite_c3/first_frame.gsd`, `zangenite_deep8/first_frame.gsd`, and `th3p4/first_frame.gsd` | Open preprocessing preservation audit | ReO3 dynamic trajectories are not stable positive examples; its ideal anchor is only a geometric preservation test. |
| `crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set/steven_example_1.gsd` and `steven_example_2.gsd` | Final external challenge after architecture freeze | Historically viewed/scored; therefore “prospectively sealed from this preregistration,” not epistemically untouched. They cannot influence any fit, choice, early stop, or threshold. |
| All Phase-5 external candidates, untouched-test payloads, fused examples, and candidate-novel v3 payloads | Out of development | Remain unopened during this architecture decision unless a later, separately approved final external step is reached. |

Paths containing `...` in the table are descriptive abbreviations only. Execution manifests must resolve and hash the complete absolute source path before a job runs.

## 3. Canonical prototype policy

Before scoring, the 44-row expanded anchor list is converted into a canonical table containing one physical anchor per prototype. Known aliases are grouped as follows:

| Canonical family/prototype | Aliases that must remain in the same group |
|---|---|
| CsCl/B2 | `cscl`, `cscl_b2_anchor`, and the original CsCl copy |
| Th3P4 | the two hashed Th3P4 entries, `th3p4_anchor`, the Materials Project primitive entry, and the original Th3P4 copy |
| Cu3Au/L12 | `cu3au_l12`, its Materials Project conventional entry, and the original Cu3Au copy |
| Ni3Sn/D019 | `ni3sn_d019`, `li3fe`, and the original Ni3Sn copy |

The canonical map is frozen and hashed before any representation is scored. `Li3Fe` is treated consistently as the Ni3Sn/D019 structural prototype for this study; it must not simultaneously act as a distinct negative. The exact same canonical-anchor intersection is used for SOAP, the supervised GAT diagnostic, and the self-supervised encoder.

The old reported nearest false distance is not imported as a threshold. The old `D=0.40` acceptance rule is retired, and the conflict between the reported 0.359 hard neighbor and the 0.1837 recommendation output is resolved by recomputing the canonical development table without external candidates.

## 4. Representation-unseen family folds

There are exactly four primary family-removal folds:

| Fold | Entirely unseen representation family | Active known families for OOD scoring | Why this fold is required |
|---|---|---|---|
| H-CSCL | CsCl/B2 | Th3P4 and amorphous | Dense ordered binary crystal and known hard relation to BiF3-like anchors |
| H-TH3P4 | Th3P4 | CsCl and amorphous | Open-framework family and the historical catastrophic per-particle Mahalanobis failure |
| H-CU3AU | Cu3Au/L12 | CsCl, Th3P4, and amorphous | Ordered alloy with the known Al3Ti/D022 retrieval confusion |
| H-NI3SN | Ni3Sn/D019 | CsCl, Th3P4, and amorphous | Ordered hexagonal prototype and the canonical Li3Fe/D019 ambiguity |

For a held-out fold, “entirely unseen” means exclusion from:

- self-supervised encoder training and validation;
- supervised baseline training, where a corrected model exists;
- normalization, PCA, covariance, bandwidth, and all score-model fitting;
- threshold calibration and early stopping;
- reference/prototype banks and all augmented views;
- mined negatives, batches, cached embeddings, and fitted preprocessing statistics.

Family names may be used only to build the exclusion manifest and evaluate the frozen predictions. They are not inputs to the self-supervised objective.

## 5. Self-supervised pretraining split

The common pretraining pool consists of the v3-A expert-only train and calibration source groups, the canonical non-held-out prototype anchors, and non-held-out stable thermal runs. The v3-A validation groups, held-out family, all external candidates, and all study evaluation candidates remain excluded.

Within each fold:

- training groups fit the encoder;
- a deterministic 15% group holdback from the eligible training pool monitors self-supervised loss only;
- no class-identity accuracy, retrieval result, OOD result, or external score controls training duration;
- augmentations inherit the source group and can never cross a fold boundary.

The deterministic holdback is produced by sorting the SHA-256 of `fold_id || source_group_id` and taking the lowest 15%. Its resolved membership and hash are written before the first GPU submission.

## 6. Candidate, incremental-update, and second-test roles

For each held-out family in the verified thermal library:

| Role | Frozen source | Permitted use |
|---|---|---|
| Initial representation-unseen candidate | seed 1 run | First OOD/retrieval measurement after the fold encoder, downstream model, threshold, and prototype bank are frozen |
| Small reference update | seed 11 run | Add one independent run to the reference/statistical model without retraining the encoder |
| Independent post-update test | seed 12 run | Test whether the update generalizes; never used to choose the update or threshold |

Each run is summarized once at the run level. Per-frame scores may be plotted as dependent descriptive trajectories but do not inflate denominators or confidence intervals.

The update test asks whether a frozen representation supports adding a family cheaply. Encoder weights and feature normalization remain frozen. Only the explicitly permitted downstream reference/model update is allowed.

## 7. Prototype-retrieval split

Retrieval is tested only after the encoder and canonical prototype map are frozen. The correct held-out prototype is absent from encoder training but is inserted into the retrieval bank at evaluation time. This separates representation transfer from memorization.

For every candidate, report:

- top-1 and complete rank of the correct prototype;
- correct-prototype distance, nearest-wrong distance, and their ratio;
- the exact nearest hard neighbor;
- the result after removing the correct prototype: abstention versus confident wrong match.

The preregistered hard-neighbor audits are CsCl versus BiF3, Th3P4 versus zangenite/open-framework neighbors, Cu3Au versus Al3Ti/D022, and Ni3Sn versus the closest automatically determined non-equivalent canonical prototype. The Ni3Sn neighbor is determined from the sealed canonical development bank rather than hard-coded after looking at its candidate.

## 8. Open preprocessing split

Preprocessing is evaluated before representation training, using only open controls:

- dense positive controls: ideal/stable CsCl, Cu3Au, rocksalt where available;
- low-coordination positive controls: ideal ReO3, cuprite, zangenite, and Th3P4;
- negative controls: the four documented gas-only trajectories;
- mixed gas/coherent controls only when their ground-truth role was already open before this preregistration.

Two and only two screens are compared:

1. the current four-neighbor frozen gas screen;
2. one preregistered cohesive-component alternative: seed edges use pair-contact distance `s <= 1.35`, seeded components expand over the frozen `s < 1.62` contact graph, components require at least 20 particles, and every particle in a retained component is kept.

No cutoff sweep is permitted. Connectivity persistence from `s=1.35` to `s=1.62` is descriptive, not a tuning axis. Success requires retaining the low-coordination positive while keeping all four gas-only controls rejected and not materially damaging the dense controls. If that fails, the final report records a preprocessing scope limitation and restricts representation claims to a common pre-extracted coherent-domain input.

## 9. Sealed data and access order

The access order is fixed:

1. open preprocessing controls;
2. non-held-out training and self-supervised validation groups;
3. family-removal candidate seed 1;
4. incremental reference seed 11;
5. independent post-update seed 12;
6. only after architecture and thresholds are frozen, any separately approved external-candidate step.

During this study, the following remain closed unless their numbered step is reached: v3 untouched test trajectories, v3 candidate-novel payloads, Phase-5 external OOD coordinates and scores, fused FEW/MANY and blocked ZERO children, prototype identities not already present in the open canonical library, RDF/S(k) external panels, and the Steven candidates. Existing historical outputs are provenance evidence only and cannot be queried to make a decision.

## 10. Required executable manifests

Before submission, code must materialize and hash:

- `data/source_manifest.csv`: absolute path, source group, family, construction, role, frame selection, and file SHA-256;
- `data/canonical_prototypes.csv`: alias-to-canonical mapping and selected anchor;
- `data/fold_membership.csv`: every source group’s status in every fold;
- `data/pretraining_holdback.csv`: deterministic SSL training/validation allocation;
- `data/access_ledger.csv`: timestamped file access purpose and phase;
- `data/sealed_paths.txt`: path patterns forbidden before final freeze.

Any source not present in the frozen manifest is ineligible. Missing or unreadable data cause an explicit `DEPENDENCY_BLOCKED` or `SOURCE_MISSING` result; they do not justify substitution after scores are known.
