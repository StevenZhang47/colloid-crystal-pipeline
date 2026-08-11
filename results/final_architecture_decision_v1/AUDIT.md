# Final Architecture Decision v1 — Phase 0 Evidence Audit

**Audit date:** 2026-08-11  
**Workspace:** `/projects/rps/gmh4/hockygroup/sz5808`  
**Study directory:** `/projects/rps/gmh4/hockygroup/sz5808/results/final_architecture_decision_v1`  
**Status:** audit and preregistration only; no new model has been trained or scored.

## 1. Scope and authority

This study compares exactly three representation philosophies:

1. fixed species-aware SOAP;
2. the existing supervised GAT/GAT hidden representation;
3. one modest self-supervised geometric graph encoder.

It does not authorize a model zoo, broad hyperparameter search, test-set tuning, or changes to any frozen production result. All new files must remain inside this study directory.

The audit distinguishes frozen implementations, completed evidence, diagnostic evidence, data sealed from this preregistration, and historically viewed data. Prior access cannot be undone or redescribed as an untouched test.

## 2. Source-of-truth map

### 2.1 Frozen gas and M3 preprocessing

Implementation:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/frozen_gas_m3_pipeline_v1/gas_m3_pipeline.py`

SHA-256:

`9244c1527f857d46fcd280c15bcd8985c7436685d1b5ed81cdd6df46b1c40696`

| Quantity | Value |
|---|---:|
| gas graph/contact threshold | `s < 1.62` |
| M3 graph multiplier | `1.35` |
| M3 exposure threshold | `0.14` |
| M3 same-coordination threshold | `13` |

This remains the production incumbent. Phase 2 may test one study-local repair for low-coordination cohesive structures, but may not overwrite this file or its prior results.

### 2.2 Frozen SOAP representation

Implementation:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/soap_atlas_core3_phase2_v1/scripts/core3_pipeline.py`

SHA-256:

`acc9ae34d3b4bc22b15ffc8d535d7689e1bd86144c9cf280fe5092798888c356`

Descriptor:

```text
952 dimensions
canonical species LARGE=11 and SMALL=17
n_max=8
l_max=6
r_cut=3.0 d0
sigma=0.10 d0
per-particle SOAP
GLOBAL_D0 scaling
```

Parameter hash:

`5723502cb4c59b0e4faeb2533c54d1c697886798b7fd43be7179eb425fe490c6`

For multi-domain frames, `d0` must be calculated per predicted structural region, never once for the whole GSD/box. SOAP centers use the full non-gas physical-component context. This policy is inherited unchanged from Phase 4.

### 2.3 Current SOAP + Mahalanobis OOD evidence

There are three generations of evidence. They must not be conflated.

#### Historical Phase 5 v1

Workspace:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1`

The per-particle median Mahalanobis variant failed strict held-out Th3P4 decisively. MMD and REMatch passed the three small LOFO folds, but robustness acceptance was poor. This is historical evidence about the per-particle representation, not the present incumbent.

#### Expanded comparator v2

Workspace:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v2_expanded`

Primary report:

`reports/V2_EXPANDED_LOFO_RESULTS.md`

The decisive change was from per-particle distribution scoring to a species-conditioned region-mean summary. Global Ledoit–Wolf Mahalanobis then achieved strict-LOFO AUROC/TPR `1.0/1.0` for CsCl, Th3P4, and amorphous. It accepted 7/8 clean Phase-4 controls and 6/16 graded perturbations, outperforming MMD on robustness. The report recommends global region-summary Ledoit–Wolf as the production comparator and rejects per-particle Mahalanobis as the primary OOD gate.

The safer dimensionality is PCA16 per species, yielding a 32-dimensional `[LARGE|SMALL]` summary. The higher-dimensional version was poorly conditioned and behaved partly as a subspace-residual detector.

#### Atlas v3, the newest controlled evidence

Workspace:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3`

| File | SHA-256 |
|---|---|
| `scripts/v3_common.py` | `3634f2af441a09fc77180be17c83489c04def94b6b1569b6c89730fd4899f73f` |
| `scripts/v3_03_compare.py` | `aa4128a82204dc0351799122851aaa17095054e07719eebd6fe16690d24a0c77` |

Primary report:

`reports/V3_A_VS_B_COMPARISON.md`

The v3 study re-featurized 386 trajectories. All frozen gas/M3/SOAP checks passed, and 49 overlapping trajectories reproduced the phase-2 descriptors bit-for-bit.

| Cohort | Regions | Groups | Training | Calibration | Validation |
|---|---:|---:|---:|---:|---:|
| v3-A, expert-supported only | 246 | 101 | 126 / 68 groups | 70 / 19 groups | 50 / 14 groups |
| v3-B, provisional labels added only to training | 380 | 103 | 260 / 70 groups | same 70 / 19 | same 50 / 14 |

Reserved and unopened in that study: six untouched-test condition groups containing 24 trajectories, plus a six-trajectory candidate-novel panel.

Both A and B achieved AUROC/AP/TPR `1.0/1.0/1.0` for all three strict family holdouts. Cohort A had better worst-fold known FPR and substantially more stable thresholds. Provisional labels added no meaningful benefit and degraded amorphous LOFO FPR and threshold stability. Cohort A is therefore the correct expert-only data source for this new study.

Important unresolved point: v3 explicitly states **no model selected**. Its preregistered threshold over-rejected validation regions, and a p99 calibration-region option was only costed, not adopted. This new study must not pretend that an immutable v3 production threshold exists. It will preserve the v3-A representation/fitting protocol and use a study-local, method-symmetric 5% known-FPR operating rule without modifying v3.

### 2.4 Existing supervised GAT

Production training implementation:

`/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/scripts/motif_pipeline/train_pyg_gat_particle_classifier_v7.py`

SHA-256:

`6f2e0178d38c25350c4dc1ce406982dd79424b6e87e2fd1d3683aa9156830a7a`

Checkpoint and summary:

```text
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/results/motif_pipeline/pyg_gat_v7_stratified/gat_model.pt
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/results/motif_pipeline/pyg_gat_v7_stratified/gat_summary.json
```

Architecture/training facts:

```text
three GATv2 layers; hidden width 64; four heads; dropout 0.15
11 node features; four edge features
AdamW, lr 1e-3, weight decay 1e-4
120 epochs; batch size 4; inverse-frequency class weighting
720 train graphs; 156 validation graphs; six held-out reference graphs
```

The classifier was supervised on gas, amorphous, CsCl, and Th3P4. Its class probabilities are not a valid open-world score.

Corrected strict-LOFO implementation/results:

`/projects/rps/gmh4/hockygroup/sz5808/experiments/gat_vs_soap_ood/strict_lofo_v2/`

Read:

```text
STRICT_LOFO_V2_AUDIT.md
corrected_two_family_report.md
primary_table.csv
per_heldout_domain_scores.csv
```

The original GAT LOFO was invalid because it retained a dead output neuron and allowed held-out-family nodes to participate in message passing. Corrected v2 used a true three-class head and dropped every graph containing any held-out-family node.

Corrected result:

- GAT hidden/domain-mean embedding + global Mahalanobis detected 17/17 held-out Th3P4 and 23/23 held-out CsCl for all three seeds, matching SOAP on those two families.
- GAT classifier confidence remained unsafe: CsCl confidence AUROC was below 0.3 and often inverted.
- Amorphous was not rerun under the corrected protocol.
- The corrected study used its own 70-domain regime and did not perform CIF retrieval.

Therefore, GAT hidden embeddings are a legitimate diagnostic representation; GAT softmax/entropy/margin scores are not candidate production OOD detectors.

### 2.5 Current physical RDF/S(k)/EMD matcher

Workspace:

`/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1`

Primary implementation and SHA-256:

`build_and_run_matcher.py`  
`76f6c5af354515ae3f553dbcca301540cd989c7f8318a22ca7cf883ffc4890a2`

The current matcher uses species partial-RDF EMD, total-RDF EMD, deterministic powder-S(k) EMD, dimensionless coordinates, global per-channel anchor scales, and the mean normalized channel distance `D`.

The historical open-set rule was `D < 0.40` with runner-up/best margin at least `1.15`. The expanded library showed that `0.40` is too permissive.

Expanded self-match artifacts:

```text
expanded_library_selfmatch.py
expanded_selfmatch_anchors.csv
expanded_selfmatch_results.csv
expanded_selfmatch_full_ranking.csv
expanded_selfmatch_summary.json
```

Implementation SHA-256:

`1c6e220728c8947c081b53392a06c15497616cfee2be7dfe26928f7ec0b85899`

There are 24 clean thermal cases: four families × six late frames. Frames within one run are correlated and must not be counted as 24 independent realizations. The observed clean self-match maximum is `D = 0.2582`.

The expanded library exposes close confusers, notably Cu3Au versus Al3Ti D0_22 and CsCl versus BiF3. The old `0.40` rule can confidently accept the wrong close prototype after the correct prototype is removed.

### 2.6 Existing SOAP matcher diagnostic

Implementation and SHA-256:

`/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1/soap_vs_emd_matcher.py`  
`9b400266623c5ab6c3aa4fd7d6443282b13596cb45b59687aa946e275900240e`

Representation:

```text
frozen 952-D SOAP
per-particle L2 normalization
separate LARGE and SMALL means
concatenate to 1904 dimensions
L2-normalize domain fingerprint
Euclidean/cosine-equivalent distance
```

The diagnostic achieved 6/6 correct self-match for CsCl, Th3P4, Ni3Sn, and Cu3Au and showed strong separation from the nearest wrong prototype. It is not yet a fair production comparison because:

1. EMD scored 44 anchor rows, including seven `orig_*` anchors, while SOAP scored 37 `lib_*` anchors. Most missing rows are duplicate anchors, but the library identity was not literally identical.
2. SOAP ideal anchors bypassed the gas/M3 screen and used all particles, while thermal candidates used frozen gas/M3. EMD used its own full-domain extractor.
3. SOAP has no frozen acceptance threshold under realistic nuisance variation.
4. The prototype-equivalence map differs across saved artifacts.

The final study must create one canonical prototype table and one common eligible anchor set before any head-to-head claim.

### 2.7 Expanded CIF library

Primary inventory:

`/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1/expanded_selfmatch_anchors.csv`

It contains 44 anchor rows: 37 `lib_*` CIF-derived anchor runs plus seven `orig_*` stable-family anchors. Multiple anchors map to the same prototype.

Important equivalences already used in code include:

| Anchor aliases | Canonical prototype |
|---|---|
| `cscl`, `cscl_b2_anchor`, `orig_CsCl` | CsCl |
| four `th3p4_*` anchors plus `orig_Th3P4` | Th3P4 |
| `cu3au_l12`, `cu3au_mp-2258_conventional_standard`, `orig_Cu3Au` | Cu3Au |
| `ni3sn_d019`, `li3fe`, `orig_Ni3Sn` | Ni3Sn |
| `fluorite_c1`, `antifluorite_c1b` | fluorite under role relabeling |

The saved CSV does not consistently encode all equivalences; for example it lists `li3fe` as its own prototype, whereas `soap_vs_emd_matcher.py` maps it to Ni3Sn based on a StructureMatcher result. A canonical, hashed equivalence manifest is a mandatory pre-score artifact.

### 2.8 Independently simulated structural domains

Verified source inventory:

`/projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/verified_thermal_reference_library_v1/source_inventory.csv`

Seven stable families have ideal CIF anchors, independent thermal seeds 11 and 12, and a separate seed-1 candidate trajectory:

```text
CsCl
Th3P4
Cu3Au
Ni3Sn
zangenite
Cr3Si_A15
rocksalt_B1
```

Cu3Au additionally has a different-condition stable realization, `cu3au_l12_big`.

Each thermal seed contributes fixed late frames 300/350/400/450/500. These five frames are correlated blocks from one run; the independent sample size is the number of simulation seeds/runs.

This seven-family set is the cleanest source for representation-unseen training, calibration, retrieval, and incremental-addition tests.

### 2.9 Existing family-removal evidence

Two removal benchmarks exist:

1. `verified_matcher_benchmark_v1/reference_removal_results.csv`: seven same-construction families; 7/7 abstained, but stoichiometry and physical gates were load-bearing and true/removed distance distributions overlapped.
2. `reference_conditioned_matcher_v1/expanded_selfmatch_results.csv`: 24 thermal frames across four families with close expanded-library negatives.

The second is closer to this study, but its independent sample size is four candidate runs, not 24 frames.

### 2.10 External candidates and prior-access status

The two matcher candidates are:

```text
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set/steven_example_1.gsd
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set/steven_example_2.gsd
```

Earlier work reported nearest identities Ni3Sn and Cu3Au with EMD distances about 0.347 and 0.373. The script `recalibrate_accept_d.py` explicitly loaded and reported both candidates, although it did not fit thresholds from them.

Consequences:

- They are **sealed from this preregistration forward** and may not affect splits, architecture, augmentations, thresholds, early stopping, or branch termination.
- They are not epistemically untouched; prior results are known. The final report must say “held out from the final decision protocol,” not “never previously viewed.”
- Their next scoring may occur only once after every architecture and acceptance/abstention rule is hashed.

The broader Phase-5 D1 manifest remains at:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_comparator_benchmark_v1/data/external_evaluation_manifest.csv`

Those additional candidates are out of scope unless explicitly added in a later amendment.

### 2.11 Open/low-coordination preprocessing failure

The clearest mechanistic record is in `soap_vs_emd_matcher.py`.

The frozen gas screen removes particle `i` when the mean normalized distance over its four physical-nearest neighbors exceeds approximately 1.2698. This assumes a cohesive particle has roughly four nearby contacts.

That assumption is false for open networks. On the ideal ReO3-like anchor, whose majority site has coordination two, the screen removed the entire majority sublattice: 5,324 particles entered and only 1,330 remained, exactly the LARGE-particle count. Nine of 37 cohesive ideal anchors were lost in the earlier screened SOAP attempt, including zangenite.

Available open-structure ideal anchors:

```text
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/runs/lib_reo3_d09/first_frame.gsd
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/runs/lib_cuprite_c3/first_frame.gsd
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/runs/zangenite_deep8/first_frame.gsd
/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/runs/th3p4/first_frame.gsd
```

Important qualification: the available dynamic `reo3_d09` simulations melted or dispersed and are failed-stability controls. They cannot serve as independent thermal ReO3 crystal positives. The ideal `lib_reo3_d09/first_frame.gsd` is valid for a geometric preprocessing audit only.

Four true gas-control trajectories are identified by `label_status=gas_only_control` in:

`/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/soap_atlas_core3_phase2_v1/data/source_trajectory_audit.csv`

### 2.12 Existing reliable low-coordination extraction idea

The verified thermal library already documents a safer physical pattern:

```text
identify solid-like seeds
then retain the connected component in a fuller neighbor graph
```

This preserved the low-coordination Th3P4 sublattice where a universal coordination cutoff failed. Phase 2 will test one analogous, radius-normalized cohesive-component screen rather than a parameter sweep.

## 3. Current train/calibration/test roles

The final study will inherit v3-A expert-supported roles:

| Role | Regions | Condition groups | Permitted use |
|---|---:|---:|---|
| training | 126 | 68 | representation/statistical fitting |
| calibration | 70 | 19 | thresholds and predefined early stopping only |
| validation | 50 | 14 | closed architecture comparison |
| untouched test | 24 trajectories | 6 | sealed until final architecture freeze |
| candidate-novel panel | 6 trajectories | — | sealed; not part of this checkpoint |

The self-supervised representation uses no crystal labels in its loss. Family/prototype labels are nevertheless required for leakage exclusion and evaluation bookkeeping.

## 4. Discrepancies that must be resolved before scoring

### 4.1 EMD hard-negative minimum

The prompt states:

```text
worst genuine D ≈ 0.2582
nearest false holdout D ≈ 0.3590
```

The genuine maximum is confirmed. The false minimum depends on canonical prototype grouping:

- Cu3Au → Al3Ti reaches `D = 0.3590`.
- `accept_d_recommendation.json` currently reports a global holdout minimum of `0.1837`, caused by an inconsistent Ni3Sn/Li3Fe prototype treatment in the recalibration path.

No EMD threshold may be chosen until a canonical equivalence manifest is frozen and the 24-case removal audit is recomputed without external candidates.

### 4.2 SOAP/EMD anchor fairness

The earlier comparison used nonidentical anchor rows and different preprocessing. The final study must use one canonical anchor per prototype and exactly the same successfully preprocessed candidate/reference domains for every representation. Non-overlap must be reported as a preprocessing result, not silently removed from one method.

### 4.3 OOD threshold status

v3-A is the correct expert-only source and representation protocol, but no final v3 absolute threshold was selected. The final study will compare all representations at the same grouped-calibration known-FPR operating rule and preserve v3 unchanged.

### 4.4 GAT evidence coverage

Corrected representation-unseen GAT evidence exists only for CsCl and Th3P4. Cu3Au and Ni3Sn were absent from the production four-class GAT and can be tested without retraining; amorphous is not a completed corrected LOFO fold.

## 5. Environment audit

Approved interpreter:

`/home/sz5808/.conda/envs/crystal_ml/bin/python`

Confirmed importable:

```text
torch
torch_geometric
ase
dscribe
scikit-learn
gsd
```

`e3nn` is not installed. The preregistered self-supervised encoder will use existing PyTorch Geometric distance-based message-passing components and will not add a dependency unless the implementation gate demonstrates a specific blocker.

## 6. Audit conclusion

The minimum decisive study is feasible with existing data, but model execution must wait for four study-local freezes:

1. a canonical prototype-equivalence and anchor manifest;
2. a common preprocessing/eligibility manifest after the open-structure audit;
3. the family-level split manifest in `DATA_SPLITS.md`;
4. the comparison, acceptance, and early-stop rules in `DECISION_CRITERIA.md`.

No GPU job or new benchmark score was run during this audit.
