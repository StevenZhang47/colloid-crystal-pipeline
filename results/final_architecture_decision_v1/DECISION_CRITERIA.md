# Final architecture-decision study: decision criteria

Status: **preregistered; no study model has been run**  
Purpose: choose between fixed SOAP, the corrected supervised GAT diagnostic, and one fixed self-supervised graph representation without open-ended tuning.

## 1. Scientific questions

This study answers four separate questions:

1. Can a representation recognize a family that was wholly absent from representation training?
2. Can it retrieve the correct newly supplied prototype rather than a nearby wrong prototype?
3. Can it incorporate one small independent reference run without encoder retraining and then generalize to a second run?
4. Can preprocessing preserve genuinely cohesive low-coordination crystals without admitting gas?

OOD rejection, nearest-family identity, prototype retrieval, calibrated acceptance, and numerical failure remain separate outcomes. A rejected candidate with the correct nearest family is not a wrong-family classification. A numerical failure is not credited as a successful rejection.

## 2. Frozen comparison arms

### A. Fixed SOAP incumbent

OOD/domain feature:

- frozen 952-dimensional per-particle GLOBAL_D0 SOAP;
- species-conditioned, training-only centering/scaling and PCA16 per species;
- mean LARGE and mean SMALL embeddings concatenated to 32 dimensions;
- global Ledoit-Wolf Mahalanobis model;
- equal LARGE/SMALL contribution;
- group-aware, study-local 5% known-FPR threshold.

Prototype-retrieval feature:

- separate LARGE and SMALL mean 952-dimensional SOAP vectors;
- concatenate to 1904 dimensions and L2-normalize;
- cosine-equivalent Euclidean distance;
- the same canonical single-anchor prototype bank used by every admissible representation.

The old per-particle Mahalanobis variant is historical only; its held-out-Th3P4 failure is not retuned. The old `D < 0.40` EMD acceptance rule is also historical only.

### B. Corrected supervised GAT diagnostic

- architecture source: `crystal_discovery/preprocessing_pipeline_final/unified_pipeline/scripts/motif_pipeline/train_pyg_gat_particle_classifier_v7.py`;
- only true family-removal models with a three-class head and all held-out-family nodes removed from message passing are valid;
- domain representation is mean pooled hidden state (`GAT_MEAN`) followed by the same study-local global Ledoit-Wolf scoring protocol;
- softmax confidence is never used as an OOD score because the corrected audit showed invalid/inverted confidence behavior;
- existing corrected CsCl and Th3P4 LOFO models are evaluated as a supervised diagnostic;
- Cu3Au and Ni3Sn are reported `NOT_AVAILABLE_UNDER_FROZEN_SUPERVISED_BASELINE` unless an already-preregistered, label-clean model exists before execution.

The supervised arm cannot win the universal architecture decision on only two folds. It provides an upper-context diagnostic for methods trained with family labels.

### C. One self-supervised graph encoder

Implementation is fixed to the existing PyTorch Geometric stack in `crystal_ml`; no e3nn dependency or architecture search is introduced.

Encoder:

- binary species embedding, width 64;
- 32 Gaussian radial basis functions over normalized distance;
- three continuous-filter/message-passing interaction blocks, hidden width 64;
- cutoff `3.0 * d0`, at most 64 neighbors per center;
- distances expressed as `r/d0` (or the exactly equivalent pair-contact normalization recorded in the cache manifest);
- separate LARGE and SMALL mean pooling, concatenated to 128 dimensions and L2-normalized.

Objective and optimization:

- instance-level InfoNCE, temperature 0.10;
- positives are two augmented views of the same source domain;
- other source-domain views in the batch are negatives; family labels are not read by the loss;
- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- maximum 200 epochs; batch size is the largest deterministic value that fits and is frozen before the first fold;
- early stopping uses only non-held-out self-supervised validation loss, patience 20 epochs;
- primary seeds 17 and 29.

Fixed augmentations:

- arbitrary rigid rotation and translation;
- one mild nuisance transform per view, selected from Gaussian coordinate noise (`sigma=0.02*d0`), anisotropic strain independently sampled within +/-2%, or a connected crop retaining 70-100% of centers;
- uniform scale within +/-5% is allowed only because distances are d0-normalized;
- species identities are never swapped and connectivity-changing deletion is not used as an augmentation.

No embedding width, depth, temperature, augmentation magnitude, crop range, optimizer, or score model may be tuned after downstream results are visible.

## 3. Common downstream protocol

For the OOD comparison, each representation feeds the simplest common statistical head that is numerically admissible: training-only standardization followed by a global Ledoit-Wolf covariance model. No representation receives a method-specific nonlinear classifier to rescue it.

Thresholds are estimated from independent non-held-out condition groups with a symmetric target of 5% known FPR. Report the threshold, achieved FPR, leave-one-group-out range, bootstrap-by-group interval where meaningful, and effective group count. A threshold based on fewer than five independent groups is flagged low-confidence.

For each family-removal fold and seed, report:

- known and held-out run counts;
- AUROC and AUPRC;
- held-out TPR at the frozen 5% known-FPR rule and achieved known FPR;
- minimum, median, and maximum normalized rho;
- raw run-level, condition-group macro, and family-balanced metrics;
- LARGE and SMALL scores/nearest families and species disagreement;
- numerical failures and runtime.

No failed family is averaged away. Worst-family held-out TPR is the primary OOD summary.

## 4. Retrieval and removal protocol

All methods receive the same canonical prototype bank and candidate coordinates. Retrieval reports top-1, rank, correct and nearest-wrong distances, and `d_correct/d_wrong` at the run level.

The correct-prototype-removal arm is mandatory. After removal, a safe system should abstain; a confident wrong match is a failure. Any acceptance/margin rule is calibrated on non-held-out development groups before candidate access and shared across comparable distance outputs when mathematically meaningful.

The self-supervised encoder is considered to provide a meaningful retrieval improvement only if, on both primary seeds, it does at least one of the following without increasing confident-wrong removals:

- fixes a preregistered hard-neighbor top-1 error made by the fixed representation; or
- improves the median correct-to-wrong separation by at least 25% while preserving top-1 accuracy.

Tiny rank or distance changes below this criterion do not justify replacing the incumbent.

## 5. Incremental-reference protocol

After initial held-out-family evaluation, one seed-11 run is added to the downstream reference/statistical model. The encoder, normalization recipe, architecture, and scientific thresholds remain frozen. Seed 12 is then opened once as the independent post-update test.

A representation passes incremental adaptation when:

- the seed-12 family is correctly recognized and accepted;
- no encoder retraining occurred;
- old-known family accuracy and known FPR change by no more than 2 percentage points;
- no new numerical failure or confident wrong prototype removal appears.

Report preprocessing/cache time, downstream update time, memory, and the number of new labeled/reference runs required.

## 6. Preprocessing decision

The alternative cohesive-component screen replaces the current screen only if it:

- preserves each preregistered low-coordination ideal positive, including ReO3;
- rejects all four gas-only controls;
- causes no material loss on dense controls;
- produces physically connected retained components under periodic-boundary-aware inspection.

There is no cutoff optimization. If the single alternative fails, preprocessing is declared not universally solved. Representation comparisons may proceed on identical pre-extracted coherent domains, but the final architecture must carry that scope limitation.

## 7. Preregistered early stopping

The user-authorized principle is implemented as a statistical and operational rule: a decisively losing branch stops; it is not hyperparameter-tuned in response.

### Sentinel order

Run H-TH3P4 first because it is the open-framework and historical catastrophic-failure case. Run H-CU3AU second because it contains the strongest known prototype-neighbor ambiguity. Only a surviving self-supervised branch proceeds to H-CSCL and H-NI3SN.

### OOD branch stop

Stop the self-supervised OOD branch after the two sentinels if both seeds show either:

- held-out TPR at least 0.20 below SOAP on either sentinel at the common operating rule; or
- achieved known FPR more than twice SOAP's and at least 5 percentage points higher.

A single catastrophic family is sufficient when it repeats across both seeds. Mean performance cannot hide it.

### Retrieval branch stop

Stop the self-supervised retrieval branch after the two sentinels if, for both seeds:

- top-1 is no better than SOAP;
- median separation improvement is below 25%; and
- confident-wrong prototype-removal errors are not fewer.

### Entire experimental encoder stop

Stop all further self-supervised training when both the OOD and retrieval branches meet their stop rules. If only one branch survives, finish all four folds only for that scientific claim. Mixed results or uncertainty intervals overlapping the stop margins proceed to the remaining folds.

### Numerical/reproducibility stop

If more than 1% of required scores are nonfinite, or deterministic reruns fail invariant tests, permit one implementation correction before any closed score is opened. After closed scores are opened, no scientific or implementation change is allowed. Persistent failure is labeled `NUMERICALLY_INELIGIBLE`, not tuned around.

### Conditional third seed

Seed 43 is permitted only when seeds 17 and 29 disagree by more than 0.05 in a primary metric or reach different hard-neighbor/pass decisions. It quantifies instability; it cannot be used to select hyperparameters. Otherwise two seeds are final.

## 8. Final architecture decision rule

Selection priority is frozen:

1. preserve known natural structures;
2. avoid a catastrophic representation-unseen family failure;
3. retrieve the correct new prototype and abstain safely when it is absent;
4. support the one-run incremental update without forgetting;
5. maintain numerical reliability;
6. prefer lower runtime and operational complexity when scientific performance is comparable.

SOAP remains the OOD/domain-recognition incumbent unless the self-supervised encoder is no worse by more than 0.05 in worst-family TPR, no worse by more than 2 percentage points in known FPR, has no catastrophic fold on either seed, and provides either a meaningful retrieval improvement or a material system simplification.

A tie does not displace SOAP. The supervised GAT diagnostic cannot be selected as a universal winner without valid coverage of all four folds and cannot use softmax confidence for OOD.

The architecture may be split by task only when the evidence is decisive—for example, SOAP for OOD and another representation for retrieval. Such a split must include its operational cost; it is not called a single universal representation.

## 9. Required validation before any closed score

Each implementation must pass:

- manifest and sealed-path audit;
- held-out-family absence search over paths, rows, aliases, caches, and batches;
- mathematical identity and self-distance checks;
- permutation invariance and rigid-motion invariance;
- deterministic repeat test for embeddings and scores;
- finite-value, symmetry, and covariance conditioning checks;
- augmentation bounds and connected-crop checks;
- gradient/update test for the self-supervised encoder;
- hash audit for code, environment, inputs, split manifests, and initial weights.

Failure stops before candidate access.

## 10. Compute and scheduler plan

No GPU job is authorized by this preregistration checkpoint. The proposed logical jobs are:

| Stage | Logical jobs | Resource | Maximum budget |
|---|---:|---|---:|
| Open preprocessing comparison | 1 | CPU | 4 core-hours |
| Common graph/SOAP caches and invariant tests | 1 | CPU | 12 core-hours |
| SSL primary folds: 4 folds x 2 seeds | 8 | 1 GPU each | 6 GPU-hours/job (48 total) |
| Conditional seed 43 | up to 4 | 1 GPU each | 6 GPU-hours/job (24 conditional) |
| Common OOD heads and scoring | 4 | CPU | 16 core-hours total |
| Prototype retrieval/removal | 4 | CPU | 12 core-hours total |
| Incremental seed-11/seed-12 evaluation | 4 | CPU | 16 core-hours total |
| Final all-eligible-data encoder, only if selected | 1 | 1 GPU | 6 GPU-hours |
| Final external scoring, only after separate approval | 1 | CPU | 4 core-hours |

Primary pre-external workload is 22 logical jobs, up to 26 if every conditional third seed is triggered. Scheduler arrays may consolidate submissions but must retain per-fold/per-seed logs and hashes. Early stopping can reduce this workload after the two sentinel folds.

Wall-time exhaustion does not relax convergence or numerical gates. An unfinished branch is reported `COMPUTATIONALLY_INCOMPLETE`.

## 11. Freeze boundaries and forbidden adaptations

Before the first candidate run is opened, freeze and hash:

- all three representation definitions;
- source/canonical/fold manifests;
- preprocessing policy;
- downstream model and threshold algorithm;
- augmentation and optimization parameters;
- early-stop and final-selection rules;
- exact code and environment manifests.

After any closed candidate score is opened, do not alter cutoffs, normalization, pooling, loss, architecture, hyperparameters, threshold rule, canonical prototype identities, family aliases, support rule, or decision criteria. Failed results are results.

External candidates, prior external score files, v3 untouched tests, fused examples, and candidate-novel payloads cannot be used to choose between arms. A later external run requires a separate user checkpoint after the architecture decision is frozen.
