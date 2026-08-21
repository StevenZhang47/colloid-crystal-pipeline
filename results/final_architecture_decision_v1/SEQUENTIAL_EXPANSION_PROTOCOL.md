# Sequential active-atlas expansion protocol

Status: **MATHEMATICS FROZEN; EXECUTION BLOCKED BY PROVENANCE AUDIT**

This document separates the active statistical atlas from the passive canonical
prototype library and freezes the update before any new trajectory scoring.

## 1. Fixed order and libraries

The only permitted order is:

```text
A0 = {CsCl, amorphous}
A1 = A0 + Th3P4
A2 = A1 + Cu3Au
A3 = A2 + Ni3Sn
```

No alternative insertion order may be tested. The passive 30-prototype CIF bank is
constant at every stage. Membership in that bank does not make a family active or
in-distribution.

## 2. Frozen representation

The stage-0 transform is exactly the transform in
`baselines/soap/ood_model_H-TH3P4.joblib`, trained on the cohort-A CsCl and
amorphous development groups. The following remain byte/parameter frozen:

- domain-local physical-component extraction;
- M3 reliable-center rule and constants;
- component-local `GLOBAL_D0`;
- SOAP species, `r_cut=3.0`, `n_max=8`, `l_max=6`, and 952-D output;
- per-center L2 normalization;
- two 952-D species means and scales;
- the shared PCA16 components, PCA mean, and dimensionality;
- the resulting 32-D `[LARGE mean | SMALL mean]` summary definition;
- the normalized 1904-D retrieval fingerprint definition;
- the canonical 30-prototype bank and map;
- retrieval distance threshold `0.2404109239578247`;
- retrieval margin threshold `1.117297887802124`;
- absolute OOD threshold `tau0 = 7.982486955871957`;
- all preprocessing and minimum support rules.

No gradients, GPU optimization, SOAP retraining, normalization refit, PCA refit,
prototype update, retrieval-threshold refit, or OOD-threshold refit is permitted.

## 3. Exact feature and run aggregation

For each center `i` of species `s`, let `x_i` be its 952-D SOAP row and
`u_i = x_i / ||x_i||_2`. With frozen species mean `mu_s`, scale `sigma_s`, and PCA
map `P`, the frame summary is

```text
z_frame = concat_s mean_{i:species(i)=s} P((u_i - mu_s) / sigma_s) in R^32.
```

Every dynamical run uses the five prospectively fixed late frames
`300, 350, 400, 450, 500`. They are correlated observations from one run:

```text
z_run = (1/5) * sum_frame z_frame.
```

Thus one new family contributes exactly one independent 32-D summary. Frame
replication never increases independent `n`.

The retrieval fingerprint at each frame is the concatenation of the LARGE and SMALL
means of `u_i`, L2-normalized. The run fingerprint is the arithmetic mean of the five
frame fingerprints followed by one L2 normalization. Euclidean distance to the
unchanged 30-anchor bank and the frozen distance/margin gates define identity.

## 4. Stage-0 statistical state

`A0` is the original H-TH3P4 SOAP LOFO model:

| Quantity | Value |
|---|---:|
| training regions | 80 |
| condition groups | 49 |
| covariance rows after group weighting | 196 |
| pure-CsCl groups | 14 |
| pure-amorphous groups | 26 |
| mixed CsCl/amorphous groups | 9 |
| CsCl-containing groups | 23 |
| amorphous-containing groups | 35 |
| summary dimension | 32 |
| Ledoit–Wolf shrinkage | 0.0017230805644895143 |
| frozen threshold | 7.982486955871957 |

Mixed-label condition groups are retained exactly as in the frozen implementation.
There is no new family balancing.

## 5. Exact covariance update

Let `G=4` be the frozen `group_balance_regions_per_training_group`. Reconstruct the
original fit matrix in deterministic `condition_group_id`, then `region_id`, order.
Each old group contributes four summaries through the existing cyclic tiling rule.
Before any addition, refitting `LedoitWolf()` to this reconstructed `196 x 32` matrix
must reproduce the stored A0 location, covariance, precision, and shrinkage within
maximum absolute error `1e-12`; otherwise execution stops as implementation-blocked.

For a new family `X` with one run summary `z_X`, append

```text
[z_X, z_X, z_X, z_X]
```

to the covariance-fit matrix and refit only `sklearn.covariance.LedoitWolf` with the
same defaults. The four identical rows implement the pre-existing equal-group weight;
they are one independent run and one independent condition group, not four samples.

The sequential fit matrices are:

| Stage | Independent groups | Covariance rows | Newly added group's matrix weight |
|---|---:|---:|---:|
| A0 | 49 | 196 | — |
| A1 | 50 | 200 | 2.000% |
| A2 | 51 | 204 | 1.961% |
| A3 | 52 | 208 | 1.923% |

Each previously added run remains as its same four weighting rows at later stages.
Adding one family therefore changes its statistical weight, but does not rebalance
families: old families retain many more condition groups. This asymmetry is part of
the frozen model and may not be tuned away.

Quantities allowed to change after each addition are only:

- the 32-D Ledoit–Wolf location;
- the symmetric `32 x 32` covariance and derived precision;
- shrinkage and condition-number diagnostics;
- fit-matrix row/group counts and provenance;
- the active-family membership metadata;
- the stored new run summary and audit fingerprint.

There are zero trainable neural parameters. In independent-moment terms the refit
updates 32 mean values plus 528 unique covariance values; the 528 unique precision
entries and shrinkage scalar are derived/stored statistics. Actual changed-entry
counts must be measured stage by stage.

## 6. Old data and computation

The original joblib stores the fitted covariance but not its fit matrix. Therefore old
32-D summaries must be deterministically re-derived from frozen SOAP shards under the
frozen transform. Old raw trajectories and old SOAP rows are not regenerated. The
reconstruction time must be reported separately from the incremental four-row
Ledoit–Wolf update time.

## 7. Score and threshold semantics

For stage `k`,

```text
D_k(z) = sqrt((z - location_k)^T precision_k (z - location_k))
rho_k(z) = D_k(z) / tau0
KNOWN iff rho_k <= 1.
```

`tau0` is retained unchanged at A0–A3. This is mathematically executable and avoids
an unpreregistered threshold update. However, its original interpretation as the
higher 95th percentile of A0 calibration-group scores is not automatically preserved
after covariance refitting. At A1–A3 it must be called the **frozen A0 operating
threshold**, not a newly calibrated 5% threshold. Empirical old-known FPR at every
stage is therefore a primary endpoint. This calibration limitation is an important
scientific result, not a reason to modify the rule.

## 8. Sequential evaluation, if provenance is repaired

At each stage, and only in the frozen order:

1. score the blind pre-addition candidate against the current active covariance;
2. retrieve against the unchanged passive bank;
3. require OOD plus correct accepted prototype before addition;
4. encode exactly one independent addition run into one `z_run`;
5. refit the permitted covariance statistics and freeze the new stage;
6. open the blind post-addition run once;
7. rescore identical old-known validation groups and every previously added blind
   post-addition run;
8. score candidates of not-yet-added families to detect premature acceptance.

Run-level OOD is the arithmetic mean of the five frame scores divided by `tau0`.
Group-level old-known OOD is the mean score within each independent condition group
divided by `tau0`.

Identity accuracy uses only validation groups whose family has a canonical prototype.
Amorphous remains in the OOD/FPR denominator but has no passive CIF identity target;
it is reported separately rather than counted as an identity error. Group retrieval
averages region fingerprints and renormalizes before matching.

The correct-prototype-removal diagnostic is applied to every new blind test under the
same frozen thresholds. A confident wrong identity is a family failure, not an OOD
success.

## 9. Frozen success criteria

For each addition, report separately:

- pre-addition candidate OOD;
- correct and accepted passive-CIF retrieval;
- exactly one independent addition run;
- post-addition test accepted known;
- post-addition correct identity;
- every earlier added family remains known and correctly identified;
- not-yet-added families remain OOD;
- absolute old-known FPR change at most 2 percentage points;
- absolute eligible old-known identity-accuracy change at most 2 percentage points;
- no new numerical or confident-wrong failure;
- no gradient, GPU, PCA, representation, bank, or threshold refit.

No aggregate score may hide a failed family or stage.

## 10. Feasibility decision and stop

The growing covariance is numerically and mathematically well-defined. The fixed
threshold becomes calibration-stale but remains a deterministic operating point whose
stability can be tested. The model therefore does not require an unpreregistered
statistical change to execute.

Execution nevertheless stops because the provenance audit found only two blind Cu3Au
run keys and zero blind Ni3Sn run keys. No partial Th3P4 scoring, insertion-order
change, viewed-run substitution, threshold change, or special-case code is allowed.

The minimum path to the primary experiment is to generate and provenance-seal one new
Cu3Au run and three new Ni3Sn runs before any of the currently eligible Th3P4/Cu3Au
payloads are decoded. A later role-conditioned seed-1/11/12 replication requires
separate authorization and must label identity as confirmatory.

## 11. Frozen file pins

```text
acc9ae34d3b4bc22b15ffc8d535d7689e1bd86144c9cf280fe5092798888c356  core3_pipeline.py
3634f2af441a09fc77180be17c83489c04def94b6b1569b6c89730fd4899f73f  v3_common.py
c4dc4b9efc896e53a7971a99af8289d158cba4b9dbcc5d637d3840b55d4bacfe  scripts/06_prepare_soap_baseline.py
1148a28f98210a8e44940013e7335df258339762bc6204b63802d518fb7e25ce  scripts/11_score_fixed_sentinels.py
c74dab6f5e268487103802deb7b740e4dc8f30ddb4a0030efbdce1b1cdf3f1eb  baselines/soap/ood_model_H-TH3P4.joblib
5003016d567a8f51a9f5890057791db8d226456aad466a5e08719efea66931ef  baselines/soap/canonical_anchor_fingerprints.npz
988162504a6f3efda8ab8abbae13a5f32538907fceca9e3c2042f083457ca349  data/canonical_prototypes.csv
54354efee5837d38bdfde22316b5895933efb63c384114a1b553d25acbfcc482  V3_PREDECLARATION.json
```

## 12. Firewall

This protocol authorizes no scoring by itself. H-CSCL, H-NI3SN final-study scoring,
SSL/GAT expansion, Steven examples, external candidates, untouched v3 tests,
candidate-novel data, fused examples, and external RDF/S(k) remain closed.
