# v3-A vs v3-B — predeclared comparison report

**Status: no model selected. The untouched test has not been opened.**

Predeclaration hash `29c1cf5d25d758bc147b5992d257b10f7c68ee3d7a9d374992e4a66ac6330bcc`,
fixed in `data/V3_PREDECLARATION.json` before either model was fitted. Run:
SLURM job 15389330, `cpu_short`, 3 m 19 s, MaxRSS 3.0 GB. Cohort A reproduced
bit-for-bit across two machines (login node and cs623) before the login-node run
was killed.

---

## 1 · What was built

386 trajectories re-featurized through the frozen gas + M3 pipeline into a new
workspace. Every trajectory passed five asserted gates (no gas in SOAP, context
complete, M3 mask non-empty and gas-free, all-finite, support ≥20 centres and ≥8
per species). Zero failures. The SOAP parameter hash matches the frozen phase-2
hash, and on the 49 trajectories that overlap the frozen phase-2 atlas the
re-derived vectors are **bit-identical** (max |Δ| = 0.000e+00), with M3 masks,
species IDs and component IDs identical throughout.

| | regions | groups | train | calibration | validation |
|---|---|---|---|---|---|
| **v3-A** expert-supported only | 246 | 101 | 126 / 68 grp | 70 / 19 grp | 50 / 14 grp |
| **v3-B** + qualified provisional in training | 380 | 103 | 260 / 70 grp | 70 / 19 grp | 50 / 14 grp |

Calibration and validation membership are **identical across cohorts** and 100 %
expert-labelled — asserted, not assumed. The cohorts differ only in training
content: B carries 134 additional GAT-provisional regions. Architecture is the
same for both: hard condition-group split (no cross-fitting), PCA16,
species-conditioned, group-balanced at both fitting stages, Ledoit–Wolf on 32-D
`[LARGE|SMALL]` summaries.

Reserved and untouched: 6 sealed test condition groups (24 trajectories), and the
6-trajectory candidate-novel panel.

---

## 2 · The five predeclared metrics

### M1 — strict LOFO

Both cohorts reach **AUROC 1.000, average precision 1.000, TPR 1.000** on all
three held-out families. The metric that separates them is the false-positive
rate at τ.

| holdout | n pos / neg | A FPR | B FPR | A margin | B margin |
|---|---|---|---|---|---|
| CsCl | 31 / 37 | 0.027 | 0.027 | 5.10× | 4.34× |
| Th3P4 | 27 / 39 | 0.103 | **0.077** | 2.13× | 2.10× |
| amorphous | 62 / 24 | **0.083** | 0.292 | 3.00× | 2.86× |

*margin = min positive ρ / max negative ρ.*

B is better on Th3P4 by one region (3/39 vs 4/39). A is better on amorphous by a
wide margin — 2/24 vs 7/24, a 3.5× degradation. Separation margins favour A on
two of three families.

### M2 — known false-positive rate

| cohort | τ | false positives | rate | CsCl | Th3P4 | amorphous |
|---|---|---|---|---|---|---|
| A | 4.7266 | 12 / 50 | 0.240 | 0.231 | 0.000 | 0.346 |
| B | 4.9374 | 11 / 50 | 0.220 | 0.231 | 0.000 | 0.308 |

A one-region difference. Not a discriminating result. Both are high, and
amorphous drives it in both.

### M3 — clean-control retention

Both cohorts: **0 / 8 clean controls accepted, 16 / 16 perturbed controls
rejected.** v2's frozen global LW accepted 7/8 of these same controls. See §3 —
this is not the threshold artefact it first appears to be.

### M4 — threshold stability (1000 bootstraps of the calibration groups)

| cohort | τ | τ p5 | relative width | decision flips |
|---|---|---|---|---|
| **A** | 4.7266 | 4.5788 | **0.031** | **0.0063** |
| B | 4.9374 | 4.4551 | 0.098 | 0.0234 |

**A is 3.1× more stable on threshold width and 3.7× on decision flips.** Both
cohorts calibrate on the identical 19 groups, so this difference is caused
entirely by the training content: B's covariance produces a more dispersed
calibration-score distribution, which the bootstrap amplifies.

### M5 — context shift

| variant | A accept rate | B accept rate |
|---|---|---|
| frozen M3 centres | 0.76 | 0.78 |
| all non-gas centres in component | 0.00 | 0.00 |
| all non-gas centres whole frame | 0.00 | 0.00 |
| random species-matched, median of 25 | 0.00 | 0.00 |

Indistinguishable between cohorts. Both models are tightly coupled to the exact
M3 centre-selection convention.

---

## 3 · Why the clean controls fail — and why it is not the threshold rule

The obvious first hypothesis was that τ is calibrated on per-group *mean* scores
and then applied to *individual* regions, so the statistic setting τ is narrower
than the statistic it is compared against. The diagnostic supports that for the
validation FPR but **rules it out for the clean controls**:

| quantity | A | B |
|---|---|---|
| calibration region scores, median | 3.76 | 3.64 |
| calibration region scores, p95 | 5.51 | 5.54 |
| calibration region scores, **max** | **7.27** | **6.75** |
| **clean-control scores, median** | **9.14** | **8.15** |

The clean controls' *median* score sits above the *maximum* calibration region
score. No threshold derivable from calibration data can accept them — confirmed
by costing six rules, of which five give 0/8 in both cohorts and the sixth
(max group mean × 1.5) reaches 2/8 in B only.

The threshold rule *is* responsible for the validation FPR, and the cost of
fixing it is small: p99 of calibration region scores drops A from 0.240 to
**0.040** and B from 0.220 to **0.060**. That is a live option, not a selection.

The clean-control failure is a different phenomenon. Ledoit–Wolf shrinkage is
**0.0062 (A) / 0.0059 (B)** — essentially the raw empirical covariance, because
272–280 group-balanced rows in 32-D is a well-posed estimation problem. v2's d32
model fitted 35 summaries in 64-D, where *n < p* forces heavy shrinkage and
inflates the acceptance ellipsoid enormously. **v2 did not accept the Phase-4
controls because it recognised them; it accepted them because its covariance was
too degenerate to reject anything.**

Interpretation, offered as interpretation: the Phase-4 Method-A controls are
synthetic conservative *re-segmentations*, so they carry a different centre set by
construction. M5 shows that any departure from the frozen M3 centre set scores
outside the acceptance region. M3 and M5 are therefore measuring the same thing,
and for v3 the clean-control panel reads as a segmentation-convention mismatch
rather than a detector-quality result. It should probably be rebuilt on the M3
convention before it is used to judge v3.

---

## 4 · Comparison against v2

Like-for-like against v2's `global_lw_d16`, which is the same method family:

| holdout | v2 AUROC / TPR / FPR | v3-A | v3-B |
|---|---|---|---|
| CsCl | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.027 | 1.000 / 1.000 / 0.027 |
| Th3P4 | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.103 | 1.000 / 1.000 / 0.077 |
| amorphous | 1.000 / 1.000 / 0.000 | 1.000 / 1.000 / 0.083 | 1.000 / 1.000 / 0.292 |

v2 was already saturated on this metric, so v3 cannot beat it — and v2's zeros
are not achievements. Its folds carried 5–8 known regions; with 7 negatives the
smallest expressible non-zero FPR is 0.143, so v2 could not have detected a 3 %
error rate. v3's folds carry 61–86 regions. The gain from the expansion is
resolution, not score.

Where v3 is unambiguously better than v2: threshold stability. v2's failure
analysis found threshold resampling spanned ρ = 1, making the accept/reject
boundary the fragile component. v3-A's τ has relative width 0.031 with 0.6 %
decision flips.

---

## 5 · What the predeclared metrics say

| metric | favours |
|---|---|
| M1 LOFO | **A** (amorphous FPR 0.083 vs 0.292; margins on 2 of 3 families) |
| M2 known FPR | tie (1 region) |
| M3 clean controls | tie (0/8 both) |
| M4 threshold stability | **A** (3.1× tighter, 3.7× fewer flips) |
| M5 context shift | tie |

**The 134 provisional training regions bought no measurable improvement and cost
threshold stability.** B's only win is one Th3P4 region.

The likely mechanism is worth noting for the record: B's worst result is the
amorphous holdout, and amorphous is exactly where the provisional labels
concentrate (109 of B's 260 training regions, versus 52 of A's 126). It is also
the class whose GAT calls the expert adjudication overturned in 2 of 3 cases.
Provisional amorphous labels being the noisiest is consistent with everything
else seen in this project.

---

## 6 · Open items — decisions not taken here

1. **No model selected.** The above is the evidence, not a choice.
2. **Threshold rule.** The predeclared rule over-rejects individual regions.
   Six alternatives are costed in `data/v3_threshold_rule_options.csv`. None adopted.
3. **Clean-control panel.** Recommend rebuilding it on the M3 centre convention
   before it is used as a v3 acceptance criterion.
4. **Untouched test.** Sealed. 6 condition groups, 24 trajectories.
5. **Candidate-novel panel.** 6 trajectories, not yet scored under either model.

### Files

| file | contents |
|---|---|
| `data/V3_PREDECLARATION.json` | hashed split + metric definitions |
| `data/v3_region_manifest.csv` | 380 regions, roles, provenance, gas/M3 provenance |
| `data/v3_m1_lofo.csv`, `_per_region.csv` | LOFO results and per-region scores |
| `data/v3_m2_false_positive.csv` | known FPR |
| `data/v3_m3_clean_controls.csv` | 24 Phase-4 controls under both models |
| `data/v3_m4_threshold_stability.csv` | bootstrap thresholds |
| `data/v3_m5_context_shift.csv` | centre-set variants |
| `data/v3_threshold_diagnostic.csv`, `v3_threshold_rule_options.csv` | §3 |
| `data/v3_shard_manifest.csv` | 386 trajectories, gas/M3 gates |
