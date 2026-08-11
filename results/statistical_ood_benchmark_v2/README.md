# Statistical OOD benchmark v2 — cached pooled stage

## Scope

This stage uses cached descriptors only. It ran no raw-particle SOAP recomputation, external K4C60 scan, simulation, or SLURM job. Thresholds were calibrated with five-fold condition-grouped (`Debye|volume_fraction`) known-only folds. Zangenite, K4C60, Steven1, Steven2, and the leave-one-family-out target were excluded from every fit and calibration step.

The primary feature space is the complete cached SOAP power spectrum (952 dimensions) plus all nine compatible physical/motif features. Ablations are SOAP only, physical only, SOAP + physical + GAT hidden, and GAT hidden only. GAT softmax probabilities are never OOD scores.

## Main results

- Best primary/no-GAT pooled OOD methods: global Mahalanobis and kNN tie on the ten named controls (gas abstained): 0 known false positives, 100% detection of Zangenite, K4C60, Candidate A, and Candidate B, balanced accuracy 1.0 with SOAP + physical.
- Family-wise Mahalanobis with SOAP + physical detects 3/4 pooled positives; relative Mahalanobis detects 2/4 and has a 40% known false-positive rate. Relative Mahalanobis does not improve near-OOD here.
- Isolation Forest and PCA residual each detect all four positives but have a 40% known false-positive rate in the primary space. They are not competitive with global Mahalanobis/kNN on this small named-control set.
- GAT is not necessary: the main conclusions hold with SOAP + physical. Adding GAT creates additional perfect-scoring combinations but does not improve on the primary tie. GAT hidden should remain auxiliary because it was learned from phase labels.
- MMD is unavailable for the required named-subject comparison: compatible particle descriptor sets do not exist for K4C60/Steven/Zangenite. It was not faked.

These rates cover a very small subject-level test set with one pooled vector per subject. They are useful stress tests, not precise population estimates; subject-level bootstrap uncertainty is unavailable.

## Leave-one-family-out

With SOAP + physical:

- Leave-CsCl-out succeeds for global Mahalanobis (1.52× threshold), family Mahalanobis (1.53×), relative Mahalanobis, kNN (1.15×), and PCA residual (2.05×). Isolation Forest misses it.
- Leave-Th3P4-out is the harder scientific failure: only relative Mahalanobis flags the cached Th3P4 control in the primary space, while its nearest remaining family is amorphous. Global/family Mahalanobis, kNN, Isolation Forest, and PCA residual incorrectly keep it in-distribution. Thus the primary methods do **not** reliably make held-out Th3P4 novel, and several collapse its nearest-family relation toward amorphous.

The summary metric mixes all named positives; consult `method_comparison_scores.csv` for the explicit held-out-family control rows. Threshold ratios for relative Mahalanobis are not intrinsically interpretable when its calibrated score/threshold may be signed; use its raw margin and binary call.

## Amorphous versus Th3P4

The labeled trajectory analysis contains 17 Th3P4 and 27 amorphous trajectories, with grouped out-of-fold predictions. Unresolved/boundary cases are kept separate and excluded from binary metrics.

SOAP-only and SOAP + physical achieve perfect grouped separation for family Mahalanobis, relative Mahalanobis, kNN, and class-specific PCA residual: balanced accuracy/ROC-AUC/PR-AUC 1.0, false-amorphous rate for true Th3P4 0, and false-Th3P4 rate for amorphous 0. Isolation Forest is weaker; see the summary CSV. GAT-hidden variants also separate perfectly, so success is not dependent on GAT.

This result is strong for the cached labels but must be read beside the documented Nicole label contamination. The evaluation uses expert-corrected trajectory labels; it does not prove performance on every surface-dominated structure. Boundary/defective predictions are descriptive, never silently counted as Th3P4 or amorphous.

### Dedicated difficult Th3P4 holdout

`data/known_examples/trajectory (5) organized.gsd` was audited separately. It is absent from the pooled SOAP/physical atlas, and a freshly extracted final-frame ordered-core vector was scored only after model fitting. SOAP and SOAP + all nine physical/motif features call it Th3P4 under all five algorithms. Motif recurrence alone calls it Th3P4 under all methods; its recurrence is 0.08218 versus cached trajectory means 0.08359 for Th3P4 and 0.01960 for amorphous. Isolation Forest is the fragile exception in physical-only spaces, while family/relative Mahalanobis and kNN remain correct.

The existing GAT-v4 checkpoint trained on late frames/core particles from this exact trajectory. GAT-hidden results for this target are therefore unavailable due to leakage, not counted as a success. See `trajectory5_organized_holdout_report.md` and its exact result CSV.

## Interpretable phase features and motif recurrence

All nine compatible physical/motif features were used, not cherry-picked:

1. coordination;
2. neighbor fraction;
3. same-type fraction;
4. normalized mean neighbor distance;
5. neighbor-distance standard deviation;
6. mean neighbor diameter/type delta;
7. motif recurrence;
8. q4;
9. q6.

Training-only univariate ranking identifies motif recurrence as the strongest interpretable separator both for crystal versus noncrystal and Th3P4 versus amorphous (absolute trajectory-level AUC 1.0 in this cache). For Th3P4 versus amorphous, the next strongest are q4, diameter/type delta, shell-distance spread, and same-type fraction. q6 is highly useful across all phases but, in this dataset, its Th3P4 and amorphous trajectory means are nearly identical; using q6 alone would reproduce the exact failure the user is concerned about.

Representative trajectory means:

| label | motif recurrence | q4 | q6 | coordination | shell-distance std | same-type fraction |
|---|---:|---:|---:|---:|---:|---:|
| CsCl | 0.1321 | 0.1108 | 0.5304 | 11.55 | 0.0762 | 0.4094 |
| Th3P4 | 0.0836 | 0.2365 | 0.3727 | 8.81 | 0.1064 | 0.3360 |
| amorphous | 0.0196 | 0.2059 | 0.3730 | 9.11 | 0.0990 | 0.3626 |
| boundary/defective | 0.0454 | 0.1683 | 0.4060 | 10.35 | 0.1035 | 0.4043 |

These are dataset summaries, not universal cutoffs. The full rankings and means/medians/stds are in `phase_feature_rankings.csv` and `phase_feature_profiles.csv`. Cached RDF, partial RDF, S(k), Bragg, PTM/stacking, and temporal features are scientifically valuable but are not available in a compatible form for every pooled subject; they belong in the order/matcher validation path rather than being concatenated into this benchmark with missingness leakage.

## Feature definitions for the current writeup

For one coherent ordered core with particle descriptors \(f_i\), domain mean pooling is

\[
x_{\mathrm{domain}}=\frac{1}{N_{\mathrm{core}}}\sum_{i\in\mathrm{core}} f_i.
\]

SOAP expands neighbor density into \(c_{nlm}(i)\) and uses the rotationally invariant power spectrum

\[
p_{nn'l}(i)=\sum_m c_{nlm}(i)c^*_{n'lm}(i).
\]

For bond order,

\[
\bar q_{lm}(i)=\frac{1}{N_b(i)}\sum_{j\in\mathrm{nn}(i)}Y_{lm}(\hat r_{ij}),\qquad
q_l(i)=\left[\frac{4\pi}{2l+1}\sum_{m=-l}^{l}|\bar q_{lm}(i)|^2\right]^{1/2},\quad l\in\{4,6\}.
\]

With median first-neighbor distance \(d_0\), \(r_s=1.35d_0\) and \(\mathcal N_i=\{j\ne i:\|r_{ij}\|\le r_s\}\). Then \(z_i=|\mathcal N_i|\), \(f_i^{\rm nbr}=z_i/k\) for \(k=12\), and

\[
f_i^{\rm same}=\frac{1}{z_i}\sum_{j\in\mathcal N_i}\mathbf 1(t_j=t_i),
\quad
\bar d_i=\frac{1}{z_i}\sum_{j\in\mathcal N_i}\frac{d_{ij}}{d_0},
\]

\[
s_{d,i}=\left[\frac{1}{z_i}\sum_{j\in\mathcal N_i}\left(\frac{d_{ij}}{d_0}-\bar d_i\right)^2\right]^{1/2},
\quad
\Delta D_i=\frac{1}{z_i d_0}\sum_{j\in\mathcal N_i}|D_j-D_i|.
\]

After sorting neighbors by distance, the chemical motif signature is \(\sigma_i=(t_{j_1},\ldots,t_{j_{\min(k,z_i)}})\). If \(n(\sigma_i)\) particles share it among \(N_{\rm dense}\) valid particles,

\[
R_i=\frac{n(\sigma_i)}{N_{\rm dense}},\qquad R_i=0\text{ for }z_i<6.
\]

High recurrence indicates a repeated local chemical environment, but it is not sufficient evidence of a crystal without core/order and reciprocal/real-space validation.

## Matcher and held-out controls

K4C60 and Zangenite are verified as training-excluded pooled positives; provenance limits are documented in `heldout_positive_controls_verification.md`. Existing matcher outputs demonstrate an abstention path is necessary. Candidate A/B have Cu3Au-family evidence in one cached library but fail distance/margin or cross-variant consistency checks, so neither identity is confirmed. K4C60 matching and targeted reference-removal controls require small future fingerprint runs.

## Deliverables

- `method_comparison_scores.csv`, `method_comparison_summary.csv`
- `amorphous_th3p4_failure_modes.csv`, `amorphous_th3p4_separation_summary.csv`
- `phase_feature_rankings.csv`, `phase_feature_profiles.csv`
- `heldout_positive_controls_verification.md`
- `reference_matcher_stress_tests.csv`, `reference_matcher_stress_tests.md`
- required separation figures plus an OOD correctness heatmap in `figures/`
- reproducible Python scripts in this directory

The corrected `crystal_ml` Conda environment is available and its NumPy, pandas, scikit-learn, and Matplotlib imports were validated. The completed analysis itself used the already-active compatible Python environment; no descriptor values depend on the environment switch.
