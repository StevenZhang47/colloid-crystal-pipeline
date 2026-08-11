# Final status

Draft compiles with `tectonic -X compile main.tex` (the `texlive` conda env is broken —
missing Perl `TeXLive::TLUtils`, so `fmtutil` cannot build `pdflatex.fmt`; use
`/home/sz5808/.conda/envs/tex/bin/tectonic`).

## READY

- **Methods §2.2–2.8** (gas removal, reliable-centre selection, SOAP, standardisation/PCA,
  fingerprint, detection). Every constant read from current code today: s<1.62, M_GRAPH=1.35,
  TAU_E=0.14, TAU_C=13, n_max=8, l_max=6, r_cut=3.0·d0, σ=0.10·d0, 952-D (asserted at
  runtime), PCA16, 32-D fingerprint, group-balance R=4, support gate 20/8.
- **Methods §2.10 matcher** — library/anchor architecture, channels, EMD closed form,
  ranking vs acceptance, all frozen constants.
- **Results §3.1–3.4, §3.6–3.7** — all numbers traced to v3 artifacts and matcher CSVs.

## NEEDS HUMAN REVIEW

1. **The 952-D derivation.** I assert DScribe's crossover formula gives 952 for
   N_s=2, n_max=8, l_max=6. The code only asserts the *value*, not the formula. Confirm the
   crossover convention is the default.
2. **Clean-control interpretation (§3.5).** I argue the earlier configuration accepted 7/8
   clean controls because its covariance was degenerate, not because it recognised them.
   That is my inference from the shrinkage values, and it reframes an earlier positive result
   as an artifact. Check it.
3. **Introduction citations** — all are \TODO{VERIFY CITATION}; `bibliography.bib` is empty.
   Nothing was fabricated, so the Introduction currently cites nothing.
4. **Overfull hbox** in `03_results` around the split table — cosmetic.

## MISSING EXPERIMENT

1. **Sequential detect→match→add→recognise.** Does not exist anywhere in the tree. §3.8
   states the design and claims nothing.
2. **Re-derive the reference-removal arm** with the anisotropically relaxed zangenite anchor
   *and* a regenerated thermal envelope (envelope frames come from an MD run started from the
   pre-fix lattice).
3. **Rebuild the 24-control panel** on the current reliable-centre convention.
4. **GAT comparison** — NOT WRITTEN. `reports/gat_pipeline/` was not audited before the
   session limit was hit. The critical unresolved question: does the held-out-family
   comparison retrain the GAT, or only re-fit statistics on frozen features? The
   SOAP-vs-GAT claim depends entirely on this and no version of it is currently in the draft.
5. **Adjudication overturn rate by class**, to support the §3.3 attribution quantitatively.

## LITERATURE TO READ

Not yet verified against any source in the repository — `reports/*/references.bib` exist and
should be checked first, since they may already contain correct entries.

1. Bartók, Kondor, Csányi — SOAP representation
2. Himanen et al. — DScribe
3. Steinhardt, Nelson, Ronchetti — bond-orientational order
4. Lechner, Dellago — averaged local bond order
5. Ledoit, Wolf — shrinkage covariance
6. Lee et al. — Mahalanobis OOD detection
7. Vaze et al. — open-set recognition / maximum-logit baseline
8. Boattini et al. — unsupervised local environments in colloids
9. Reinhart et al. — autonomous crystal structure identification
10. Spellings, Glotzer — machine learning for crystal identification

## Files

manuscript/{main.tex, sections/01–05, bibliography.bib (empty)}
manuscript/audit/{SOURCE_OF_TRUTH.md, DRAFT_STALENESS.md, FINAL_STATUS.md}
