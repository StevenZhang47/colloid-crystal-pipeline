# AI agent log

## Session 2026-07-15 (Codex) -- GAT-free distribution rebuild Phase 1

**Asked:** execute Phase 1 only: immutable inventory/leakage audit, trajectory-safe split design, generalized GAT-free extractor foundation, bounded physical/SOAP pilots, caches, and tests. Do not fit detectors, run MMD, calibrate thresholds/matcher acceptance, or launch large jobs.

### Built

- New isolated work area: `results/ood_matcher_distribution_rebuild_v1/`.
- Inventory: 90 files, SHA256, inspected-frame geometry metadata, duplicate/source groups, five provisional split manifests.
- Versioned arbitrary-GSD extractor with diameter-canonical LARGE=0/SMALL=1 species, periodic connected components, preserved original indices, isolated re-featurization, and immutable content-addressed caches.
- Six domain caches across five pilots: full physical 11D, nonredundant 10D audit view, q4m/q6m, metadata, and full per-particle 952D SOAP for 25,423 particles.
- Full details and exact commands: `results/ood_matcher_distribution_rebuild_v1/PHASE1_REPORT.md`.

### Load-bearing findings

1. `expert_corrected_label` is **not manual expert ground truth**. Its generating code normally returns the GAT majority, and the 67-case subset was selected from GAT outputs. The 23 CsCl/17 Th3P4/27 amorphous split design is provisional and **must not be fitted** until an independent GAT-free label/corpus audit.
2. Exact duplicates: Steven1 == `crystal_stability/runs/ni3sn/trajectory_big.gsd`; Steven2 == `.../cu3au/trajectory_big.gsd`; `trajectory (5) organized.gsd` == phase-map source `debye5.35_volumefraction.004294716531_seed1/trajectory.gsd`. All aliases are grouped/quarantined.
3. The active zangenite ideal anchor and strict zangenite thermal target come from the same run/prototype group, so they are not independent matcher validation.
4. Example1 frame 253 is recovered GAT-free as two domains: 15,352 particles near 1:1 and 6,705 near 2:3. Trajectory5 gives one 876-particle near-3:4 domain.
5. The amorphous pilot still produces a large connected balanced-mask component. Spatial extraction is not a crystallinity, identity, or OOD decision.
6. SOAP pilot: 25,423 particles, 952D float32, all finite, exact repeat sample, 97 MiB matrices, max process RSS 453 MiB. q_lm is cheap (~176 bytes/particle).
7. Nine Phase-1 tests pass. No Phase-2 work ran.

### Stop gate

Stopped after Phase 1. Next approval must first resolve the GAT-derived label/subset blocker; do not silently proceed to representation selection or detector/matcher benchmarking.

## Session 2026-07-15 (Codex) -- Cr3Si deep8 v5 free-volume 10x continuation

**Asked:** run the successful `cr3si_a15_deep8_v5_free_volume` simulation much longer.

### Submitted

- Added `crystal_stability/prepare_cr3si_deep8_v5_free_volume_long10x.py`.
- New run: `runs/cr3si_a15_deep8_v5_free_volume_long10x/`; the completed v5 source is
  untouched.
- Source trajectory and checkpoint are both at 10,000,000 steps. PACSim advances
  `run_steps` after checkpoint load, so the continuation adds 90,000,000 steps for
  100,000,000 total: 15 microseconds at dt 0.15 ps (10x the v5 duration).
- Configuration, first frame, source trajectory, and checkpoint were copied; SHA256
  confirmed the first frame, source trajectory, and checkpoint are byte-identical.
- CUDA job `13812589`, initially pending for priority. Requested wall time 4 hours;
  measured v5 throughput predicts about 2.2--2.5 hours.
- Interactions, 298 K temperature, timestep, 13,429 nm free-volume box, walls-off,
  gravity-off, and dynamics are unchanged. Only reporter intervals were thinned.
- Manifest: `runs/cr3si_a15_deep8_v5_free_volume_long10x/protocol_manifest.json`.

### Verification still needed

- After job start, confirm the output advances beyond step 10,000,000, proving the
  checkpoint was loaded rather than starting from the initial frame.

## Session 2026-07-15 (Codex) -- Cr3Si free-volume reruns and deep8 v2 10x continuation

**Asked:** remove the small-box/caging concern for Cr3Si A15 baseline and deep8, then
continue `cr3si_a15_deep8_v2` for about ten times its original duration on CUDA.

### Large-free-volume comparison (running)

- Added `crystal_stability/prepare_cr3si_v5_free_volume.py`.
- New versioned runs: `runs/cr3si_a15_v5_free_volume/` and
  `runs/cr3si_a15_deep8_v5_free_volume/`.
- Same 4096-particle structures and interactions; 298 K from step 0, no ramp, dt 0.15 ps,
  10,000,000 steps, CUDA. Walls and gravity are off.
- Box expanded from the densely periodic 3608 nm v4 cell to 13,429 nm at volume fraction
  0.0087, giving about 4939 nm vacuum per side around the rebuilt crystal. This is a
  finite-crystal/free-surface test, not a thermodynamic bulk NPT test.
- Slurm jobs `13809444` (baseline) and `13809445` (deep8); both running when handed off.

### Deep8 v2 long continuation (running)

- Added `crystal_stability/prepare_cr3si_deep8_v2_long10x.py`.
- New run: `runs/cr3si_a15_deep8_v2_long10x/`; original v2 files are untouched.
- Byte-identical configuration, first frame, and checkpoint copied from deep8 v2.
- Source checkpoint and last trajectory frame are at exactly 5,000,000 steps. PACSim
  advances `run_steps` *after* checkpoint load, so the job adds 45,000,000 steps for
  50,000,000 total (15 microseconds at dt 0.3 ps), exactly 10x the original duration.
- CUDA job `13809593`. Checkpoint load verified: trajectory advanced to step 5,400,000.
- Only reporter intervals were thinned to control disk use; interactions, box, temperature,
  timestep, and dynamics are unchanged. Manifest: `protocol_manifest.json` in the run dir.

## Session 2026-07-14 (Codex) -- Current matcher limits and periodic-bulk melt audit

**Asked:** clarify Th3P4/amorphous status, determine whether removing CsCl yields an
ordered-unknown verdict, and check whether the newest stability crystals melted.

### Matcher status

- Do **not** quote the invalidated v1 7/7 removal or 0/12 amorphous headlines.
- Corrected-RDF matcher-v2 external Th3P4/amorphous rows are also not valid conclusions:
  the old S(k) collapsed all 17 Th3P4 and 27 amorphous trajectories toward Ni3Sn, while
  the extractor selected 85--91% of the box and corrupted Th3P4 stoichiometry.
- Validated-S(k) matcher-v3 has so far scored only the 23 CsCl trajectories. With CsCl
  removed, zangenite is the nearest residual anchor for 23/23, but its distance is
  2.631--4.263x the true CsCl distance (median 3.636). This is separation evidence, not
  a final abstention result because repaired-score thresholds are not calibrated.
- See `matcher_v3_validated_structure_factor/no_cscl_distance_diagnostic.md`.

### Stability audit

Added `crystal_stability/quick_v4_stability_audit.py` and ran it on all 23 newest v4
directories. Results: 7 STABLE runs, 5 BORDERLINE, 9 MELTED, 2 Nickeline numerical
failures (Energy NaN, one frame only). CsCl is BORDERLINE, not melted: Lindemann 0.161,
site occupancy 1.000, q6 retention 0.981. Th3P4 is STABLE: 0.041 / 1.000 / 1.129.
See `crystal_stability/V4_QUICK_STABILITY_AUDIT.md` and
`crystal_stability/v4_quick_stability_audit.csv`.

## Session 2026-07-14 (Codex) -- Versioned S(k) repair and CsCl-only external validation

**Area:** `results/statistical_ood_benchmark_v2/matcher_v3_validated_structure_factor/`

**Asked:** execute the approved narrow next step: repair/validate S(k) without changing
failed matcher-v2 outputs, then score only Nicole's 23 independent CsCl trajectories.

### Result

- Corrected partial RDF freshly rerun: **19/19 checks pass**.
- Repaired S(k): **11/11 checks pass** (size, analytic Bragg position, relative
  intensities, deterministic ordering, periodic wrapping, thermal jitter, disorder).
- Nicole external CsCl ranking: **23/23 CsCl top-1** for RDF-only, S(k)-shape-only,
  and anchor-scaled RDF+S(k).
- Minimum combined top-1/top-2 ratio 2.63; no channel contributes more than 60% for
  the CsCl comparison. Runtime 74.4 s.

### Important correction

With equal scattering weights CsCl's total-density lattice follows BCC extinction:
the first allowed peak is (110), `k*d0 = pi*sqrt(6) = 7.695`. The earlier test expecting
the chemically contrast-weighted (100) peak at 5.44 was wrong. Chemical ordering remains
in the LL/LS/SS RDF channels.

### Scope boundary / still open

This is ranking validation, not a production acceptance benchmark. No threshold was
fitted; family removal was not claimed; Th3P4/amorphous were not scored. The next gate is
the domain extractor repair and mask audit. See
`matcher_v3_validated_structure_factor/structure_factor_validation_report.md`.

## Session 2026-07-14 (part 2) -- Thermal-ensemble matcher (0 K vs 298 K fix)

**Areas:** `results/statistical_ood_benchmark_v2/{verified_thermal_reference_library_v1,
thermal_ensemble_matcher_benchmark_v1}/`
**Asked:** fix the 0 K-reference vs 298 K-candidate mismatch WITHOUT weakening the matcher
or tuning thresholds on the seven test answers.

### Result: it worked, and the failure condition was not triggered

| metric | old (0.35 rule) | new (thermal envelope) |
|---|---|---|
| ideal-anchor top-1 recovery | 7/7 | **7/7** |
| confident match | 4/7 | **6/7** |
| family-removal abstention | 7/7 | **7/7** |
| wrong-family confident acceptance | 0 | **0** |
| all-control false crystal assignment | 0/12 | **0/12** |
| same-stoichiometry false assignment | not measured | **0/7** |
| Cu3Au independent (ONE pair) | pass | **pass** |

Predictions were PRE-REGISTERED before any thermal reference was generated. Cr3Si_A15 and
rocksalt_B1 flipped `weak_match_abstain` -> `confident_reference_match` exactly as predicted.

### Architecture (keep this)
- **Ideal CIF anchor = identity, does the RANKING.**
- **Independent thermal ensemble = tolerance, does the ACCEPTANCE.** Built from independent
  seeds (11, 12) at each family's own already-validated parameters; only the RNG seed changed.
  **A family's envelope is NEVER built from its own candidate.**
- The single `mean physical deviation < 0.35` rule is dead. Channels now have roles:
  stoichiometry = hard gate; partial RDF + S(k) = decisive; coordination = important;
  q4/q6/motif = supporting only.

### The Ni3Sn finding (most scientifically interesting)
Ni3Sn is STILL `weak_match_abstain` -- but for a completely different reason. Its thermal
envelope now PASSES; what holds it back is a **1.13x margin against Cr3Si_A15**. Ni3Sn
(D0_19), Cr3Si (A15) and Cu3Au (L1_2) are all A3B/1:3 close-packed lattices with nearly
identical partial RDFs, separated mainly by **STACKING** (hcp ABAB vs fcc ABC) -- exactly the
distinction the paper uses to separate Li3Fe-like from Cu3Au-like crystals.
**The matcher is correctly refusing to guess.** The missing evidence is a **PTM/stacking
channel**, which is designed but NOT IMPLEMENTED. That is the highest-value next step.
**Do NOT "fix" Ni3Sn by lowering the margin threshold** -- that trades a correct abstention
for a coin flip between two genuinely similar lattices.

### Also done
- 14 new independent thermal seed runs, ALL verified STABLE before use.
- Stage-8 audit: 7/12 controls pass the stoichiometry gate (so RDF/S(k) must reject them) ->
  **0/7 false assignment**. Coverage GAP: no 3:4 disordered control exists to challenge
  Th3P4/zangenite. Plan written (`SAME_STOICHIOMETRY_AMORPHOUS_GENERATION_PLAN.md`,
  ~4 short GPU jobs, awaiting approval).
- Two transient `CUDA_ERROR_MPS_CONNECTION_FAILED` job failures -- infrastructure flake,
  resubmitted, fine. Watch for this.

### Still open
- **PTM/stacking channel** (unblocks Ni3Sn; the A3B family is degenerate without it).
- Envelopes are **low-n (2 independent runs)** -- robust ranges, NOT coverage guarantees.
- Only Cu3Au has a genuine Tier-3 independent pair; the other six are Tier 1.
- 3:4 same-stoichiometry controls (approval pending).

---

## >>> PICK UP HERE (left 2026-07-14, overnight) <<<

**6 SLURM jobs queued, will finish on their own.** They generate the missing
size-pair cells for detector-v2 validation (each ~2 min GPU once it starts):

    13552765 cscl_eta081      13552766 cscl_eta081_s2
    13552619 cscl_eta085_s2   13552621 th3p4_eta085_s2
    13552734 th3p4_eta081     13552735 th3p4_eta081_s2

Already done: `runs/cscl_eta085/`, `runs/th3p4/`.
Check with: `squeue -u sz5808` then `ls crystal_discovery/crystal_stability/runs/`

**Next 3 steps when you're back:**
1. `cd results/statistical_ood_benchmark_v2/topology_conditioned_detector_v2 && python make_quality_variants.py`
   (the reviewed v2 generator — species-stratified vacancies, real seeds, group_ids,
   unwrapped truncation, stress-tests separated). Needs the 6 runs above to exist.
2. Show the **grouped split design** (fold by `group_id`, never by variant) and confirm
   whether **leave-one-size-ratio-out** is actually possible. **STOP there for review.**
3. Only then implement the two-view descriptors (`x_topology` / `x_full`) and fit v2.

**Do NOT** recompute detector v2 before step 2 is reviewed. Detector v1 stays frozen.

**Wins from the last session (all verified):**
- Reference library rebuilt: **7/7 audit PASS**, exact stoichiometry.
- Matcher: family-removal abstention **7/7 = 100%**, amorphous false-crystal **0/12 = 0%**.
- **Tier-3 Cu3Au PASSED 2/2** — identified across a 2-3% lattice change, different psi
  (+25 vs +5 mV) and ~2x Debye length (6.0 vs 11.59 nm). First real independence evidence.

**Open caveats:** all other matcher evidence is Tier 1 (same run). Detector-v2 quality
axis is only PARTIALLY balanced (no nucleated 204/240 — not run, by decision). The two
size ratios are minimal validation controls, NOT universality; a held-out intermediate
ratio is the clean interpolation test (logged, not run).



A running, append-only record of what AI agents do in this workspace: what was
built, what was decided, and what is safe to reuse. Newest session at the TOP.

**Convention for future sessions:** append a new `## Session <date>` block at the
top. Each block: what the agent was asked, what it built (with paths), what it
found, what it got WRONG (keep the wrong turns -- they are the expensive lessons),
and what is still open. Link to the detailed docs rather than duplicating them.

---

## Session 2026-07-13 (part 3) -- Rebuilt matcher: reference library + benchmark (Stages 0-8)

**Areas:** `results/statistical_ood_benchmark_v2/{verified_stable_reference_library_v1,
verified_matcher_benchmark_v1, topology_conditioned_detector_v2}/`
**Asked:** freeze v1, rebuild the matcher reference library from the verified
`crystal_stability` structures, benchmark it (recovery / abstention / amorphous), then
design a topology-conditioned detector v2 and stop for approval.

### Done
- **Stage 0:** v1 frozen; `V1_LIMITATIONS_AND_NEXT_STEPS.md` written.
- **Stage 1:** new reference library, **7/7 passed audit** with EXACT stoichiometry
  (CsCl 1:1, Th3P4 3:4, Cu3Au 1:3, Ni3Sn 1:3, zangenite 3:4, Cr3Si 1:3, rocksalt 1:1).
- **Stages 2-4:** matcher benchmarked. **Top-1 ranking 7/7 correct. Family-removal
  abstention 7/7 = 100%. Amorphous false-crystal-assignment 0/12 = 0%.**
- **Stages 5-8:** independent-validation plan, two-view feature design, atlas design,
  execution plan. **STOPPED at the approval gate — no detector recomputation run.**

### Key findings
1. **The audit caught a real bug in my own code.** Th3P4 initially FAILED with domain
   stoichiometry 1080:8121 instead of 3:4, because the order gate required `coord>=6`
   and Th (the LARGE species) sits at ~4 neighbours in Th3P4. The gate was silently
   discarding 5/6 of the Th atoms -- **the same class of defect that invalidated the
   legacy library.** Fixed by seeding on solid-like particles then taking the connected
   component of the FULL neighbour graph, so low-coordination embedded species are kept.
2. **A distance threshold alone is NOT safe.** True-match and answer-absent top-1
   distance distributions **overlap** (0.0183-0.1637 vs 0.1022-0.4692). The 100%
   abstention rate is carried by the **stoichiometry and physical-channel gates**, not
   by distance. A distance/margin-only matcher would fail here.
3. **A q6 order gate alone would have let every melted control through.** All 12 melted
   structures PASSED the order gate (87-99% "ordered" core) -- they are dense disordered
   aggregates, not gas. The RDF/S(k) + physical/stoichiometry gates rejected them.
4. **CONTAMINATION:** `crystal_stability/runs/cu3au/trajectory_big.gsd` and
   `runs/ni3sn/trajectory_big.gsd` are **steven_example_2 and steven_example_1** (exact
   match to `steven_recovered_params.json`). They are the UNKNOWN CANDIDATES the project
   is trying to identify. Using them as Cu3Au/Ni3Sn references would be circular.
   **Quarantined.**
5. **The detector-v2 factorial is unidentifiable from existing data.** Only 8 of 24
   atlas cells exist; **`CsCl @ 204/240` is entirely empty (0/6)**. Size ratio and
   crystal quality are currently **aliased** -- the atlas is all nucleated@170/210 and
   our stable runs are all relaxed@204/240. No re-featurisation fixes this; the missing
   cells must be generated. This is the decision blocking Stage 9.

### Honest weakness
Tier-1 confident-match is only 4/7. Ni3Sn/Cr3Si/rocksalt were vetoed into
`weak_match_abstain` because the physical-agreement gate compares a **0 K ideal**
reference to a **298 K thermal** candidate. That is a gate-calibration defect, not a
ranking error (top-1 was still correct for all 7). **Do not fix it by loosening the gate
against the test structures** -- fix it with thermalised references.

### Still open
- **All matcher evidence is Tier 1** (same run, same construction). No independent
  generalisation is demonstrated. A free **Tier-3 test exists today**: cross-match
  `cu3au_l12_big` (psi +5/-45, lambda 11.59) against the `cu3au_d6` reference
  (psi +25/-45, lambda 6.0) -- same crystal, different interaction conditions, no new
  simulation needed. **Run this first.**
- Approval needed on detector-v2 Option 1 (cheap, ~15 min GPU, accepts partial aliasing)
  vs Option 2 (full factorial, needs SLURM + nucleation runs that may not nucleate).

---

## Session 2026-07-13 (part 2) -- Stable-GSD x detector/CIF holdout benchmark

**Area:** `results/statistical_ood_benchmark_v2/stable_gsd_cif_holdout_benchmark_v1/`
**Asked:** use the 7 newly stabilized GSDs to test the detector + matcher pipeline with
CIF/reference holdouts, detector-atlas holdouts, and amorphous abstention (a 2x2:
"not in detector training" is NOT the same as "not in CIF library").

**Outcome: detector half (Part D) COMPLETE. Matcher half (Parts E-J) DEFERRED --
the legacy matcher references turned out to be invalid.**

### What was found

1. **The legacy matcher references are not the crystals they claim to be.** The
   library's own metadata: the "CsCl" reference has **3:1** stoichiometry (real CsCl is
   1:1) and "th3p4" has **8:3** (real Th3P4 is 3:4). They were built from nucleation
   boxes at the experimental ~3:1 composition, and the "ordered core" mask selected ~90%
   of the whole box instead of isolating a crystal. **Any prior result depending on
   these two references should be re-examined.** This confirmed the user's suspicion
   that "the old GSD files were not good".

2. **Two conflicting typeid conventions exist in the project.** Detector atlas GSDs use
   `typeid 0 = LARGE` (210 nm); matcher ground-truth GSDs use `typeid 0 = SMALL`
   (204 nm). Anything consuming both (partial RDFs gr_AA/gr_BB) will silently swap
   species. Canonicalising to the atlas convention before featurisation is load-bearing.

3. **The detector holdout works, and it repairs a documented failure.** Remove CsCl from
   the atlas -> our CsCl domain flags OOD at 14.2x threshold; remove Th3P4 -> 43.6x. The
   previous v2 README recorded leave-Th3P4-out as "the harder scientific failure" where
   global Mahalanobis and kNN *incorrectly kept it in-distribution*. With a clean relaxed
   Th3P4 domain they now flag it decisively. The nearest-family-collapses-to-amorphous
   pathology, however, persists.

4. **Relative Mahalanobis is unusable on this atlas** (19% accuracy; calls almost
   everything known). Global Mahalanobis and kNN tie. Independently reproduces the prior
   finding.

5. **Honest negative result -- the atlas does not span crystal QUALITY.** Even with
   their family present, our CsCl (2.2x) and Th3P4 (18.7x) score OOD. Their nearest
   family is still correct, so the detector knows what they are -- but the atlas contains
   only nucleated/defective crystals, and our relaxed near-ideal lattices are genuinely
   unusual relative to it. **"OOD" from this detector currently conflates "new structure"
   with "unusually perfect structure."**

6. **Confound:** six of the seven use the paper's 204/240 nm colloid pair; the atlas was
   built on 170/210. SOAP is d0-normalised (scale-free) but NOT size-ratio invariant, so
   part of their OOD signal is the colloid pair, not the crystal. **CsCl is the one
   clean test** (same pair as the atlas): 2.2x with family present -> 14.2x with family
   removed, nearest flipping CsCl->Th3P4. That holdout is uncontaminated.

### Structural finding about the atlas
Only **three real families**: CsCl (23), Th3P4 (17), amorphous (27) + gas/boundary.
So Cu3Au, Ni3Sn, zangenite, Cr3Si, rocksalt are `absent_by_default` -- there is nothing
to remove, and the full 4-cell 2x2 only exists for CsCl and Th3P4. Recorded as such
rather than faking a removal.

### Still open
- Rebuild the matcher reference library from the verified `crystal_stability` structures
  (correct stoichiometry), then run Parts E-J (matcher, calibration, abstention, amorphous).
- MMD / SOAP-REMatch still impossible: only pooled vectors cached, per-particle SOAP
  discarded. See `future_set_level_matcher_channels.md`.

---

## Session 2026-07-13 -- Colloidal crystal stabilization pipeline

**Area:** `crystal_discovery/crystal_stability/` (new)
**Asked:** understand how PACSim builds a CsCl crystal from a CIF; build a
reusable folder to relax any crystal to a stable multi-frame GSD; then find
stabilization points for a list of ~25 AFLOW prototypes.

### What was built (all in `crystal_discovery/crystal_stability/`)

| file | what it does |
|------|--------------|
| `systems/*.yaml` | the ONE file you edit per crystal (cif, radii, charges, salt, protocol) |
| `build.py` | expands a system yaml -> `runs/<name>/{configuration.yaml, run.yaml}` for PACSim |
| `design.py` | **the core tool.** Given a CIF, finds charges + Debye length that make it stable |
| `tune.py` | per-site binding-energy screen (the metric `design.py` ranks on) |
| `verify_run.py` | verdicts on a relaxed run (STABLE / CORE-ONLY / MELTED) |
| `make_cifs.py` | builds prototype CIFs from AFLOW spacegroup+Wyckoff specs |
| `check_library.py` | re-checks the task14 reference library's stability claims |
| `submit_gpu.sbatch` | build + relax on a GPU (~2 min for 500k steps) |
| `DECISIONS.md` | full reasoning path, every decision and mistake |
| `RESCUE.md` | how the melted structures were (or weren't) stabilized |
| `run_verdicts.csv` | the results table |

Usage: `sbatch submit_gpu.sbatch systems/<name>.yaml` -> `runs/<name>/trajectory.gsd`.

### The physics that came out of it

- **Set the salt from the pair well depth, not by guessing.** lambda_D is solved so
  the unlike-pair well = -5 kT (where CsCl and Cu3Au both hold). A -28 kT well
  crashes the integrator (`Energy is NaN`) via irreversible aggregation.
- **Rank on the WEAKEST species' per-site binding, never the average.** The average
  cohesive energy rates a structure whose majority species is held by 0 kT as
  "most bound." It is a trap.
- **Deeper wells rescue weak-binding lattices; nothing rescues low coordination.**
  Coordination <= 4 (zincblende, fluorite's majority site) melts at any well depth
  we can safely reach -- isotropic colloids cannot make diamond.

### Results

**Stable set (7):** CsCl, Th3P4, Cu3Au, Ni3Sn (at -5 kT -- these are exactly the
four structures the paper reports experimentally, an independent rediscovery) plus
rocksalt_B1, Cr3Si_A15, zangenite (rescued at a -8 kT well).

**Not accessible with the fixed 102/120 nm colloid pair:** zincblende, fluorite,
antifluorite, ReO3, I3P, nickeline, Fe4N.

### What the agent got WRONG (worth remembering)

1. **Equal radii for Cu3Au.** Killed the size-ratio mechanism that suppresses
   like-charge repulsion (the paper's central point). The crystal melted.
2. **Reasoned from stoichiometry (3:1) instead of the neighbour shell (8 like /
   4 unlike).** Wrong charge ratio.
3. **Ranked by average energy, then by `min()` when the weakest species is `max()`.**
   Recommended a structure whose majority species had zero binding.
4. **Read lambda_D = 12 nm out of a figure caption** that was a repulsion-comparison
   condition, not a crystallization one. Produced a -28 kT well and a NaN crash.
5. **Used RDF-vs-first-frame correlation as a melt test.** Unreliable: the first
   frame is the unthermalized ideal lattice with delta-sharp peaks, so thermal
   broadening alone tanks the correlation and flags stable CsCl as melted.
   **Trust core Lindemann instead.**
6. **Ran too-small supercells (`lattice_repeats: 8`).** For a 4-atom cell that is
   only 2048 particles -- mostly surface. Cu3Au at N=2048 reads MELTED (0.183); the
   SAME parameters at N=6912 are STABLE (0.075). Caught only because the user asked
   me to double-check the Cu3Au FCC parameters. `design.py` now sizes the supercell
   to ~6000 particles. **Any borderline verdict at N <= 4096 is untrustworthy.**

### Findings about EXISTING work in this repo

- `task14_relaxed_reference_library`: its `survived_short_md` column is **NaN for
  all 60 rows** -- the stability check was never run. And those `short_md_stability`
  frames are only ~6 ps of MD (2000 steps x 0.0032 ps), far too short for anything
  to melt. The "120/120 PASS" means files were written, not crystals held. The
  library carries **no stability evidence in either direction**.
- `data/reference_structures/zangenite_reference.gsd` is **unusable**: every
  diameter is 1.0, i.e. a bare lattice with colloid radii never applied. A corrected
  `zangenite.cif` (real element symbols; the original used `L`/`S`, which are not
  elements and break `LatticeBuilder`) is now in `crystal_stability/structures/`.

### Environment gotcha (will bite again)

The `pacsim` env's OpenMM has its plugin path baked in as
`/home/sz5808/.conda/envs/pacsim/lib`, but the env lives on `/scratch`. Without a
fix, OpenMM loads no plugins, only the slow `Reference` platform registers, and
`pacsim-run` dies with `There is no registered Platform called "CPU"`.
`crystal_stability/env.sh` works around it by exporting `OPENMM_PLUGIN_DIR`.
Permanent fix: `ln -s /scratch/sz5808/.conda/envs/pacsim /home/sz5808/.conda/envs/pacsim`

### Still open

- ~10 low-symmetry prototypes (cr2s3, zr3al2, fe3sn2, pt2ge3, zr4al3, ni4b3,
  rh4p3, al4c3, bi4te3, wal5) have free internal Wyckoff coordinates and cannot be
  built from the AFLOW label alone. They need Materials Project data:
  `export MP_API_KEY=...` then run
  `unified_pipeline/scripts/motif_pipeline/acquire_task14_mp_cifs.py`.
  (The key is NOT stored in the task14 folder -- only `MP_API_KEY_set: true`.)
- The rescued three need lower salt than the paper's conditions, i.e. assembly
  closer to the aggregation boundary. We tested "stable once formed"; assembly
  kinetics is untested.
- Colleague's MD `run.yaml` for Cu3Au was never found, so lambda_D = 6 nm was
  anchored on the CsCl pair-well depth rather than read from his file.
