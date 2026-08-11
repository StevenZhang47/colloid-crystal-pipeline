# Reference-conditioned open-set matcher v1

Adds the **acceptance layer** that `matcher_v3_validated_structure_factor` was
missing. v3 validated the repaired channels (corrected partial-RDF + deterministic
S(k), 23/23 on external Nicole CsCl) but did RANKING only — it emitted no
accept/reject label because no acceptance threshold had been fitted after the S(k)
repair. This directory closes that gap and runs the two requested tests.

`build_and_run_matcher.py` — everything below. Reads the v2/v3 channel code and the
crystal_stability reference GSDs read-only; writes only into this directory.

---

## What this directory now contains (session log)

| capability | script | status |
|---|---|---|
| Open-set matcher: rank by anchor-scaled RDF+S(k), accept/reject/abstain | `build_and_run_matcher.py` | ✅ |
| CsCl holdout (remove family → library_unmatched) | `build_and_run_matcher.py` | ✅ 23/23 |
| Steven 1/2 identification | `build_and_run_matcher.py` | ✅ Ni3Sn / Cu3Au |
| Multi-domain: separated crystals scored independently | `match_multi_domain.py` | ✅ |
| Attached DIFFERENT crystals split (CsCl+Th3P4 fused) | `match_multi_domain.py` (magnitude sub-seg) | ✅ |
| Attached SAME-structure grains split (two CsCl crystallites) | `match_multi_domain.py` (orientational grains) | ✅ |
| Amorphous negative control | `amorphous_negative_control.py` | ✅ 8/8 unmatched |
| Generalise to molecular systems (snowmen) | `snowmen_sweep_match.py` | ✅ (binary regime) |
| Full snowman sweep survey | `snowmen_sweep_match.py` | ✅ 156 dirs |

### Domain extractor — species-agnostic, no q6
`gas_safe_domain` / `order_mask` reject gas by **coordination ≥ 4** only, then take the
largest connected cluster and keep ALL its particles. The old q6-based extractor was
species-biased (Th3P4 large sublattice sits on the q6 cut → extracted L/S collapsed
0.79→0.41 → Th3P4 failed to match itself). Coordination-cluster inclusion preserves
composition (Th3P4 0.787 vs true 0.786) without distorting the RDF/S(k) patterns.

### Two kinds of domain splitting (`subsegment`)
1. **Orientational grains** — cut first-shell bonds with low complex-q6m coherence, take
   connected components. Separates two grains of the SAME structure (e.g. two CsCl
   crystallites in contact) that share identical bond-order magnitudes but differ in
   orientation. A single crystal stays one grain.
2. **Magnitude split** (fallback) — KMeans on averaged q4/q6/q8; separates two DIFFERENT
   structures fused at an interface (CsCl+Th3P4).
A split is kept only if ≥2 pieces match confidently. Relations are reported as
`separated` (crystals with a gap) vs `attached` (grains in one connected component) — so
"two CsCl crystallites joined by a few particles" reads as **2 attached CsCl domains**,
not 1.

Validated: single→1, separated CsCl+Th3P4→2, attached CsCl+Th3P4→2, two-CsCl-grain
snowman frame→2 attached CsCl.

## Snowmen generalisation & its limit — read `snowmen_sweep_analysis/REPORT.md`
Each snowman is body(+) + lobe(−) + head. Treated as a two-species ionic system
(body=large/+, lobe=small/−, head dropped) so the ionic matcher applies. Swept all
final frames; results in `snowmen_sweep_analysis/`.

**KNOWN LIMITATION (honest):** `seed_ratio` = head diameter / body diameter. At low
seed_ratio the head is small/absent → body+lobe IS the structure → binary matching is
valid and the CsCl/Th3P4 matches are trustworthy. At **high seed_ratio the head grows to
~body size → the system is TERNARY, not binary** → dropping the head makes the analysis
incomplete. All confident matches occur at low seed_ratio; high-seed-ratio rows
(disordered / novel_ordered) are **low confidence** and must not be over-read. Real fix:
include the head as a third species (3-species descriptor) or add ternary references.

## Method

**Reference library:** the 7 verified stable families, each represented by its
CIF-derived energy-optimized ideal lattice (`crystal_stability/runs/<run>/first_frame.gsd`)
as the identity **anchor**. Correct stoichiometry (audited in
`verified_stable_reference_library_v1`).

Anchor validation (the two that Steven matched):

| family | source CIF | anchor | N | stoich (L:S) | optimize_energy | build warnings |
|---|---|---|---|---|---|---|
| Ni3Sn | `ni3sn_D019.cif` | `ni3sn_d019/first_frame.gsd` | 4096 | 3072:1024 = 1:3 ✓ (D0₁₉) | true | none (.out/.err clean) |
| Cu3Au | `Cu3Au_mp-2258.cif` | `cu3au_d6/first_frame.gsd` | 6912 | 1728:5184 = 1:3 ✓ (L1₂) | true | none (.out/.err clean) |

Each anchor is a single frame (the built ideal lattice **before** any MD), so it is the
energy-optimized ideal crystal, not a run frame. The pacsim-create energy scan found an
interior minimum for both — the "minimum at scan boundary" warning did not fire.

**Channels (frozen v3):** `rdf_species` (mean EMD over LL/LS/SS partial RDFs),
`rdf_total`, `sk_shape`. All dimensionless (r/nn, k·nn), so absolute length scale
is removed.

**No stoichiometry filter.** Only radius and charge are fixed by the experiment; the
crystal structure — and therefore its L/S composition — is emergent. A candidate is
allowed to match a family of different nominal stoichiometry; the structural channels
decide. L/S is recorded as a diagnostic only.

**Two separate quantities (the "nearest reference vs accepted reference" distinction):**

- **Ranking — anchor-scaled distance** `D_f = mean_c( d_c / global_scale_c )`, where
  `global_scale_c` is the median pairwise inter-anchor distance in channel c. A GLOBAL
  scale, so it is **not** confounded by per-family envelope width. `nearest = argmin_f D_f`.
  This is v3's validated ranking metric.
- **Acceptance** — `D_nearest < 0.40`, a cutoff sitting in the empirical gap between a
  legitimate match (~0.16–0.23) and a non-match (~0.60–0.71). Plus a margin gate
  (second-best/best ≥ 1.15) to catch ambiguity.

**Open-set decision:** `matched(f*)` if `D_{f*} < 0.40` and margin ok; `ambiguous` if
the top two are too close; `library_unmatched` (ordered_unknown / novel-and-not-in-
active-library) if `D_nearest ≥ 0.40`.

## Equations

Notation: candidate domain `D`, reference family `f`, anchor = family `f`'s ideal
lattice. `W1` = 1-Wasserstein (earth-mover) distance between two curves; all curves are
made non-negative and area-normalized before `W1`. Length scale `nn` = robust first-shell
distance (median **fourth**-neighbor distance — the nearest-neighbor statistic is biased
low after thermal broadening). All radial/reciprocal coordinates are dimensionless
(`r/nn`, `k·nn`), so absolute size is removed.

**1. Corrected partial RDF** (finite domain in a gas/vacuum box), species a,b ∈ {L,S}:

```
g_ab(r) = N_ab(r) / [ N_a^centres · ρ_b · 4π r² dr · S_a(r) ]
```

`N_ab(r)` = a-centre/b-neighbor pairs in bin r; `N_a^centres` = # species-a centres (drawn
from the domain interior); `ρ_b` = local number density of species b; `S_a(r)` = shell-
containment (fraction of the radius-r shell around a centre lying inside the domain — the
finite-size truncation correction). `g_ab → 1` at large r. Total:
`g_tot = (x_L² g_LL + 2 x_L x_S g_LS + x_S² g_SS)/(x_L² + 2 x_L x_S + x_S²)`, `x` = species fraction.

**2. Structure factor** (deterministic Debye powder average over the domain):

```
S(k) = 1 + 2/(N(N-1)) · Σ_{i<j} w_ij · sin(k r_ij)/(k r_ij)          (Lorch window w_ij)
ŝ(k) = [S(k)]_+ / ∫ [S(k)]_+ dk                                       (unit-area shape)
```

**3. Three channel distances** between candidate `D` and family `f`:

```
d_rdf_species(D,f) = (1/3) Σ_{ab∈{LL,LS,SS}} W1( ĝ^D_ab , ĝ^f_ab )
d_rdf_total(D,f)   = W1( ĝ^D_tot , ĝ^f_tot )
d_sk_shape(D,f)    = W1( ŝ^D , ŝ^f )
```

**4. Global (non-confounded) channel scale** — median pairwise inter-anchor distance:

```
σ_c = median_{f≠g} d_c(anchor_f, anchor_g)          c ∈ {rdf_species, rdf_total, sk_shape}
```

**5. Anchor-scaled combined distance and ranking:**

```
D_f(D) = (1/3) Σ_c  d_c(D,f) / σ_c
f*     = argmin_f D_f(D)                              (nearest reference)
```

**6. Open-set acceptance decision** (accept cutoff `θ = 0.40`, margin `m = 1.15`):

```
matched(f*)          if D_{f*} < θ  and  D_second/D_{f*} ≥ m
ambiguous            if D_{f*} < θ  and  D_second/D_{f*} < m
library_unmatched    if D_{f*} ≥ θ                    (ordered_unknown)
```

**Diagnostic only (NOT used for ranking) — the per-family thermal envelope:**

```
τ_{f,c} = 1.30 · Q_0.90( { d_c(thermal_i^f , anchor_f) : i } )       (own 298 K frames)
ρ_{f,c}(D) = d_c(D,f) / τ_{f,c} ;   S_f(D) = max_c ρ_{f,c}(D)          (conservative max-ρ)
```

`S_f` is reported but not used to rank, because `τ` is confounded with stability — see below.

### Why not the per-family thermal-envelope τ (conservative max-ρ)?

The first version used per-family acceptance thresholds τ_{f,c} from each family's own
298 K thermal frames, with S_f = max_c(d_c/τ_{f,c}) — exactly the conservative max-ρ in
the design spec. **It failed, and the reason is scientifically important:** envelope
width is confounded with structural stability. The well-ordered families (CsCl) get a
*tight* envelope; the marginally-stable "rescued" families (Cr3Si_A15, rocksalt_B1 —
which we independently found melt) get a *huge* one (τ_sk ≈ 1.2 vs CsCl's 0.12). Dividing
distance by τ then turns the unstable families into distance magnets and **inverts the
ranking**: true CsCl scored closer to rocksalt than to CsCl, and Steven's "nearest"
came out Cr3Si/rocksalt. The τ table is still written (`calibration_thresholds.csv`) as
a diagnostic — the width confound is itself a useful readout — but ranking uses the
global anchor scale instead. Fixing the τ approach properly needs independent
finite-size acceptance ensembles per family (only CsCl/Th3P4 have them), which is the
documented next step.

## Results

### Test A — CsCl holdout (23 independent Nicole CsCl trajectories)

| condition | verdict | nearest | D_best (min / median / max) |
|---|---|---|---|
| full library | **matched → CsCl (23/23)** | CsCl | 0.158 / 0.181 / 0.227 |
| CsCl removed | **library_unmatched (23/23)** | zangenite | 0.597 / 0.665 / 0.708 |

Textbook leave-family-out: with CsCl present every trajectory matches CsCl; remove CsCl
and every trajectory moves ~3.6× further and is correctly declared `library_unmatched`
(the "it didn't find any structure" behaviour). The 0.40 cutoff sits cleanly in the gap
(0.227 max match vs 0.597 min non-match).

### Test B — Steven examples (full library)

| candidate | L/S | verdict | matched | D_best | 2nd nearest (D) |
|---|---|---|---|---|---|
| steven_example_1 | 0.364 | **matched → Ni3Sn** | Ni3Sn | 0.348 | zangenite (0.645) |
| steven_example_2 | 0.333 | **matched → Cu3Au** | Cu3Au | 0.373 | Ni3Sn (0.720) |

**Steven 1 → Ni3Sn, Steven 2 → Cu3Au**, each a confident match (D < 0.40) with a large
margin to the runner-up (1.86× and 1.93×). This independently reproduces the task14
full-consensus hypothesis, now on the repaired/validated channels and with no
stoichiometry gate doing the work.

Caveat: Steven's D (~0.35–0.37) is looser than the CsCl self-matches (~0.18), as expected
for independent experimental-condition domains, and sits just under the 0.40 cutoff. The
matches are robust in ranking (huge margins) but the acceptance cutoff has been calibrated
only against the CsCl match/non-match separation; a Steven-specific tightening would need
independent Ni3Sn/Cu3Au finite-size ensembles.

## Multi-domain (coexistence) — `match_multi_domain.py`

Does the matcher work when a frame holds more than one crystal? The single-domain
path extracts only the largest ordered cluster, so `match_multi_domain.py` adds:
order mask (q6>0.28 & coord>=6) -> connected-component segmentation on the first-shell
graph -> **re-featurise each domain in isolation** (own particles, own nn) -> match each
against the library. Tested on the synthetic frames in
`mixed_domain_diagnostics_v1/test_gsds/`:

| frame | domains found | relation | per-domain result |
|---|---|---|---|
| single_cscl | 1 | – | CsCl (D=0.175, margin 3.8) ✓ |
| single_th3p4 | 1 | – | **library_unmatched** (Th3P4 top but D=0.699) — known extractor bug |
| separated_cscl_th3p4 | **2** | **separated** | d0→**CsCl** (D=0.175); d1→Th3P4 top (D=0.699, unmatched) |
| attached_cscl_th3p4 | **1 (merged)** | – | pooled → **CsCl** (D=0.184) — WRONG, see below |
| gas_plus_cscl | 1 | – | CsCl (D=0.175), gas correctly excluded ✓ |

**Separated crystals: works.** Two crystals in one box are segmented into two domains
and matched independently. The isolation re-featurisation is confirmed sound: the CsCl
domain scores D=0.175 whether alone, beside a Th3P4 crystal, or beside a gas background —
identical to three decimals, so a neighbouring domain does not perturb the match.

**Attached crystals: does NOT work yet (honest gap).** When the two ordered cores touch
directly, the connected-component segmentation merges them into one domain, whose pooled
RDF/S(k) is dominated by CsCl and false-matches to CsCl. Fixing this needs the
sub-segmentation step from `diagnose_mixed_domains_v1` (split a mixed component into
label-contiguous subdomains), which is not yet wired into the matcher.

**Th3P4 self-match fails** in single/separated frames: the gas-safe domain extractor is
the known-broken one that mis-selects non-1:1 stoichiometry (extracted L/S=0.41 vs the
Th3P4 anchor's 0.75). Th3P4 still ranks *first* for its own domain — the ranking is
right, but the extracted composition is wrong so D never drops below the cutoff. Repairing
the extractor is the prerequisite for any Th3P4-family coexistence test.

## Amorphous negative control — `amorphous_negative_control.py`

An amorphous blob must not match any crystal family. 8 independent Nicole amorphous
trajectories, full library:

**8/8 `library_unmatched`, 0 false crystal matches.** Every one lands nearest to
zangenite at D = 0.40–0.43, i.e. just above the 0.40 acceptance cutoff. Correct, but the
margin is thin — amorphous is only ~0.02–0.03 beyond the cutoff, and the family it comes
closest to is zangenite (itself a wide-envelope "rescued" structure). So the amorphous
rejection is real but not comfortable; a slightly looser cutoff would start admitting
amorphous as zangenite. This argues for a dedicated amorphous/disorder gate (e.g. a
Bragg-coherence order gate from the S(k) raw peak) in front of the family matcher, rather
than relying on the family distances alone to reject disorder.

## Files
- `build_and_run_matcher.py` — the matcher.
- `library_cache.pkl` — cached anchors + τ (skip the ~8-min rebuild).
- `calibration_thresholds.csv` — per-family thermal-envelope τ (diagnostic; shows the
  stability/width confound).
- `cscl_holdout_results.csv` — 23 trajectories × {full, CsCl-removed}, per-family D.
- `steven_match_results.csv` — Steven 1/2, per-family D.
- `run_summary.json` — headline rates. (Its `calibration` string is stale — it names the
  superseded max-ρ method; the live method is anchor-scaled D ranking, described above.)

## Guardrails
- v2/v3 channel code and reference GSDs read-only; nothing outside this directory written.
- No candidate used to build its own envelope or scale.
- No threshold tuned on the Steven answers; the 0.40 cutoff comes from the CsCl
  match/non-match separation only.
- Stoichiometry not used as a filter (emergent, not fixed).
