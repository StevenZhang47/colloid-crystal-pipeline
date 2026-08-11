# frozen_gas_m3_pipeline_v1 — continuous gas-removal + M3 surface-center pipeline

One frozen, phase-blind preprocessing pass that turns a raw frame into the set of **SOAP pooling
centers**: it removes gas, keeps the material, and flags exterior/surface particles so only clean
interior sites are pooled. Combines two previously-frozen pieces into a single call:

- **gas removal** — `radius_normalized_coord_r4_gas_screen_v1`
- **M3 surface filter** — `phase_blind_coord_exposure_v1` (`exposure > 0.14 AND same_coord < 13`)

**Status:** FROZEN. `gas_m3_pipeline.py` sha256 `9244c1527f857d46fcd280c15bcd8985c7436685d1b5ed81cdd6df46b1c40696`.
Self-test: `python gas_m3_pipeline.py` → PASS (reproduces the validated removed-counts; see
`pipeline_selftest.csv`).

---

## Quick use

```python
import sys; sys.path.insert(0, ".../results/motif_pipeline/frozen_gas_m3_pipeline_v1")
import gas_m3_pipeline as P

# positions (n,3), diameters (n,), box: length-3 (ortho) / length-6 (HOOMD tilt) / 3x3 / None (isolated)
res = P.run(positions, diameters, box, type_ids=type_ids)   # type_ids optional (species by diameter)

res.is_soap_center        # bool[n]  -> POOL these particles as SOAP centers
res.excluded_as_surface   # bool[n]  -> predicted surface (removed from pooling only)
res.retain_mask           # bool[n]  -> survived the gas screen (non-gas)
res.exposure, res.same_coord, res.component_id, res.canonical_species_id
res.summary()             # dict: n_gas, n_non_gas, n_excluded_surface, n_soap_centers, n_components, ...
res.to_dataframe()        # per-particle table
```

Reading a GSD frame is the caller's job (e.g. `gsd.hoomd`): pass `fr.particles.position`,
`fr.particles.diameter`, `fr.configuration.box`, `fr.particles.typeid`.

### The one rule you must honor downstream — the mask is REMOVABLE
Compute and **keep** SOAP for **every non-gas particle**. Surface particles stay in the neighbourhood as
context; `is_soap_center` only chooses which particles are **pooled/averaged** as centers. Never delete
SOAP rows — store `excluded_as_surface` as metadata so any cut is reversible.

---

## How it works (4 steps)

```
raw frame
  [1] gas screen        radius-normalized coordination + mean_r4* -> drop diffuse "gas"
  [2] components        connected components on retained particles, physical graph s<1.62 (phase-blind)
  [3] geometry          per-particle over SAME-component s<1.35 neighbours:
                          same_coord = # contacts
                          exposure   = || Σ_j unit(r_j - r_i) || / same_coord      (one-sidedness, 0..1)
  [4] M3 filter         exclude_center := exposure > 0.14  AND  same_coord < 13
                        SOAP centers = retained non-gas AND NOT excluded
```

- **`exposure`** measures directional one-sidedness of the neighbour shell. A buried atom in a *dense*
  lattice has a balanced shell → exposure ≈ 0; a surface atom is one-sided → exposure high.
- **`same_coord`** is the plain contact count within the particle's own physical component
  (radius-normalized graph `s = d/(R_i+R_j) < 1.35`).
- The **AND of both signals** is the key. Exposure alone fails on **open frameworks** (e.g. Zangenite),
  whose interior atoms are one-sided *even when fully coordinated* (intrinsic bulk anisotropy ≈ 0.19). The
  coordination term rescues them: a porous deep core has `same_coord ≥ 13` and is protected, while true
  surface is both one-sided **and** under-coordinated, so it is removed. This lets the exposure threshold
  drop to 0.14 (recovering far more real surface) without ever cutting into the protected core.

### Phase-blind by construction
No crystal phase, construction domain, GAT/Nicole label, or `BULK_COORD[phase,species]` enters any step.
The only inputs are geometry (positions, physical radii/diameters, box) and species-by-size. Coordination
and exposure are computed over the **physical connected component** from the gas screen, not any known
domain. Frozen constants (`M_GRAPH=1.35`, `TAU_E=0.14`, `TAU_C=13`, plus the gas-screen constants) are not
retunable.

---

## Why these thresholds (validation summary)

Selected in `../phase_blind_surface_filter_v1` (see `reports/FINAL_SURFACE_FILTER_DECISION.md`). On the 7
labelled crystal families, with all in calibration:

- **Surface recall:** worst-family **0.934** (Ni3Sn), macro **0.979**, Zangenite **0.997**.
- **Core safety:** strict-deep-core false-positive rate **0.0 on every family and species** (0/210 Zangenite
  deep cores removed). OVITO renders confirm **zero over-removal** on all shapes.
- **Beats the alternatives:** exposure-only (worst-family 0.820) and the phase-blind GF4 (0.353).
- **Improves downstream SOAP matching** (frozen paired-site matcher, mask-swap only, 62 domains): top-1
  **0.871** and accepted-correct **0.661** — best of all masks; keeping *all* centers collapses accepted-
  correct to 0.145 and cannot identify Zangenite or Cr3Si_A15. Surface removal genuinely helps.

---

## Scope and caveats (read before deploying beyond the current library)

1. **Calibrated current-library filter, not universally proven.** Thresholds are tuned on the 7 families.
   In strict leave-one-family-out, holding out **Th3P4** breaks (its deep cores sit at `same_coord ≈ 9`):
   the family that stresses the filter must be represented in calibration. Deploy within/near the calibrated
   library; treat a genuinely novel open lattice as out-of-scope until added to calibration.
2. **Open frameworks are removed aggressively.** Zangenite loses ~74% of particles (it is genuinely porous —
   most sites are surface-like); only the coordinated backbone is kept. This is correct, not over-removal,
   but expect low center counts for such structures.
3. **A species can be fully removed in a domain.** Zangenite's SMALL sublattice is entirely surface-exposed,
   so a domain can end with 0 SMALL centers (the matcher still IDs it from the LARGE framework). If you need
   a guarantee, add a per-species center floor (keep the N highest-`same_coord` centers per species) — the
   removable-mask design already makes this safe.
4. **Coordination reference = physical component.** On single-crystallite domains this equals the cached
   `same_coord`; on multi-domain frames the component is the phase-blind physical merge (never construction
   domain).

The old phase-aware F4 (`frozen_f4_surface_center_v1`) and the earlier exposure-only rule
(`phase_blind_surface_center_exposure_v1`, τ=0.190) are **superseded** by this pipeline and kept only for
provenance / as non-deployable diagnostics.

---

## Files
- `gas_m3_pipeline.py` — the frozen pipeline (`run(...)`, `PipelineResult`, `_selftest()`).
- `pipeline_selftest.csv` — end-to-end reproduction of validated removed-counts.
- `README.md` — this file.
