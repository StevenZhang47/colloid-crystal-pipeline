# CIF prototype library

Canonical, deduplicated home for every crystal prototype used anywhere in this
project, plus the provenance needed to answer "which CIF did method X actually
read?" — which was previously unanswerable.

## Layout

```
cif_library/
  README.md                    this file
  catalog.csv                  one row per DISTINCT CIF content
  audit_ambiguity.csv          prototypes with >1 distinct CIF, resolved
  audit_coverage.csv           target prototype list vs what we hold
  audit_build_validity.csv     every built crystal, cohesive or exploded
  prototypes/<slug>.cif        the deduplicated CIFs
  build_catalog.py             regenerates catalog.csv + prototypes/
  audit_library.py             regenerates the three audit CSVs
```

`catalog.csv` columns worth knowing:

| column | meaning |
|---|---|
| `slug` | canonical key; `<stem>__<hash8>` when two files share a stem |
| `content_sha256` | first 16 hex of the file hash — the real identity |
| `n_duplicate_files` | how many byte-identical copies exist in the tree |
| `canonical_source` | where the copy was taken from |
| `spacegroup_symbol/number` | determined **from coordinates**, not the CIF header (symmetry-expanded CIFs are routinely written as P1) |
| `large_small_ratio` | minority:majority = **LARGE:SMALL** colloid, matching `FAMILY_STOICH` |
| `axis_groups`, `n_free_lattice_params` | from the patched `LatticeBuilder._axis_groups` |
| `requires_anisotropic` | `True` when `n_free_lattice_params > 1` |
| `crystal_stability_runs`, `consumed_by` | which runs and which downstream methods use this exact content |

## Starting state

**21 distinct CIF contents spread over 161 files.** Four prototypes had more than
one distinct CIF, so "the Th3P4 reference" was ambiguous.

## Anisotropic scaling

The reference builder optimises lattice scale by energy. Until
`hockyg/PACSim@582a2d0` it searched a **single uniform scale factor**, which
preserves the CIF's axial ratios — ratios that describe the *atomic* crystal and
are generally wrong for a colloidal crystal of the same structure type, because
the ideal ratios depend on the particle radius ratio.

`anisotropic_energy: true` relaxes each symmetry-distinct axis separately.

- **Cubic → 1 free parameter, bit-identical with the flag on** (regression-tested
  upstream, and reproduced here). Setting the flag on a cubic system is a no-op.
- Hexagonal / tetragonal → 2. Orthorhombic → 3.

**Therefore: set `anisotropic_energy: true` unconditionally on every new build.**
It changes nothing for cubic prototypes and is required for the rest. The flag is
opt-in and defaults to `False`, so existing `configuration.yaml` files will *not*
pick it up on a rebuild — it must be added.

Measured effect on the two non-cubic references currently in the matcher:

| | change |
|---|---|
| Ni₃Sn (hexagonal) | **none** — already at its colloid optimum |
| Zangenite (orthorhombic) | a −9.2%, b −6.9%, c +12.6%; **c/a 0.480 → 0.596** |

6 of the 21 library CIFs are non-cubic and need the flag.

## The four ambiguous prototypes, resolved

None is a wrong structure. All are cell-setting or labelling differences — but
each still changes what gets built, which is why they matter.

| prototype | verdict |
|---|---|
| **Cu₃Au** ×3 | Structurally identical (StructureMatcher). Lattice constants differ by ≤1% (3.735 / 3.748 / 3.774 Å), which the energy scan removes anyway. **Benign.** |
| **Th₃P₄** `Th3P4.cif` ×2 | Byte-different, structurally identical. **Benign.** |
| **Th₃P₄** vs `th3p4_anchor.cif` | **Different cell setting**: 28-site conventional cubic (90°) vs 14-site primitive rhombohedral (109.5°). Same crystal, but `lattice_repeats: 8` means 8³ of *different* cells — different crystal size and shape. **Pick one deliberately.** |
| **Zangenite** ×2 | Identical geometry (5.09 / 9.23 / 2.54 Å, 14 sites). Differ **only in species labels**: `L`/`S` placeholders vs `La`/`S`. Build configs key radii and potentials by element symbol, so they are **not interchangeable in a config**. The matcher uses the `La`/`S` variant. |
| **CsCl** vs `cscl_B2_anchor` | Same formula, sites, lattice and angles; StructureMatcher still separates them. Almost certainly an origin/site-ordering difference. **Lowest priority, but unconfirmed.** |

## Build validity

All **122** built runs under `crystal_stability/runs/` pass the cohesion check:
nearest-neighbour spacing is 1.02–1.11 × the steric contact distance. None shows
the exploded-lattice failure mode (a cell so dilute that nothing interacts, which
is what the uniform scan produced for AlB₂ upstream).

Thresholds used: `≤1.25` cohesive, `≤2.0` suspect-dilute, `>2.0` exploded.

## Coverage against the 31-prototype target list

See `audit_coverage.csv`. **Matching is by space group + stoichiometry, which is
necessary but not sufficient** — several distinct prototypes share both:

- Fluorite and antifluorite are both `Fm-3m` 1:2 (species-swapped)
- Cu₃Au (L1₂) and ReO₃ (D0₉) are both `Pm-3m` 1:3
- Ni₃Sn (D0₁₉) and Li₃Fe are both `P6₃/mmc` 1:3

So a `have = True` row means "a CIF with the right symmetry and stoichiometry
exists", not "the right prototype is confirmed". Rows listing multiple
`matched_cif` values need a StructureMatcher check against a trusted reference
before being treated as covered.

## Fetching the missing prototypes

**Blocked: no Materials Project credentials on this system.** No `MP_API_KEY`
environment variable, no `~/.pmgrc.yaml`, no `mp_api` or `pymatgen.ext.matproj`
usage anywhere in the tree, and `mp_api` is not installed in either conda
environment. The single `Cu3Au_mp-2258_conventional_standard.cif` was evidently
downloaded by hand.

To enable fetching:

```bash
pip install mp-api                       # into crystal_ml or pacsim_fep
export MP_API_KEY=<key from materialsproject.org/api>
```

Once that exists, prototypes should be pulled **by `mp-id`, not by formula** —
formula search returns many polymorphs and picks the wrong one silently.
