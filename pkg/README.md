# crystal-discovery

One command over a trajectory:

```bash
crystal-discovery analyze trajectory.gsd
```

```
Detected domains: 1

Domain 0
==============================================================
  particles 481   SOAP centres 222   L/S ratio 0.345

  SEEN BEFORE      NO - novel
                   OOD score 42.446   tau 6.7152   rho 6.321   [OOD]

  IDENTIFIED AS    Ni3Sn
                   distance 0.0497   margin 4.628
  VERIFICATION     PASS

  RANKED GUESSES
     #  prototype             distance   vs #1  verdict
     1. Ni3Sn                   0.0497   1.00x  best match
     2. al3ti_d022              0.2300   4.63x  beyond accept range
     3. mosi2_c11b              0.2674   5.38x  beyond accept range
     4. Cu3Au                   0.2809   5.65x  beyond accept range
     5. bif3_d03                0.2816   5.67x  beyond accept range
        anchors: li3fe, al3ti_d022, mosi2_c11b
        gates: distance <= 0.1619, margin >= 2.45
```

That is the whole workflow: the detector says it has never seen this structure, and the
matcher says it is Ni3Sn anyway. Those are different questions and the answers are allowed
to disagree.

## Install

```bash
python -m pip install -e pkg
crystal-discovery info                        # the frozen model this build will use
crystal-discovery analyze traj.gsd --json     # machine-readable
crystal-discovery analyze traj.gsd --top-k 10 # more ranked guesses
crystal-discovery analyze traj.gsd --frame 250
```

## Reading the output

| line | question it answers |
|---|---|
| `SEEN BEFORE` | is this structure in the training atlas? `rho = d/tau`, novel when `rho > 1` |
| `IDENTIFIED AS` | which library prototype it is, or `nothing in the library` |
| `VERIFICATION` | `PASS`, or exactly which gate failed and by how much |
| `RANKED GUESSES` | the full ranking, with each candidate's distance relative to rank 1 |

The ranking is the part to read when a result looks surprising. A rank 1 sitting just
ahead of rank 2 means the library cannot separate those two on this domain, however small
the absolute distance is — which is why the margin is a gate and not a footnote.

`SEEN BEFORE: YES` with `IDENTIFIED AS: nothing` is the normal amorphous answer, and is
not a contradiction: amorphous domains are in the atlas, so the detector recognises them,
but no crystalline prototype describes them.

## Pipeline

```
GSD -> gas screen -> M3 reliable centres -> SOAP 952-D
                                              |
             +--------------------------------+--------------------------------+
             |                                                                 |
    DETECT: in the atlas?                                    IDENTIFY: which prototype?
    PCA16 -> 32-D fingerprint                                one-sided chamfer, median
    Mahalanobis vs frozen Gaussian                           vs 41 CIF-ideal anchors
    rho = d/tau, novel if rho > 1                            distance gate + margin gate
```

Detection and identification are separate on purpose. The detector never names a family;
retrieval names the nearest prototype and abstains when nothing is close enough.

## The matcher

For every centre in the candidate, the distance to its nearest environment in the anchor;
the **median** of those, per species, averaged over species.

It is not a distance between summaries. The previous matcher averaged each sublattice to
one 1904-D vector and compared centroids — which is what let amorphous domains through,
since a diffuse cloud whose *centre* lands near a prototype's centre reads as that
prototype no matter how spread out it is. Chamfer forms no centroid, so a candidate is
charged for every particle whose environment nothing in the library can account for.

Two choices, both measured rather than argued:

**One-sided, candidate → anchor.** The reverse question — *are the anchor's sites present
in the candidate?* — is satisfied trivially by a diffuse cloud, so being more disordered
makes it pass more easily. Measured on 50 held-out trajectories against 41 anchors:

```
forward (used)   1.44x     reverse   1.02x     symmetric (textbook Chamfer)   1.25x
```

**Median, not mean.** A clean CsCl crystal has median nearest-anchor distance 0.117 but
p99 = 0.726 — about 1% of centres are defects or surface particles M3 missed, and they
inflate the mean. Amorphous is uniformly far and has no such tail to lose, so the median
tightens only the crystal side: gap 1.44 → 1.50. A 90% trimmed mean scores identically,
so the median is simply the version with no free parameter.

## Acceptance test

Reproduce with `python pkg/verify_acceptance.py`. On the 50 **held-out validation-split**
trajectories, run through the shipped code path:

```
CsCl       13/13  ->  CsCl                 PASS
Th3P4      11/11  ->  Th3P4                PASS
amorphous  26/26  correctly rejected       PASS
```

Both gates were fitted on that split and each separates the populations completely:

```
             crystals      amorphous     window             gap
distance     <= 0.1294     >= 0.1943     [0.1294, 0.1943]   1.50x
margin       >= 3.309      <= 1.590      [1.590, 3.309]     2.08x
```

## Frozen artifacts

| artifact | what | size |
|---|---|---|
| `results/motif_pipeline/core3_atlas_v3/frozen/ood_model_v3A_p99.npz` | species scaling, PCA16, Ledoit-Wolf location/precision, tau | 180 KB |
| `references/chamfer_bank.npz` | 41 anchors, 9512 distinct environments, float16 | 17 MB |

The bank stores environments, not particles. An ideal lattice has few distinct
neighbourhoods — every particle on a Wyckoff site has the same one — so a greedy cover at
radius 0.02 collapses lib_cscl's 2744 particles to 55 representatives. Rebuild with
`python pkg/build_chamfer_bank.py`.

## Design notes

**Nothing is reimplemented.** `descriptors.py` delegates preprocessing and SOAP to the
frozen modules that produced the frozen model. A second implementation is a second thing
to drift.

**Retrieval collapses to one row per prototype.** The library holds several cell settings
of the same structure (three Cu3Au, four Th3P4, two CsCl). Two settings sit at ~0 from
each other, which would break the runner-up margin test for a reason having nothing to do
with the candidate.

**Anchors are not preprocessed.** They are CIF-perfect, so every particle the gas screen
removes is a false positive. Candidates *are* preprocessed, and that asymmetry was tested
rather than assumed — removing candidate preprocessing made every method worse, taking
`soap_mean` from 0 to 26 of 26 amorphous domains leaking through.

**Thermalising the anchors was tried and rejected.** Running all 42 through their own MD
protocol gave a comparable gap (1.53 vs 1.50) but only 14 of 43 anchors survived at 298 K,
and one external candidate dropped from accepted to borderline. The 0 K anchors with a
full library are the better configuration; chamfer's per-particle lookup already tolerates
thermal displacement.

## Limitations

Run `crystal-discovery info` — these are carried in the freeze record.

- **Open frameworks are not supported.** The gas screen assumes ~4 near contacts and M3
  protects interior particles via `same_coord < 13`, a clause no open structure ever
  satisfies. On a 14-family prototype benchmark the seven failures were exactly the seven
  families that lose most of their particles to screening (zincblende_b3: 2048 -> 168).
  The acceptance test covers CsCl, Th3P4 and amorphous only.
- **The detector is validated on three well-separated families.** The crowded 4-5 family
  regime, where sequential expansion saw known-FPR rise 0.09 -> 0.36, is not covered.
- **Four prototypes are unrealisable with the 102/120 nm colloid pair** — the three Laves
  phases and NaTl. Independently confirmed: they were the only anchors that disintegrated
  completely under thermalisation (contact retention 0.000).
