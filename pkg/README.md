# crystal-discovery

One command over a trajectory:

```bash
crystal-discovery analyze trajectory.gsd
```

```
Detected domains: 1

Domain 0
---------
Particles: 1,006
SOAP centres: 525   L/S ratio: 1.108
OOD score: 2.736   (tau = 6.7152, rho = 0.407)
Status: KNOWN

Nearest CIF prototypes:
1. CsCl               distance=0.117
2. Th3P4              distance=0.422
3. zangenite          distance=0.444

Retrieval: CsCl   (margin 3.625)
Verification: PASS
```

## Install

```bash
python -m pip install -e pkg
crystal-discovery info          # shows the frozen model this build will use
```

## What it does

```
GSD -> gas screen -> M3 reliable centres -> SOAP 952-D -> PCA16 -> 32-D fingerprint
                                                                        |
                            +-------------------------------------------+
                            |                                           |
                   DETECT: in the atlas?                    IDENTIFY: which prototype?
                   Mahalanobis vs frozen Gaussian           1904-D fingerprint vs 37 CIF anchors
                   rho = d/tau, OOD if rho > 1              nearest, with a margin test
```

Detection and identification are separate on purpose. The detector never names a family;
it only says whether the domain resembles anything in the atlas. Retrieval names the
nearest prototype, and abstains when nothing is close enough — a novel structure still
has a nearest neighbour, and reporting it as an identification would be wrong.

## Frozen configuration

Every constant is in `configs/production.yaml`. The two shipped artifacts:

| artifact | what |
|---|---|
| `results/motif_pipeline/core3_atlas_v3/frozen/ood_model_v3A_p99.npz` | species scaling, PCA16, Ledoit-Wolf location/precision, tau |
| `references/prototype_fingerprints.npz` | 37 CIF-ideal anchor fingerprints, 1904-D |

```
tau = 6.715183      known false-positive rate 0.04 (2 of 50 validation domains)
```

Validated separation:

```
known validation domains    rho 0.360 -  1.019    accepted
Th3P4-look candidates       rho 2.551 -  4.928    rejected
K4C60 (external crystal)    rho         22.492    rejected, retrieval abstains
```

## Design notes worth knowing

**Nothing is reimplemented.** `descriptors.py` delegates preprocessing and SOAP to the
frozen modules that produced the frozen model, rather than carrying a second copy. A
second implementation is a second thing to drift.

**The two sublattices are averaged separately.** The fingerprint is
`[mean LARGE | mean SMALL]`, not a pooled mean. Measurement on this library showed the
discriminating signal lives almost entirely in the minority sublattice; pooling discards
it, which is why radial-only channels confuse Cu3Au with Al3Ti.

**Retrieval collapses to one row per prototype.** The library holds several cell settings
of the same structure (three Cu3Au, four Th3P4, two CsCl). Two settings of one prototype
sit at distance ~0 from each other, which would break the runner-up margin test for a
reason having nothing to do with the candidate.

## Limitations

Run `crystal-discovery info` — they are carried in the freeze record, not buried here.
The two that bite most often:

- the gas screen assumes ~4 near contacts, so **open frameworks lose their majority
  sublattice** before the descriptor sees them (on an ideal ReO3 anchor: 5324 particles
  in, 1330 out);
- the detector is validated on **three well-separated families**. The crowded 4-5 family
  regime, where sequential expansion saw known-FPR rise 0.09 -> 0.36, is not covered.
