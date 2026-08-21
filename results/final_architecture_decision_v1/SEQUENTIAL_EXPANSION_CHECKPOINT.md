# Sequential active-atlas expansion checkpoint

Status: **COMPLETED_A3**

Stop stage: `none`

Stop reason: none; A3 completed

The threshold at every stage is the frozen A0 operating threshold `tau0=7.982486955871957`. It is not described as a newly calibrated 5%-FPR threshold after A0.

## Stage results

### A0

- Active atlas: `CsCl;amorphous`
- Empirical old-known FPR: `0.09090909090909091`
- Family-balanced old-known FPR: `0.13333333333333333`
- Old-known identity accuracy: `1.0`
- Candidate gate: `True`
- Newly added independent post-test known: `nan`
- Newly added independent post-test correct identity: `nan`
- Changed decisions from prior stage: `0`

### A1

- Active atlas: `CsCl;Th3P4;amorphous`
- Empirical old-known FPR: `0.35714285714285715`
- Family-balanced old-known FPR: `0.4222222222222223`
- Old-known identity accuracy: `1.0`
- Candidate gate: `True`
- Newly added independent post-test known: `True`
- Newly added independent post-test correct identity: `True`
- Changed decisions from prior stage: `1`
- Newly rejected old-known condition group: `debye5.3_volumefraction.010890058823` (`rho 0.762455 -> 2.319057`)

### A2

- Active atlas: `CsCl;Th3P4;Cu3Au;amorphous`
- Empirical old-known FPR: `0.35714285714285715`
- Family-balanced old-known FPR: `0.4222222222222223`
- Old-known identity accuracy: `1.0`
- Candidate gate: `True`
- Newly added independent post-test known: `False`
- Newly added independent post-test correct identity: `True`
- Changed decisions from prior stage: `0`

### A3

- Active atlas: `CsCl;Th3P4;Cu3Au;Ni3Sn;amorphous`
- Empirical old-known FPR: `0.35714285714285715`
- Family-balanced old-known FPR: `0.3666666666666667`
- Old-known identity accuracy: `1.0`
- Candidate gate: `nan`
- Newly added independent post-test known: `True`
- Newly added independent post-test correct identity: `True`
- Changed decisions from prior stage: `1`

## Family matrix

```text
CsCl       A0  rho=0.6055842375142033 KNOWN
amorphous  A0  rho=0.7508722059340631 KNOWN
Th3P4      A0  rho=5.1119422971657595 OOD
Cu3Au      A0  rho=12.01698637764064 OOD
Ni3Sn      A0  rho=10.326484347431022 OOD
CsCl       A1  rho=0.5754228927333392 KNOWN
amorphous  A1  rho=0.7099321240840569 KNOWN
Th3P4      A1  rho=0.9955601768662636 KNOWN
Cu3Au      A1  rho=11.22790533193938 OOD
Ni3Sn      A1  rho=8.829912088429342 OOD
CsCl       A2  rho=0.556043773094521 KNOWN
amorphous  A2  rho=0.6816502742995867 KNOWN
Th3P4      A2  rho=0.98053784643174 KNOWN
Cu3Au      A2  rho=10.533245811820153 OOD
Ni3Sn      A2  rho=7.764283569498887 OOD
CsCl       A3  rho=0.5176623302569031 KNOWN
amorphous  A3  rho=0.637169219274328 KNOWN
Th3P4      A3  rho=0.9533489913130608 KNOWN
Cu3Au      A3  rho=7.786534397817497 OOD
Ni3Sn      A3  rho=0.8918584207524808 KNOWN
```

## Frozen facts

- representation retraining: none
- PCA refit: none
- gradient optimization: none
- GPU optimization: none
- CIF bank update: none
- threshold refit: none

No H-CSCL/H-NI3SN final fold, SSL/GAT expansion, Steven example, external candidate, untouched test, fused example, candidate-novel payload, or external RDF/S(k) was opened.

## Reporting audit correction

The initial one-shot report omitted one mixed condition-group transition because its decision-change table initially tracked condition-family units only. The frozen atlas states and every score were unchanged. Initial reports are preserved under `data/SEQUENTIAL_EXPANSION_INITIAL_REPORTING/`.
