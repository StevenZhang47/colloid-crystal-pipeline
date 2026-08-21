# Sequential expansion provenance audit

Status: **DEFICIT REPAIRED — GLOBAL THREE-ROLE MANIFEST ELIGIBLE, FROZEN, AND UNSCORED**

This audit used filenames, `run.yaml`, `configuration.yaml`, completion logs,
content hashes, manifests, cache manifests, access ledgers, and already-written
historical provenance. It did not decode any previously unopened trajectory frame
or calculate any new structural descriptor or score.

## Current result

| Family | Pre-repair eligible | New independent runs | Current eligible | Required | Three roles frozen? |
|---|---:|---:|---:|---:|---|
| Th3P4 | 3 | 0 | 3 | 3 | **Yes** |
| Cu3Au | 2 | 1 | 3 | 3 | **Yes** |
| Ni3Sn | 0 | 3 | 3 | 3 | **Yes** |

The original preflight correctly stopped the sequence because Cu3Au and Ni3Sn lacked
the required runs. After explicit approval, exactly that minimum deficit was repaired:
one Cu3Au run and three Ni3Sn runs. All four completed under their pre-launch freeze
and passed the unchanged fail-closed cohesion gate. The primary sequence is now
**provenance-eligible**, but no expansion descriptor or score has been calculated.

`RUNS` below means
`/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/runs`.
"Initial QC" means that a public starting lattice/first frame entered the historical
CIF build-validity inventory; no late thermal trajectory frame or identity score was
found. This does not reveal the thermal realization, so it is not by itself a blind
thermal-payload exclusion. "Late" means that one or more dynamical frames were
decoded previously.

## Deterministic eligibility and selection rule

This rule was written without opening candidate payloads or using candidate
performance.

1. A run must have `trajectory.gsd`, `run.yaml`, `configuration.yaml`, and a
   completion record reaching its configured terminal time.
2. Its independent-run key is
   `(canonical family, configuration.yaml SHA-256, randomNumberSeed,
   velocity_seed)`. Versioned/longer runs with the same key are continuations or
   reruns, not new independent observations.
3. A distinct configuration with the same random seed is a distinct physical-run
   key, consistent with the pre-existing `TierA_independent_conditions` convention.
   The shared random stream must still be disclosed.
4. Exclude a run if a late trajectory frame was previously used in representation
   training/validation, normalization/PCA fitting, OOD calibration, retrieval
   calibration/evaluation, a matcher benchmark, or a stability evaluation.
5. Initial-lattice catalog QC alone does not exclude the thermal payload. After
   selection, generic support/cohesion checks are fail-closed; failure stops the
   sequence and does not permit substitution.
6. For each family, sort eligible independent keys by integer simulation seed and
   then SHA-256 of the absolute trajectory path. Assign the first to pre-addition
   candidate, second to one-run addition, and third to post-addition test.
7. If fewer than three keys remain for any family, assign no global primary manifest
   and stop. Historically viewed seed-1/11/12 runs may only enter a later, separately
   labeled role-conditioned replication.

## Approved minimum-deficit repair

The complete pre-launch specification was frozen before Slurm submission in
`data/SEQUENTIAL_EXPANSION_SIMULATION_FREEZE.json`, SHA-256
`7c816f6acfb6f9c83092367e7242ea41a3b001ecb44906f6691041921ad8976a`.

The seed rule was performance-blind: after excluding historically present family seeds
1, 11, and 12, use the smallest unused positive integers. Velocity seeds equal the
simulation seeds. Only these four simulations were generated:

| Family | New run | Simulation / velocity seed | Frozen physical source | Final role |
|---|---|---:|---|---|
| Cu3Au | `cu3au_l12_big_seqexp_blind_seed2` | 2 / 2 | validated periodic-bulk `cu3au_l12_big_v4` | post-addition test |
| Ni3Sn | `ni3sn_d019_seqexp_blind_seed2` | 2 / 2 | validated periodic-bulk `ni3sn_d019_v4` | pre-addition candidate |
| Ni3Sn | `ni3sn_d019_seqexp_blind_seed3` | 3 / 3 | validated periodic-bulk `ni3sn_d019_v4` | one-run addition |
| Ni3Sn | `ni3sn_d019_seqexp_blind_seed4` | 4 / 4 | validated periodic-bulk `ni3sn_d019_v4` | post-addition test |

No physical parameter was tuned. Relative to the validated sources, the only changed
fields were `randomNumberSeed` and `velocity_seed`. The frozen periodic-bulk protocol
used 298 K, Debye length 11.59 nm, 0.15 ps steps, 10,000,000 steps, 500 reported
frames plus the initial frame, no walls, no gravity, and no temperature ramp.

Slurm array `15815271` completed all four tasks with exit code 0. Dependent QC job
`15815316` also completed with exit code 0. Every state table reached
`1499999.9998346088 ps`, and every stdout carried its frozen completion marker.

The unchanged v4 generic cohesion audit gave:

| Run | Frames | Particles | Lindemann | Site occupancy | q6 retention | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Cu3Au seed 2 | 501 | 6912 | 0.09130 | 1.000 | 0.89214 | **STABLE** |
| Ni3Sn seed 2 | 501 | 4096 | 0.03127 | 1.000 | 0.91370 | **STABLE** |
| Ni3Sn seed 3 | 501 | 4096 | 0.03099 | 1.000 | 0.91247 | **STABLE** |
| Ni3Sn seed 4 | 501 | 4096 | 0.03005 | 1.000 | 0.91332 | **STABLE** |

The QC table is `data/sequential_expansion_simulation_qc.csv`, SHA-256
`5e3fd7ad6a273dce8eb3cecabac2b5ed47a138dc24882bc4a7a362ee2067fb69`.
QC calculated no SOAP, PCA, Mahalanobis, OOD, prototype, retrieval, identity, or
sequential-expansion quantity.

Trajectory content hashes for the newly generated runs are:

```text
073168bf1b6fe7900ec4354b0b639d59016fed0278169edd60696092aab6c991  cu3au_l12_big_seqexp_blind_seed2
ccd7a1b432aa33d42bfef0e21bae665bcaab7b40caf647490011f60b411a4820  ni3sn_d019_seqexp_blind_seed2
74e59106d30c287dcedb14fbf7b1a44ba8f75f2e5aae103ebdc5ca0e22eba50c  ni3sn_d019_seqexp_blind_seed3
a0e43d0afa786c8a95cb90c4070e3db23f113030b4ea9c71c4552f08b124d900  ni3sn_d019_seqexp_blind_seed4
```

## Frozen global primary roles

The deterministic rule now yields:

| Family | Pre-addition candidate | One-run atlas addition | Post-addition test |
|---|---|---|---|
| Th3P4 | `th3p4_eta081` | `th3p4_eta085_s2` | `th3p4_eta081_s2` |
| Cu3Au | `cu3au_l12` | `cu3au` | `cu3au_l12_big_seqexp_blind_seed2` |
| Ni3Sn | `ni3sn_d019_seqexp_blind_seed2` | `ni3sn_d019_seqexp_blind_seed3` | `ni3sn_d019_seqexp_blind_seed4` |

The machine-readable manifest is
`data/SEQUENTIAL_EXPANSION_GLOBAL_ROLE_MANIFEST.json`, SHA-256
`5f5eca12a9e0e19eda891a864182e97369883b3943cf387a8a474b03fc6728f3`.

Two dependencies remain disclosed rather than hidden: the Th3P4 addition and test use
different physical configurations but share seed 2; the Cu3Au candidate and addition
use different physical configurations but share seed 1. These satisfy the frozen
independent-key definition but are not independent random streams.

## Historical-use audit

Flags are `Y`, `N`, or `QC` (initial-lattice QC only). `Rep` is any representation
training/validation, including the SSL encoder; `PCA` is SOAP normalization/PCA;
`OOD-cal` is OOD threshold/statistical calibration; `Retr` is retrieval calibration
or evaluation; `ID seen` means a prototype-match outcome was already observed.

### Th3P4

| Run (`$RUNS/<run>/trajectory.gsd`) | Seed / config SHA prefix | Independent key? | Rep | PCA | OOD-cal | Retr | ID seen | Coordinates opened | Blind add/test eligible? | Reason |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `th3p4` | 1 / `6df2f725` | Y | Y | N | N | Y | Y | Late | No | Final sentinel and older matcher candidate |
| `th3p4_v2` | 1 / `6df2f725` | No | N | N | N | N | N | Initial QC | No | Same key as viewed seed-1 run |
| `th3p4_v3` | 1 / `6df2f725` | No | N | N | N | N | N | Initial QC | No | Same key and incomplete trajectory |
| `th3p4_v4` | 1 / `6df2f725` | No | N | N | N | N | N | Late | No | Same key; v4 stability audit opened late frames |
| `th3p4_eta081` | 1 / `e2da2047` | Y | N | N | N | N | N | Initial QC | **Yes** | Completed, no historical late-frame result found |
| `th3p4_eta081_s2` | 2 / `e2da2047` | Y | N | N | N | N | N | Initial QC | **Yes** | Completed independent seed, no late-frame result found |
| `th3p4_eta085_s2` | 2 / `6df2f725` | Y | N | N | N | N | N | Initial QC | **Yes** | Completed distinct condition/seed key, no late-frame result found |
| `thermref_th3p4_seed11` | 11 / `6df2f725` | Y | Y | N | N | Y | Y | Late | No | SSL training plus thermal/retrieval calibration |
| `thermref_th3p4_seed12` | 12 / `6df2f725` | Y | Y | N | N | Y | Y | Late | No | SSL validation plus thermal/retrieval calibration |

If Th3P4 were run alone, the deterministic assignments would be:

1. candidate: `th3p4_eta081` (seed 1);
2. addition: `th3p4_eta085_s2` (seed 2; lower path-hash among seed-2 keys);
3. post-addition test: `th3p4_eta081_s2` (seed 2).

The two seed-2 runs have distinct configuration hashes but share RNG seed 2. That
dependence must be reported; no Th3P4 payload is opened because the global sequence
is blocked.

### Cu3Au

| Run | Seed / config SHA prefix | Independent key? | Rep | PCA | OOD-cal | Retr | ID seen | Coordinates opened | Blind add/test eligible? | Reason |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `cu3au` | 1 / `a1f64ab8` | Y | N | N | N | N | N | Initial QC | **Yes** | Completed distinct construction; no late-frame result found |
| `cu3au_v2` | 1 / `a1f64ab8` | No | N | N | N | N | N | Initial QC | No | Same key as `cu3au` |
| `cu3au_v3` | 1 / `a1f64ab8` | No | N | N | N | N | N | None | No | No trajectory payload |
| `cu3au_v4` | 1 / `a1f64ab8` | No | N | N | N | N | N | Late | No | Same key; v4 stability audit |
| `cu3au_l12` | 1 / `da50c893` | Y | N | N | N | N | N | Initial QC | **Yes** | Completed distinct construction; no late-frame result found |
| `cu3au_d6` | 1 / `5a4e07de` | Y | Y | N | N | Y | Y | Late | No | Final sentinel and older matcher candidate |
| `cu3au_d6_v2` | 1 / `5a4e07de` | No | N | N | N | N | N | Initial QC | No | Same key as viewed candidate |
| `cu3au_d6_v3` | 1 / `5a4e07de` | No | N | N | N | N | N | Initial QC | No | Same key and incomplete trajectory |
| `cu3au_d6_v4` | 1 / `5a4e07de` | No | N | N | N | N | N | Late | No | Same key; v4 stability audit |
| `cu3au_l12_big` | 1 / `8ef7fb93` | Y | Y | N | N | Y | Y | Late | No | Historical Tier-A calibration and SSL training |
| `cu3au_l12_big_v2` | 1 / `8ef7fb93` | No | N | N | N | N | N | Initial QC | No | Same key as viewed Tier-A run |
| `cu3au_l12_big_v3` | 1 / `8ef7fb93` | No | N | N | N | N | N | Initial QC | No | Same key and incomplete trajectory |
| `cu3au_l12_big_v4` | 1 / `8ef7fb93` | No | N | N | N | N | N | Late | No | Same key; v4 stability audit |
| `thermref_cu3au_d6_seed11` | 11 / `5a4e07de` | Y | Y | N | N | Y | Y | Late | No | SSL training plus thermal/retrieval calibration |
| `thermref_cu3au_d6_seed12` | 12 / `5a4e07de` | Y | Y | N | N | Y | Y | Late | No | SSL training plus thermal/retrieval calibration |

Only `cu3au` and `cu3au_l12` survive. Because three roles are required, no Cu3Au
primary assignment is made. One genuinely new independent Cu3Au simulation is the
minimum repair.

### Ni3Sn

| Run | Seed / config SHA prefix | Independent key? | Rep | PCA | OOD-cal | Retr | ID seen | Coordinates opened | Blind add/test eligible? | Reason |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| `ni3sn_d019` | 1 / `24df402c` | Y | Y | N | N | Y | Y | Late | No | Older matcher candidate and SSL training |
| `ni3sn_d019_v2` | 1 / `24df402c` | No | N | N | N | N | N | Initial QC | No | Same key as viewed seed-1 run |
| `ni3sn_d019_v3` | 1 / `24df402c` | No | N | N | N | N | N | Initial QC | No | Same key; changed construction size does not create an independent seed |
| `ni3sn_d019_v4` | 1 / `24df402c` | No | N | N | N | N | N | Late | No | Same key; v4 stability audit |
| `thermref_ni3sn_d019_seed11` | 11 / `24df402c` | Y | Y | N | N | Y | Y | Late | No | Thermal library and SSL training |
| `thermref_ni3sn_d019_seed12` | 12 / `24df402c` | Y | Y | N | N | Y | Y | Late | No | Thermal library and SSL training |
| `ni3sn` | N/A | N/A | N | N | N | N | N | Historical build files | No | No compatible GSD thermal trajectory |

Ni3Sn has no blind eligible run. Three new independent Ni3Sn simulations are needed.
The final-study H-NI3SN fold remains closed; the disqualification comes from older
matcher/library use and cross-fold SSL training, not from opening that fold here.

## Full trajectory SHA-256 inventory

Content hashing was byte-level provenance only; no GSD frame was decoded.

```text
af6c4784eeb61171bda752394a4e3a1ce531ba8fcc72ba738f612a35d823c238  th3p4
62f32db90e2eb1b5833ca3f5d8e32d0602e96df59e150c7e0d7379a10d641d4b  th3p4_v2
f0ce3032dad4f70f57782a0d20d89b862a6f5d548253d9a7b327a96d59e5cbd7  th3p4_v3
c2835a49f85bb984b954f1db315aa7bafe0ca61918a2b4cf9af3bc2528197d11  th3p4_v4
042a6913892adf5569cb0aa473e6c2cfba953b0860fbf705332c03170298ce75  th3p4_eta081
f83c30e1248bd0611545aee12fd8f363ad3c516ea1346f453c15cc8a24dc42c4  th3p4_eta081_s2
6c8e158373a3d480c781234f738cf9b8049d2944be8bb92932849b5568d279be  th3p4_eta085_s2
e627e93b383ed076dbd9b4eed1d5eb6c4e843ba20403f143d3d92245e79ba2c6  thermref_th3p4_seed11
89ab0321ef0451467bf08abe098c7e9c227866e9f53ff165cfc030572d31db67  thermref_th3p4_seed12
c879d719bf020fa85c395123ba130c282dbb56377deff99162a0c41276f591ae  cu3au
f86f700fd67284cb49a4366904c6ce7f0c82e87d88483df49250554c4ceec4af  cu3au_v2
816367020e9a6cc9f786b0c88dd89321fc85cfc95f1fd6cb2b776f7a9f6d6d4e  cu3au_v4
fe2aaad632b7e6b9b301018953826f3c0169bec69dee629ba9beb407a2f6599d  cu3au_l12
19f54894ff4f2df2bd4c2f64762955750186d1d7536dede7aa2500756ea7caed  cu3au_d6
ab0645855700e53daee3f62de96861772f40c07a4af32b4bd790c68de0e4896c  cu3au_d6_v2
6ef3c78a74203504f4dfc77743318d57b208c72fdae05f90d6245891df4e0567  cu3au_d6_v3
45de061f891df794fe83fd70216a22c317c3f7b1badf5da6ef7c95dfc76aad7a  cu3au_d6_v4
1a3f6468e10f5a61c6ab646064c301dd2575a8ff6257c4db215b04f32452d9e7  cu3au_l12_big
4af99e7987b5104c79cc73c1d9ea29cc7bdfddbe609dd66684352bd7d2811248  cu3au_l12_big_v2
fcc450b94100c7e26b9bf8106f4d1e80bc6cbd0ccc0f7383d7c4b1aa34f48195  cu3au_l12_big_v3
74530e4e2a6497c26ce9960ab3132284313017edf58e26a1a23c192f00c9749d  cu3au_l12_big_v4
6fda876923702da51cc4d6454f5e02394e316a5e51c4aa32a5ea8a4bad275194  thermref_cu3au_d6_seed11
053eb66665be706c9b8c4fbbb94233dcddb4b941b3e74b7cee87da0136c466d7  thermref_cu3au_d6_seed12
a26896d003315f5b656f513889823afe305ce604ae42b52c90c3172ed7d3b8e1  ni3sn_d019
95da3f1f5beb1559351b1e385bf7d49423127e53297be5af28d2d81970c9f6d3  ni3sn_d019_v2
511af2ba3855a89bb74965ddf479b912a3358291e725b046b9a7a08843b14aa7  ni3sn_d019_v3
833bb00ebb8e04347bd0770ad1a26eedca2384a8e6f40ea47b83bbbaddb975cc  ni3sn_d019_v4
52ded59c3e8888b7e143fc47f24dbf83eabfa983e9e436c9c095467dc89d9946  thermref_ni3sn_d019_seed11
a3e41a4b438d3a1ea13fe4aad9d87c4bbf792a383b2e93aff9a637402e3bb14a  thermref_ni3sn_d019_seed12
```

## Firewall conclusion

- The global primary run manifest is now provenance-eligible and frozen at exactly
  three roles for each of Th3P4, Cu3Au, and Ni3Sn.
- The only coordinate access added during repair was the preregistered generic v4
  cohesion QC for the four new runs. No selected run has been featurized or scored.
- No selected Th3P4 or pre-existing Cu3Au late payload was opened opportunistically.
- Seeds 11/12 remain available only for a separately authorized, role-conditioned
  replication with confirmatory identity endpoints.
- H-CSCL, H-NI3SN scoring, SSL/GAT expansion, external candidates, Steven examples,
  untouched v3 tests, fused examples, candidate-novel data, and external RDF/S(k)
  remain closed.
