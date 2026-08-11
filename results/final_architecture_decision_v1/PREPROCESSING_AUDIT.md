# Phase 2 preprocessing audit

Exactly two preregistered screens were compared; no parameter sweep was run.

| control                        | screen                 |   n_input |   n_retained |   retention_fraction |   n_retained_components |   largest_retained_component | criterion_pass   | obvious_failure_mode                               |
|:-------------------------------|:-----------------------|----------:|-------------:|---------------------:|------------------------:|-----------------------------:|:-----------------|:---------------------------------------------------|
| dense_CsCl                     | A_frozen_four_neighbor |      6750 |         6750 |            1         |                       1 |                         6750 | True             | none                                               |
| dense_CsCl                     | B_cohesive_component   |      6750 |         6750 |            1         |                       1 |                         6750 | True             | none                                               |
| dense_Cu3Au                    | A_frozen_four_neighbor |      6912 |         6909 |            0.999566  |                       1 |                         6909 | True             | none                                               |
| dense_Cu3Au                    | B_cohesive_component   |      6912 |         6912 |            1         |                       1 |                         6912 | True             | none                                               |
| dense_rocksalt                 | A_frozen_four_neighbor |      4096 |         4096 |            1         |                       1 |                         4096 | True             | none                                               |
| dense_rocksalt                 | B_cohesive_component   |      4096 |         4096 |            1         |                       1 |                         4096 | True             | none                                               |
| Th3P4                          | A_frozen_four_neighbor |     14336 |        14336 |            1         |                       1 |                        14336 | True             | none                                               |
| Th3P4                          | B_cohesive_component   |     14336 |        14336 |            1         |                       1 |                        14336 | True             | none                                               |
| zangenite                      | A_frozen_four_neighbor |      7168 |         7166 |            0.999721  |                       1 |                         7166 | True             | none                                               |
| zangenite                      | B_cohesive_component   |      7168 |         7168 |            1         |                       1 |                         7168 | True             | none                                               |
| ideal_ReO3                     | A_frozen_four_neighbor |      5324 |         1330 |            0.249812  |                    1330 |                            1 | False            | retention_below_0.95;fragmented_largest_below_0.90 |
| ideal_ReO3                     | B_cohesive_component   |      5324 |         5324 |            1         |                       1 |                         5324 | True             | none                                               |
| ideal_cuprite                  | A_frozen_four_neighbor |      6000 |         1729 |            0.288167  |                    1729 |                            1 | False            | retention_below_0.95;fragmented_largest_below_0.90 |
| ideal_cuprite                  | B_cohesive_component   |      6000 |         5994 |            0.999     |                       2 |                         3000 | False            | fragmented_largest_below_0.90                      |
| gas_debye5.15_phi.008655_seed2 | A_frozen_four_neighbor |      1024 |          244 |            0.238281  |                      10 |                          229 | False            | false_cohesive_component                           |
| gas_debye5.15_phi.008655_seed2 | B_cohesive_component   |      1024 |          246 |            0.240234  |                       1 |                          246 | False            | false_cohesive_component                           |
| gas_debye5.15_phi.008655_seed3 | A_frozen_four_neighbor |      1024 |           93 |            0.0908203 |                      28 |                           14 | True             | none                                               |
| gas_debye5.15_phi.008655_seed3 | B_cohesive_component   |      1024 |           21 |            0.0205078 |                       1 |                           21 | False            | false_cohesive_component                           |
| gas_debye5.15_phi.002704_seed2 | A_frozen_four_neighbor |      1024 |            0 |            0         |                       0 |                            0 | True             | none                                               |
| gas_debye5.15_phi.002704_seed2 | B_cohesive_component   |      1024 |            0 |            0         |                       0 |                            0 | True             | none                                               |
| gas_debye5.2_phi.002704_seed2  | A_frozen_four_neighbor |      1024 |            0 |            0         |                       0 |                            0 | True             | none                                               |
| gas_debye5.2_phi.002704_seed2  | B_cohesive_component   |      1024 |            0 |            0         |                       0 |                            0 | True             | none                                               |

## Frozen decision

- Alternative preserved every positive: **False**
- Alternative rejected all four gas controls: **False**
- Replacement eligible: **False**
- Study-local input policy: **COMMON_PREEXTRACTED_COHERENT_DOMAINS**

If replacement eligibility is false, no third screen is invented; representation comparisons use identical pre-extracted coherent domains.
