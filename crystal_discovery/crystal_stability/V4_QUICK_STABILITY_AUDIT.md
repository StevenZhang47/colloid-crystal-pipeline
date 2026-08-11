# Periodic-bulk v4 stability audit

This bounded audit answers whether the newest periodic-bulk trajectories retained their
lattices. It sequentially unwraps all stored coordinates and evaluates ten fixed frames
from the trailing 20% (frames 400--500) using three predeclared criteria:

- median Lindemann displacement `< 0.15`;
- species-aware lattice-site occupancy `> 0.90`;
- mean q6 retention relative to frame 0 `> 0.80`.

The full per-run values are in `v4_quick_stability_audit.csv`. This check deliberately
omits the much more expensive per-frame partial-RDF calculation. Therefore `BORDERLINE`
means that the run retained some strong lattice evidence but did not pass all three strict
criteria; it must not be reported as either cleanly stable or melted.

## Results

| verdict | runs |
|---|---|
| **STABLE** | `cu3au_d6_v4`, `cu3au_l12_big_v4`, `cu3au_v4`, `ni3sn_d019_v4`, `th3p4_v4`, `zangenite_deep8_v4`, `zangenite_v4` |
| **BORDERLINE** | `cscl_v4`, `cr3si_a15_v4`, `cr3si_a15_deep8_v4`, `rocksalt_b1_v4`, `rocksalt_b1_deep8_v4` |
| **MELTED** | `antifluorite_c1b_v4`, `fe4n_lp10_v4`, `fe4n_lp10_deep8_v4`, `fluorite_c1_v4`, `fluorite_c1_sr069_v4`, `i3p_a3b_hp8_v4`, `reo3_d09_v4`, `zincblende_b3_v4`, `zincblende_b3_deep10_v4` |
| **NUMERICAL FAILURE / INCOMPLETE** | `nickeline_b81_v4`, `nickeline_b81_deep8_v4` |

## Important borderline cases

- `cscl_v4`: Lindemann 0.161 (slightly above 0.15), but site occupancy 1.000 and q6
  retention 0.981. The crystal clearly retained its sites and orientational order, so this
  is **borderline, not melted**.
- `cr3si_a15_v4`: 0.169 / 1.000 / 0.864 (Lindemann / occupancy / q6 retention).
- `cr3si_a15_deep8_v4`: 0.245 / 0.962 / 0.842.
- `rocksalt_b1_deep8_v4`: 0.102 / 1.000 / 0.584; displacement and occupancy pass, but
  bond-orientational order does not.
- `rocksalt_b1_v4`: 0.183 / 1.000 / 0.699; only site occupancy passes.

Both Nickeline trajectories contain only the initial frame. Their production logs end
with OpenMM `ValueError: Energy is NaN`; they are failed simulations, not melt tests.

## Reproducibility

Run with the project environment:

```bash
MPLCONFIGDIR=/tmp/mpl-crystal \
  /projects/rps/gmh4/hockygroup/sz5808/conda_storage/envs/crystal_ml/bin/python \
  crystal_discovery/crystal_stability/quick_v4_stability_audit.py
```

