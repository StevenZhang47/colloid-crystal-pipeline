#!/usr/bin/env python
"""
Evaluate the task14 reference library's short_md_stability frames.

reference_family_qc.csv has a `survived_short_md` column that is NaN for all 60
rows -- the check was never run. Each short_md_stability GSD is the final frame of
a short MD run, and each has a matching `perfect` GSD (the ideal relaxed lattice).
So compare them directly, using the same criteria as the paper:

  RDF Pearson correlation of the post-MD frame vs the ideal lattice, computed on
  all particles and on each species separately; > 0.5 = survived.
  Plus the Lindemann ratio and the fraction of particles that left their sites.
"""
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr

LIB = Path("/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/preprocessing_pipeline_final/"
           "unified_pipeline/results/motif_pipeline/gat_v4_qc_phase_map/novel_ordered_detector_v1/"
           "audit_reanalysis_v2/task14_relaxed_reference_library")


def rdf_of(frame, species=None, r_max=900.0, bins=200):
    positions = frame.particles.position
    if species is not None:
        names = np.array(frame.particles.types)[frame.particles.typeid]
        positions = positions[names == species]
    rdf = freud.density.RDF(bins=bins, r_max=r_max)
    rdf.compute((frame.configuration.box, positions))
    return rdf.rdf


def nearest_neighbour(frame, sample=500):
    p = frame.particles.position[:sample]
    d = distance_matrix(p, p)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def main():
    rows = []
    for condition in ["steven_example_1", "steven_example_2"]:
        md_dir = LIB / "reference_family_gsd" / condition
        for md_path in sorted(md_dir.glob("*__short_md_stability.gsd")):
            family = md_path.name.replace("__short_md_stability.gsd", "")
            perfect_path = LIB / "relaxed_gsd" / condition / f"{family}__perfect.gsd"
            if not perfect_path.is_file():
                continue

            ideal = gsd.hoomd.open(str(perfect_path))[0]
            after = gsd.hoomd.open(str(md_path))[0]

            nn = nearest_neighbour(ideal)
            species = list(ideal.particles.types)

            correlations = {"all": pearsonr(rdf_of(ideal), rdf_of(after))[0]}
            for s in species:
                correlations[s] = pearsonr(rdf_of(ideal, s), rdf_of(after, s))[0]

            displacement = np.linalg.norm(after.particles.position - ideal.particles.position, axis=1)
            lindemann = float(np.median(displacement) / nn)
            escaped = float((displacement > nn / 2).mean())

            worst = min(correlations.values())
            rows.append(dict(
                condition=condition.replace("steven_example_", "Steven"),
                prototype=family.split("__")[0],
                variant=family.split("__")[1] if "__" in family else "",
                r_all=correlations["all"],
                r_worst_species=worst,
                lindemann=lindemann,
                escaped=escaped,
                survived=bool(worst > 0.5),
            ))

    df = pd.DataFrame(rows).sort_values(["condition", "survived", "r_worst_species"],
                                        ascending=[True, False, False])
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = Path(__file__).resolve().parent / "library_stability.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print("\nsurvived (worst-species RDF correlation > 0.5):")
    print(df.groupby("condition").survived.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
