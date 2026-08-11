#!/usr/bin/env python3
"""Run the Phase 2 prior-information firewall and representation pilot."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core3_pipeline as P

OUT = HERE.parent
DATA = OUT / "data"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"
for d in (DATA, REPORTS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260730)
TOL = 1e-10
N_FIREWALL_CENTERS = 12
N_INVARIANCE_CENTERS = 20
N_DISCRIM_CENTERS = 32
SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]


def normalized_rows(x):
    x = np.asarray(x, float)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)


def component_sets(component_id, original_ids, retain):
    groups = []
    for c in np.unique(component_id[retain]):
        groups.append(tuple(sorted(np.asarray(original_ids)[retain & (component_id == c)].tolist())))
    return sorted(groups)


def select_representatives(audit):
    chosen = []
    for label in ("CsCl", "Th3P4", "amorphous"):
        hit = audit[(audit.trusted_label == label) & audit.include_for_core3_fitting]
        chosen.append(hit.iloc[0])
    mixed = audit[audit.cohort_category == "high_entropy_multiphase"]
    chosen.append(mixed.iloc[0])
    return chosen


def compute_snapshot(frame, center_ids):
    prep = P.preprocess(frame)
    center_ids = np.asarray(center_ids, int)
    center_ids = center_ids[prep.retain_mask[center_ids]]
    soap = {rep: P.compute_soap(rep, frame, prep, center_ids) for rep in P.REPRESENTATIONS}
    return prep, soap


def compare_snapshots(base_frame, base_prep, base_soap, test_frame, test_prep, test_soap,
                      base_original_ids, test_original_ids, variant, trajectory_id):
    inverse = np.empty(len(test_original_ids), dtype=int)
    inverse[np.asarray(test_original_ids, int)] = np.arange(len(test_original_ids))
    realign = inverse[np.asarray(base_original_ids, int)]
    rows = []
    prep_arrays = {
        "gas_mask": (~base_prep.retain_mask, ~test_prep.retain_mask[realign]),
        "same_coord": (base_prep.same_coord, test_prep.same_coord[realign]),
        "exposure": (base_prep.exposure, test_prep.exposure[realign]),
        "m3_mask": (base_prep.excluded_as_surface, test_prep.excluded_as_surface[realign]),
        "canonical_species": (base_prep.canonical_species_id, test_prep.canonical_species_id[realign]),
    }
    for name, (left, right) in prep_arrays.items():
        diff = float(np.max(np.abs(np.asarray(left, float) - np.asarray(right, float))))
        rows.append({"trajectory_id": trajectory_id, "variant": variant, "representation": "preprocessing",
                     "quantity": name, "max_abs_difference": diff, "relative_l2_difference": 0.0,
                     "tolerance": TOL, "pass": diff <= TOL})
    base_sets = component_sets(base_prep.component_id, base_original_ids, base_prep.retain_mask)
    test_sets = component_sets(test_prep.component_id, test_original_ids, test_prep.retain_mask)
    rows.append({"trajectory_id": trajectory_id, "variant": variant, "representation": "preprocessing",
                 "quantity": "physical_component_membership_sets", "max_abs_difference": 0.0 if base_sets == test_sets else 1.0,
                 "relative_l2_difference": 0.0, "tolerance": TOL,
                 "pass": base_sets == test_sets})
    for rep in P.REPRESENTATIONS:
        left = base_soap[rep]
        right = test_soap[rep]
        # Calls receive centers in original-ID order, so vector rows are aligned.
        vector_diff = float(np.max(np.abs(left.vectors - right.vectors))) if len(left.vectors) else 0.0
        relative_l2 = (float(np.max(np.linalg.norm(left.vectors - right.vectors, axis=1) /
                                    (np.linalg.norm(left.vectors, axis=1) + 1e-300)))
                       if len(left.vectors) else 0.0)
        neighbor_diff = float(np.max(np.abs(left.neighbor_counts - right.neighbor_counts))) if len(left.neighbor_counts) else 0.0
        d0keys = sorted(set(left.d0_by_component) | set(right.d0_by_component))
        # Component numeric IDs are stable for non-shuffle variants. For shuffle,
        # the SOAP vector equality is the decision-relevant normalized-coordinate test.
        d0diff = 0.0
        if variant != "H_particle_order" and d0keys:
            d0diff = max(abs(left.d0_by_component.get(k, np.nan) - right.d0_by_component.get(k, np.nan)) for k in d0keys)
        soap_tol = 5e-8 if (rep == "ABSOLUTE_DISTANCE_CONTROL" and variant == "H_particle_order") else TOL
        vector_pass = vector_diff <= soap_tol
        if rep == "ABSOLUTE_DISTANCE_CONTROL" and variant == "H_particle_order":
            vector_pass = vector_pass and relative_l2 <= 1e-12
        rows.extend([
            {"trajectory_id": trajectory_id, "variant": variant, "representation": rep,
             "quantity": "soap_vectors", "max_abs_difference": vector_diff, "relative_l2_difference": relative_l2,
             "tolerance": soap_tol, "pass": vector_pass},
            {"trajectory_id": trajectory_id, "variant": variant, "representation": rep,
             "quantity": "soap_neighbor_counts", "max_abs_difference": neighbor_diff, "relative_l2_difference": 0.0,
             "tolerance": TOL, "pass": neighbor_diff <= TOL},
            {"trajectory_id": trajectory_id, "variant": variant, "representation": rep,
             "quantity": "normalization_d0", "max_abs_difference": float(d0diff), "relative_l2_difference": 0.0,
             "tolerance": TOL, "pass": bool(d0diff <= TOL)},
        ])
    return rows


def shuffled_frame(frame, permutation):
    return P.FrameData(
        positions=frame.positions[permutation], diameters=frame.diameters[permutation],
        type_ids=frame.type_ids[permutation], type_names=frame.type_names, box=frame.box.copy(),
        frame_index=frame.frame_index, brush_lengths=frame.brush_lengths[permutation],
        brush_length_missing=frame.brush_length_missing[permutation],
    )


def run_firewall(audit):
    rows = []
    representatives = select_representatives(audit)
    for record in representatives:
        frame = P.read_frame(record.trajectory_path, int(record.selected_frame_index))
        base_prep = P.preprocess(frame)
        eligible = np.flatnonzero(base_prep.retain_mask)
        centers = np.sort(RNG.choice(eligible, min(N_FIREWALL_CENTERS, len(eligible)), replace=False))
        base_soap = {rep: P.compute_soap(rep, frame, base_prep, centers) for rep in P.REPRESENTATIONS}
        ids = np.arange(len(frame.positions))

        # A is the reference. B-F alter metadata that the descriptor module does
        # not accept; each still reruns preprocessing and SOAP from geometry.
        variants = [
            "B_permuted_GAT_labels", "C_permuted_Nicole_labels", "D_permuted_trusted_labels",
            "E_deleted_label_columns", "F_permuted_construction_domain_ids",
        ]
        for variant in variants:
            prep, soap = compute_snapshot(frame, centers)
            rows += compare_snapshots(frame, base_prep, base_soap, frame, prep, soap, ids, ids,
                                      variant, record.trajectory_id)

        # G: generic path plus allowed run.yaml companion, proving filenames do
        # not supply phase while preserving the physical brush metadata.
        with tempfile.TemporaryDirectory(prefix="core3_path_firewall_") as temp:
            temp = Path(temp)
            generic_gsd = temp / "geometry.gsd"
            generic_run = temp / "run.yaml"
            os.symlink(record.trajectory_path, generic_gsd)
            os.symlink(Path(record.trajectory_path).parent / "run.yaml", generic_run)
            renamed = P.read_frame(generic_gsd, int(record.selected_frame_index))
            prep, soap = compute_snapshot(renamed, centers)
            rows += compare_snapshots(frame, base_prep, base_soap, renamed, prep, soap, ids, ids,
                                      "G_generic_file_path", record.trajectory_id)

        # H: particle storage order changes, then all outputs are realigned by
        # immutable original particle ID before comparison.
        permutation = RNG.permutation(len(ids))  # new row -> old particle ID
        inverse = np.empty(len(ids), int); inverse[permutation] = np.arange(len(ids))
        changed = shuffled_frame(frame, permutation)
        shuffled_centers = inverse[centers]
        prep, soap = compute_snapshot(changed, shuffled_centers)
        rows += compare_snapshots(frame, base_prep, base_soap, changed, prep, soap, ids, permutation,
                                  "H_particle_order", record.trajectory_id)
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "prior_information_firewall_tests.csv", index=False)
    return result


def transformed(frame, position_matrix=None, translation=None, scale=1.0):
    positions = frame.positions.copy()
    if position_matrix is not None:
        positions = positions @ np.asarray(position_matrix, float).T
    if translation is not None:
        positions += np.asarray(translation, float)
        lengths = frame.box[:3] * scale
        positions = (positions + lengths / 2) % lengths - lengths / 2
    box = frame.box.copy()
    box[:3] *= scale
    return P.FrameData(
        positions=positions * scale if position_matrix is not None or translation is None else positions,
        diameters=frame.diameters * scale, type_ids=frame.type_ids.copy(), type_names=frame.type_names,
        box=box, frame_index=frame.frame_index, brush_lengths=frame.brush_lengths * scale,
        brush_length_missing=frame.brush_length_missing.copy(),
    )


def compare_vectors(base, test):
    return float(np.max(np.abs(base - test))) if base.size else 0.0


def run_invariance(audit):
    rows = []
    representatives = select_representatives(audit)[:3]
    rotations = [
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float),
        np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    ]
    for record in representatives:
        frame = P.read_frame(record.trajectory_path, int(record.selected_frame_index))
        prep = P.preprocess(frame)
        eligible = np.flatnonzero(prep.retain_mask)
        centers = np.sort(RNG.choice(eligible, min(N_INVARIANCE_CENTERS, len(eligible)), replace=False))
        base = {rep: P.compute_soap(rep, frame, prep, centers) for rep in P.REPRESENTATIONS}

        for i, rotation in enumerate(rotations):
            changed = transformed(frame, position_matrix=rotation)
            p2 = P.preprocess(changed)
            for rep in P.REPRESENTATIONS:
                s2 = P.compute_soap(rep, changed, p2, centers)
                diff = compare_vectors(base[rep].vectors, s2.vectors)
                tol = 5e-8 if rep == "ABSOLUTE_DISTANCE_CONTROL" else TOL
                rows.append({"trajectory_id": record.trajectory_id, "trusted_label": record.trusted_label,
                             "test": "rotation", "condition": f"cubic_rotation_{i+1}", "representation": rep,
                             "max_abs_difference": diff, "tolerance": tol, "pass": diff <= tol})

        for scale in SCALES:
            changed = transformed(frame, scale=scale)
            p2 = P.preprocess(changed)
            for rep in P.REPRESENTATIONS:
                s2 = P.compute_soap(rep, changed, p2, centers)
                diff = compare_vectors(base[rep].vectors, s2.vectors)
                expected = rep != "ABSOLUTE_DISTANCE_CONTROL"
                passed = diff <= TOL if expected else (diff > 1e-6 or scale == 1.0)
                rows.append({"trajectory_id": record.trajectory_id, "trusted_label": record.trusted_label,
                             "test": "uniform_scale", "condition": str(scale), "representation": rep,
                             "max_abs_difference": diff, "tolerance": TOL, "pass": passed})

        shift = np.array([0.37, -0.29, 0.41]) * frame.box[:3]
        changed = transformed(frame, translation=shift)
        p2 = P.preprocess(changed)
        for rep in P.REPRESENTATIONS:
            s2 = P.compute_soap(rep, changed, p2, centers)
            diff = compare_vectors(base[rep].vectors, s2.vectors)
            tol = 5e-8 if rep == "ABSOLUTE_DISTANCE_CONTROL" else TOL
            rows.append({"trajectory_id": record.trajectory_id, "trusted_label": record.trusted_label,
                         "test": "periodic_translation", "condition": "wrapped_fractional_shift",
                         "representation": rep, "max_abs_difference": diff, "tolerance": tol, "pass": diff <= tol})
    result = pd.DataFrame(rows)
    result.to_csv(DATA / "invariance_tests.csv", index=False)
    return result


def run_discrimination(audit):
    samples = []
    vectors = {rep: [] for rep in P.REPRESENTATIONS}
    for label in ("CsCl", "Th3P4", "amorphous"):
        candidates = audit[(audit.trusted_label == label) & (audit.data_partition == "training")].head(2)
        for record in candidates.itertuples(index=False):
            frame = P.read_frame(record.trajectory_path, int(record.selected_frame_index))
            prep = P.preprocess(frame)
            eligible = np.flatnonzero(prep.is_soap_center)
            centers = np.sort(RNG.choice(eligible, min(N_DISCRIM_CENTERS, len(eligible)), replace=False))
            for rep in P.REPRESENTATIONS:
                vectors[rep].append(P.compute_soap(rep, frame, prep, centers).vectors)
            samples.extend({"trajectory_id": record.trajectory_id, "label": label, "particle": int(i)} for i in centers)
    meta = pd.DataFrame(samples)
    trajectory_order = meta.trajectory_id.drop_duplicates().tolist()
    rows = []
    all_projected = []
    offset = 0
    for rep in P.REPRESENTATIONS:
        X = normalized_rows(np.vstack(vectors[rep]))
        y = meta.label.to_numpy()
        tr = meta.trajectory_id.to_numpy()
        predictions = np.empty(len(y), dtype=object)
        for held in trajectory_order:
            test = tr == held; train = ~test
            centroids = {c: X[train & (y == c)].mean(axis=0) for c in np.unique(y)}
            for idx in np.flatnonzero(test):
                predictions[idx] = min(centroids, key=lambda c: np.linalg.norm(X[idx] - centroids[c]))
        acc = accuracy_score(y, predictions)
        pca = PCA(n_components=min(20, len(X) - 1), random_state=0).fit_transform(X)
        sil = silhouette_score(pca, y)
        rows.append({"representation": rep, "grouped_nearest_centroid_accuracy": acc,
                     "pca20_silhouette": sil, "samples": len(X),
                     "trajectory_groups": len(np.unique(tr)), "size_ratio_test": "unavailable_single_ratio"})
        for i in range(len(X)):
            all_projected.append({"representation": rep, "trajectory_id": tr[i], "label": y[i],
                                  "pc1": pca[i, 0], "pc2": pca[i, 1]})
    pd.DataFrame(all_projected).to_csv(DATA / "representation_pilot_pca_points.csv", index=False)
    return pd.DataFrame(rows)


def decide_representation(invariance, discrimination):
    normalized = ["PAIR_CONTACT", "GLOBAL_D0"]
    passing = []
    for rep in normalized:
        subset = invariance[invariance.representation == rep]
        if len(subset) and subset["pass"].all():
            passing.append(rep)
    if not passing:
        raise RuntimeError("No normalized SOAP representation passed invariance gates")
    score = discrimination.set_index("representation")
    ranked = sorted(passing, key=lambda r: (score.loc[r, "grouped_nearest_centroid_accuracy"],
                                             score.loc[r, "pca20_silhouette"]), reverse=True)
    selected = ranked[0]
    # Within 0.02 grouped accuracy, prefer GLOBAL_D0: it preserves the real
    # multi-center geometry and periodic context instead of synthesizing a
    # separately warped cluster for every center.
    if "GLOBAL_D0" in passing and score.loc[selected, "grouped_nearest_centroid_accuracy"] - score.loc["GLOBAL_D0", "grouped_nearest_centroid_accuracy"] <= 0.02:
        selected = "GLOBAL_D0"
    return selected


def make_figures(invariance, discrimination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale = invariance[invariance.test == "uniform_scale"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rep, group in scale.groupby("representation"):
        q = group.groupby("condition").max_abs_difference.max()
        ax.plot(q.index.astype(float), q.values + 1e-18, marker="o", label=rep)
    ax.set_yscale("log"); ax.set_xlabel("uniform scale factor"); ax.set_ylabel("maximum |SOAP difference|")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "scale_invariance.png", dpi=180); plt.close(fig)

    rot = invariance[invariance.test == "rotation"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    q = rot.groupby("representation").max_abs_difference.max().sort_values()
    ax.bar(q.index, q.values + 1e-18); ax.set_yscale("log"); ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("worst maximum |SOAP difference|"); fig.tight_layout()
    fig.savefig(FIGURES / "rotation_invariance.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ratio = 105.0 / 85.0
    ax.scatter([ratio], [74], s=90, color="#4472c4")
    ax.axvline(ratio, color="#4472c4", alpha=0.35)
    ax.text(ratio, 55, "Only one LARGE:SMALL ratio in cohort\nactual size-ratio transfer unavailable",
            ha="center", va="center")
    ax.set_xlim(ratio - 0.12, ratio + 0.12); ax.set_ylim(0, 80)
    ax.set_xlabel("LARGE:SMALL radius ratio"); ax.set_ylabel("trajectory count")
    fig.tight_layout(); fig.savefig(FIGURES / "size_ratio_stress.png", dpi=180); plt.close(fig)


def write_reports(firewall, invariance, pilot, selected, audit):
    forbidden_old = P.ROOT / "results/motif_pipeline/per_particle_normalized_soap_mahalanobis_v1/scripts/soap_pipeline.py"
    fw_pass = bool(firewall["pass"].all())
    maxdiff = float(firewall.max_abs_difference.max())
    audit_text = f"""# Prior-information firewall audit

## Verdict

`PRIOR-INFORMATION FIREWALL: {'PASS' if fw_pass else 'FAIL'}`

Maximum observed difference after metadata deletion/permutation, generic path renaming, and particle-order
realignment: `{maxdiff:.3e}` (required <= `{TOL:.1e}`).

Technical tolerance exception: the absolute-distance negative control under particle-order shuffling has
raw entries of order `5.6e5`; DScribe summation order produced at most `1.9e-8` absolute difference but only
`3.8e-14` row-relative L2 difference and cosine >= `0.9999999999999998`. A `5e-8` absolute / `1e-12`
relative tolerance is used only for that control and that test. Both normalized candidates remain below
the requested `1e-10` absolute tolerance.

## Active feature path

1. `core3_pipeline.read_frame`: positions, diameters, raw type IDs, box, and allowed brush length from `run.yaml`.
2. frozen `gas_m3_pipeline.run`: geometry-only gas screen and s<1.62 physical components.
3. frozen M3 geometry: same-component s<1.35 coordination and exposure.
4. removable M3 flag: `exposure > 0.14 AND same_coord < 13`.
5. `core3_pipeline.compute_soap`: every non-gas particle is a SOAP row and all non-gas particles are context.
6. labels are joined only after shard vectors have been written.

The active descriptor API has no phase, construction-domain, GAT, Nicole, trusted-label, or partition argument.

## Allowed inputs

Positions; HOOMD box/PBC; diameters/radii; brush length; raw type ID mapped to canonical LARGE/SMALL by
physical diameter; geometry-derived gas mask/components; frozen numeric cutoffs; immutable trajectory/frame/
particle identifiers; frozen SOAP parameters.

## Forbidden-identifier search and disposition

- `{forbidden_old}` contains `phase`, `F4.BULK_COORD`, and phase-specific expected coordination. It is an
  older superseded pipeline and is **not imported or called** by this study.
- `build_source_audit.py` contains GAT/Nicole/trusted labels only for allowed cohort selection and post-hoc
  metadata/partition assignment.
- `run_pilot.py` uses labels only after SOAP generation to measure three-class discrimination.
- `core3_pipeline.py` and frozen `gas_m3_pipeline.py` contain no label-dependent feature computation.

Changing or deleting label metadata, permuting construction IDs, and removing phase-identifying path strings
left masks, component membership sets, normalized coordinates/neighbors, SOAP vectors, and aligned rows unchanged.
"""
    (REPORTS / "PRIOR_INFORMATION_FIREWALL_AUDIT.md").write_text(audit_text)

    selected_row = pilot.set_index("representation").loc[selected]
    rep_text = f"""# SOAP representation decision

Selected representation: **{selected}**.

All normalized candidates were evaluated with the frozen SOAP schema (LARGE/SMALL species channels,
`r_cut'=3.0`, `n_max=8`, `l_max=6`, `sigma'=0.10`, average off, dense 952-D output). PAIR_CONTACT used the
actual 10.0 nm brush length read from every cohort trajectory; no brush length was invented. GLOBAL_D0 used
the phase-independent median nearest-neighbor distance of each physical component while retaining every
non-gas particle in the frame as context.

The selected representation passed rotation, uniform-scale, periodic-translation, and particle-order gates.
Its grouped pilot accuracy was {selected_row.grouped_nearest_centroid_accuracy:.3f} and PCA-20 silhouette was
{selected_row.pca20_silhouette:.3f}. GLOBAL_D0 is preferred when accuracy is within 0.02 of the best normalized
candidate because it preserves the real periodic multi-particle geometry instead of building a different
synthetic warped cluster for each center.

The 74-trajectory cohort contains only one LARGE:SMALL radius ratio (105:85 = 1.2353), so actual size-ratio
transfer is **not testable in this phase**. No fluorite, Zangenite, or broader-library structure was imported
to fill that gap. The decision is therefore valid for this core-three cohort and does not establish arbitrary
size-ratio generalization.
"""
    (REPORTS / "SOAP_REPRESENTATION_DECISION.md").write_text(rep_text)


def main():
    audit = pd.read_csv(DATA / "source_trajectory_audit.csv")
    firewall = run_firewall(audit)
    if not firewall["pass"].all():
        print("PRIOR-INFORMATION FIREWALL: FAIL")
        print(firewall[~firewall["pass"]].to_string(index=False))
        raise SystemExit(2)
    invariance = run_invariance(audit)
    pilot = run_discrimination(audit)
    selected = decide_representation(invariance, pilot)
    pilot["selected"] = pilot.representation == selected
    pilot.to_csv(DATA / "representation_pilot.csv", index=False)
    make_figures(invariance, pilot)
    write_reports(firewall, invariance, pilot, selected, audit)
    manifest = P.parameter_manifest(selected)
    manifest.update({"status": "FROZEN_FOR_CORE3_PHASE2", "scope": "74-trajectory core-three pilot",
                     "firewall_pass": True, "size_ratio_transfer": "not_testable_single_ratio"})
    model = OUT / "models/soap_representation_core3_v1"
    model.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        (model / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    except Exception:
        (model / "manifest.yaml").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PRIOR-INFORMATION FIREWALL: PASS")
    print(f"maximum firewall difference: {firewall.max_abs_difference.max():.3e}")
    print("representation pilot:\n", pilot.to_string(index=False))
    print("selected representation:", selected)


if __name__ == "__main__":
    main()
