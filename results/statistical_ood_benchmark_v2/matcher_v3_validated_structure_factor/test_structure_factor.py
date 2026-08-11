#!/usr/bin/env python3
"""Known-answer regression tests for the versioned structure-factor channel."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from structure_factor import (  # noqa: E402
    coherence_metrics,
    peak_positions,
    positive_shape,
    raw_structure_factor,
    relative_peak_intensities,
    robust_first_shell_distance,
)


FAIL: list[str] = []
RESULTS: list[dict] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    passed = bool(condition)
    RESULTS.append({"test": name, "passed": passed, "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not passed:
        FAIL.append(name)


def cscl(repeats: int, jitter: float = 0.0, seed: int = 0):
    points = []
    for i in range(repeats):
        for j in range(repeats):
            for z in range(repeats):
                points.extend(([i, j, z], [i + 0.5, j + 0.5, z + 0.5]))
    p = np.asarray(points, dtype=np.float64)
    if jitter:
        p += np.random.default_rng(seed).normal(0.0, jitter, size=p.shape)
    box = np.full(3, float(repeats))
    wrapped = np.mod(p, box)
    return wrapped, robust_first_shell_distance(wrapped, box), box


def disordered(n: int, box_length: float, seed: int):
    return np.random.default_rng(seed).uniform(0.0, box_length, size=(n, 3))


def emd_shape(k, a, b):
    aa = positive_shape(a, k)
    bb = positive_shape(b, k)
    if aa.sum() == 0 or bb.sum() == 0:
        return np.inf
    return float(wasserstein_distance(k, k, aa, bb))


def nearest_peak(peaks, target):
    return float(peaks[np.argmin(np.abs(peaks - target))]) if len(peaks) else np.nan


def main() -> int:
    print("STRUCTURE FACTOR v3 REGRESSION TESTS")
    curves = {}
    for repeats in (8, 11, 15):
        p, nn, box = cscl(repeats)
        k, raw = raw_structure_factor(p, nn, box)
        curves[len(p)] = (k, raw, coherence_metrics(raw, len(p), k))
        print(
            f"CsCl N={len(p):5d}: raw_peak={curves[len(p)][2]['raw_positive_peak']:.6g}, "
            f"N^(2/3)-scaled={curves[len(p)][2]['n23_scaled_peak']:.4f}, "
            f"peaks={np.round(peak_positions(k, raw, 5), 3)}"
        )

    sizes = sorted(curves)
    ref_k, ref_raw, _ = curves[sizes[-1]]
    shape_distances = [emd_shape(ref_k, curves[n][1], ref_raw) for n in sizes[:-1]]
    check(
        "normalized Bragg-window shapes are stable across particle count",
        max(shape_distances) < 0.03 * (ref_k[-1] - ref_k[0]),
        f"EMD={np.round(shape_distances, 4).tolist()} (<3% of the k window)",
    )

    # Equal scattering weights make CsCl's total-density lattice BCC: (100) is
    # extinct and the first allowed family is (110), k*nn = pi*sqrt(6).
    expected_first = float(np.pi * np.sqrt(6.0))
    firsts = []
    for n in sizes:
        k, raw, _ = curves[n]
        firsts.append(nearest_peak(peak_positions(k, raw, 8), expected_first))
    check(
        "equal-weight CsCl/BCC first allowed Bragg peak is at pi*sqrt(6)",
        all(np.isfinite(firsts)) and max(abs(x - expected_first) for x in firsts) < 0.25,
        f"expected={expected_first:.3f}, observed={np.round(firsts, 3).tolist()}",
    )
    check(
        "Bragg peak position is stable across particle count",
        max(firsts) - min(firsts) < 0.20,
        f"observed={np.round(firsts, 3).tolist()}",
    )

    ref_peaks = peak_positions(ref_k, ref_raw, 4)
    rel = []
    for n in sizes:
        k, raw, _ = curves[n]
        rel.append(relative_peak_intensities(k, raw, ref_peaks))
    rel_dev = max(float(np.max(np.abs(x - rel[-1]))) for x in rel[:-1])
    check("relative peak intensities are stable across particle count", rel_dev < 0.30, f"max_delta={rel_dev:.4f}")

    scaled = [curves[n][2]["n23_scaled_peak"] for n in sizes]
    check(
        "finite-size-scaled coherence remains within a factor of three",
        max(scaled) / max(min(scaled), 1e-15) < 3.0,
        f"scaled={np.round(scaled, 4).tolist()}",
    )

    p, nn, box = cscl(11)
    k, base = raw_structure_factor(p, nn, box)
    shifted = np.mod(p + np.array([0.37, -1.91, 2.44]), box)
    _, wrapped = raw_structure_factor(shifted, nn, box)
    check("equivalent periodic wrapping is invariant", np.allclose(base, wrapped, atol=5e-6), f"max_delta={np.max(np.abs(base-wrapped)):.2e}")
    permutation = np.random.default_rng(7).permutation(len(p))
    _, permuted = raw_structure_factor(p[permutation], nn, box)
    check("particle ordering is invariant", np.allclose(base, permuted, atol=2e-12), f"max_delta={np.max(np.abs(base-permuted)):.2e}")

    p_th, nn_th, box_th = cscl(11, jitter=0.045, seed=4)
    _, thermal = raw_structure_factor(p_th, nn_th, box_th, k=k)
    thermal_emd = emd_shape(k, thermal, base)
    check(
        "thermal jitter preserves Bragg shape",
        thermal_emd < 0.05 * (k[-1] - k[0]),
        f"EMD={thermal_emd:.4f} (<5% of the k window)",
    )
    thermal_peak = nearest_peak(peak_positions(k, thermal, 8), expected_first)
    check("thermal jitter preserves the first Bragg position", abs(thermal_peak - expected_first) < 0.30, f"observed={thermal_peak:.3f}")

    disorder_scaled = []
    for n in (1024, 2662, 6750):
        # Match the CsCl number density: 2 particles per unit cube.
        length = (n / 2.0) ** (1.0 / 3.0)
        random_p = disordered(n, length, seed=100 + n)
        _, random_raw = raw_structure_factor(random_p, np.sqrt(3.0) / 2.0, np.full(3, length), k=ref_k)
        disorder_scaled.append(coherence_metrics(random_raw, n, ref_k)["n23_scaled_peak"])
    ordered_min = min(scaled)
    disordered_max = max(disorder_scaled)
    check(
        "raw/scaled Bragg evidence separates crystal from disordered controls",
        ordered_min > 3.0 * disordered_max,
        f"ordered={np.round(scaled, 4).tolist()}, disordered={np.round(disorder_scaled, 4).tolist()}",
    )
    random_shape = positive_shape(random_raw, ref_k)
    check(
        "shape normalization is explicitly separate from the disorder gate",
        np.isclose(np.trapezoid(random_shape, ref_k), 1.0, atol=1e-6) and disordered_max < ordered_min,
        "a unit-area random curve is not accepted without raw coherence evidence",
    )

    output = Path(__file__).resolve().parent / "structure_factor_test_results.json"
    output.write_text(json.dumps({"passed": len(RESULTS) - len(FAIL), "failed": len(FAIL), "tests": RESULTS}, indent=2) + "\n")
    print(f"\n{len(RESULTS) - len(FAIL)} passed, {len(FAIL)} failed; wrote {output}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
