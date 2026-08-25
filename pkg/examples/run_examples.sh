#!/usr/bin/env bash
# Four trajectories, one for each answer the pipeline can give.
#
#   bash pkg/examples/run_examples.sh
#
# Paths are absolute and point at data in the working tree, not at the repository --
# trajectories are excluded from git. On another machine, substitute your own GSD files.
set -u
ROOT=/projects/rps/gmh4/hockygroup/sz5808
ATLAS=$ROOT/crystal_discovery/preprocessing_pipeline/nicoles_dataset/phase_diagram
TEST=$ROOT/crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set

run () {  # run <label> <expected> <path>
  printf '\n\033[1m=== %s ===\033[0m  expect: %s\n' "$1" "$2"
  if [ ! -f "$3" ]; then echo "  missing: $3"; return; fi
  crystal-discovery analyze "$3"
}

# KNOWN structure, correctly identified.
run "CsCl" "in-distribution, IDENTIFIED AS CsCl" \
    "$ATLAS/debye5.45_volumefraction.002704544521_seed2/trajectory.gsd"

# KNOWN structure, a different family.
run "Th3P4" "in-distribution, IDENTIFIED AS Th3P4" \
    "$ATLAS/debye5.15_volumefraction.012521039450_seed1/trajectory.gsd"

# Disordered. The detector recognises it (amorphous IS in the atlas) but no crystalline
# prototype describes it, so retrieval must abstain. This is the case that killed the
# previous mean-SOAP matcher.
run "amorphous" "in-distribution, IDENTIFIED AS nothing" \
    "$ATLAS/debye5.3_volumefraction.004294716531_seed1/trajectory.gsd"

# The interesting one: NOVEL to the detector, but the matcher still names it. Detection
# and identification are different questions and are allowed to disagree.
run "steven_example_1" "NOVEL (rho > 1), IDENTIFIED AS Ni3Sn" \
    "$TEST/steven_example_1.gsd"

run "steven_example_2" "NOVEL (rho > 1), IDENTIFIED AS Cu3Au" \
    "$TEST/steven_example_2.gsd"
