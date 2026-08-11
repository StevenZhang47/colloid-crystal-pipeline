#!/bin/bash
# v2 PROTOCOL rerun — postdoc's three changes:
#   1. BIGGER BOX: target volume fraction 0.0087 (Nicole's median). Our old boxes were
#      0.032-0.165, i.e. 4x-19x too dense.
#   2. NO RAMP: start at 298 K from step 0. A genuinely stable crystal should survive it,
#      and the ramp made the first 40% of every trajectory a transient.
#   3. LONGER: 5,000,000 steps (was 500,000) = 1.5 us at dt=0.3 ps. ~8 min/GPU.
#
# Each v2 run reuses its ORIGINAL run's configuration.yaml + run.yaml verbatim, so the
# interaction parameters (psi, Debye, radii, well depth) are bit-identical to the run it
# reproduces. Only box, temperature schedule, and length change.
#
# Outputs go to runs/<name>_v2 . NOTHING existing is overwritten -- the reference library
# and matcher benchmarks depend on the originals.
set -euo pipefail
CS=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability
cd "$CS"

STEPS=5000000
INTERVAL=10000        # -> 500 frames
TARGET_VF=0.0087

# every distinct crystal/condition in crystal_stability (stable + melted controls)
RUNS=(
  # --- the 7 stable ---
  cscl th3p4 cu3au_d6 ni3sn_d019 zangenite_deep8 cr3si_a15_deep8 rocksalt_b1_deep8
  # --- independent Cu3Au realisation ---
  cu3au_l12_big
  # --- melted / failed-stability controls ---
  zincblende_b3 zincblende_b3_deep10 fluorite_c1 fluorite_c1_sr069
  nickeline_b81 nickeline_b81_deep8 fe4n_lp10 fe4n_lp10_deep8
  reo3_d09 i3p_a3b_hp8 antifluorite_c1b cu3au
  # --- baselines that melted before rescue ---
  cr3si_a15 rocksalt_b1 zangenite
)

n=0
for r in "${RUNS[@]}"; do
  src="runs/$r"
  dst="runs/${r}_v2"
  [ -f "$src/run.yaml" ] || { echo "SKIP $r (no run.yaml)"; continue; }
  [ -f "$dst/trajectory.gsd" ] && { echo "skip $r (v2 exists)"; continue; }

  mkdir -p "$dst"
  cp "$src/configuration.yaml" "$dst/"
  cp "$src/run.yaml"           "$dst/"
  cp "$src"/*.cif              "$dst/" 2>/dev/null || true

  py=$(command -v python)
  # 1) no ramp: integrator starts at potential_temperature; drop the update reporter
  # 2) longer run + matching trajectory interval
  python - "$dst/run.yaml" "$STEPS" "$INTERVAL" <<'PY'
import re, sys
p, steps, interval = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(p).read()
# target temperature = potential_temperature
m = re.search(r"^potential_temperature:.*?\n\s*unit:.*?\n\s*value:\s*([\d.]+)", t, re.S | re.M)
T = m.group(1)
# integrator temperature -> T (handles both !Quantity and !Copy forms)
t = re.sub(r"(integrator_parameters:.*?temperature:)\s*!Copy\s*\n\s*key:\s*\w+",
           r"\1 !Quantity\n    unit: kelvin\n    value: " + T, t, flags=re.S)
t = re.sub(r"(integrator_parameters:.*?temperature:\s*!Quantity\s*\n\s*unit:\s*kelvin\s*\n\s*value:\s*)[\d.]+",
           r"\g<1>" + T, t, flags=re.S)
# kill the ramp: remove BOTH the update_reporter line and the whole indented
# update_reporter_parameters block, then re-add explicit nulls. (A partial removal
# leaves an orphan parameters block and PACSim rejects it:
#  "Update-reporter parameters must not be specified if the update reporter is not on.")
t = re.sub(r"^update_reporter_parameters:\s*\n(?:^[ \t]+.*\n|^\s*\n(?=[ \t]))*", "", t, flags=re.M)
t = re.sub(r"^update_reporter:.*\n", "", t, flags=re.M)
t = t.rstrip() + "\n\nupdate_reporter: null\nupdate_reporter_parameters: null\n"
t = re.sub(r"^run_steps:.*$", f"run_steps: {steps}", t, flags=re.M)
t = re.sub(r"^trajectory_interval:.*$", f"trajectory_interval: {interval}", t, flags=re.M)
t = re.sub(r"^state_data_interval:.*$", f"state_data_interval: {int(interval)//10}", t, flags=re.M)
t = re.sub(r"^checkpoint_interval:.*$", f"checkpoint_interval: {int(interval)*10}", t, flags=re.M)
open(p, "w").write(t)
print(f"    ramp removed; T0={T} K; run_steps={steps}; interval={interval}")
PY

  cat > "$dst/job.sbatch" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=v2_${r}
#SBATCH --output=${CS}/logs/%x.%j.out
#SBATCH --error=${CS}/logs/%x.%j.err
set -euo pipefail
cd ${CS}
source env.sh
cd ${CS}/${dst}
pacsim-create configuration.yaml first_frame.gsd
python ${CS}/set_box_density.py first_frame.gsd --target-vf ${TARGET_VF}
pacsim-run run.yaml
echo "done: \$(pwd)/trajectory.gsd"
EOF
  jid=$(sbatch --parsable "$dst/job.sbatch")
  n=$((n+1))
  echo "submitted  $r -> $dst  job=$jid"
done
echo
echo "submitted $n v2 jobs (steps=$STEPS, no ramp, target VF=$TARGET_VF)"
