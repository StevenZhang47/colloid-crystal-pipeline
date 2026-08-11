#!/bin/bash
# v4 — PERIODIC BULK stability at 298 K.
#
# Removes every boundary artefact we hit:
#   v1 small box  -> high pressure          -> crystals looked MORE stable than they are
#   v2 big box    -> vacuum sublimation sink-> crystals evaporated regardless of lattice
#   v3 + gas      -> gas near the walls got LAUNCHED by the diverging wall potential
#                    (Ni3Sn: max|pos| 7,357 nm -> 322,727,936 nm in one frame, T still 298 K)
#
# All three are SURFACE / BOUNDARY artefacts, not lattice physics. A periodic bulk crystal:
#   * no free surface -> no evaporation, no sublimation sink
#   * no walls        -> nothing to launch
#   * pressure set by the (already energy-minimised) lattice constant
#   * measures exactly the question asked: IS THE BULK LATTICE STABLE AT 298 K?
#
# Surface / evaporation behaviour is now a separate, deliberate experiment.
set -euo pipefail
CS=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability
cd "$CS"

STEPS=10000000        # 1.5 us at dt = 0.15 ps
INTERVAL=20000        # -> 500 frames
DT=0.15

RUNS=(
  cscl th3p4 cu3au_d6 ni3sn_d019 zangenite_deep8 cr3si_a15_deep8 rocksalt_b1_deep8
  cu3au_l12_big
  zincblende_b3 zincblende_b3_deep10 fluorite_c1 fluorite_c1_sr069
  nickeline_b81 nickeline_b81_deep8 fe4n_lp10 fe4n_lp10_deep8
  reo3_d09 i3p_a3b_hp8 antifluorite_c1b cu3au
  cr3si_a15 rocksalt_b1 zangenite
)

n=0
for r in "${RUNS[@]}"; do
  src="runs/$r"; dst="runs/${r}_v4"
  [ -f "$src/run.yaml" ] || { echo "SKIP $r"; continue; }
  [ -f "$dst/trajectory.gsd" ] && { echo "skip $r (exists)"; continue; }
  mkdir -p "$dst"
  cp "$src/configuration.yaml" "$src/run.yaml" "$dst/"
  cp "$src"/*.cif "$dst/" 2>/dev/null || true
  CIF=$(grep -oE "lattice_specification: *\S+" "$src/configuration.yaml" | awk '{print $2}')
  REP=$(grep -oE "lattice_repeats: *[0-9]+" "$src/configuration.yaml" | awk '{print $2}')

  python - "$dst/run.yaml" "$STEPS" "$INTERVAL" "$DT" <<'PY'
import re, sys
p, steps, interval, dt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
t = open(p).read()
T = re.search(r"^potential_temperature:.*?\n\s*unit:.*?\n\s*value:\s*([\d.]+)", t, re.S|re.M).group(1)
# start AT temperature, no ramp
t = re.sub(r"(integrator_parameters:.*?temperature:)\s*!Copy\s*\n\s*key:\s*\w+",
           r"\1 !Quantity\n    unit: kelvin\n    value: " + T, t, flags=re.S)
t = re.sub(r"(integrator_parameters:.*?temperature:\s*!Quantity\s*\n\s*unit:\s*kelvin\s*\n\s*value:\s*)[\d.]+",
           r"\g<1>" + T, t, flags=re.S)
t = re.sub(r"^update_reporter_parameters:\s*\n(?:^[ \t]+.*\n|^\s*\n(?=[ \t]))*", "", t, flags=re.M)
t = re.sub(r"^update_reporter:.*\n", "", t, flags=re.M)
# PERIODIC BULK: no walls, no gravity  (both are boundary/surface effects)
t = re.sub(r"^wall_directions:\s*\n(?:^\s*-\s*\w+\s*\n)+",
           "wall_directions:\n  - false\n  - false\n  - false\n", t, flags=re.M)
t = re.sub(r"^use_gravity:.*$", "use_gravity: false", t, flags=re.M)
# walls OFF => the wall SLJ parameters MUST be null, or PACSim refuses to start:
#   "Epsilon must not be specified if walls are not active."  (run_parameters.py:399/401)
# This killed all 23 of the first v4 jobs instantly.
t = re.sub(r"^epsilon:\s*!Quantity\s*\n(?:^[ \t]+.*\n)*", "epsilon: null\n", t, flags=re.M)
t = re.sub(r"^alpha:.*$", "alpha: null", t, flags=re.M)
# gravity OFF => its parameters must ALSO be null (run_parameters.py:454):
#   "Gravitational acceleration must not be specified if gravity is not on."
for key in ("gravitational_acceleration", "water_density", "particle_density"):
    t = re.sub(rf"^{key}:\s*!Quantity\s*\n(?:^[ \t]+.*\n)*", f"{key}: null\n", t, flags=re.M)
t = re.sub(r"(stepSize:\s*!Quantity\s*\n\s*unit:\s*picosecond\s*\n\s*value:\s*)[\d.]+",
           r"\g<1>" + dt, t)
t = re.sub(r"^run_steps:.*$", f"run_steps: {steps}", t, flags=re.M)
t = re.sub(r"^trajectory_interval:.*$", f"trajectory_interval: {interval}", t, flags=re.M)
t = re.sub(r"^state_data_interval:.*$", f"state_data_interval: {int(interval)//10}", t, flags=re.M)
t = re.sub(r"^checkpoint_interval:.*$", f"checkpoint_interval: {int(interval)*10}", t, flags=re.M)
t = t.rstrip() + "\n\nupdate_reporter: null\nupdate_reporter_parameters: null\n"
open(p,"w").write(t)
PY

  cat > "$dst/job.sbatch" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=v4_${r}
#SBATCH --output=${CS}/logs/%x.%j.out
#SBATCH --error=${CS}/logs/%x.%j.err
set -euo pipefail
cd ${CS}
source env.sh
cd ${CS}/${dst}
pacsim-create configuration.yaml first_frame.gsd
python ${CS}/make_periodic.py first_frame.gsd --cif ${CIF} --repeats ${REP}
pacsim-run run.yaml
echo "done: \$(pwd)/trajectory.gsd"
EOF
  jid=$(sbatch --parsable "$dst/job.sbatch")
  n=$((n+1)); echo "submitted $r -> $dst  job=$jid"
  sleep 2
done
echo; echo "launched $n periodic-bulk (v4) jobs"
