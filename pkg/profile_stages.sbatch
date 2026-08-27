#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=profstage
#SBATCH --output=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.err
set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability && source env.sh
cd /projects/rps/gmh4/hockygroup/sz5808
export OMP_NUM_THREADS=8
UP=crystal_discovery/preprocessing_pipeline_final/unified_pipeline/data/test_set
echo "########## small: steven_example_1 (6.7 MB) ##########"
python pkg/profile_stages.py $UP/steven_example_1.gsd
echo; echo "########## large: example1_trajectory (844 MB) ##########"
python pkg/profile_stages.py $UP/example1_trajectory.gsd
