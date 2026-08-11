#!/bin/bash
#SBATCH --job-name=snowmen_sweep
#SBATCH --account=torch_pr_146_general
#SBATCH --partition=cpu_short
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate crystal_ml
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MPLCONFIGDIR=/tmp/mpl-snowmen-$$
python -u snowmen_sweep_match.py
echo "DONE"
