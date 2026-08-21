#!/bin/bash
#SBATCH --job-name=covstruct
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3/data/slurm_compare_%j.log

set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/results/motif_pipeline/core3_atlas_v3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
srun /home/sz5808/.conda/envs/crystal_ml/bin/python -u scripts/v3_10_covariance_structure.py
echo "=== threshold diagnostic ==="
