#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=bank
#SBATCH --output=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.err
set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability
source env.sh
cd /projects/rps/gmh4/hockygroup/sz5808/results/statistical_ood_benchmark_v2/reference_conditioned_matcher_v1
export OMP_NUM_THREADS=8
cd /projects/rps/gmh4/hockygroup/sz5808/pkg && python build_chamfer_bank.py
