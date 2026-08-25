#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=03:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=anchorframes
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability
source env.sh
python select_anchor_frames.py
