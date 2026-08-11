#!/bin/bash
# Self-match specificity of the EMD matcher against the expanded reference library.
# Submit with a dependency on the crystal-generation array so it only starts once the
# anchors exist:
#
#   sbatch --dependency=afterany:<arrayjobid> run_expanded_selfmatch.sbatch
#
# afterany, not afterok: individual prototypes are allowed to fail to build, and the
# test should run over whatever anchors did succeed rather than be cancelled by one
# bad prototype. The script records exactly which anchors it used.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=selfmatch
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/sz5808/.conda/envs/pacsim
export OPENMM_PLUGIN_DIR=/scratch/sz5808/.conda/envs/pacsim/lib/plugins

python expanded_library_selfmatch.py
