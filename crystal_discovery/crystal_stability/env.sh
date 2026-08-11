# Source this before running anything: source env.sh
#
# OPENMM_PLUGIN_DIR is not optional. The openmm in the pacsim env was built with
# its plugin path baked in as /home/sz5808/.conda/envs/pacsim/lib, but the env
# actually lives on /scratch. Without this, openmm silently loads no plugins and
# only the (very slow) Reference platform exists -- pacsim-run then dies with
#   OpenMMException: There is no registered Platform called "CPU"
# The permanent fix is a symlink:
#   ln -s /scratch/sz5808/.conda/envs/pacsim /home/sz5808/.conda/envs/pacsim

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/sz5808/.conda/envs/pacsim
export OPENMM_PLUGIN_DIR=/scratch/sz5808/.conda/envs/pacsim/lib/plugins
