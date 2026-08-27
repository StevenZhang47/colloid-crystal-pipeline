#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=vercleanup
#SBATCH --output=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.out
#SBATCH --error=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability/logs/%x.%j.err
set -euo pipefail
cd /projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability && source env.sh
cd /projects/rps/gmh4/hockygroup/sz5808
export OMP_NUM_THREADS=8

echo "### 1. rebuild the bank with the DECOUPLED builder ###"
python pkg/build_chamfer_bank.py --out /tmp/chamfer_bank_rebuilt.npz

echo; echo "### 2. does it match the shipped bank? ###"
python - <<'PY'
import numpy as np
a=np.load("pkg/references/chamfer_bank.npz"); b=np.load("/tmp/chamfer_bank_rebuilt.npz")
ka=set(a.files)-{"__meta__"}; kb=set(b.files)-{"__meta__"}
print(f"  shipped {len(ka)} arrays, rebuilt {len(kb)}; identical keys: {ka==kb}")
worst=0.0
for k in sorted(ka & kb):
    x,y=a[k].astype(float),b[k].astype(float)
    if x.shape!=y.shape: print(f"  SHAPE DIFF {k}: {x.shape} vs {y.shape}"); continue
    worst=max(worst,float(np.abs(x-y).max()))
print(f"  max |shipped - rebuilt| = {worst:.2e}")
PY

echo; echo "### 3. acceptance test through the shipped path ###"
cd pkg && python verify_acceptance.py 2>&1 | tail -14
