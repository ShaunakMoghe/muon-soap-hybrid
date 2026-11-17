#!/bin/bash
#SBATCH -J nsnorm_cpu
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/nsnorm_cpu_%j.out
# CPU-only: NS update-norm / spectrum-shape check on existing ckpt_02750
# momentum buffers (Muon s1/s2, COSMOS-r32). No GPU.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-25.10.sif
export APPTAINERENV_PYTHONPATH=$PROJ
export APPTAINERENV_CUDA_VISIBLE_DEVICES=
export APPTAINER_CACHEDIR=$PROJ/.cache/apptainer
mkdir -p "$PROJ/.cache/apptainer"
cd $PROJ || exit 1
apptainer exec --cleanenv "$SIF" python runs/ns_norm_shape_scratch.py > runs/nsnorm_run.log 2>&1 || { echo NSNORM_FAIL; tail -30 runs/nsnorm_run.log; exit 1; }
echo NSNORM_DONE
