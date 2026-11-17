#!/bin/bash
#SBATCH -J deflspec_cpu
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/deflspec_cpu_%j.out
# CPU-only: deflation spectra (M vs M_res normalized spectra + NS error on
# nonzero subspace). No GPU.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-25.10.sif
export APPTAINERENV_PYTHONPATH=$PROJ
export APPTAINERENV_CUDA_VISIBLE_DEVICES=
export APPTAINER_CACHEDIR=$PROJ/.cache/apptainer
mkdir -p "$PROJ/.cache/apptainer"
cd $PROJ || exit 1
apptainer exec --cleanenv "$SIF" python runs/deflation_spectra_scratch.py > runs/deflspec_run.log 2>&1 || { echo DEFLSPEC_FAIL; tail -30 runs/deflspec_run.log; exit 1; }
echo DEFLSPEC_DONE
