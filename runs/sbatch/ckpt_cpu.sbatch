#!/bin/bash
#SBATCH -J ckpt_cpu
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/ckpt_cpu_%j.out
# CPU ckpt round-trip narrowing test (no GPU needed; container without --nv).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-25.10.sif
export APPTAINERENV_PYTHONPATH=$PROJ
export APPTAINER_CACHEDIR=$PROJ/.cache/apptainer
mkdir -p "$PROJ/.cache/apptainer"
apptainer exec --cleanenv "$SIF" python "$PROJ/research_stack/test_ckpt_roundtrip_cpu.py"
