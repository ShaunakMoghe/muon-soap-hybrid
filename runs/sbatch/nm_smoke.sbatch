#!/bin/bash
#SBATCH -J nm_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/nm_smoke_%j.out
# Newton-Muon smoke: zero-recompile warm-cache rerun (test_nm_smoke.sh).
# Account passed via CLI (sbatch --account=...) — never committed.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ || exit 1
APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash research_stack/test_nm_smoke.sh
