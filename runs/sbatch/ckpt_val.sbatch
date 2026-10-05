#!/bin/bash
#SBATCH -J ckpt_val
#SBATCH -A paceship-moereasoning
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:45:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/ckpt_val_%j.out
# Checkpoint validation R0/R1/R2 (research_stack/test_resume.sh as-is).
# Standard shape: 1x Blackwell + 8 CPUs + 96GB. No vacancy at submit -> inferno.
cd /storage/project/ps-moereasoning-0/smoghe6 || exit 1
bash research_stack/test_resume.sh
