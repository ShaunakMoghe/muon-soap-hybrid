#!/bin/bash
#SBATCH -J run18_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run18_full_%j.out
# Run 18 full: COSMOS r32, 3250 steps, saving ON (CKPT_EVERY=250).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run18_cosmos_r32 || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_cosmos_r32_seed0.py 1 > stdout.log 2>&1 || { echo RUN18_FAIL; tail -50 stdout.log; exit 1; }
echo RUN18_FULL_DONE
