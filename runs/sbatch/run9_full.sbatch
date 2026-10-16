#!/bin/bash
#SBATCH -J run9_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run9_full_%j.out
# Run 9 full: COSMOS seed1, 3250 steps, saving ON (CKPT_EVERY=250).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run9_cosmos_s1 || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_cosmos_seed1.py 1 > stdout.log 2>&1 || { echo RUN9_FAIL; tail -50 stdout.log; exit 1; }
echo RUN9_FULL_DONE
