#!/bin/bash
#SBATCH -J run15_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run15_full_%j.out
# Run 15 full: COSMOS r128, 3250 steps, saving ON (CKPT_EVERY=250).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run15_cosmos_r128 || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_cosmos_r128_seed0.py 1 > stdout.log 2>&1 || { echo RUN15_FAIL; tail -50 stdout.log; exit 1; }
echo RUN15_FULL_DONE
