#!/bin/bash
#SBATCH -J run12_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run12_full_%j.out
# Run 12 full: amortized T=10 (freq=10), 3250 steps, saving ON (CKPT_EVERY=250).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run12_amortT10 || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_sinksoap_amortT10_seed0.py 1 > stdout.log 2>&1 || { echo RUN12_FAIL; tail -50 stdout.log; exit 1; }
echo RUN12_FULL_DONE
