#!/bin/bash
#SBATCH -J run8_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run8_full_%j.out
# Run 8 full: amortized SinkSOAP, 3250 steps, saving ON (CKPT_EVERY=250).
# Account passed via CLI (sbatch --account=...) — never committed.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run8_sinksoap_amort || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_sinksoap_amort_seed0.py 1 > stdout.log 2>&1 || { echo RUN8_FAIL; tail -50 stdout.log; exit 1; }
echo RUN8_FULL_DONE
