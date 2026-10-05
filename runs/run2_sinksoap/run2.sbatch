#!/bin/bash
# Fallback: rerun run2 (SinkSOAP @3250) as a batch job if the interactive
# allocation expires first. UNTESTED — adjust partition/account as needed.
# Submit from project root: sbatch runs/run2_sinksoap/run2.sbatch
#SBATCH --job-name=run2_sinksoap
#SBATCH --partition=gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=runs/run2_sinksoap/slurm_%j.out
#SBATCH --open-mode=append
# #SBATCH --account=<paceship-account-if-needed>
cd /storage/project/ps-moereasoning-0/smoghe6/runs/run2_sinksoap
/storage/project/ps-moereasoning-0/smoghe6/run_container.sh torchrun --standalone --nproc_per_node=1 train_gpt_sinksoap.py 1 >>stdout_batch.log 2>&1
