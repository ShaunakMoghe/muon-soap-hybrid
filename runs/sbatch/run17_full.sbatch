#!/bin/bash
#SBATCH -J run17_full
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 3:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run17_full_%j.out
# Run 17 full: Muon seed2, 3250 steps, saving ON (CKPT_EVERY=250).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run17_muon_s2 || exit 1
APPTAINERENV_CKPT_EVERY=250 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $PROJ/variants/train_gpt_baseline_seed2.py 1 > stdout.log 2>&1 || { echo RUN17_FAIL; tail -50 stdout.log; exit 1; }
echo RUN17_FULL_DONE
