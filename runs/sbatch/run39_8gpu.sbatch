#!/bin/bash
#SBATCH -J run39_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run39_8gpu_%j.out
# Run 39 8-GPU: Muon at COSMOS hypers seed0 8-GPU, 500 steps, saving OFF for 8-GPU probe (ckpt asserts world_size 1; 20-min run needs no resume).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run39_muon_hswap_s0 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_baseline_hswap_seed0.py 1 > stdout.log 2>&1 || { echo RUN39_FAIL; tail -50 stdout.log; exit 1; }
echo RUN39_8GPU_DONE
