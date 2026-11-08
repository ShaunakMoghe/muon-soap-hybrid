#!/bin/bash
#SBATCH -J run55_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run55_8gpu_%j.out
# Run 55 8-GPU: Muon spec-norm NS6 seed0 500-step 8-GPU, 500 steps, saving OFF (ckpt asserts world_size 1).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run55_muon_specns6_s0 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_baseline_specns6_seed0.py 1 > stdout.log 2>&1 || { echo RUN55_FAIL; tail -50 stdout.log; exit 1; }
echo RUN55_8GPU_DONE
