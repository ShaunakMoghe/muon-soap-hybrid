#!/bin/bash
#SBATCH -J run45_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run45_8gpu_%j.out
# Run 45 8-GPU: COSMOS r32 QR/4 seed0 8-GPU, 3250 steps, saving OFF (ckpt asserts world_size 1).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run45_cosmos_qrk4_s0 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_cosmos_r32_qrk4_seed0.py 1 > stdout.log 2>&1 || { echo RUN45_FAIL; tail -50 stdout.log; exit 1; }
echo RUN45_8GPU_DONE
