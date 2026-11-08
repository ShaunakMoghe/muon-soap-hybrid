#!/bin/bash
#SBATCH -J run50_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run50_8gpu_%j.out
# Run 50 8-GPU: COSMOS r32 Sink5 seed1 8-GPU, 3250 steps, saving OFF (ckpt asserts world_size 1).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run50_cosmos_sink5_s1 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_cosmos_r32_sink5_seed1.py 1 > stdout.log 2>&1 || { echo RUN50_FAIL; tail -50 stdout.log; exit 1; }
echo RUN50_8GPU_DONE
