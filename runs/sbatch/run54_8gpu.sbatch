#!/bin/bash
#SBATCH -J run54_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run54_8gpu_%j.out
# Run 54 8-GPU: COSMOS r32 Sink1 seed1 8-GPU, 3250 steps, saving OFF (ckpt asserts world_size 1).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run54_cosmos_sink1_s1 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_cosmos_r32_sink1_seed1.py 1 > stdout.log 2>&1 || { echo RUN54_FAIL; tail -50 stdout.log; exit 1; }
echo RUN54_8GPU_DONE
