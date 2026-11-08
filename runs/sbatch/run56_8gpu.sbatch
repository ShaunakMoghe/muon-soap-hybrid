#!/bin/bash
#SBATCH -J run56_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run56_8gpu_%j.out
# Run 56 8-GPU: Muon spec-norm NS6 seed1 500-step 8-GPU, 500 steps, saving OFF (ckpt asserts world_size 1).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run56_muon_specns6_s1 || exit 1
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_baseline_specns6_seed1.py 1 > stdout.log 2>&1 || { echo RUN56_FAIL; tail -50 stdout.log; exit 1; }
echo RUN56_8GPU_DONE
