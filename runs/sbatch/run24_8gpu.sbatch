#!/bin/bash
#SBATCH -J run24_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run24_8gpu_%j.out
# Run 24 8-GPU: Muon seed1 8-GPU 3250-step, 500 steps, saving OFF for 8-GPU probe (ckpt asserts world_size 1; 20-min run needs no resume).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run24_muon_s1_8gpu || exit 1
nvidia-smi topo -m 2>/dev/null | head -20; nvidia-smi nvlink -s 2>/dev/null | head -5 || echo NO_NVLINK_REPORTED
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 $PROJ/variants/train_gpt_baseline_seed1.py 1 > stdout.log 2>&1 || { echo RUN24_FAIL; tail -50 stdout.log; exit 1; }
echo RUN24_8GPU_DONE
