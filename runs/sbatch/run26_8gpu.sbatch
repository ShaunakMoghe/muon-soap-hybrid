#!/bin/bash
#SBATCH -J run26_8gpu
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:8
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 1:00:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run26_8gpu_%j.out
# Run 26 8-GPU: SinkSOAP seed0 8-GPU 500-step, 500 steps, saving OFF for 8-GPU probe (ckpt asserts world_size 1; 20-min run needs no resume).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ/runs/run26_sinksoap_8gpu || exit 1
nvidia-smi topo -m 2>/dev/null | head -20; nvidia-smi nvlink -s 2>/dev/null | head -5 || echo NO_NVLINK_REPORTED
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=8 script_used.py 1 > stdout.log 2>&1 || { echo RUN26_FAIL; tail -50 stdout.log; exit 1; }
echo RUN26_8GPU_DONE
