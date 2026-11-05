#!/bin/bash
#SBATCH -J r32_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/r32_smoke_%j.out
# COSMOS seed2 (COSMOS r32) 12-step smoke.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 12/' $PROJ/variants/train_gpt_cosmos_r32_seed0.py > /tmp/r32_smoke.py
rm -rf /tmp/r32smoke && mkdir -p /tmp/r32smoke && cd /tmp/r32smoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/r32_smoke.py 1 > $PROJ/runs/sbatch/r32_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/r32_smoke_run.log; exit 1; }
grep -q "^step:12/12" $PROJ/runs/sbatch/r32_smoke_run.log || { echo "SMOKE NO STEP12"; exit 1; }
[ -f checkpoints/ckpt_00010.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
echo R32_SMOKE_OK
