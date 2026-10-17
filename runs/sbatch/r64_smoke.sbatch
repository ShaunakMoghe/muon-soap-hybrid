#!/bin/bash
#SBATCH -J r64_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/r64_smoke_%j.out
# COSMOS seed2 (COSMOS r64) 12-step smoke.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 12/' $PROJ/variants/train_gpt_cosmos_r64_seed0.py > /tmp/r64_smoke.py
rm -rf /tmp/r64smoke && mkdir -p /tmp/r64smoke && cd /tmp/r64smoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/r64_smoke.py 1 > $PROJ/runs/sbatch/r64_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/r64_smoke_run.log; exit 1; }
grep -q "^step:12/12" $PROJ/runs/sbatch/r64_smoke_run.log || { echo "SMOKE NO STEP12"; exit 1; }
[ -f checkpoints/ckpt_00010.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
echo R64_SMOKE_OK
