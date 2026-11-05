#!/bin/bash
#SBATCH -J t10_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/t10_smoke_%j.out
# Amortized T=10 (run 12) 12-step smoke.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 12/' $PROJ/variants/train_gpt_sinksoap_amortT10_seed0.py > /tmp/t10_smoke.py
rm -rf /tmp/t10smoke && mkdir -p /tmp/t10smoke && cd /tmp/t10smoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/t10_smoke.py 1 > $PROJ/runs/sbatch/t10_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/t10_smoke_run.log; exit 1; }
grep -q "^step:12/12" $PROJ/runs/sbatch/t10_smoke_run.log || { echo "SMOKE NO STEP12"; exit 1; }
[ -f checkpoints/ckpt_00010.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
echo T10_SMOKE_OK
