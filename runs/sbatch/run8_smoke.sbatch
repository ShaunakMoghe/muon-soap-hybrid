#!/bin/bash
#SBATCH -J run8_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/run8_smoke_%j.out
# Run 8 (amortized SinkSOAP) 12-step smoke: script validity + save path.
# Account passed via CLI (sbatch --account=...) — never committed.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 12/' $PROJ/variants/train_gpt_sinksoap_amort_seed0.py > /tmp/run8_smoke.py
rm -rf /tmp/run8smoke && mkdir -p /tmp/run8smoke && cd /tmp/run8smoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/run8_smoke.py 1 > $PROJ/runs/sbatch/run8_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/run8_smoke_run.log; exit 1; }
grep -q "^step:12/12" $PROJ/runs/sbatch/run8_smoke_run.log || { echo "SMOKE NO STEP12"; exit 1; }
[ -f checkpoints/ckpt_00010.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
echo RUN8_SMOKE_OK
