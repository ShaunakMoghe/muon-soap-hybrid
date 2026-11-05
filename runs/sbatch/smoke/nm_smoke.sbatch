#!/bin/bash
#SBATCH -J nm_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/nm_smoke_%j.out
# COSMOS seed2 (run 7 Newton-Muon v2) 12-step smoke.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 12/' $PROJ/variants/train_gpt_newton_muon_seed0.py > /tmp/nm_smoke.py
rm -rf /tmp/nmsmoke && mkdir -p /tmp/nmsmoke && cd /tmp/nmsmoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/nm_smoke.py 1 > $PROJ/runs/sbatch/nm_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/nm_smoke_run.log; exit 1; }
grep -q "^step:12/12" $PROJ/runs/sbatch/nm_smoke_run.log || { echo "SMOKE NO STEP12"; exit 1; }
[ -f checkpoints/ckpt_00010.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
V0=$(grep -oE "step:0/12 val_loss:[0-9.]+" $PROJ/runs/sbatch/nm_smoke_run.log | grep -oE "[0-9.]+$")
V12=$(grep -oE "step:12/12 val_loss:[0-9.]+" $PROJ/runs/sbatch/nm_smoke_run.log | grep -oE "[0-9.]+$")
python3 -c "import sys; sys.exit(0 if float(sys.argv[2]) < float(sys.argv[1]) else 1)" "$V0" "$V12" || { echo "SMOKE LOSS NOT DECREASING $V0 -> $V12"; exit 1; }

echo "recompile lines: $(grep -ci recompil $PROJ/runs/sbatch/nm_smoke_run.log || true)"
echo NM_SMOKE_OK
