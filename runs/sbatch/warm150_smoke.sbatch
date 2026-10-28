#!/bin/bash
#SBATCH -J warm150_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/warm150_smoke_%j.out
# Exp 1 smoke: 155 steps (must cross the 150->151 handoff).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
sed 's/train_steps = 3250/train_steps = 155/' $PROJ/variants/train_gpt_cosmos_r32_warm150_seed0.py > /tmp/warm150_smoke.py
rm -rf /tmp/warm150smoke && mkdir -p /tmp/warm150smoke && cd /tmp/warm150smoke
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=10 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/warm150_smoke.py 1 > $PROJ/runs/sbatch/warm150_smoke_run.log 2>&1 || { echo SMOKE_FAIL; tail -50 $PROJ/runs/sbatch/warm150_smoke_run.log; exit 1; }
grep -q "^step:155/155" $PROJ/runs/sbatch/warm150_smoke_run.log || { echo "SMOKE NO STEP155"; exit 1; }
[ -f checkpoints/ckpt_00150.pt ] || { echo "SMOKE NO CKPT"; ls checkpoints/ 2>/dev/null; exit 1; }
V0=$(grep -oE "step:0/155 val_loss:[0-9.]+" $PROJ/runs/sbatch/warm150_smoke_run.log | grep -oE "[0-9.]+$")
V125=$(grep -oE "step:125/155 val_loss:[0-9.]+" $PROJ/runs/sbatch/warm150_smoke_run.log | grep -oE "[0-9.]+$")
python3 -c "import sys; sys.exit(0 if float(sys.argv[2]) < float(sys.argv[1]) else 1)" "$V0" "$V125" || { echo "SMOKE LOSS NOT DECREASING $V0 -> $V125"; exit 1; }
N=$(grep -c "warm150 handoff" $PROJ/runs/sbatch/warm150_smoke_run.log || true)
[ "$N" = "72" ] || { echo "SMOKE HANDOFF COUNT $N != 72"; exit 1; }
python3 - "$PROJ/runs/sbatch/warm150_smoke_run.log" <<'EOF' || exit 1
import re, sys
rs = [float(m.group(1)) for line in open(sys.argv[1])
      for m in [re.search(r"ratio=([0-9.]+)", line)] if m]
bad = [r for r in rs if not 0.95 <= r <= 1.05]
print(f"handoff ratios: n={len(rs)} min={min(rs):.4f} max={max(rs):.4f}")
sys.exit(0 if not bad else 1)
EOF
[ "$(grep -c "^nsresid " $PROJ/runs/sbatch/warm150_smoke_run.log || true)" = "9" ] || { echo "SMOKE NSRESID COUNT WRONG"; exit 1; }
echo "recompile lines: $(grep -ci recompil $PROJ/runs/sbatch/warm150_smoke_run.log || true)"
echo WARM150_SMOKE_OK
