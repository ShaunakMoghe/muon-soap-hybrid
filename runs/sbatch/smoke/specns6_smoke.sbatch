#!/bin/bash
#SBATCH -J specns6_smoke
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/specns6_smoke_%j.out
#SBATCH --exclude=atl1-1-03-020-6-0
# 12-step 1-GPU smoke of variants/train_gpt_baseline_specns6_seed0.py: first GPU execution of this
# variant family (compile + optimizer path), ahead of the queued 8-GPU runs.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
LOG=$PROJ/runs/sbatch/smoke/specns6_smoke_run.log
W=/tmp/specns6_smoke_$SLURM_JOB_ID
mkdir -p $W && cd $W || exit 1
sed -E 's/train_steps = [0-9]+/train_steps = 12/' $PROJ/variants/train_gpt_baseline_specns6_seed0.py > $W/smoke.py
[ "$(grep -c 'train_steps = 12' $W/smoke.py)" = "1" ] || { echo "SMOKE SED MISS"; exit 1; }
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=1000000 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 $W/smoke.py 1 > $LOG 2>&1 || { echo SMOKE_FAIL; tail -50 $LOG; exit 1; }
grep -q "^step:12/12 val_loss" $LOG || { echo "SMOKE NO STEP12"; exit 1; }
echo "$(grep -oE 'step:12/12 val_loss:[0-9.]+' $LOG | tail -1)  recompile lines: $(grep -ci recompil $LOG || true)"
echo SPECNS6_SMOKE_OK
