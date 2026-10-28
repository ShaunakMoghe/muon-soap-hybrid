#!/bin/bash
#SBATCH -J nsresid
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/nsresid_%j.out
# Step 0b: NS residuals M vs M_res at 12/8/6 from ckpt states (no training).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ || exit 1
$PROJ/run_container.sh python profile/ns_residual_check.py runs/run18_cosmos_r32/checkpoints/ckpt_02750.pt runs/run16_muon_s1/checkpoints/ckpt_02750.pt > $PROJ/runs/sbatch/nsresid_run.log 2>&1 || { echo NSRESID_FAIL; tail -30 $PROJ/runs/sbatch/nsresid_run.log; exit 1; }
grep -q "mean-over-params" $PROJ/runs/sbatch/nsresid_run.log || { echo "NSRESID NO OUTPUT"; exit 1; }
echo NSRESID_OK
