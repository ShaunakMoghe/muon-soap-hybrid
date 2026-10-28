#!/bin/bash
#SBATCH -J prof_cosmos
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:30:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/prof_cosmos_%j.out
# Step 0a: COSMOS-r32 overhead breakdown (extends profile_optimizers.py §6).
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd $PROJ || exit 1
$PROJ/run_container.sh python profile/profile_optimizers.py > $PROJ/runs/sbatch/prof_cosmos_run.log 2>&1 || { echo PROF_FAIL; tail -30 $PROJ/runs/sbatch/prof_cosmos_run.log; exit 1; }
grep -q "== 6. COSMOS-r32" $PROJ/runs/sbatch/prof_cosmos_run.log || { echo "PROF NO SECTION 6"; exit 1; }
echo PROF_COSMOS_OK
