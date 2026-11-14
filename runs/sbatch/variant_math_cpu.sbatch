#!/bin/bash
#SBATCH -J variant_math_cpu
#SBATCH -p cpu-small
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0:20:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/variant_math_cpu_%j.out
# CPU-only: (1) optimizer-math checks for the r0 / qrk / sink / specns6
# variants, (2) the GPU preflight block from a generated 8-GPU sbatch, run
# verbatim on this GPU-less node: it must report failure with exit code 42.
PROJ=/storage/project/ps-moereasoning-0/smoghe6
SIF=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-25.10.sif
export APPTAINERENV_PYTHONPATH=$PROJ
export APPTAINERENV_CUDA_VISIBLE_DEVICES=
export APPTAINER_CACHEDIR=$PROJ/.cache/apptainer
mkdir -p "$PROJ/.cache/apptainer"
cd $PROJ || exit 1
LOG=runs/sbatch/variant_math_cpu_run.log
apptainer exec --cleanenv "$SIF" python research_stack/test_variant_math_cpu.py > $LOG 2>&1 || { echo VARIANT_MATH_FAIL; tail -40 $LOG; exit 1; }
cat $LOG
sed -n "/^\$PROJ\/run_container.sh python -c '\$/,/^'\$/p" runs/sbatch/run57_8gpu.sbatch > /tmp/preflight_block_$SLURM_JOB_ID.sh
echo "preflight block lines: $(wc -l < /tmp/preflight_block_$SLURM_JOB_ID.sh)"
unset APPTAINERENV_CUDA_VISIBLE_DEVICES
PROJ=$PROJ bash /tmp/preflight_block_$SLURM_JOB_ID.sh
echo "PREFLIGHT_RC=$?"
