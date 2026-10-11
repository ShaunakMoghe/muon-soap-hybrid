#!/bin/bash
#SBATCH -J eigh_det
#SBATCH -p gpu-rtxpro-blackwell
#SBATCH --gres=gpu:rtx_pro_6000_blackwell:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0:20:00
#SBATCH -q inferno
#SBATCH -o runs/sbatch/eigh_det_%j.out
# Eigh determinism: 20x in-process (in-script) + 3x cross-process hash compare.
cd /storage/project/ps-moereasoning-0/smoghe6 || exit 1
for i in 1 2 3; do
  ./run_container.sh python research_stack/test_eigh_determinism.py > runs/sbatch/eigh_run$i.log 2>&1 \
    || { echo "RUN $i FAILED"; tail -5 runs/sbatch/eigh_run$i.log; exit 1; }
done
grep -h "20x identical" runs/sbatch/eigh_run1.log > /tmp/eigh_h1
grep -h "20x identical" runs/sbatch/eigh_run2.log > /tmp/eigh_h2
grep -h "20x identical" runs/sbatch/eigh_run3.log > /tmp/eigh_h3
cmp -s /tmp/eigh_h1 /tmp/eigh_h2 && cmp -s /tmp/eigh_h2 /tmp/eigh_h3 \
  && echo "EIGH_CROSSPROC_OK" || { echo "EIGH_CROSSPROC_DIFF"; exit 1; }
