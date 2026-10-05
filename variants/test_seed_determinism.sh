#!/bin/bash
# Determinism check for seeded variants: run <script> twice at 12 steps and
# require identical final (step-12) val_loss (same init => paired comparisons valid).
# NOTE: step-0 val is NOT a valid check — the output proj is zero-initialized,
# so step-0 loss is ln(vocab)=10.82583 for ANY init. Assert on post-training val.
# Needs an idle GPU — do NOT run alongside a live training run (port + contention).
# Usage: ./test_seed_determinism.sh variants/train_gpt_baseline_seed0.py
set -u
SCRIPT=${1:?usage: test_seed_determinism.sh <variant-script.py>}
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd "$PROJ" || exit 1
sed 's/train_steps = 3250/train_steps = 12/' "$SCRIPT" > /tmp/det_check.py
rm -rf /tmp/det_check_run && mkdir -p /tmp/det_check_run && cd /tmp/det_check_run
ln -sfn "$PROJ/modded-nanogpt-public/data" data
for i in 1 2; do
  "$PROJ/run_container.sh" torchrun --standalone --nproc_per_node=1 /tmp/det_check.py 1 > run$i.log 2>&1 \
    || { echo "RUN $i FAILED"; tail -5 run$i.log; exit 1; }
done
V1=$(grep "val_loss" run1.log | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p')
V2=$(grep "val_loss" run2.log | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p')
echo "run1 final val_loss=$V1"
echo "run2 final val_loss=$V2"
[ -n "$V1" ] && [ "$V1" = "$V2" ] && echo DETERMINISM_OK || { echo DETERMINISM_FAIL; exit 1; }
