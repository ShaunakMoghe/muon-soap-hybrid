#!/bin/bash
# Chain two full runs sequentially (each ~105 min): run4 = SOAP rerun
# (same seed0 script as killed run3 — second chaotic draw + early-spread pair),
# then run5 = seeded Muon baseline (control arm of the paired set).
# Launched detached via setsid; legs are independent (`;`), each logs to its
# own run dir. To stop everything: pkill -f train_gpt_soap.py; pkill -f train_gpt_baseline.py
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd "$PROJ/runs/run4_soap_rerun" || exit 1
date -u +"LEG1_START:%Y-%m-%dT%H:%M:%SZ" >> leg1.log
"$PROJ/run_container.sh" torchrun --standalone --nproc_per_node=1 train_gpt_soap.py 1 >>stdout.log 2>&1
echo "LEG1_EXIT:$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> leg1.log
cd "$PROJ/runs/run5_baseline_seed0" || exit 1
date -u +"LEG2_START:%Y-%m-%dT%H:%M:%SZ" >> leg2.log
"$PROJ/run_container.sh" torchrun --standalone --nproc_per_node=1 train_gpt_baseline.py 1 >>stdout.log 2>&1
echo "LEG2_EXIT:$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> leg2.log
