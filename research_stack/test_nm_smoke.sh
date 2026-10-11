#!/bin/bash
# Newton-Muon GPU smoke: correctness + zero-per-step-recompiles. Needs idle GPU (~10 min).
# 12-step x2 runs with TORCH_LOGS=recompiles.
# Run 1: must complete + val must decrease (hooks fire, preconditioner sane).
# Run 2 (warm file cache): must print ZERO recompile lines + stable step times.
#
# Why this invariant (NOT "zero graph breaks"): the gated hooks read a mutable
# step counter, so they are @torch.compiler.disable BY DESIGN -- dynamo breaks
# at each hook boundary (bounded, one-time fragmentation) and runs hooks eager.
# Zero breaks is therefore unachievable AND undesirable; the catastrophic failure
# would be per-step recompiles (a global_step guard failing every step), which
# recur in EVERY run -- hence run 2 must be recompile-silent.
set -u
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd "$PROJ" || exit 1
sed 's/train_steps = 3250/train_steps = 12/' variants/train_gpt_newton_muon_seed0.py > /tmp/nm_smoke.py
run_one() { # $1 = rundir
  rm -rf /tmp/$1 && mkdir -p /tmp/$1 && cd /tmp/$1
  ln -sfn $PROJ/modded-nanogpt-public/data data
  APPTAINERENV_TORCH_LOGS=recompiles $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/nm_smoke.py 1 > sm.log 2>&1
}
run_one nm_sm1 || { echo "SMOKE RUN1 FAILED"; tail -50 /tmp/nm_sm1/sm.log; exit 1; }
V0=$(sed -n 's/^step:0\/[0-9]* val_loss:\([0-9.]*\).*/\1/p' /tmp/nm_sm1/sm.log)
V12=$(grep "val_loss" /tmp/nm_sm1/sm.log | tail -1 | sed -n 's/.*val_loss:\([0-9.]*\).*/\1/p')
python3 -c "import sys; sys.exit(0 if float(sys.argv[2]) < float(sys.argv[1]) else 1)" "$V0" "$V12" \
  || { echo "SMOKE: LOSS DID NOT DECREASE $V0 -> $V12"; exit 1; }
echo "run1 OK: $V0 -> $V12"
run_one nm_sm2 || { echo "SMOKE RUN2 FAILED"; tail -50 /tmp/nm_sm2/sm.log; exit 1; }
N=$(grep -ci "recompil" /tmp/nm_sm2/sm.log 2>/dev/null || true)
[ "$N" = "0" ] || { echo "SMOKE: $N RECOMPILE LINES IN RUN2"; grep -i "recompil" /tmp/nm_sm2/sm.log | head -5; exit 1; }
echo "run2 OK: zero recompile lines"
python3 - <<'EOF'
import re, sys
ts = []
for line in open('/tmp/nm_sm2/sm.log'):
    m = re.match(r"step:(\d+)/\d+ train_time:([0-9.]+)s", line)
    if m and 3 <= int(m.group(1)) <= 12:
        ts.append(float(m.group(2)))
ds = [b - a for a, b in zip(ts, ts[1:])]
print("step deltas:", ["%.2f" % d for d in ds])
sys.exit(0 if max(ds) < 2 * min(ds) else 1)
EOF
[ $? -eq 0 ] && echo "SMOKE ALL OK" || { echo "SMOKE: UNSTABLE STEP TIMES"; exit 1; }
