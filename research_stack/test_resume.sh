#!/bin/bash
# Resume validation for research_stack/ckpt.py (needs idle GPU; ~15 min).
# V1 continuity: 130-step baseline, CKPT_EVERY=50 -> resume from ckpt@100,
#   step-125 val must match an uninterrupted reference within 2e-3
#   (natural same-seed spread is ~2e-4; a real resume bug shows >>1e-2).
# V2 preemption drill: SIGTERM the whole process group mid-run (like SLURM),
#   expect emergency checkpoint + "preempted at step" marker, then resume to
#   completion and match the reference within 2e-3.
set -u
PROJ=/storage/project/ps-moereasoning-0/smoghe6
cd "$PROJ" || exit 1
sed 's/train_steps = 3250/train_steps = 130/' variants/train_gpt_baseline_seed0.py > /tmp/resume_test.py
val125() { sed -n 's/^step:125\/[0-9]* val_loss:\([0-9.]*\).*/\1/p' "$1"; }
cmp_close() { python3 -c "import sys; sys.exit(0 if abs(float(sys.argv[1])-float(sys.argv[2])) <= 0.002 else 1)" "$1" "$2"; }

echo "=== V1 reference (uninterrupted) ==="
rm -rf /tmp/rs_ref && mkdir -p /tmp/rs_ref && cd /tmp/rs_ref
ln -sfn $PROJ/modded-nanogpt-public/data data
$PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/resume_test.py 1 > ref.log 2>&1 \
  || { echo "V1 REF FAILED"; tail -50 ref.log; exit 1; }
REF125=$(val125 ref.log); echo "ref step-125 val=$REF125"

echo "=== V1 checkpointed run (ckpts at 50, 100) ==="
rm -rf /tmp/rs_ckpt && mkdir -p /tmp/rs_ckpt && cd /tmp/rs_ckpt
ln -sfn $PROJ/modded-nanogpt-public/data data
APPTAINERENV_CKPT_EVERY=50 $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/resume_test.py 1 > ck.log 2>&1 \
  || { echo "V1 CKPT RUN FAILED"; tail -50 ck.log; exit 1; }
[ -f checkpoints/ckpt_00100.pt ] || { echo "V1 MISSING ckpt_00100.pt"; ls checkpoints/ 2>/dev/null; exit 1; }

echo "=== V1 resume from ckpt@100 ==="
rm -rf /tmp/rs_res && mkdir -p /tmp/rs_res && cd /tmp/rs_res
ln -sfn $PROJ/modded-nanogpt-public/data data
$PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/resume_test.py 1 --resume /tmp/rs_ckpt/checkpoints/ckpt_00100.pt > res.log 2>&1 \
  || { echo "V1 RESUME FAILED"; tail -50 res.log; exit 1; }
grep -q "resumed at step 100" res.log || { echo "V1 NO RESUME MARKER"; exit 1; }
RES125=$(val125 res.log); echo "resumed step-125 val=$RES125"
cmp_close "$REF125" "$RES125" && echo "V1 OK" || { echo "V1 DIVERGED: ref=$REF125 resumed=$RES125"; exit 1; }

echo "=== V2 preemption drill ==="
rm -rf /tmp/rs_term && mkdir -p /tmp/rs_term && cd /tmp/rs_term
ln -sfn $PROJ/modded-nanogpt-public/data data
setsid $PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/resume_test.py 1 > term.log 2>&1 &
VICTIM=$!
for _ in $(seq 1 120); do sleep 5; grep -q "^step:60/" term.log 2>/dev/null && break; done
if ! grep -q "^step:60/" term.log 2>/dev/null; then echo "V2 NEVER REACHED STEP 60"; kill -KILL -$VICTIM 2>/dev/null; exit 1; fi
kill -TERM -$VICTIM 2>/dev/null
wait $VICTIM 2>/dev/null
grep -q "preempted at step" term.log || { echo "V2 NO PREEMPT MARKER"; tail -50 term.log; exit 1; }
LATEST=$(ls checkpoints/ckpt_*.pt 2>/dev/null | sort | tail -1)
[ -n "$LATEST" ] || { echo "V2 NO EMERGENCY CKPT"; exit 1; }
echo "emergency ckpt: $LATEST"
$PROJ/run_container.sh torchrun --standalone --nproc_per_node=1 /tmp/resume_test.py 1 --resume "$PWD/$LATEST" > res2.log 2>&1 \
  || { echo "V2 RESUME FAILED"; tail -50 res2.log; exit 1; }
RES125b=$(val125 res2.log); echo "resumed step-125 val=$RES125b"
cmp_close "$REF125" "$RES125b" && echo "V2 OK" || { echo "V2 DIVERGED: ref=$REF125 resumed=$RES125b"; exit 1; }
echo "RESUME_ALL_OK"
