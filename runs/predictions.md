# Run predictions ledger (predict before, record after)

## Run 8 smoke: amortized SinkSOAP 12-step — QUEUED
- Predict: completes; step-12 val in [6.2, 6.5] (unamortized seed0 was
  6.33337@12; amortization shifts the early curve slightly);
  checkpoints/ckpt_00010.pt written (save path exercised, CKPT_EVERY=10).
- OUTCOME (job 13945795): TECHNICAL PASS (completed + ckpt written) but
  step-12 val 7.82851 vs predicted [6.2, 6.5] — PREDICTION WRONG.
  Mechanism (hypothesized): cold-start staleness — step-1 bases built
  from near-zero grams are reused for steps 2-10. Full run tests
  whether it recovers; crossing band widened to [3100, 3250].

## Run 8 full: amortized SinkSOAP 3250 steps (precond_freq=10, T=1) — PENDING smoke
- Predict: 1.85–1.95 s/step (eigh cost ÷10: 1.83 + ~0.1 overhead);
  crossing in [3100, 3200] (staleness may cost 0–100 steps vs unamortized
  3100); final ~3.27.
- Decides: does the ~3% step saving survive at ~Muon per-step cost?
- OUTCOME (job 13949314): COST HIT (1.952s/step), QUALITY MISS — NO
  crossing, final 3.29182 (+0.012). Verdict: savings do NOT survive at
  (freq=10, T=1); amortized SinkSOAP is worse than Muon on both axes
  this draw. Caveat: T=1 and freq=10 changed together — unattributed.

## COSMOS smoke: 12-step — QUEUED
- Predict: completes; step-12 val in [6.0, 6.6]; ckpt written.
- OUTCOME (job 13945796): PASS. Step-12 val 6.33464 — prediction HIT
  (notably == unamortized SinkSOAP's 6.33337@12). 31.3GB max.

## COSMOS full (after smoke) — config LOGGED here per loadout
- Variant: variants/train_gpt_cosmos_seed0.py = Sep harness, seed0,
  3250 steps, Sep optimizer1 (embed 0.7 / proj 0.004 / 1D 0.015);
  optimizer2 = COSMOS lr=0.04, wd=0.025, rank_o=rank_u=256
  (Liming's track-3 wiring values; ranks are class defaults).
- Predict: crosses 3.28 (method is tuned for this track); s/step
  unknown — per-step QR on rank-256 subspaces, guess 2.0–2.5s.
  Wide uncertainty, stated plainly.
- OUTCOME (job 13949315): CROSS HIT (3150), COST HIT (1.986s/step),
  final 3.27399. COSMOS beats seeded Muon by 75 steps at ~same step
  cost — best seed0 final and best SOAP-family wall-clock (6258s to
  cross, ~4-7% slower than Muon 5860/5987s to target).

## Newton-Muon smoke (test_nm_smoke.sh: zero-recompile warm-cache rerun) — QUEUED
- Predict: passes (script asserts zero recompile lines + stable step times).
- Full benchmark ONLY if smoke passes. Paper claims (incl. any novelty
  framing) treated as UNVERIFIED until Liming weighs in.
- OUTCOME (job 13945798): FAIL — CUDA OOM in first forward (83/95GB,
  before any hook accumulation runs). Diagnosis: 24 compiler-disabled
  hooks fragment dynamo graphs; per-fragment cudagraph pools +
  fragmentation. FIX (one shot per stop rule): triton.cudagraphs=False
  + expandable_segments. Resubmitted; if OOM persists, lane STOPS.
- OUTCOME (job 13949316): SAME OOM (84.3GB; allocated GREW 70.6->82.9GB,
  fragmentation fixed but real bloat grew — cudagraphs-off removed memory
  planning). Lane STOPPED per stop rule. NM needs deeper surgery
  (selective-eager linears or attach/detach scheduling) — user decision.

## Smoke bands batch 2 (jobs 14050014-18)
- c9/c10 (COSMOS s1/s2): step-12 val in [6.0, 6.6] (seed0 was 6.33464).
  OUTCOME: c9 6.32481, c10 6.32774 — both HIT. Fulls submitted
  (14095529/14095530).
- soap11 (ref recipe): completes + ckpt; val band WIDE [6.0, 8.5]
  (warmup shifts early curve unpredictably; seed0 unamortized ref unknown).
  OUTCOME (job 14050016): FAIL "NO CKPT" — soap_ref lacked ckpt hooks
  (12 steps completed, val 9.70472, consistent with warmup eta<=0.05 —
  band miscalibrated, not a model failure). Fix: ckpt ported (same
  5-insertion port); resubmitted as 14095534 (one-shot allowance).
  OUTCOME (job 14095534): SMOKE_OK, step-12 val 9.70472 (same as first
  attempt — band was miscalibrated, model fine). Full run11 submitted
  as 14129567.
- t10 (freq=10/T=10): completes + ckpt; val in [6.2, 8.0] (between
  unamortized 6.33 and run8-T1 7.83 if T drives the shift).
  OUTCOME (job 14050017): 7.88492 — HIT (barely), and ~= run8-T1's 7.83:
  early read is T does NOT drive the shift (staleness does). Full
  submitted (14095532); full run decides.
- ws (warm-start): completes + ckpt; val in [6.2, 6.6] (eigh-every-step
  early should track unamortized 6.33).
  OUTCOME (job 14050018): 6.35683 — HIT; warm-start tracks unamortized
  early as predicted. Full submitted (14095533).

## COSMOS replicates seed1/seed2 (runs 9, 10) — PENDING
- Predict: both cross; crossing steps within ±100 of 3150 (noise estimate
  for the 50-75-step edge over Muon). If spread >> 75 steps, the edge is
  inconclusive per the noise rule. (run7 slot reserved for NM if revived.)
  OUTCOME (jobs 14095529/14095530): run9 cross 3150 final 3.27373
  (1.997s/step); run10 cross 3175 final 3.27430 (2.036s/step) — both
  HIT. Three-seed spread 3150/3150/3175, finals within 0.0006: the
  50-75-step edge over Muon (3225) SURVIVES the noise rule.

## Plain SOAP reference recipe (run 11: lr=0.0018 + 250 warmup, seed0) — PENDING
- Predict: crosses OR lands materially closer than run4A's +0.066 if the
  hot-lr/no-warmup hypothesis is right; no-cross again weakens it.
  Pairs with run4A (same seed, recipe change only).
  OUTCOME (job 14130113): NO cross, min 3.30265 (+0.023), 1.879s/step,
  max mem 41.4GB — materially closer than +0.066 but still no-cross:
  hypothesis WEAKENED (recipe matters, but is not the fix). Plain SOAP
  stays unresolved. (14129567 failed first on a wrong script path in
  the sbatch — infra error, resubmitted.)

## Amortized T=10 (run 12: freq=10, T=10) — PENDING
- Predict: separates the confound — if T=1 caused run8's failure, this
  recovers toward 3100-cross; if staleness, still no-cross. Cost ~2.0s/step.
  OUTCOME (job 14095532): NO cross, final 3.28985, 2.017s/step —
  staleness branch CONFIRMED, T exonerated (matches smoke read 7.88~=7.83).

## Amortized warm-start (run 13: eigh every step to 200, then every 10, T=1) — PENDING
- Predict: if cold-start staleness caused run8's failure, this recovers
  most of the gap (cross 3100-3200); cost ~2.1s/step (200 full steps).
  OUTCOME (job 14095533): cross 3125, final 3.26992, 2.104s/step — HIT
  on all three. Run8 note REVISED: failure is cold-start staleness,
  not staleness per se; warm-started freq-10 matches unamortized
  (3100/3.26769) at ~56% per-step cost.

## Batch-3: COSMOS rank sweep (runs 14/15/18/19) + Muon seeds (runs 16/17) — PENDING

Setup (all CPU-verified before submit): rank variants differ from run6's
script ONLY in rank_o=rank_u=r (varied TOGETHER; r in {32,64,128,512},
seed0, same lr/wd/harness; 6-line diff each, py_compile clean). Muon s1/s2
differ from run5's script only in the 2 seed lines. Val reported at fixed
steps 2500/3000/3250 plus crossing. Grounding: run5 Muon-s0
(2500: 3.38123, 3000: 3.30489, 3250: 3.27844, cross 3225, 1.857s/step,
31.075GB); run6 COSMOS-s0 (2500: 3.36948, 3000: 3.29779, 3250: 3.27399,
cross 3150, 1.986s/step, 31.280GB).

Scaling model (hypothesized): COSMOS overhead over Muon (~0.13-0.18s/step
at r=256) is dominated by subspace matmuls ~linear in r; overhead(r) ≈
0.01 + 0.15*(r/256) s/step. Memory = 31.075GB + state delta from
state_accounting.md (C256-calibrated: predicted 31.269 vs measured 31.280).

- run14 (r64): s/step 1.89-1.93; cross 3175-3225; final 3.274-3.279;
  max mem ~31.12GB (±0.1); fixed vals between run5 and run6 rows above.
  OUTCOME (14130114/14130122): cross 3200 (6184s), final 3.27649,
  1.933s/step, 31.120GB — HIT on all four (s/step at band edge).
- run15 (r128): s/step 1.93-1.98; cross 3150-3200; final 3.273-3.277;
  max mem ~31.16GB (±0.1).
  OUTCOME (14130115/14130123): cross 3200 (6077s), final 3.27640,
  1.899s/step, 31.167GB — s/step MISS (faster than band); rest HIT.
- run18 (r32): s/step 1.87-1.91; cross 3200-3250 WITH no-cross tail risk
  (rank may be too crude; worst case lands near Muon, NOT worse than run5
  by more than +0.005 — a worse landing would implicate rank pathology,
  not just weak curvature); final 3.276-3.283; max mem ~31.10GB (±0.1).
  OUTCOME (14130118/14130126): cross 3175 (6008s), final 3.27517,
  1.892s/step, 31.097GB — cross + final MISS on the good side (better
  than band); tail risk did NOT materialize. r32 ≈ r256 quality.
- run19 (r512): s/step 2.10-2.20; cross 3125-3175 (at least as good as
  r256; diminishing returns expected); final 3.272-3.275; max mem
  ~31.53GB (±0.15).
  OUTCOME (14130119/14130127): cross 3125 (6724s), final 3.27021,
  2.151s/step, 31.582GB — final MISS on the good side (best COSMOS
  final); rest HIT. Diminishing returns confirmed (25 steps for
  2x rank + 8% slower steps).
- run16/17 (Muon s1/s2): s/step 1.84-1.88; cross within ±75 steps of 3225
  (3150-3250 window); final 3.27844±0.004; max mem ~31.08GB (±0.1).
  These set the Muon spread; the COSMOS-vs-Muon edge stays UNESTABLISHED
  until both spreads are in.
  OUTCOME (14130116/14130124, 14130117/14130125): s1 cross 3250 final
  3.27959 (1.823s/step); s2 cross 3225 final 3.27922 (1.816s/step) —
  cross/final/mem HIT; s/step MISS (both ~2% faster than band AND than
  run5's 1.857: node/machine variance, same code). Muon 3-seed spread:
  crosses 3225/3250/3225, finals within 0.0012. COSMOS-256: 3150/3150/
  3175. Worst COSMOS (3175) still beats best Muon (3225) by 50 steps
  with no overlap → edge ESTABLISHED (measured, 3v3).

## 8-GPU probes (run20 Muon-s0, run21 COSMOS-r32-s0; 500 steps) — PENDING

Purpose: (a) measure per-step cost at 8 GPUs under the current
param-gather scheme (the scale claim's cheapest test); (b) verify the
code's "equivalently with 1/2/4/8 GPUs" claim by matching 1-GPU curves.
Same scripts as run5/run18 except train_steps 500 (executed file + its
source snapshot archived in run dir / logs/*.txt).

- Predict: step_avg within 1.6-2.0 s/step for both (compute shards 8
  ways; comm adds ~340MB param-gather + ~1.1GB grad-reduce per step);
  C-r32 within +5% of Muon-8GPU (subspace work is local; no extra
  collective vs Muon — from code, both all-gather params only).
- Predict: val at 125/250/375/500 within 1% of run5 (4.59338/4.09780/
  3.92328/3.82585) and run18 (4.66057/4.10066/3.91876/3.81929)
  respectively. A larger gap implicates all-reduce associativity drift
  or a sharding bug, and would gate any 8-GPU crossing comparison.
- NOT measured here: steps-to-cross at 8 GPUs (needs full 3250-step
  runs; this probe decides whether those are worth the queue + spend).
  OUTCOME (jobs 14395302/14395303): step_avg 0.390 (Muon) / 0.395 s/step
  (C-r32) — absolute band (1.6-2.0) MISS badly: prediction forgot compute
  shards ~linearly (1.82/8=0.23 + comm); relative prediction HIT (+1.2%,
  inside +5%). 8-GPU throughput ~4.7x 1-GPU.
  METHOD NOTE (both figures are correct, different denominators): 0.390 /
  0.395 are steady-state means over steps 376-500 (final step_avg field,
  excludes ~74s compile + startup vals). Wall-per-step including everything
  is 228.834/500=0.458 and 235.158/500=0.470 (+2.8%). Quote as "within
  1-3%"; never mix the two denominators.
  Equivalence: FAIL beyond step ~125. Gaps (8-GPU vs 1-GPU): -0.2% @125,
  -1.0% @250, -2.0% @375, -2.3% @500, same sign both optimizers. Mechanism
  (hypothesized, supported): NCCL/reduction-order nondeterminism amplified
  by training chaos — NOT a sharding bug. Evidence: (1) step-0 vals
  identical (same init, broadcast works); (2) attempt-1 vs attempt-2 @125
  differ 0.08-0.28% run-to-run at fixed seed0, so 8-GPU noise alone
  accounts for the divergence scale; (3) grad/val math verified identical
  in exact arithmetic (local sums + all_reduce SUM = global sum).
  CONSEQUENCE: 8-GPU absolute curves are NOT portable to 1-GPU numbers —
  do not compare 8-GPU crossings to 1-GPU crossings. But 8-GPU PAIRED
  comparisons (both opts at 8 GPUs) remain FAIR; they need >=2 seeds each
  since 8-GPU run noise (0.1-0.3% @125) exceeds 1-GPU seed spread.
  FOLLOW-UP (zero-GPU checks, 2026-10-26): (a) attempt-1v2 @250 infeasible
  — attempt 1 died in the step-250 ckpt before logging val@250 (0 such
  lines in both attempt-1 logs); @125 spread stands (0.08%/0.28%).
  (b) Data order VERIFIED IDENTICAL across world sizes from code
  (distributed_data_generator: global pos += batch_size on every rank;
  rank windows tile [pos, pos+batch+1] exactly; file reload keyed on same
  pos). Microbatch count differs (8x64 sequential at 1 GPU vs 1x64 per
  rank at 8) but that is summation order only (fp rounding). Val sharding
  exact (shards tile, SUM / val_tokens). Refined mechanism: DETERMINISTIC
  fp offset (different kernel tilings at local-batch 64 vs 512 + different
  reduction trees), identical every run, amplified by chaotic dynamics —
  plus nondeterministic NCCL jitter on top (proven by attempt spread).
  Supporting: all four 8-GPU @125 readings sit below their 1-GPU
  counterparts (p~0.06 under symmetric noise), i.e. even the earliest
  measurable point carries a systematic offset. Neither component is a
  correctness bug; paired-8-GPU fairness unaffected.
  STATUS 2026-10-27: mechanism RESOLVED (was UNRESOLVED; root cause = LR-schedule
  confound, see runs 22-26 outcome; chaos-amplification critique below retained
  for the record: chaos predicts divergence,
  chaos predicts divergence, not same-sign same-size gaps in two different
  optimizers; p~0.06 is weak). FREE TEST when runs 22/24 land: table val
  @125/250/375/500 for Muon on both setups (1-GPU seeds 0/1/2 = runs
  5/16/17; 8-GPU seeds 0/1 = runs 22/24). Verdict rule: if both 8-GPU vals
  sit OUTSIDE the 1-GPU seed range @500 AND agree with each other ->
  systematic gap, not noise. Paired directional hypothesis stands as
  written; a miss (Muon first at either seed) gets reported plainly.
  SCOPE CAVEAT (cite with the parity number): single node, data-parallel
  param-gather sharding — NOT FSDP or TP. Interconnect unprofiled; RTX Pro
  class carries no datacenter NVLink fabric (verify via nvidia-smi topo,
  piggybacked into next 8-GPU sbatch prolog), so comm here likely runs
  over PCIe and overstates gather cost vs a DGX-class node.

## 8-GPU fulls (runs 22-25: Muon/C-r32 x seed0/seed1, 3250 steps) + SinkSOAP probe (run26, 500 steps) — SUBMITTED (jobs 14467048-52)

Design: paired 8-GPU crossing comparison (both opts at 8 GPUs each seed).
NOT comparable to 1-GPU absolute crossings (gate stands). ckpt OFF all five
(world_size assert; ~25-min inferno runs, preemption risk accepted).
nvidia-smi topo/nvlink prolog in every sbatch (interconnect profiling).

- Primary hypothesis (directional, paired): C-r32 crosses at an earlier
  observed step than Muon at BOTH seeds. Absolute steps UNPREDICTABLE
  (label untested): 8-GPU noise + deterministic fp offset mean 1-GPU
  crossings (3225 vs 3175) do not port; any 3100-3250 outcome consistent.
- Wall-clock: ~0.40 s/step steady-state both (from probes); to-cross wall
  = steps x 0.40. C-r32 within 5% per-step of Muon (probe: +1.2%).
- run26 SinkSOAP 500-step: predict 0.5-1.0 s/step (WIDE band, hypothesized:
  eigh work shards over params across ranks — each rank handles ~9 params —
  but 3072x3072 grams stay heavy per rank; compile unknown). Forms the
  missing contrast half: COSMOS-cheap-at-scale vs SinkSOAP-cost-at-scale.
  Expect val@125 within ~5% of a 1-GPU SinkSOAP early curve (no seeded 1-GPU
  SinkSOAP full exists — compare shape only, label loose).
- Topo expectation (hypothesized): no NVLink fabric reported (RTX Pro
  class); comm over PCIe. If NVLink IS present, the parity number is
  stronger than claimed (less comm handicap).
  OUTCOME (jobs 14467048-52, all COMPLETED): directional hypothesis HIT at
  BOTH seeds — seed0: C-r32 3200 < Muon 3225; seed1: C-r32 3200 < Muon
  3250. Finals: Muon 3.27790/3.27978, C-r32 3.27635/3.27690. Per-step:
  steady 0.389-0.395 (+1.4%), wall 0.402-0.416 (+2.5-3.5%) — within-5%
  HIT. To-cross wall: Muon 1294/1305s, C-r32 1330/1318s (fewer steps but
  slower steps net ~flat — the honest 8-GPU wall-clock read).
  run26 SinkSOAP: 0.639 steady / 0.646 wall — inside 0.5-1.0 band;
  ~1.6x Muon-8GPU (vs 2.0x at 1 GPU: eigh shards partially over ranks).
  Contrast half complete: at 8 GPUs COSMOS-r32 costs ~Muon while SinkSOAP
  costs ~1.6x Muon. Topo CONFIRMED: "Device does not have or support
  Nvlink", GPU pairs PIX / cross-pair NODE — PCIe-only, so parity was
  measured with a comm handicap; NVLink hardware can only help the
  relative case.
  FREE-TEST VERDICT (mechanism): NOT systematic — both 8-GPU Muon @500
  vals (3.82961/3.82909) sit INSIDE the 1-GPU seed range [3.82585,
  3.83128]; @125-375 gaps <=0.15%. Root cause of the probe -2.3% FOUND:
  LR-SCHEDULE CONFOUND, not NCCL/chaos — the 500-step probes cool down
  from step 150 (cooldown_frac=0.7 x 500) while 3250-step runs stay hot
  to 975, so run20/21 were never on run5/18's schedule. Same-schedule
  1v8 gaps are <=0.15% (noise). Chaos-amplification story RETRACTED;
  1v8-crossing gate LIFTED for matched schedules (Muon crossings match
  exactly both seeds: 3225/3250; C-r32 3200/3200 vs 3175 = 1 interval).

## Newton-Muon v2 (run 7: custom-op capture; hooks deleted) — SUBMITTED (smoke 14508442, full 14508443 afterok (attempt-1 smoke failed in 11s on unannotated custom_op signature; fixed))

V1 failed twice (OOM 70-83GB, first forward): eager hooks graph-broke every
Linear under model.compile. V2 ports the ref's compiled nm::accum_xtx op +
bool-flag gating; optimizer math unchanged (CPU-validated); baseline compile
settings restored. Paper (short track, H100): 6% fewer steps, +1.8%
per-step, -4% wall vs Muon. Different track/harness here, so bands wide.

- Smoke (12 steps): completes; step-12 val in [6.2, 6.6] (first 9 steps are
  pure Muon — identity until refresh at step 10 — so init should match
  Muon-family ~6.38); ckpt written; no OOM (the v1 failure mode).
  OUTCOME (job 14508442): SMOKE_OK, step-12 val 6.40805 (in band), ckpt
  written, ZERO recompile lines — the custom-op port traces cleanly, no
  graph breaks, v1 failure mode eliminated. Full auto-released (14508443).
- Full run7: crosses 3.28; crossing 3000-3200 (paper-analogous 6% under
  Muon's 3225 would be ~3030; band covers track differences); s/step
  1.9-2.1; final <=3.278 (at least Muon); max mem ~32GB (±1).
  OUTCOME (job 14508443): NO CROSSING, min 3.31761 (+0.038), 1.956s/step,
  33.394GB — crossing MISS, final MISS, s/step HIT, mem marginal MISS
  (+1.4GB over band; 2.3GB over Muon baseline — covariance/inverse/grad
  buffers + transients exceed the ~1GB static estimate). Early deficit vs
  Muon (+0.068 @500) never closed. Paper's 6%-fewer-steps gain does NOT
  transfer to this Track-3 harness (measured). Candidate transfer gaps
  (hypothesized, untested): NS12+rect-scale harness vs paper's NS5;
  refresh_k=10 vs 32; squared-ReLU MLP + RMSNorm layout vs paper's GPT-2;
  possibly their lr tuning. CALIBRATION (do not overcite): this was ONE run
  of a port with several deviations — it shows this port didn't cross in
  this harness, nothing more. Not evidence against the paper; keep as a
  footnote. Lane closed: no further NM runs planned.
- A miss on crossing still leaves a useful datum (NM cost at Track 3);
  an OOM again kills the lane permanently (same-reason x3).

## ckpt job 13945668 — pre-loadout, record only
- Tests the RNG-device fix (265f37f). Whatever it says, Run 8 is NOT
  gated on it; resume gets fixed only if a run is actually cut off.
