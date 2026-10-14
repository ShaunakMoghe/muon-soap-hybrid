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
- soap11 (ref recipe): completes + ckpt; val band WIDE [6.0, 8.5]
  (warmup shifts early curve unpredictably; seed0 unamortized ref unknown).
- t10 (freq=10/T=10): completes + ckpt; val in [6.2, 8.0] (between
  unamortized 6.33 and run8-T1 7.83 if T drives the shift).
- ws (warm-start): completes + ckpt; val in [6.2, 6.6] (eigh-every-step
  early should track unamortized 6.33).

## COSMOS replicates seed1/seed2 (runs 9, 10) — PENDING
- Predict: both cross; crossing steps within ±100 of 3150 (noise estimate
  for the 50-75-step edge over Muon). If spread >> 75 steps, the edge is
  inconclusive per the noise rule. (run7 slot reserved for NM if revived.)

## Plain SOAP reference recipe (run 11: lr=0.0018 + 250 warmup, seed0) — PENDING
- Predict: crosses OR lands materially closer than run4A's +0.066 if the
  hot-lr/no-warmup hypothesis is right; no-cross again weakens it.
  Pairs with run4A (same seed, recipe change only).

## Amortized T=10 (run 12: freq=10, T=10) — PENDING
- Predict: separates the confound — if T=1 caused run8's failure, this
  recovers toward 3100-cross; if staleness, still no-cross. Cost ~2.0s/step.

## Amortized warm-start (run 13: eigh every step to 200, then every 10, T=1) — PENDING
- Predict: if cold-start staleness caused run8's failure, this recovers
  most of the gap (cross 3100-3200); cost ~2.1s/step (200 full steps).

## ckpt job 13945668 — pre-loadout, record only
- Tests the RNG-device fix (265f37f). Whatever it says, Run 8 is NOT
  gated on it; resume gets fixed only if a run is actually cut off.
