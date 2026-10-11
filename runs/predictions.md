# Run predictions ledger (predict before, record after)

## Run 8 smoke: amortized SinkSOAP 12-step — QUEUED
- Predict: completes; step-12 val in [6.2, 6.5] (unamortized seed0 was
  6.33337@12; amortization shifts the early curve slightly);
  checkpoints/ckpt_00010.pt written (save path exercised, CKPT_EVERY=10).

## Run 8 full: amortized SinkSOAP 3250 steps (precond_freq=10, T=1) — PENDING smoke
- Predict: 1.85–1.95 s/step (eigh cost ÷10: 1.83 + ~0.1 overhead);
  crossing in [3100, 3200] (staleness may cost 0–100 steps vs unamortized
  3100); final ~3.27.
- Decides: does the ~3% step saving survive at ~Muon per-step cost?

## COSMOS smoke: 12-step — QUEUED
- Predict: completes; step-12 val in [6.0, 6.6]; ckpt written.

## COSMOS full (after smoke) — config LOGGED here per loadout
- Variant: variants/train_gpt_cosmos_seed0.py = Sep harness, seed0,
  3250 steps, Sep optimizer1 (embed 0.7 / proj 0.004 / 1D 0.015);
  optimizer2 = COSMOS lr=0.04, wd=0.025, rank_o=rank_u=256
  (Liming's track-3 wiring values; ranks are class defaults).
- Predict: crosses 3.28 (method is tuned for this track); s/step
  unknown — per-step QR on rank-256 subspaces, guess 2.0–2.5s.
  Wide uncertainty, stated plainly.

## Newton-Muon smoke (test_nm_smoke.sh: zero-recompile warm-cache rerun) — QUEUED
- Predict: passes (script asserts zero recompile lines + stable step times).
- Full benchmark ONLY if smoke passes. Paper claims (incl. any novelty
  framing) treated as UNVERIFIED until Liming weighs in.

## ckpt job 13945668 — pre-loadout, record only
- Tests the RNG-device fix (265f37f). Whatever it says, Run 8 is NOT
  gated on it; resume gets fixed only if a run is actually cut off.
