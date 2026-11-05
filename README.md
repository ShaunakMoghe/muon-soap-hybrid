# muon-soap-hybrid

Can curvature information buy step-efficiency for LLM pre-training at
Muon-like compute and communication cost? This repo characterizes
Muon-family vs SOAP-family optimizers on modded-NanoGPT Track 3 (124M
parameters, 3250 steps) to find out, and to identify concrete hybrid
ingredients.

Headline findings (runs 1–42; caveats in `runs/analysis.md`):

- The COSMOS track-3 variant crosses 25–50 steps before Muon at roughly
  Muon-like per-step cost: 3 seeds on 1 GPU, 2 paired seeds on 8 GPUs.
  Quality is nearly flat across ranks 32–256.
- That edge is not just hyperparameter tuning. With hyperparameters
  swapped, Muon never matches COSMOS, and COSMOS stays ahead or tied in
  all four matched comparisons — but margins are small and paired stats
  are suggestive, not significant (p~0.07).
- COSMOS tolerates 6 Newton-Schulz iterations where Muon collapses (no
  crossing, +0.010 loss), pointing at fewer residual iterations as a
  possible hybrid ingredient.
- Plain SOAP and a Newton-Muon port did not cross under the tested
  recipes. SOAP-family runs show large same-seed spread, so single-seed
  gaps near the noise floor are inconclusive.

## Layout

- `variants/` — single-file training scripts (one optimizer variant each;
  provenance + deviations in file headers). This is the unit of a run.
- `research_stack/` — shared infra: checkpoint/resume (`ckpt.py`),
  determinism tests, optimizer modules (Newton-Muon, Sinkhorn).
- `runs/` — the ledger: `runs.csv` (every run), `RUNS.md` (experiment
  index + incident notes), `analysis.md` (tradeoff write-up),
  `method_notes.md` (protocols), `state_accounting.md`
  (optimizer-state model), `curves.png`, `sbatch/` (batch scripts +
  job outputs; smoke checks under `sbatch/smoke/`), per-run dirs.
- `profile/` — optimizer microbenchmarks (time / memory / noise floor).

## Upstreams (all public)

- `KellerJordan/modded-nanogpt` (Track 3 baseline harness)
- `lliu606/Modded-NanoGPT-Private` (SinkSOAP / COSMOS sources — public
  despite the name; "Private" only distinguishes it from upstream)
- `zhehangdu/Newton-Muon` + Du & Su, arXiv:2604.01472

## Compute

Georgia Tech PACE Phoenix, `gpu-rtxpro-blackwell`, via `sbatch`
(see `runs/sbatch/`): 1-GPU shape is 1 GPU + 8 CPUs + 96GB, 8-GPU shape
is 8 GPUs + 16 CPUs + 400GB. Work from the login node; GPU work only
through queued jobs.

## Status

Measured (runs 1–42, 2026-11-04): Muon 3-seed spread; COSMOS (track-3
variant) edge over Muon at r256 (3 seeds, 1-GPU) and r32 (2 paired
seeds, 8-GPU); rank sweep r32–r512 (quality nearly flat); warm-start
amortization finding; SOAP recipes; Newton-Muon port (no-cross
footnote); 150-step Muon warmup (no effect); NS-count sweep
(COSMOS holds at 6, Muon collapses); hyperparameter swap
(tuning ruled out as the full explanation).
Queued: square-matrix NorMuon (runs 37/38) + Muon-NS8 seed-1 retry
(run30).
Open: COSMOS lr/wd tuning, rank 8/16 knee, asymmetric ranks, larger
scale; SOAP-QR spread mechanism (hypothesis only).
