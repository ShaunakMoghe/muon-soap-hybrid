# muon-soap-hybrid

Can curvature information buy step-efficiency for LLM pre-training at
Muon-like compute and communication cost? This repo characterizes
Muon-family vs SOAP-family optimizers on modded-NanoGPT Track 3 (124M
parameters, 3250 steps) to find out, and to identify concrete hybrid
ingredients.

Headline findings (runs 1–96; caveats in `runs/analysis.md`):

- The COSMOS track-3 variant beats Muon by a small, well-resolved
  margin: 0.0026 lower final loss and 25–50 fewer steps to 3.28 on all
  five paired seeds at 8 GPUs (paired t, one-sided p = 0.0006), at
  +1.4% step time for rank 32.
- That edge comes from the variant's NorMuon update, not from the
  tracked subspace. With the subspace removed (rank 0: no QR, no
  Sinkhorn) the final loss is the same within noise (+0.0003, 95% CI
  −0.0006 to +0.0012, 5 seeds) at Muon's step time. The subspace does
  give an early lead (0.10 in loss at step 125) that is gone by step
  2500. This holds at rank 32 only (next point).
- Higher ranks do add to NorMuon. Against the rank-0 control (3 paired
  seeds, 8 GPUs), final loss is lower by 0.0021 at rank 256, 0.0052 at
  rank 512 and 0.0078 at full rank (SinkSOAP), all p < 0.01. At this
  scale each costs more step time than it saves: +5.5%, +10.9% and
  +64% per step, and none reaches 3.28 sooner in wall-clock than Muon
  or the control (same node). The subspace and NorMuon substitute:
  NorMuon's gain falls from 0.0051 at rank 32 to 0.0011 at full rank.
  Raising only the 768-wide side of the MLP matrices to rank 512
  recovers 37% of the rank-512 gain at +3.6% step time (2 seeds).
- Subspace upkeep can be made cheaper with little or no visible harm
  (QR refresh every 2/4/8 steps: 0 to +0.0023 final loss; Sinkhorn 5
  or 3 iterations: within noise; 1 iteration: +0.0024). The saving is
  under 1% of step time, and the rank-0 result makes it moot at this
  rank.
- COSMOS tolerates 6 Newton-Schulz iterations where Muon collapses (no
  crossing, +0.010 loss). Offline spectra support the premise of the
  deflation explanation, but a spectral-norm start recovers only
  13–15% of Muon's early gap, so the cause is still open.
- SinkSOAP reaches 3.28 100–150 steps before Muon (3100–3125 vs
  3225–3250, 3 paired seeds at 8 GPUs) at 1.6x (8 GPUs) to 2x (1 GPU)
  the step time; a warm-started
  amortization recovers the crossing at ~56% of that cost. Plain SOAP
  and a Newton-Muon port did not cross under the tested recipes.

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
  `parse_results.py` reads the 8-GPU logs into one table;
  `attribution_stats.py` and `rank_normuon_stats.py` print the
  seed-paired statistics.
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

Measured (runs 1–96, 2026-11-23): the findings above. Muon, the rank-0
control and COSMOS-r32 at five paired seeds on 8 GPUs; COSMOS-r256,
COSMOS-r512 and SinkSOAP at three; rank 0, 32, 512 and SinkSOAP with
the NorMuon update switched off at three; rank 512 on one group of
matrix sides at two. Earlier:
the 1-GPU rank sweep, amortized SinkSOAP (cold start fails, warm start
recovers), SOAP recipes, Newton-Muon port (no-cross footnote), 150-step
Muon warmup and square-matrix NorMuon (no effect), hyperparameter swap,
QR-cadence and Sinkhorn-iteration sweeps.
Queued: nothing.
Open: why rank 0 with NorMuon off trails Muon at the same lr/wd (the
fixed-norm rescale is the suspect); per-cell lr tuning for the
NorMuon-off cells; unequal ranks on the two sides of the MLP matrices;
the rank-0 control at 6 Newton-Schulz iterations; a larger model (on
hold until the 124M picture is settled).
