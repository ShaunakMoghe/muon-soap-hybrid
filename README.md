# muon-soap-hybrid

Active research: characterizing Muon-family vs SOAP-family optimizer
tradeoffs for LLM pre-training, toward hybrid updates that match SOAP's
step efficiency at Muon's compute and communication cost.

## Layout

- `variants/` — single-file training scripts (one optimizer variant each;
  provenance + deviations in file headers). This is the unit of a run.
- `research_stack/` — shared infra: checkpoint/resume (`ckpt.py`),
  determinism tests, optimizer modules (Newton-Muon, Sinkhorn).
- `runs/` — the ledger: `runs.csv` (every run), `analysis.md` (tradeoff
  write-up), `method_notes.md` (protocols), `state_accounting.md`
  (optimizer-state model), `curves.png`, `sbatch/` (batch scripts +
  job outputs), per-run dirs.
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

Measured (runs 1–28): Muon 3-seed spread; COSMOS (track-3 variant)
3-seed edge over Muon at r256, rank sweep r32–r512 (quality nearly
flat), warm-start amortization finding, SOAP recipes, Newton-Muon
port (no-cross footnote); 8-GPU paired Muon/COSMOS-r32 crossings +
SinkSOAP cost probe (PCIe-only node).
In progress: audit batch Exps 1–3 (warmup handoff, NS sweep,
square-matrix NorMuon).
Open: COSMOS lr/wd tuning, rank 8/16 knee, asymmetric ranks; SOAP-QR
chaos mechanism (hypothesis only).
