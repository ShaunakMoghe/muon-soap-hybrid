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
  write-up), `method_notes.md` (forensics + load-bearing lessons),
  `curves.png`, `sbatch/` (batch scripts + job outputs), per-run dirs.
- `profile/` — optimizer microbenchmarks (time / memory / noise floor).

## Upstreams (all public)

- `KellerJordan/modded-nanogpt` (Track 3 baseline harness)
- `lliu606/Modded-NanoGPT-Private` (SinkSOAP / COSMOS sources — public
  despite the name; "Private" only distinguishes it from upstream)
- `zhehangdu/Newton-Muon` + Du & Su, arXiv:2604.01472

## Compute

Georgia Tech PACE Phoenix, `gpu-rtxpro-blackwell`, standard shape 1 GPU +
8 CPUs + 96GB via `sbatch` (see `runs/sbatch/`). Work from the login node;
GPU work only through queued jobs.
