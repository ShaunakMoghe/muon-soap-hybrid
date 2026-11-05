# Run ledger — Muon vs SOAP optimizer tradeoffs

One row per training run. Log file from the script is `logs/<uuid>.txt`
inside the repo checkout the run was launched from (it logs the full
script + `step:... val_loss:... train_time:... step_avg:...` lines).

Columns: run_id,date,variant,script,git_hash,steps,val@328_step,val_final,wallclock_s,log_uuid,notes

- `variant`: baseline-muon | sinksoap | soap | sinkhorn-embed | hybrid-...
- `script`: path to the exact script used (copy per-variant scripts to `variants/` so the
  logged code stays reproducible even if the checkout moves on)
- `steps`: train_steps used; `val@328_step`: first step where val_loss <= 3.28 (empty if never)
- `val_final`: val loss at final step; `wallclock_s`: train_time at final step

## Experiments

| experiment | runs | question |
|---|---|---|
| pilots | 1, 2 | unseeded Muon vs SinkSOAP baselines |
| Muon seeds | 5, 16, 17 | seeded Muon 3-seed spread (controls) |
| COSMOS seeds | 6, 9, 10 | COSMOS-256 3-seed edge over Muon |
| rank sweep | 14, 15, 18, 19 | quality vs rank (64/128/32/512) |
| amortized SinkSOAP | 8, 12, 13 | cached eigenbases; cold-start staleness |
| SOAP recipes | 3, 4, 11 | QR recipe, killed draw, reference recipe |
| Newton-Muon port | 7 | ported v2 run (no crossing) |
| 8-GPU pairs | 20–26 | paired seeds; 1v8 check; SinkSOAP cost probe |
| warmup | 27, 28 | 150 Muon steps then COSMOS |
| NS sweep | 29–36 | NS iterations 12/8/6 × Muon/COSMOS |
| hyper-swap | 39–42 | attribution: hyperparameters vs method |
| square NorMuon | 37, 38 | queued |

## Incidents

Operational failures with no scientific content. Rows stay factual;
details live here.

- run4b (2026-09-28): accidental duplicate launch. The leg-1 shell
  wrapper executed a stale pre-edit image after run4A completed,
  relaunching the same seed0 SOAP script. Killed; partial curve
  discarded.
- run30 attempt 1 (2026-11-03, job 14820476): failed at 71s with a
  stale NFS file handle inside Triton compile — transient
  infrastructure (the variant differs from healthy run29 by seed
  only). Log preserved as `stdout_attempt1_14820476.log`; resubmitted
  as job 15003180.
