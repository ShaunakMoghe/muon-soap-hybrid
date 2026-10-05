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
