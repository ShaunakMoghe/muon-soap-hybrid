# Optimizer state accounting (from code, Track 3 124M config)

Status: analytic (computed from the variant sources, not measured per-tensor).
Cross-checked against run max-memory readings where noted (measured).

## Model geometry (variants/train_gpt_baseline_seed0.py)

12 blocks, dim 768, 6 heads x 128, vocab 50304. `Linear` weight is (out, in).

| Group | Shapes | Count | Params |
|---|---|---|---|
| Block attn q/k/v/proj | (768,768) | 48 | 28.3M |
| Block MLP fc / proj | (3072,768) / (768,3072) | 12 + 12 | 56.6M |
| Embedding (bf16) | (50304,768) | 1 | 38.6M |
| Output proj (fp32) | (50304,768) | 1 | 38.6M |
| Biases + RMSNorm gains (fp32) | 1D | — | 0.1M |
| Total | | | 162.3M |

"124M" = non-embedding params (123.7M). Block params are fp32 (only the
embedding is cast to bf16); only fp32/bf16 state dtypes appear below.

Optimizer split (all variants): AdamW (optimizer1) owns embed + output proj +
all 1D params; the 2D optimizer (optimizer2) owns the 72 block matrices.

## Per-state inventory (from code)

- Muon (`muon_update`): momentum, `zeros_like(p)` -> 1x fp32 per param.
- SOAP (`train_gpt_soap_seed0.py`, incl. QR branch — same shapes): exp_avg +
  exp_avg_sq (2x fp32 mn), GG grams (m^2+n^2 fp32), Q eigenbases (m^2+n^2 fp32).
- SinkSOAP (`train_gpt_sinksoap_seed0.py`): momentum (mn fp32), left/right grams
  (m^2+n^2 fp32), second_momentum (max(m,n) fp32). U/V are recomputed per step,
  not stored.
- COSMOS (`train_gpt_cosmos_seed0.py`): momentum (mn fp32), second_momentum
  (max(m,n) fp32), O (m x r_o fp32), U (n x r_u fp32), S (r_u^2 fp32),
  R (r_o^2 fp32). Ranks capped by layer dims; no cap binds below r=768.
- AdamW common (fused): 2 states per param; embed states bf16 (`zeros_like(p)`
  per torch AdamW source — assumed, not profiled), rest fp32.

## Per-layer state bytes (MiB)

| Layer (768x768) | Muon | SOAP | SinkSOAP | C32 | C64 | C128 | C256 | C512 |
|---|---|---|---|---|---|---|---|---|
| MiB | 2.25 | 13.50 | 6.75 | 2.45 | 2.66 | 3.13 | 4.25 | 7.25 |

| Layer (3072x768) | Muon | SOAP | SinkSOAP | C32 | C64 | C128 | C256 | C512 |
|---|---|---|---|---|---|---|---|---|
| MiB | 9.00 | 94.50 | 47.26 | 9.49 | 9.98 | 11.01 | 13.26 | 18.51 |

(The (768x3072) rows are identical to (3072x768).)

## Total optimizer-state bytes (GiB, incl. 0.433 GiB AdamW common)

| Muon | SOAP | SinkSOAP | C32 | C64 | C128 | C256 | C512 |
|---|---|---|---|---|---|---|---|
| 0.749 | 3.280 | 1.857 | 0.770 | 0.791 | 0.837 | 0.943 | 1.206 |

COSMOS-32 is within 3% of Muon's state; COSMOS-256 carries ~26% more than Muon
and ~49% less than SinkSOAP. SOAP's full Q + GG pairs dominate (3.28 GiB).

## Cross-check vs measured max memory (measured)

Same harness/activations, so run-to-run deltas isolate optimizer state +
transients. Measured `max_memory_allocated`:

| Run | Opt | Measured |
|---|---|---|
| run5 | Muon | 31.075 GB |
| run6/run9/run10 | COSMOS-256 | 31.280 GB (all three) |
| run8/run12/run13 | SinkSOAP-family | 33.458 GB (all three) |

- COSMOS-256 minus Muon: predicted 0.194 GB, measured 0.205 GB. MATCH.
- SinkSOAP minus COSMOS-256: predicted 0.914 GB, measured 2.178 GB. MISMATCH:
  residual ~1.26 GB, hypothesized (untested) to be transient eigh/cuSolver
  workspace on the 3072x3072 grams plus compiled autograd buffers, which
  `max_memory_allocated` includes but this table does not.

## Current-scheme comms (from code, 1-GPU config)

`Muon.step` all-gathers the 2D params once per step after the local update:
340 MB/step (fp32). (At 1 GPU this is a no-op copy; it prices the 8-GPU
dense-sync cost of the current scheme.)

## FSDP 8-GPU all-gather model (ANALYTIC, not measured)

Full-shard FSDP, 8x GPUs: params + grads + states sharded 8 ways. Per step each
rank all-gathers full params twice (forward + backward):

- Param bytes: embed 77.3 MB (bf16) + rest 494.7 MB (fp32) = 572 MB.
- All-gather receive per rank per step: ~2 x 572 MB = ~1.14 GB, IDENTICAL for
  all four optimizers (states stay sharded; only params move).
- Per-rank memory (P+G+S)/8: Muon 243 MB, COSMOS-256 267 MB, SinkSOAP 375 MB,
  SOAP 556 MB (params+grads 1144 MB + state row above, /8).

Implication: under FSDP the optimizer choice is a per-rank memory question, not
a per-step comm question — the rotation matrices price into storage (SOAP 3.28
GiB, SinkSOAP 1.86 GiB vs COSMOS-256 0.94 GiB), exactly the concern raised
about full-rank curvature. Under tensor-parallel sharding, by contrast, the
m x m / n x n factors would enter the per-step comm volume; that model is
untested here.

## Width scaling (ANALYTIC MODEL, not measured)

Same architecture scaled to width d: 12 layers, MLP 4d, vocab 50304 fixed,
seq/batch fixed, fp32 states (embed bf16). Formulas above; d=768 row
reproduces the measured-anchored totals. GiB total optimizer state (incl.
shared AdamW) and FSDP 8-GPU all-gather bytes/step (optimizer-independent).

| d | Params | Muon | SOAP | SinkSOAP | C32 | C256 | all-gather/step |
|---|---|---|---|---|---|---|---|
| 768 | 0.16B | 0.75 | 3.28 | 1.86 | 0.77 | 0.94 | 1.06 |
| 2048 | 0.81B | 3.40 | 21.40 | 11.28 | 3.45 | 3.86 | 5.65 |
| 4096 | 2.83B | 11.30 | 83.30 | 42.80 | 11.41 | 12.18 | 20.30 |
| 8192 | 10.49B | 40.61 | 328.61 | 166.61 | 40.82 | 42.33 | 76.61 |

At d=8192: SOAP state is 8.1x Muon's (329 GiB — exceeds single-device memory
on its own), SinkSOAP 4.1x (167 GiB), while COSMOS-32 is within 0.5% of Muon
and COSMOS-256 within 5%. The rotation matrices (m^2+n^2 per layer) grow with
the square of width; rank-r state grows linearly in r and is width-dominant
only through the shared momentum term. This is the quantitative form of the
scale argument: full-rank curvature becomes unshardable while low-rank
curvature rides at ~Muon cost. Model only — no multi-hundred-GB run exists
to confirm it.
