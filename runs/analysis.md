# Muon vs SOAP tradeoff analysis

Single-GPU Track-3 optimizer comparisons plus an analytical
distributed-cost model, toward hybrid updates matching SOAP quality at
Muon cost. Rig: 1x RTX PRO 6000 Blackwell 96GB. Run ledger: `runs.csv`.
Methods and provenance: `method_notes.md`. Last updated 2026-10-28.

Terminology (2026-10-28): "COSMOS" throughout means the **track-3 variant**
(4-block subspace Sinkhorn + Newton-Schulz on the residual + NorMuon
second-moment update) as adapted in the upstream fork — not COSMOS in
general. No "Pareto winner" language: 8-GPU wall-clock is flat and 1-GPU
is ~1.5% slower than Muon; the claim is a step-count edge at ~cost parity.
Newton-Muon lane closed (run7 footnote).

## 1. Setup

- Harness: modded-nanoGPT Track 3 (@bc3a0c2), 124M GPT, 3250 steps,
  global batch 524288 tokens (8 microbatches x 64 with grad accumulation;
  exact data-parallel math), stable then linear cooldown
  (cooldown_frac=0.7), validation every 125 steps (every 25 after 2875).
- Protocol: run1/run2 are unseeded pilots. All later runs are seed0-paired
  (`torch.manual_seed(0)` + `cuda.manual_seed_all(0)`) and swap ONLY the
  2D block-parameter optimizer against the seeded Muon control (run5).
- Per-step times are 1-GPU measurements (Sec. 3 states the bound this is).

## 2. Convergence (target val_loss 3.28)

| run | method | seed | first obs <=3.28 | window | final | train | time-to-cross |
|---|---|---|---|---|---|---|---|
| run1 | Muon | unseeded | 3200 | (3175,3200] | 3.27714 | 5952s | 5860s |
| run2 | SinkSOAP | unseeded | 3100 | (3075,3100] | 3.26769 | 12193s | 11628s |
| run5 | Muon | 0 | 3225 | (3200,3225] | 3.27844 | 6034s | 5987s |
| run6 | COSMOS | 0 | 3150 | (3125,3150] | 3.27399 | 6456s | 6258s |
| run8 | amort SinkSOAP (freq=10/T=1) | 0 | — | — | 3.29182 | 6473s | — |
| run4 | SOAP-QR (lr=0.002, no warmup) | 0 | — | — | 3.34559 | 6155s | — |
| run3 | SOAP-QR (killed@968) | 0 | — | — | 3.769@875 | — | — |

Final-loss order: SinkSOAP 3.26769 > COSMOS 3.27399 > Muon 3.27714 >
seeded Muon 3.27844 > amort-SinkSOAP 3.29182 > SOAP-QR 3.34559.
Crossing grid is ±25 steps late; run1/run2 carry unpaired init noise
(Sec. 5 for the noise floor this sits on). Run4 is one chaotic draw.

## 3. Per-step cost (measured, 1x Blackwell)

| method | s/step | note |
|---|---|---|
| Muon | 1.83–1.86 | run1/run5 |
| SinkSOAP (eigh every step) | 3.75 | 96–98% eigh; Sinkhorn ~0.5ms |
| amort SinkSOAP (freq=10/T=1) | 1.95 | ~1.8 normal, ~3.7 refresh steps |
| COSMOS (rank-256 QR) | 1.99 | per-step subspace QR |
| SOAP-QR (eigh once + track) | 1.89 | full eigh at init only |

Cost-model closure: microbenchmark-predicted SinkSOAP overhead (1.88s)
matches the measured run2−run1 delta (1.92s). The 2x SinkSOAP slowdown
is an implementation choice (unamortized eigh), not an inherent cost —
but run8 shows aggressive amortization can erase the quality edge
(Sec. 8). 1-GPU timing flatters QR-SOAP (hides distributed sync) and
punishes unamortized SinkSOAP (shards ÷W on multi-GPU); treat these as
1-GPU bounds.

## 4. Memory

Optimizer state, 2D block params, fp32 exact: Muon 340MB <
SinkSOAP 1.53GB (grams) < SOAP 3.06GB (m/v + grams + Qs).
Observed peak (`max_memory_allocated`): run5 31.1, run6 31.3,
run8 33.5, run4A 33.8 GB.

## 5. Noise and reproducibility

Same-seed, same-GPU, same-code val spread:

| steps | Muon | SinkSOAP | SOAP-QR |
|---|---|---|---|
| 12 | 0 (5-dec) | 7e-5 | ~0.15 (4 runs, chaotic) |
| 125–130 | 1.9e-4 | 4.8e-3 | ~0.16 (persists) |

Measured: `torch.linalg.eigh` is bit-identical 20x in-process + 3x
cross-process on CUDA, including rank-deficient 3072 grams
(`research_stack/test_eigh_determinism.py`, job 13848105). fp64
per-step traces show both methods receive last-bit harness flips at
step 1 (consistent with CUDA atomic races in compiled backward);
Muon contracts them below print precision in one step while SinkSOAP
amplifies ~360x (small Gram eigengaps, Davis-Kahan).
Hypothesized (unproven): SOAP-QR chaos via low-rank Grams, near-tied
eigenvalues, and argsort/QR divergence; decisive tests pending
(tolerance-quantized sort, deterministic rerun, autotune check).
Practical consequence: ±5e-3 loss jitter maps to ~±70 crossing steps
near target, so single-seed gaps near that floor are inconclusive,
and SOAP-QR needs multi-seed spread before any crossing claim.

## 6. Distributed cost (analytical; no multi-GPU runs yet)

Two regimes, with different bottlenecks:
- FSDP/ZeRO-style (this harness: data-parallel + optimizer param
  sharding). Each rank owns whole 2D matrices, so eigh/QR is purely
  rank-local compute; SOAP-family methods pay local compute + memory
  only, and per-GPU communication matches Muon's (gradient allreduce +
  parameter all_gather). The 1-GPU compute/memory numbers characterize
  exactly this regime.
- Tensor parallelism (single 2D weights sharded across GPUs).
  Assembling each mode's Gram matrix needs a cross-rank AllReduce over
  the sharded dimension, and the eigendecomposition becomes a
  distributed eigensolve — cubic work that parallelizes poorly plus
  fine-grained synchronization. Muon's Newton-Schulz iteration is
  structurally local here (no Gram eigenbasis needed).
Implementation notes: the transplanted SOAP.step does not shard
(redundant optimizer compute + full states per rank; correct but
wasteful, fixable). No multi-GPU measurements exist yet; 2-GPU
strong-scaling probes would convert regime-1 scaling into numbers.

## 7. Method landscape

- Muon: no curvature anywhere; orthogonalization by Newton-Schulz GEMMs.
- Newton-Muon (Du & Su, arXiv:2604.01472): unilateral right
  preconditioning by input activation covariance (ZZᵀ, Cholesky-tracked)
  under an isotropic-output assumption. The paper's §6 flags single-GPU
  evaluation as future work; the observation that ZZᵀ accumulates locally
  under column-parallel TP is ours (untested). Status: module implemented
  and CPU-validated; OOMs at 83/95GB under hooks+compile on this rig
  (two attempts, cudagraphs on and off); lane paused pending decision.
- Plain SOAP: bilateral gradient covariance (dual eigh/QR). Status:
  unresolved — one no-cross draw (+0.066, lr=0.002 no warmup), one
  draw killed early, 0.037 same-seed spread. No claim either way.
- SinkSOAP: bilateral dual eigh every step + Sinkhorn coordinate
  balance. Same rotation cost as SOAP for ~0.5ms of post-processing.
- COSMOS: rank-256 subspace QR tracking + compressed Sinkhorn blocks;
  crossed 3150 at 1.99s/step. Best seed0 final (3.27399) and best
  SOAP-family wall-clock to target (6258s, ~4–7% slower than Muon).

## 8. Claim register

- [measured] Profiler closure: 1.88s predicted vs 1.92s observed;
  SinkSOAP overhead ~97% eigh.
- [measured, low-confidence] SinkSOAP crossed ~100 steps before Muon
  (3100 vs 3200); gap is near the ±70-step noise floor, single
  unseeded draws — suggestive, not conclusive.
- [measured] COSMOS (seed0): cross 3150, final 3.27399, 1.99s/step.
- [measured] Amortized SinkSOAP (freq=10/T=1): 1.95s/step, NO crossing,
  final 3.29182 — the step savings do not survive at these settings.
  T-vs-frequency unattributed (both changed in run8).
- [measured] SOAP-QR (seed0, lr=0.002, no warmup): no crossing, 3.34559.
- [measured] Same-seed SOAP-QR @875 differs by 0.037 (run3 vs run4A).
- [measured] eigh bit-identical 20x + 3x on CUDA (job 13848105).
- [hypothesized] SOAP-QR chaos mechanism (low-rank Grams, near-tied
  eigenvalues, argsort/QR divergence). Decisive battery pending.
- [hypothesized] Noise-robustness axis (NS contraction vs eigengap
  amplification); supported by fp64 traces, needs tensor-level proof.
- [hypothesized] run4A shortfall partly from no-warmup + 3250-step
  horizon (reference recipe uses warmup 250 + 7000 steps at lr 0.0018).
- [untested] Gentler amortization (freq=3/5, T=10); COSMOS replicates;
  reference-recipe SOAP; Newton-Muon full benchmark; anything multi-GPU.
- [analytical] FSDP-vs-TP distributed-cost split.

## 9. Collaborator questions (two answered 2026-10-16)

1. [answered] Distributed regime: both FSDP and TP matter, framed as
   step-count reduction under limited overhead — at scale, gradient
   compute+comm dominate, so fewer steps to target wins if the
   optimizer adds little overhead and its state shards (or is cheap to
   replicate). Full rotation matrices (SOAP/SinkSOAP) are the expensive
   case; low-rank (COSMOS) the promising one.
2. [open] SOAP recipe: is lr=0.002 with no warmup a fair test, or
   should the reference recipe (lr=0.0018 + 250 warmup, 7000 steps)
   be matched?
3. [open] Newton-Muon priority: worth deeper surgery for the 96GB OOM?
4. [answered] COSMOS config: lr=0.04/wd=0.025/ranks=256 confirmed as
   his initial speedrun config, but untuned — room for improvement.
   Directed next step: rank–convergence–runtime tradeoff at lower
   ranks, then larger scale.
