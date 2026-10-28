# Muon vs SOAP tradeoff analysis

Single-GPU Track-3 optimizer comparisons plus an analytical
distributed-cost model, toward hybrid updates matching SOAP quality at
Muon cost. Rig: 1x RTX PRO 6000 Blackwell 96GB. Run ledger: `runs.csv`.
Methods and provenance: `method_notes.md`. Last updated 2026-10-28 (runs 1–28).

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
| run9/10 | COSMOS-256 | 1/2 | 3150/3175 | — | 3.27373/3.27430 | 6491/6616s | — |
| run12 | amort SinkSOAP (freq=10/T=10) | 0 | — | — | 3.28985 | 6556s | — |
| run13 | amort SinkSOAP (warm-start ≤200) | 0 | 3125 | (3100,3125] | 3.26992 | 6837s | — |
| run11 | SOAP (lr=0.0018 + warmup) | 0 | — | — | 3.30265 | 6106s | — |
| run14/15/18/19 | COSMOS r64/r128/r32/r512 | 0 | 3200/3200/3175/3125 | — | 3.27649/3.27640/3.27517/3.27021 | — | — |
| run16/17 | Muon | 1/2 | 3250/3225 | — | 3.27959/3.27922 | 5924/5904s | — |
| run7 | Newton-Muon port (v2) | 0 | — | — | 3.31761 | 6357s | — |

Best to worst final loss (1-GPU): SinkSOAP 3.26769, warm-start amort
3.26992, COSMOS-512 3.27021, COSMOS-256 ~3.274, COSMOS-32 3.27517,
Muon 3.277–3.280 (3 seeds), amort-SinkSOAP ~3.290, SOAP-ref 3.30265,
Newton-Muon port 3.31761, SOAP-QR 3.34559.
Crossing grid is ±25 steps late; run1/run2 carry unpaired init noise
(Sec. 5 for the noise floor this sits on). Run4 is one chaotic draw.

8-GPU (paired seeds, same 3250-step protocol): Muon s0/s1 cross
3225/3250 (finals 3.27790/3.27978); COSMOS-r32 s0/s1 cross 3200/3200
(3.27635/3.27690) — directional edge holds, ~0.40 s/step both, to-cross
wall roughly flat (1294/1305s vs 1330/1318s). SinkSOAP 500-step probe:
0.64 s/step (~1.6x Muon-8GPU).

## 3. Per-step cost (measured, 1x Blackwell)

| method | s/step | note |
|---|---|---|
| Muon | 1.82–1.86 | run1/run5/run16/run17 |
| SinkSOAP (eigh every step) | 3.75 | 96–98% eigh; Sinkhorn ~0.5ms |
| amort SinkSOAP (freq=10/T=1) | 1.95 | ~1.8 normal, ~3.7 refresh steps |
| COSMOS (rank-256 QR) | 1.99–2.04 | run6/run9/run10 |
| COSMOS r32/r64/r128/r512 | 1.89/1.93/1.90/2.15 | rank sweep, seed0 |
| SOAP-QR (eigh once + track) | 1.89 | full eigh at init only |
| Newton-Muon port | 1.96 | run7 |
| 8-GPU Muon / COSMOS-r32 | 0.39–0.42 | steady-state; SinkSOAP 0.64 |

Cost-model closure: microbenchmark-predicted SinkSOAP overhead (1.88s)
matches the measured run2−run1 delta (1.92s). The 2x SinkSOAP slowdown
is an implementation choice (unamortized eigh), not an inherent cost —
but run8 shows aggressive amortization can erase the quality edge
(Sec. 8). 1-GPU timing flatters QR-SOAP (hides distributed sync) and
punishes unamortized SinkSOAP (shards ÷W on multi-GPU); treat these as
1-GPU bounds.

## 4. Memory

Optimizer state, 2D block params, fp32 exact: Muon 340MB <
COSMOS-32 360MB < COSMOS-256 548MB < SinkSOAP 1.53GB (grams) <
SOAP 3.06GB (m/v + grams + Qs). Full table: `state_accounting.md`.
Observed peak (`max_memory_allocated`): Muon 31.1, COSMOS-256 31.3,
COSMOS-32 31.1, COSMOS-512 31.6, SinkSOAP-family 33.5, SOAP-ref 41.4
(large transient), Newton-Muon port 33.4 GB.

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

## 6. Distributed cost (8-GPU measured + analytical)

Two regimes, with different bottlenecks:
- Data-parallel + optimizer param sharding (this harness). Each rank
  owns whole 2D matrices, so eigh/QR is rank-local compute; per-step
  comm matches Muon's (gradient allreduce + parameter all_gather).
  Measured at 8 GPUs (single node, PCIe-only, no NVLink): Muon and
  COSMOS-r32 both ~0.40 s/step (within 1–3%); SinkSOAP 0.64 s/step.
- Tensor parallelism (single 2D weights sharded across GPUs).
  Assembling each mode's Gram matrix needs a cross-rank AllReduce over
  the sharded dimension, and the eigendecomposition becomes a
  distributed eigensolve — cubic work that parallelizes poorly plus
  fine-grained synchronization. Muon's Newton-Schulz iteration is
  structurally local here (no Gram eigenbasis needed). Untested.
Implementation notes: the transplanted SOAP.step does not shard
(redundant optimizer compute + full states per rank; correct but
wasteful, fixable). FSDP all-gather model + width scaling table:
`state_accounting.md` (analytic).

## 7. Method landscape

- Muon: no curvature anywhere; orthogonalization by Newton-Schulz GEMMs.
- Newton-Muon (Du & Su, arXiv:2604.01472): unilateral right
  preconditioning by input activation covariance (ZZᵀ, Cholesky-tracked)
  under an isotropic-output assumption. Status: closed as a footnote —
  v1 (eager hooks) OOM'd twice under compile; v2 (compiled custom op,
  NS12, refresh k=10) ran clean but did not cross (3.31761, run7).
  One ported run; not evidence against the paper.
- Plain SOAP: bilateral gradient covariance (dual eigh/QR). Status:
  unresolved — hot recipe no-cross (+0.066), reference recipe no-cross
  (+0.023, run11), one draw killed early, 0.037 same-seed spread.
  Recipe matters but is not the fix. No claim either way.
- SinkSOAP: bilateral dual eigh every step + Sinkhorn coordinate
  balance. Same rotation cost as SOAP for ~0.5ms of post-processing.
  Warm-started freq-10 amortization (run13) matches it at ~56% cost.
- COSMOS (track-3 variant): rank-r subspace QR tracking + compressed
  Sinkhorn blocks + NorMuon update. 3-seed edge over Muon at r256
  (3150/3150/3175 vs 3225/3250/3225); quality nearly flat in rank
  (r32 ≈ r256); directional edge holds at 8 GPUs. Step-count edge at
  ~cost parity; wall-clock flat-to-slower at measured scales.

## 8. Claim register

- [measured] Profiler closure: 1.88s predicted vs 1.92s observed;
  SinkSOAP overhead ~97% eigh.
- [measured, low-confidence] SinkSOAP crossed ~100 steps before Muon
  (3100 vs 3200); gap is near the ±70-step noise floor, single
  unseeded draws — suggestive, not conclusive.
- [measured] COSMOS-256 edge over Muon, 3 seeds each: 3150/3150/3175
  vs 3225/3250/3225, no overlap.
- [measured] Quality nearly flat in rank: r32/r64/r128 cross
  3175/3200/3200 vs r256 3150–3175; r512 3125 (final 3.27021).
- [measured] Amortized SinkSOAP (freq=10/T=1): 1.95s/step, NO crossing,
  final 3.29182. T exonerated (run12 T=10 also no-cross); warm-start
  recovers (run13 crosses 3125) — the failure is cold-start staleness.
- [measured] SOAP-QR (seed0, lr=0.002, no warmup): no crossing, 3.34559.
- [measured] SOAP reference recipe (lr=0.0018 + warmup): no crossing,
  3.30265 (run11) — closer, but recipe is not the fix.
- [measured] Same-seed SOAP-QR @875 differs by 0.037 (run3 vs run4A).
- [measured] eigh bit-identical 20x + 3x on CUDA (job 13848105).
- [measured] 8-GPU: COSMOS-r32 within 1–3%/step of Muon, crosses first
  at both seeds (3200 vs 3225/3250); SinkSOAP 1.6x/step. PCIe-only node.
- [measured] Newton-Muon port (v2): no crossing, 3.31761. Footnote only.
- [hypothesized] SOAP-QR chaos mechanism (low-rank Grams, near-tied
  eigenvalues, argsort/QR divergence). Decisive battery pending.
- [hypothesized] Noise-robustness axis (NS contraction vs eigengap
  amplification); supported by fp64 traces, needs tensor-level proof.
- [untested] COSMOS lr/wd tuning; rank 8/16 knee; asymmetric ranks.
- [analytical] FSDP-vs-TP distributed-cost split; width scaling table.

## 9. Collaborator questions

1. [answered] Distributed regime: both FSDP and TP matter, framed as
   step-count reduction under limited overhead — at scale, gradient
   compute+comm dominate, so fewer steps to target wins if the
   optimizer adds little overhead and its state shards (or is cheap to
   replicate). Full rotation matrices (SOAP/SinkSOAP) are the expensive
   case; low-rank (COSMOS) the promising one.
2. [answered] SOAP recipe: reference recipe (lr=0.0018 + 250 warmup)
   tested in run11 — no crossing at +0.023. Fair test; method still
   open, lane deprioritized.
3. [answered] Newton-Muon priority: v2 port ran clean (custom-op
   capture) and did not cross; lane closed as a footnote per
   cost/benefit, not worth further surgery.
4. [answered] COSMOS config: lr=0.04/wd=0.025/ranks=256 confirmed as
   the initial speedrun config, but untuned — room for improvement.
   Directed next step: rank–convergence–runtime tradeoff at lower
   ranks (done, Sec. 2), then larger scale.
