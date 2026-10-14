# Muon vs SOAP tradeoff analysis — v1 (interim, 2026-09-28)

Living doc. Covers: run1 (Muon, full), run2 (SinkSOAP, full), run3 (SOAP,
killed@968), run4 (SOAP rerun, in progress), optimizer profiles, noise
characterization. Seeded paired runs + SOAP completion still pending — all
crossing comparisons below carry the stated error bars.

## 1. Convergence (steps to val 3.28, train_steps=3250, 1x Blackwell)

| run | method | first obs ≤3.28 | crossing window | final val | train time |
|---|---|---|---|---|---|
| run1 | Muon (unseeded pilot) | step 3200 | (3175, 3200] | 3.27714 | 99 min |
| run2 | SinkSOAP (unseeded pilot) | step 3100 | (3075, 3100] | 3.26769 | 203 min |
| run3 | SOAP-QR (killed@968) | — | — | 3.769 @875 | — |
| run4 | SOAP-QR rerun | running | — | — | est ~105 min |

Early curves (val loss; run3a/run4 are two same-seed SOAP draws):

| step | Muon | SinkSOAP | SOAP(a) | SOAP(b) |
|---|---|---|---|---|
| 125 | 4.60 | 4.56 | 5.90 | 5.68 |
| 250 | 4.10 | 4.06 | 4.92 | 4.55 |
| 500 | 3.83 | 3.80 | 4.04 | 3.96 |
| 875 | 3.68 | 3.66 | 3.77 | — |

Read: SinkSOAP tracks Muon ~0.03 ahead throughout and crosses ~100 steps
earlier with 0.01 better final loss. SOAP starts ~1.3 behind but closes to
~0.09 by step 875 — the literature's "SOAP wins late" shape, visible here.
Caveats: run1/run2 unseeded (init noise unpaired); crossing grid ±25;
SinkSOAP same-seed jitter (Sec. 4) is material vs the 100-step gap;
SOAP draws differ by up to ~0.4 early (Sec. 4), so run3a/run4 early numbers
are illustrative, not comparable.

## 2. Compute cost per step (measured 1x Blackwell + microbench)

Full-step wall: Muon 1.83s, SinkSOAP 3.75s, SOAP-QR ~1.89s.
Per-matrix update ms (attn 768² / mlp 768x3072):

| method | attn | mlp | full-model est |
|---|---|---|---|
| Muon (12x NS) | 0.6 | 0.7 | ~0.05s |
| SinkSOAP (eigh every step) | 15.6 | 49.4 | ~1.9s |
| SOAP (eigh once + QR track) | 0.75 | 2.7 | ~0.1s |

SinkSOAP breakdown: 96–98% is `eigh`; Sinkhorn itself ~0.5ms. Closure:
predicted step overhead (1.88s) matches measured run2−run1 (1.92s).
Implications:
- SinkSOAP's cost is ~entirely the eigendecomposition it shares with SOAP;
  the Sinkhorn-for-Adam substitution saves ~nothing in compute. Compute-side
  confirmation that SinkSOAP keeps SOAP's expensive setup (brief hypothesis).
- BUT this is an implementation choice, not inherent: Liming runs eigh EVERY
  step with no amortization. SOAP-style QR tracking (or eigh every K steps)
  would cut ~10x. Staleness-vs-quality is the obvious next experiment.
- 1-GPU wall-clock FLATTERS QR-SOAP (hides its distributed sync) and PUNISHES
  unamortized SinkSOAP (pays full eigh; on W GPUs it shards ÷W). Quote step
  times as 1-GPU bounds, not inherent costs.

## 3. Memory (optimizer state, 2D block params, fp32, exact)

Muon 340MB < SinkSOAP 1.53GB (grams) < SOAP 3.06GB (m/v + grams + Qs).
Observed total job footprint ~36GB / 96GB for all runs (nvidia-smi); seeded
scripts log `max_memory_allocated` for exact totals going forward.

## 4. Noise / reproducibility (same seed, same GPU, same code)

Same-seed val spread:

| steps | Muon | SinkSOAP | SOAP-QR |
|---|---|---|---|
| 12 | 0 (5-dec) | 7e-5 | ~0.15 (4 runs, chaotic) |
| 125–130 | 1.9e-4 | 4.8e-3 | ~0.16 (persists) |

~25x separates each level. Mechanism (fp64 per-step traces + cross-process
eigh tests): `eigh` is bit-identical in- and cross-process — exonerated.
BOTH methods receive last-bit flips from the shared harness at step 1
(likely CUDA atomic races in compiled backward). Then:
- Muon CONTRACTS perturbations below fp64-print precision in one step
  (Newton-Schulz forgets toward the orthogonal manifold);
- SinkSOAP AMPLIFIES ~360x in one step (eigenvector swings via small
  Gram eigengaps, Davis-Kahan);
- SOAP-QR goes chaotic: low-rank early Grams → thousands of tied ~zero
  eigenvalues → fp-arbitrary argsort → permuted columns into
  order-dependent QR → a different preconditioner per run.
Noise robustness (NS contraction vs eigengap sensitivity) is a genuine extra
tradeoff axis favoring Muon-family — and a design constraint on any hybrid
using eigendecomposition. Practical consequence: Muon/SinkSOAP crossings are
comparable within stated jitter; SOAP-QR crossings are wide draws (multi-seed
needed for a serious SOAP number).

## 5. Distributed-friendliness (ANALYTICAL — not measured; 1 GPU cannot)

Two regimes matter; conflating them is the common error:
- Regime 1 — FSDP/ZeRO-style (this harness: data-parallel + optimizer
  param-sharding): each rank owns WHOLE 2D matrices, so `eigh`/QR is purely
  rank-local compute. SOAP-family pays local compute + memory only; per-GPU
  comm ≈ Muon's (grad allreduce + param all_gather). Our 1-GPU compute/memory
  numbers characterize exactly this regime.
- Regime 2 — Tensor Parallelism (single 2D weights sharded across GPUs):
  assembling each mode's Gram matrix requires a cross-rank AllReduce over
  the sharded dimension, and the eigendecomposition itself becomes a
  distributed eigensolve — cubic work that parallelizes poorly plus
  fine-grained synchronization. THIS is the hard communication wall behind
  the brief's bottleneck claim; it lives here, is currently
  literature/analysis-backed only, and is where Muon's local NS iteration
  wins structurally (its orthogonalization never needs the Gram eigenbasis).
Implementation caveats in our harness: the transplanted SOAP.step does NOT
shard (redundant W× optimizer compute + full 3GB states per rank on multi-
GPU — correct but wasteful; fixable); SinkSOAP shards like Muon but pays
unamortized per-step eigh on its shard. 2-GPU strong-scaling probes (when a
second GPU is available) can turn regime-1 scaling into measured numbers.

## 6. Theoretical landscape (added 2026-09-28)

Newton-Muon (Du & Su, arXiv:2604.01472) minimizes a triplet quadratic
surrogate (grad G + output curvature H + input data Z); under H ∝ I it yields
the right-preconditioned polar update W ← W − η·msgn(G(ZZᵀ+γI)⁻¹). It slots
in as the **unilateral right-preconditioned Muon** comparator:

- **Muon**: isotropic input + isotropic output (no curvature anywhere); pure
  GEMMs (NS iteration). Zero distributed curvature cost by construction.
- **Newton-Muon**: ANISOTROPIC input activation covariance (ZZᵀ, n×n per
  layer, Cholesky-tracked) + isotropic output. Unilateral (right) only: the
  "left curvature" (output-space geometry) is assumed away (H ∝ I), leaving
  SOAP's bilateral preconditioning untouched. TP fit: for column-parallel
  layers (QKV, MLP-FC1) the input Z is already replicated across ranks, so
  ZZᵀ accumulates LOCALLY with zero comm — unlike SOAP's gradient Grams,
  which need cross-rank AllReduce to assemble under TP.
- **Plain SOAP**: BILATERAL gradient covariance (G_L m×m + G_R n×n, dual
  eigh/QR). Heavy TP comm (Gram AllReduce + distributed eigensolve) and, in
  QR-tracked form, basis-rotation chaos (Sec. 4).
- **SinkSOAP**: bilateral dual eigh (every step, unamortized) + Sinkhorn
  coordinate balance in the rotated space. Same rotation cost as SOAP for
  ~0.5ms of post-processing savings (Sec. 2).
- Open gap the paper itself flags (§6): single-GPU evaluation only —
  distributed Newton-Muon (multi-GPU ZZᵀ semantics) is uncharted, which is
  directly our research direction. Also open: non-isotropic output proxies
  (their App. F attempts failed — "estimating Sigma_W reliably is nontrivial").

Our implementation: `research_stack/newton_muon.py` (Alg. 1 + reference repo
zhehangdu/Newton-Muon, cloned at Newton-Muon-ref/). Defaults: beta=0.95,
ridge 0.2·tr(K)/d, K init 1e-3·I, refresh k=10 (paper: 32 — we align with our
SOAP precond cadence), 4-block rule for (d,4d) contraction, Cholesky factor +
triangular solves (reference uses explicit inverse; identical math, stabler),
harness-Muon pipeline (EMA+Nesterov+NS12+rect-scale+wd) after preconditioning.
CPU self-test passes (toy convergence, PD fallback, determinism); GPU smoke +
benchmark queued after the Sinkhorn-embed run.

## 7. Open questions / pending work
- Seeded paired set: run5 (Muon control, queued) + Amortized SinkSOAP
  (precond_freq=10, T=1, est ~1.9s/step, next window). Decides whether the
  100-step gap survives pairing.
- SOAP: run4 completes one draw; crossing needs spread data (multi-seed) to
  be comparable. SOAP lr=0.002 is a track-1-recipe guess — flag to Liming,
  along with: no-warmup soundness, full-eigh vs QR-tracking in frontier
  SOAP, and the SinkSOAP-keeps-eigh observation.
- Ckpt validation (R0/R1/R2) after the chain clears; then ckpt rollout to
  sinksoap/soap variants.
- DECIDED 2026-09-28: no full unamortized seeded-SinkSOAP rerun (already have
  3.75s/step + noise data). If rerun, amortized (precond_freq=10, T=1).
- Sinkhorn-embed/head (DeepSeek-V4.1 §2.5, K=11, tau=1e-3, eps=1e-20,
  gamma=0.18, no wd): module + variant ready; smoke + full benchmark vs
  baseline right after run5 + ckpt validation.
- Staleness study (SinkSOAP eigh every K steps: convergence vs cost) — the
  natural next experiment once the paired set lands.

## 8. Emerging hybrid directions (hypotheses only — no design yet)
1. Amortized-eigh SinkSOAP: full eigh every K steps + QR tracking between
   (cuts the 1.9s overhead ~10x). Tests whether staleness hurts quality.
2. Rotation-free bridge (the brief's sharp question): SOAP-like benefit
   without any eigenbasis — the hardest, most valuable direction; needs the
   full comparative dataset first.
3. Noise-robustness as a design constraint: anything with eigh inherits
   chaos (Sec. 4); NS-style contraction is the property to preserve.

## 9. Run4A result: SOAP-QR does not cross on this draw (2026-09-28 01:36 EDT)
Run4A (seed0, track-1 QR-SOAP, lr=0.002) completed all 3250 steps:
~1.894s/step (6155s train), 33.8GB max, final=min val 3.34559 — NO
crossing of 3.28 (gap +0.066). Curve is smooth and monotone-ish late
(5.676@125, 3.959@500, 3.688@1000, 3.554@1500, 3.469@2000, 3.407@2500,
3.35881@3000, 3.34559@3250); late slope ~-0.035/500 steps, so crossing
would need roughly +900 steps at that rate. Dense-tail val cadence
(every 25 after 2875) confirmed in this log. Head-to-head on completed
draws: Muon 3.27714 (run1), SinkSOAP 3.26769 (run2), SOAP-QR 3.34559
(run4A) — SOAP-QR trails by ~0.07-0.08 with the lr=0.002-guess caveat
(flag to Liming stands) and single-draw caveat.
Same-seed determinism check FAILED for QR-SOAP: run3@875 = 3.76932 vs
run4A@875 = 3.73234 (identical seed0 script, Δ=0.037, ~8x SinkSOAP's
4.8e-3 same-seed jitter). GPU nondeterminism swamps seeding in the
QR/argsort path — seed-fixing does NOT pair SOAP-QR draws; multi-seed
spread is required before any crossing claim.
Incident: a ghost duplicate (run4b, same script) launched at 01:37 from
the leg1 wrapper's stale pre-edit image (script edited mid-flight at
23:56; forensics + rule in method_notes.md). Killed 02:34; chain
self-healed to leg2 (run5 seeded baseline, ETA ~04:18 vs 04:38 cutoff).
Run5 progress 03:35 EDT: step 1844/3250 at 1.859s/step, last val
3.49687@1750; ETA ~04:19, margin ~19min. No action needed.
Run5 DONE 04:23 EDT (LEG2_EXIT:0): full 3250 steps at 1.857s/step,
31.1GB max. First val<=3.28 observed at step 3225 (3.27910; true
crossing in (3200,3225]); final 3.27844. vs run1 (unseeded): crossing
3200 -> 3225 (+25 steps), final 3.27714 -> 3.27844 (+0.0013) — the
seed0 draw is a touch slower, small vs the 100-step run1/run2 gap and
the +0.066 SOAP shortfall. Seeded control arm COMPLETE: all future
comparators (sinkhorn-embed, newton-muon, amortized sinksoap) pair
against run5, not run1.
External review (Claude, 2026-10-04) adjudicated: agreed Run 8
(amortized SinkSOAP) goes first and sinkhorn-embed parks ("sink soap"
meant SinkSOAP — run2 covers it); COSMOS found fully implemented in
Liming's public fork and spliced. Newton-Muon paper verified directly
(arXiv:2604.01472 real; §6 single-GPU gap quote confirmed verbatim).
Eigh "contradiction" dissolves (deterministic-but-ill-conditioned), but
the 20x/3x test artifact was lost — recommitted as
research_stack/test_eigh_determinism.py (PASS 2026-10-04, job 13848105:
20x in-process + 3x cross-process bit-identical on CUDA, full-rank and
rank-deficient 3072 grams). Chaos tie-break stays a
hypothesis: decisive battery = tolerance-quantized sort +
deterministic-algorithms rerun + autotune-setting check.
Ckpt resume bug found + fixed 2026-10-10: map_location="cuda" moved the
CPU RNG ByteTensor to GPU (TypeError in set_rng_state); fix restores
with .cpu() + guards CUDA RNG (commit 265f37f). CPU round-trip PASS in
14s (job 13945667, also proving container-without---nv on cpu-small);
GPU re-validation queued as 13945668.

## 10. Claim register (loadout rule: measured / hypothesized / untested)
- [measured] Profiler: SinkSOAP step overhead 1.92s, ~97% eigh, Sinkhorn
  ~0.5ms; cost-model closure 1.88s vs 1.92s.
- [measured, LOW-CONFIDENCE] SinkSOAP crossed ~100 steps before Muon
  (3100 vs 3200) — gap is near the ±70-step noise floor; single
  unseeded draws. Per rule: treat as suggestive, not conclusive.
- [measured] SOAP-QR seed0 draw: no crossing, final 3.34559 (single draw).
- [measured] Same-seed SOAP-QR @875 differs by 0.037 (run3 vs run4A).
- [measured] torch.linalg.eigh bit-identical 20x in-process + 3x
  cross-process on CUDA (job 13848105).
- [hypothesized] Chaos mechanism: low-rank Grams -> near-tied
  eigenvalues -> argsort/QR divergence. NOT proven; decisive battery
  pending (tolerance-quantized sort + deterministic rerun + autotune check).
- [hypothesized] Noise-robustness axis (NS contraction vs eigengap
  amplification), supported by fp64 traces, needs tensor-level bisection.
- [hypothesized] run4A shortfall partly from no-warmup + 3250-step
  horizon (SOAP-ref uses warmup 250 + 7000 steps at lr 0.0018).
- [untested] Amortized SinkSOAP: ~1.9s/step + kept step savings (Run 8).
- [untested] COSMOS and Newton-Muon full-benchmark behavior on our harness.
- [paper-text-verified; novelty-UNVERIFIED] Newton-Muon §6 single-GPU
  quote confirmed verbatim; distributed-NM direction not validated
  with Liming — no further variants built on it until he weighs in.
- [analytical] FSDP-vs-TP distributed-cost split (no multi-GPU runs).

## 11. Run 8 + run 6 results (2026-10-11)
Run 8 (amortized SinkSOAP, freq=10/T=1): 1.952s/step (cost prediction
HIT), NO crossing, final 3.29182. The crux answer is NEGATIVE at these
settings: unamortized crossed 3100 @3.75s with final 3.268; amortized
runs at Muon cost but keeps none of the step savings — worse than Muon
(3225 cross, 3.27844) on both axes this draw. Open: T-vs-frequency
attribution (both changed); gentler amortization (freq=3/5, T=10) is the
obvious follow-up, NOT yet run.
Run 6 (COSMOS, lr=0.04/wd=0.025/ranks=256): crossed at observed 3150,
final 3.27399, 1.986s/step — beats seeded Muon by 75 steps at equal
step cost; best seed0 final and best SOAP-family wall-clock to date
(6258s to cross vs Muon 5860/5987s, i.e. ~4-7% slower — the earlier
"best wall-clock" wording was wrong, corrected 2026-10-13). Trailing the
unamortized SinkSOAP step count (3100) but at half the per-step cost.
Head-to-head finals: SinkSOAP 3.26769 > COSMOS 3.27399 > Muon 3.27714 >
seeded Muon 3.27844 > amort-SinkSOAP 3.29182 > SOAP-QR 3.34559.
Newton-Muon lane STOPPED: second identical OOM (cudagraphs-off made
allocated memory WORSE, 82.9GB — my pool-bloat theory was wrong about
the dominant term; fragmentation was fixed but real dynamo-fragment
allocation dominates). Options need user input; no further NM jobs.
Plain SOAP remains UNRESOLVED: run4 drew no-cross (+0.066) on one
chaotic draw, run3 was killed early at 968, and same-seed draws vary by
0.037 — no crossing claim is supported in either direction.

## 12. Open questions for Liming (consolidated 2026-10-13)
1. Distributed regime: by "distributed-friendly" do you mean FSDP-style
   (our finding: compute/load-balance problem) or tensor-parallel
   (communication wall for SOAP Grams)?
2. SOAP recipe: is lr=0.002 with no warmup a fair test, or should we
   match the SOAP-ref recipe (lr=0.0018 + 250 warmup, 7000 steps)?
3. Newton-Muon priority: OOMs at 83/95GB under hooks+compile on our
   rig — is this direction viable/important enough for deeper surgery?
4. COSMOS confirmation: we benchmarked lr=0.04/wd=0.025/ranks=256 from
   your track-3 wiring — is that what you ran?
