# Methods and protocols

How runs are paired, checked, profiled, and checkpointed, plus variant
provenance and operational rules. Results live in `analysis.md`; the run
ledger is `runs.csv`. Last updated 2026-10-28.

## 1. Seeding and pairing

The base harness never seeded, so run1/run2 are unseeded pilots with
unpaired init noise. Every later run uses a `variants/*_seed*.py`
script (`torch.manual_seed` + `cuda.manual_seed_all` at trial start)
and swaps only the 2D block-parameter optimizer, so variants pair on
init against the seeded Muon control (run5).

## 2. Determinism checks

- Gate: `variants/test_seed_determinism.sh` runs a variant twice at
  12 steps and requires identical step-12 val_loss. Step-0 val is not a
  valid check: the zero-init output projection makes step-0 loss
  ln(50304) = 10.82583 for any init.
- Matrix (2026-09-27, 1x Blackwell): seeded baseline bit-identical
  (6.37711 == 6.37711); seeded SinkSOAP differs at 7e-5 @step 12
  (6.33337 vs 6.33330); unseeded controls differ as expected.
- eigh: `research_stack/test_eigh_determinism.py` asserts bit-identical
  results 20x in-process plus a 3-process cross-process comparison, on
  full-rank and rank-deficient 3072 grams (PASS on CUDA, job 13848105).

## 3. Jitter characterization

fp64 per-step energy traces (2 seeded runs per method): step-0 grads
and post-step-0 params identical; first divergence at step 1 at last-bit
level (rel ~1e-12), located in the shared harness forward/backward
(consistent with CUDA atomic races, e.g. flash-attention dQ). Muon
contracts the perturbation below print precision in one step
(Newton-Schulz toward the orthogonal manifold); SinkSOAP amplifies it
~360x in one step (grad rel-d 2.6e-13 to param rel-d 9.4e-11) via small
Gram eigengaps (Davis-Kahan). Growth probe (seeded SinkSOAP, 2x130
steps): 7e-5 @12, 4.8e-3 @125, 4.7e-3 @130. Saturation level unknown
(a 2x500-step probe was lost to job expiry).

## 4. Profiling

`profile/profile_optimizers.py` (Blackwell): per-shape full-update ms,
post-warmup mean — attn 768² (Muon 0.63ms, SinkSOAP 15.55ms),
mlp-fc 768x3072 (0.66ms / 49.34ms), mlp-proj 3072x768 (0.76ms /
49.46ms); 96–98% of SinkSOAP time is eigh, ~0.5ms Sinkhorn.
Full-model overhead prediction (1.88s) closes against the measured
run2−run1 delta (1.92s). SOAP tracked form (precond_freq=10): 0.75ms
attn / 2.7ms MLP rects. COSMOS-r32 full update (§6): 1.5–1.7ms/shape
vs Muon 0.6–0.7ms; overhead is subspace QR + Sinkhorn (~0.7ms/layer),
NS-on-residual ≈ full-NS cost. State memory (fp32 exact, 12-layer 2D
totals): Muon 340MB, SinkSOAP 1529MB, SOAP 3058MB; see
`state_accounting.md` for the full table incl. COSMOS ranks.

## 5. Variant provenance

Single-file scripts; the variant file is the unit of a run. Shared
infrastructure only in `research_stack/`.

| variant | base + change | source |
|---|---|---|
| baseline_seed{0,1,2} | Sep harness + seed | public Track 3 @bc3a0c2 |
| baseline_ns{8,6}_seed{0,1} | baseline + NS steps 12→8/6 | ours (Exp 2) |
| sinksoap_seed0 | baseline + SinkSOAP swap (lr=0.04, wd=0.025) | lliu606 fork, May recipe @a93a972 |
| soap_seed0 | baseline + SOAP class verbatim (lr=0.002, betas .95/.95, wd=0, freq=10) | public Track-1 record 2024-10-09 |
| soap_ref_seed0 | soap_seed0 + lr=0.0018, 250-step linear warmup (reimplemented) | recipe vs nikhilvyas SOAP-ref (read-only) |
| sinksoap_amort_seed0 | sinksoap + cached eigenbases (freq=10, T=1) | ours |
| sinksoap_amortT10_seed0 | amort + T restored to 10 | ours |
| sinksoap_warmstart_seed0 | amort + per-step eigh for steps 1–200 | ours |
| cosmos_seed{0,1,2} | baseline + COSMOS swap (lr=0.04, wd=0.025, ranks=256) | lliu606 fork track-3 wiring |
| cosmos_r{32,64,128,512}_seed0 | cosmos_seed0 + rank_o=rank_u=r | ours (rank sweep) |
| cosmos_r32_seed1 | r32 + seed1 | ours |
| cosmos_r32_warm150_seed{0,1} | r32 + Muon steps 1–150, COSMOS after | ours (Exp 1) |
| cosmos_r32_ns{8,6}_seed{0,1} | r32 + NS steps 12→8/6 | ours (Exp 2) |
| cosmos_r32_sqnorm_seed{0,1} | r32 + NorMuon on square matrices | ours (Exp 3) |
| newton_muon_seed0 | baseline + NewtonMuon, custom-op capture (v2; v1 hooks OOM'd) | port of zhehangdu/Newton-Muon Alg. 1 |
| sinkhorn_embed_seed0 | parked (built on a misread brief; run2 covers SinkSOAP) | DeepSeek-V4.1-Flash report §2.5 |

Deviations from verbatim sources are bannered in each file (dist-call
guards, import renames, warmup scope).

## 6. Checkpointing

`research_stack/ckpt.py`, single-GPU only (asserts world_size==1).
Contents: next step S + model/optimizer states + train clock + RNG
states; atomic tmp+rename writes; keep-last-2 rotation. Resume
(`--resume path`) loads state and fast-forwards the deterministic
dataloader by S batches. Periodic save every CKPT_EVERY (default 250,
via APPTAINERENV_CKPT_EVERY since the container scrubs env).
SIGTERM/SIGINT sets a flag; the loop saves and exits 143 at the next
safe point (end of step, no NCCL in flight).
Validation status: CPU round-trip PASS (job 13945667); GPU resume has
two recorded failure modes (CPU RNG device fixed in 265f37f; CUDA RNG
structure still open). Resume lane closed unless a run is cut off.

## 7. Operational rules

- Never edit a script while a process may be executing it: the running
  shell executes its buffered image, including since-removed lines
  (run4b ghost duplicate, recorded in runs.csv). Batch scripts are
  immune (sbatch spools at submit).
- GPU work only through sbatch (1-GPU: 1 GPU + 8 CPUs + 96GB; 8-GPU:
  8 GPUs + 16 CPUs + 400GB); the SLURM account passes via CLI flag,
  never committed. Batch `.out` logs stay on disk, out of the repo.

## 8. Reference recipes (read-only)

- nikhilvyas/modded-nanogpt-SOAP (`SOAP-ref/`, gitignored): lr=0.0018,
  250 warmup, wd=0, betas (0.95,0.95), freq=10, 7000 iters, same
  524288-token batch. Ships no LICENSE file: reference only, no code
  copied (the warmup above is reimplemented).
- Newton-Muon paper (Du & Su, arXiv:2604.01472): §6 flags single-GPU
  evaluation as future work (quote verified); our port defaults
  (beta=0.95, ridge 0.2·tr/d, Cholesky factor + triangular solves,
  harness-Muon pipeline) and deviations are documented in
  `research_stack/newton_muon.py`.
