# Method notes — Muon vs SinkSOAP comparison rigor

## Seeding (decided 2026-09-27)
`train_gpt_simple.py` never called `manual_seed` — every run had random init.
run1/run2 are unseeded pilots. All future comparison runs use
`variants/*_seed0.py` (`torch.manual_seed(0)` + `cuda.manual_seed_all(0)` at
trial start) so variants are paired on init. Check: `variants/test_seed_determinism.sh`
runs a variant twice at 12 steps and requires identical step-12 val_loss.
(step-0 val is NOT a valid check: zero-init output proj => loss is ln(50304)
= 10.82583 for ANY init.)

## Determinism matrix (2026-09-27, 1x RTX PRO 6000 Blackwell)
- Seeded baseline: OK — 6.37711 == 6.37711 (bit-identical across runs)
- Seeded SinkSOAP: NOT bit-identical — 6.33337 vs 6.33330 (d ~ 7e-5 @ step 12)
- Unseeded baseline (negative control): differs as expected (6.37795 vs 6.38334)
- Unseeded SinkSOAP (negative control): differs as expected (6.33503 vs 6.33374)

## Jitter mechanism (CORRECTED 2026-09-28 — eigh exonerated)
`torch.linalg.eigh` is bit-identical in-process (20 calls) AND cross-process
(3 fresh processes), including on rank-deficient 3072 grams. It is a
deterministic-but-ill-conditioned function, not a noise source.

fp64 per-step energy traces (2 seeded runs each) localize the true mechanism:
- BOTH methods: step-0 grads + post-step-0 params identical; first GRAD-E
  divergence at step 1 at last-bit level (rel ~1e-12). Source is in the shared
  harness (forward/backward, likely CUDA atomic races, e.g. flash-attn dQ) —
  NOT the optimizer, NOT eigh.
- SinkSOAP AMPLIFIES the micro-perturbation ~360x in a single step
  (grad rel-d 2.6e-13 -> param rel-d 9.4e-11): eigenvectors swing under tiny
  input changes when Gram eigengaps are small (Davis-Kahan).
- Muon CONTRACTS it below fp64-print precision (PARAM-E 2 identical):
  Newton-Schulz forgets micro-perturbations toward the orthogonal manifold.
Seeding still cuts SinkSOAP noise ~70x vs init noise, but cannot zero it:
chaos-amplified race noise is intrinsic to the eigenbasis approach. Noise
robustness (NS contraction vs eigengap sensitivity) is a genuine extra
tradeoff axis favoring Muon-family.

## Jitter growth probe, seeded SinkSOAP, 2x130 steps (2026-09-27)
- step 12:  d ~ 7e-5
- step 125: 4.61638 vs 4.61159, d ~ 4.8e-3
- step 130: 4.61041 vs 4.60567, d ~ 4.7e-3
Saturation level unknown — a 2x500-step probe (d @125/250/375/475/500) was
started but lost to job expiry before printing. TODO: redo.

## Why it matters
Near the 3.28 crossing the loss slope is ~7e-5/step, so +-5e-3 of loss jitter
maps to ~+-70 steps of crossing jitter, vs the observed ~100-step run1/run2
gap. Single-run crossing gaps need the saturation number (or a second seed)
before they are conclusive. (Earlier note said "Muon has zero same-seed
jitter" — true only at 5-decimal val precision; fp64 traces show Muon also
receives last-bit harness flips but contracts them instead of amplifying.)

## Optimizer profile (2026-09-28, RTX PRO 6000 Blackwell, profile/profile_optimizers.py)
Per-shape full-update ms (post-warmup mean):
- attn (768x768):    muon 0.63ms, sinksoap 15.55ms (24.7x)
- mlp-fc (768x3072):  muon 0.66ms, sinksoap 49.34ms (75.3x)
- mlp-proj (3072x768): muon 0.76ms, sinksoap 49.46ms (65.0x)
SinkSOAP breakdown: 96-98% is eigh (14.9/15.6ms attn, 48.3/49.3ms MLP);
Sinkhorn itself only ~0.5ms. Closure check: predicted full-model step
overhead (1.93s - 0.05s = 1.88s) matches measured run2-run1 delta (1.92s).
Implication: SinkSOAP's cost is ~entirely the eigendecomposition it shares
with SOAP; the Sinkhorn post-processing it substitutes for Adam saves ~nothing
in compute — compute-side confirmation that SinkSOAP keeps SOAP's expensive
setup. (Caveat: Liming runs eigh EVERY step; amortizing like SOAP's QR
tracking would cut this a lot — a design note for hybrid work.)
State memory, 12-layer 2D totals (fp32, exact): muon 340MB, sinksoap 1529MB,
soap 3058MB (m/v + GG + Q per mode).
SOAP per-shape update (amortized, precond_freq=10): 0.75ms attn, 2.7ms MLP
rects — only ~4x Muon (QR tracking is cheap; full eigh happens once at init).
Est full-model SOAP step overhead ~0.1s => full run ~105min.
Same-seed spread @ step 12 (4 runs): 7.87041, 7.71796, 7.78997, 7.73479 —
spread ~0.15, no clustering (chaotic, not bimodal). Mechanism: early grams are
low-rank (EMA from zero) => ~3000 near-tied ~zero eigenvalues => argsort order
is fp-noise-arbitrary per run => permuted columns into order-dependent QR =>
genuinely different Q per run => divergent preconditioning. 130-step SOAP pair: 6.41628 vs 6.25758 @125 (d~0.16) — chaos PERSISTS
(grams still low-rank vs 3072 dim, ties persist). A single full SOAP run is
one wide draw; its crossing is not comparable without spread data. Running
one full draw anyway (run3) for curve shape + memory + late dynamics.

## Checkpointing (added 2026-09-28, NOT yet GPU-validated)
`research_stack/ckpt.py`, single-GPU only (asserts world_size==1; Muon-style
param sharding spreads optimizer state, so multi-GPU resume needs a state
all-gather — unimplemented). Ckpt = next step S + model/opt states +
train_time clock + RNG states; atomic tmp+rename writes; keep-last-2 rotation.
Resume (`--resume path`): loads state, fast-forwards the deterministic
dataloader by exactly S batches (train loop consumes 1/step; verified by
construction, will be verified empirically). Periodic save every CKPT_EVERY
(default 250, via APPTAINERENV_CKPT_EVERY since the container scrubs env).
SIGTERM/SIGINT (SLURM preemption): flag-based handler, save+exit(143) at the
next safe point (end of step, no NCCL in flight). Integrated so far ONLY in
`variants/train_gpt_baseline_seed0.py` (5-hook mechanical patch); roll out to
sinksoap/soap variants after validation. run4/run5 chain uses pre-ckpt copies
and is unaffected.
Validation (runs AFTER run4/run5 chain clears the GPU):
- R0: determinism re-check on patched baseline must reproduce 6.37711 exactly
  (proves the patch is behavior-neutral when inactive).
- R1/R2: `research_stack/test_resume.sh` — V1 resume-from-ckpt@100 continuity
  vs uninterrupted (pass: step-125 val within 2e-3; natural spread 2e-4, real
  bugs show >>1e-2); V2 SIGTERM-the-process-group drill + resume to completion.
Deliberately NOT built: sweep frameworks, W&B, multi-node launcher.

## Other stack additions (2026-09-28)
- `research_stack/snapshot.py`: writes run_dir/meta.json (script sha256,
  torch/CUDA, GPU, SLURM job, git hashes). Run BEFORE training launch.
- `runs/plot_curves.py`: val curves for all runs/*/stdout.log -> runs/curves.png.
- `run_container.sh`: APPTAINERENV_PYTHONPATH=$PROJ so container imports research_stack/.

## Sinkhorn-embed/head spec (DeepSeek-V4.1-Flash report Sec 2.5 + 4.2.2)
Report: momentum-based update + Sinkhorn balancing for Engram/token-embedding/
prediction-head (replaces Adam there to save optimizer-state memory); Muon for
backbone matrices, AdamW for norms/1D. Algorithm 1 transcribed verbatim into
`research_stack/sinkhorn.py`: Nesterov beta=0.95, mask rows rho_i<=tau*rho_bar
(tau=1e-3), K=11 alternating row/col l2-norms (rows first+last), Delta =
sqrt(n)*U (unit row RMS), apply with lr*gamma, gamma=0.18 ("reutilization of
the AdamW learning rate"), eps=1e-20, NO weight decay.
Transplant decisions (ours to defend):
 - params: model.embed (50304x768 bf16) + model.proj (50304x768 fp32);
   rows=tokens, cols=features, no transpose needed.
 - TWO groups keep harness base LRs (embed 0.7, proj 0.004) x gamma=0.18,
   same stable+cooldown schedule ("reutilization" per-group). No wd (report).
 - fp32 momentum (155MB x2 = 309MB total vs AdamW m+v 618MB); Delta cast to
   param dtype on apply (same rounding convention as fused AdamW; revisit
   with fp32 master weights only if embed training looks noisy).
 - (a) aspect ratio: native to the algorithm (m>>n); cost ~K*m*n = 11 iters
   x 38.6M = ~10ms/step for both params on Blackwell (their fused kernels
   unnecessary at our scale).
 - (b) sparse rows: tau-masking is load-bearing — without it, row-norm would
   blow absent-token rows (exactly-0 grad, decayed momentum) up to unit norm
   (a full-size update for every unseen token every step). tau=1e-3: rows
   within 1000x of mean row-norm survive; long-absent rows freeze at 0.
 - (c) token frequency: row-norm equalizes update magnitudes across frequent
   and rare tokens (Adam-like normalization philosophy); masking gates stale
   rows; frequent tokens dominate rho_bar (set the bar). Column-norm balances
   hidden features. Net: every PRESENT token gets an RMS-1 update regardless
   of frequency — the intended inductive bias; whether it helps rare tokens
   or destabilizes them is an empirical question for the benchmark.

## Newton-Muon module (2026-09-28, CPU-validated, GPU smoke pending)
Paper: Du & Su arXiv:2604.01472 + ref repo zhehangdu/Newton-Muon (cloned).
`research_stack/newton_muon.py`: Alg.1 faithful port — refresh cov+Cholesky
every k steps (k=10 default; paper 32), beta=0.95, ridge 0.2*tr/d+1e-8,
K init 1e-3*I, raw-grad precondition BEFORE momentum, 4-block (d,4d) rule,
hooks gated to train+grad+refresh steps (val never pollutes cov).
Deviations (deliberate, documented in module): triangular solves instead of
explicit Cholesky inverse; harness-Muon pipeline (EMA/NS12/rect-scale) instead
of reference heavy-ball/NS5, per paper's own "replace the Muon update" protocol;
ridge-x10 retry then keep-previous-factor (ref: straight to identity).
Self-test (CPU): toy least-squares 11.77->0.81, hook/block shapes, eval-gating,
PD rescue + total-failure identity path, determinism. ALL PASS.
GPU needs: `research_stack/test_nm_smoke.sh` (12-step x2, TORCH_LOGS=recompiles)
then benchmark after sinkhorn-embed run.
Compile-risk decision (2026-09-28): hook body is @torch.compiler.disable BY
DESIGN. Rationale: the gate reads mutable global_step; inlining it would
guard-fail and recompile every step (catastrophic). Disable forces eager hooks
behind a static boundary break (bounded one-time fragmentation; backward has no
hooks and compiles whole). Consequence: literal "0 graph breaks" is the WRONG
invariant (disable works BY breaking) — the smoke asserts the meaningful one:
zero recompile lines on a warm-cache second run + stable step times.
Robustness: refresh cadence truth in per-param state['_step'] (state_dict-safe),
group['_global_step'] best-effort mirror; variants defensively re-stash
group["initial_lr"] after ckpt load (torch may drop custom group keys).

## Plain SOAP transplant (2026-09-28)
`variants/train_gpt_soap_seed0.py` = seeded baseline + SOAP class verbatim
from `records/track_1_short/2024-10-09_SOAP/train_gpt2.py` (only
`optim.Optimizer` -> `torch.optim.Optimizer` + `chain` import). 2D block
params only (1D stays AdamW, matching Muon/SinkSOAP protocol).
Hyperparams: lr=0.002 (track-1 recipe: 0.5x AdamW peak; 0.5*0.004 head-lr
here), betas=(0.95,0.95), wd=0, precondition_frequency=10, correct_bias=True.
SOAP details: full eigh once at init, then QR-tracking refresh every 10 steps;
GG EMAs updated every step; first step skipped. No param sharding in this
SOAP.step (redundant updates per rank — fine on 1 GPU; a distributed-efficiency caveat for multi-GPU). LR choice is a tuning judgment call — flag to Liming.

## Ghost relaunch: never edit a running script (2026-09-28, load-bearing)
At 01:37 run4's leg1 wrapper launched a GHOST duplicate SOAP run (run4b)
seconds after run4A completed cleanly (final 3.34559). Forensics: the
wrapper PID (age since 23:46 launch) was still alive with run4b's
apptainer as its current child; the log shows run4A's clean end
(max_memory line), then `run_container.sh: line 21:
.../.cache/hf: Is a directory`, then run4b's step 0. Root cause:
run_container.sh was EDITED at 23:56 (mtime), ten minutes after the
chain launched — the running bash had the old image buffered (with
duplicate launch lines + a stray line executing the hf dir) and blindly
executed it at 01:37. run1/run2 (single runs each) prove the harness
does not self-restart. RULE: never edit any script while a process may
be executing it — pgrep first; future chains log a wrapper checksum at
each leg start. Killed run4b 02:34 (SIGTERM worker+agent; chain has no
set -e so LEG1_EXIT:143 advanced to leg2 cleanly).
Companion footgun found during the kill: `pgrep -f <pattern>` matches
your own `sh -lc` wrapper (its cmdline contains the pattern literally)
— the kill suicided the shell after sending signals. Use self-excluding
patterns (`pgrep -f "train_gpt_soa[p]"`) and resolve PIDs before
killing; verify with nvidia-smi + leg logs after.

## Plain-SOAP lr cross-check vs nikhilvyas/modded-nanogpt-SOAP (2026-10-10, read-only)
Reference run.sh: lr=0.0018, warmup 250 iters, wd=0, betas (0.95,0.95),
precond_freq=10, 7000 iters, same 524288-token global batch as ours. Our
run4A guess (lr=0.002, no warmup, 3250 steps) is within 11% on lr — same
order, slightly hotter. No-warmup + shorter horizon remain HYPOTHESES
(not conclusions) for part of the +0.066 shortfall. No code copied: that
repo ships no LICENSE file, so it stays a read-only reference (gitignored).

## Newton-Muon first-forward OOM (2026-10-10, job 13945798)
NM smoke died at 1:07 in the first forward: 83.5/95GB used, 70.6GB
PyTorch-allocated — BEFORE any hook accumulation runs (refresh cadence
skips steps 1-9), so the hog is compile structure, not hook tensors.
Leading theory: 24 @torch.compiler.disable forward hooks fragment the
dynamo graph into ~25 cudagraph-eligible pieces; static per-fragment
pools + allocator fragmentation (11.5GB reserved-unallocated) blow the
96GB budget. One-shot fix: triton.cudagraphs=False in the NM variant +
PYTORCH_CUDA_ALLOC_CONF=expandable_segments in nm_smoke.sbatch (both
zero-experiment-change: last-bit numerics at most). If OOM persists,
deeper surgery is needed (selective-eager linears or attach/detach
scheduling) — flag for user, do not iterate blindly.
ckpt 13945668 (record-only): SECOND resume failure mode — fixed CPU RNG
line now passes, but set_rng_state_all chokes on the CUDA RNG states
(TypeError, same "must be a torch.ByteTensor"). Structure of
ck["cuda_rng"] after weights_only+map_location needs a debug print;
lane closed per loadout (fix resume only if a run is cut off).
