# Muon vs SOAP-family optimizer tradeoffs (1-GPU and 8-GPU)

Track-3 optimizer comparisons on 1 GPU plus paired 8-GPU experiments
and an analytical distributed-cost model, asking whether curvature
information can buy step-efficiency at Muon-like cost. Rig: RTX PRO
6000 Blackwell 96GB (PACE Phoenix). Run ledger: `runs.csv`. Methods
and provenance: `method_notes.md`. Last updated 2026-11-24 (runs 1–96).

"COSMOS (track-3 variant)" in this file means the optimizer as adapted
in the upstream fork: 4-block subspace Sinkhorn + Newton-Schulz on the
residual + NorMuon second-moment update — not COSMOS in general. No
"Pareto winner" language: 8-GPU wall-clock is flat and 1-GPU is ~1.5%
slower than Muon; the claim is a step-count edge at ~cost parity.

At rank 32 the variant's edge over Muon is real (five paired seeds)
and comes from the NorMuon update: with the subspace removed, the
final loss is unchanged within noise (Sec. 2.1). Higher ranks do add
to it: rank 256, rank 512 and full-rank SinkSOAP all beat the rank-0
control, and the gain grows with rank, but at this scale each costs
more step time than it saves. The subspace and NorMuon substitute for
each other: NorMuon's gain shrinks as rank grows (Sec. 2.4).

## 1. Setup

- Harness: modded-nanoGPT Track 3 (@bc3a0c2), 124M GPT, 3250 steps,
  global batch 524288 tokens (8 microbatches x 64 with grad accumulation;
  exact data-parallel math), stable then linear cooldown
  (cooldown_frac=0.7), validation every 125 steps (every 25 after 2875).
- Protocol: run1/run2 are unseeded pilots. All later runs are seed-paired
  (`torch.manual_seed` + `cuda.manual_seed_all` at trial start) and swap
  ONLY the 2D block-parameter optimizer against the seeded Muon control
  (run5). 8-GPU runs use the same 3250-step protocol and LR schedule,
  except the 500-step probes (runs 20/21/26/55/56): their LR cools
  from step 150, so they compare only with each other, or with longer
  runs before step 150.
- Per-step times are 1-GPU measurements unless marked 8-GPU (Sec. 3
  states the bound this is). 8-GPU wall-clock is compared only within
  one node (Sec. 6).

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
(Sec. 5 for the noise floor this sits on). Run4 is one high-spread draw.

8-GPU (paired seeds, same 3250-step protocol; node names in `runs.csv`):

| run | method | seed | first obs <=3.28 | final |
|---|---|---|---|---|
| run22/24 | Muon NS12 | 0/1 | 3225/3250 | 3.27790/3.27978 |
| run23/25 | COSMOS-r32 NS12 | 0/1 | 3200/3200 | 3.27635/3.27690 |
| run27/28 | Muon 150 steps then COSMOS | 0/1 | 3175/3200 | 3.27504/3.27673 |
| run29 | Muon NS8 | 0 | 3250 | 3.27939 |
| run30 | Muon NS8 | 1 | — | 3.28071 |
| run31/32 | Muon NS6 | 0/1 | — | 3.29083/3.29012 |
| run33/34 | COSMOS NS8 | 0/1 | 3175/3175 | 3.27500/3.27507 |
| run35/36 | COSMOS NS6 | 0/1 | 3225/3200 | 3.27832/3.27628 |
| run39/40 | Muon at COSMOS hypers | 0/1 | 3225/— | 3.27934/3.28142 |
| run41/42 | COSMOS at Muon hypers | 0/1 | 3225/3200 | 3.27899/3.27610 |
| run37/38 | COSMOS-r32 + NorMuon on square matrices | 0/1 | 3200/3200 | 3.27645/3.27771 |

Reading the 8-GPU rows: the warmup (runs 27/28) lands within Muon seed
spread of the COSMOS baselines — no detectable effect, no harm. At NS8
all four finals stay within 0.002 of their NS12 controls; Muon seed 1
(run30) misses the crossing at 3.28071, +0.0009 above its NS12 run.
NorMuon on the square matrices (runs 37/38) changes nothing
(+0.0001/+0.0008, same crossings). At NS6 they split: COSMOS crosses both seeds (3225/3200, within
+0.002 of NS12), Muon crosses neither (+0.010–0.013). In the
hyperparameter swap, Muon at COSMOS hypers never matches COSMOS (s1 is
worse than Muon at its own hypers), while COSMOS at Muon hypers ties
Muon at s0 and beats it by 50 steps at s1. SinkSOAP 500-step probe:
0.64 s/step (~1.6x Muon-8GPU).

### 2.1 Attribution: subspace vs NorMuon update (runs 57–69)

The COSMOS track-3 variant differs from Muon in two ways at once: the
tracked rank-r subspace (QR + compressed Sinkhorn + NS on the
residual) and the NorMuon second-moment update on the rectangular
(MLP) matrices. The rank-0 control keeps the variant's code path with
the subspace removed: Nesterov momentum, NS12 on the full momentum,
rescale to the Muon-level Frobenius norm (a no-op once NS has
converged), NorMuon on the rectangular matrices. No QR, no Sinkhorn.
`research_stack/test_variant_math_cpu.py` checks it against the
verbatim update called at rank 0.

Three arms, seeds 0–4, 8 GPUs, 3250 steps. Muon runs at lr .025 /
wd .05; the control and COSMOS-r32 at lr .04 / wd .025.

| arm | runs | final val loss, seeds 0–4 | mean | first obs <=3.28 |
|---|---|---|---|---|
| Muon | 22, 24, 61, 62, 63 | 3.27790 3.27978 3.27917 3.27778 3.27959 | 3.27884 | 3225 3250 3225 3225 3250 |
| rank-0 control | 57, 58, 64, 65, 66 | 3.27638 3.27691 3.27531 3.27688 3.27736 | 3.27657 | 3200 3200 3175 3200 3200 |
| COSMOS-r32 | 23, 25, 67, 68, 69 | 3.27635 3.27690 3.27580 3.27553 3.27680 | 3.27628 | 3200 3200 3175 3175 3200 |

Seed-paired differences in final loss (`runs/attribution_stats.py`):

| comparison | mean | se | paired t, one-sided | 95% CI |
|---|---|---|---|---|
| Muon − COSMOS-r32 | +0.00257 | 0.00031 | p = 0.0006 | +0.0017 to +0.0034 |
| Muon − control | +0.00228 | 0.00052 | p = 0.0058 | +0.0008 to +0.0037 |
| control − COSMOS-r32 | +0.00029 | 0.00031 | p = 0.20 | −0.0006 to +0.0012 |

Decision rule, fixed before the runs: the subspace contributes if
control − r32 > 0 at p < 0.05; NorMuon explains the edge if
|control − r32| < 0.001 and the control beats Muon at p < 0.05. The
second branch holds. The control reproduces 89% of the r32 edge; the
subspace's share is 0.0003, with an interval that includes zero and
tops out at 0.0012. With five seeds the exact sign-flip p-value cannot
go below 1/32, which both Muon comparisons reach.

What the subspace does do: it leads the control early, and the lead
decays through training. Mean control − r32 gap: +0.099 at step 125,
+0.017 at 500, +0.005 at 1000, +0.0016 at 2000, +0.0004 at 2500. Muon
leads both at step 125 (by 0.076 over r32) and trails from step 500 on.

At Muon's hyperparameters (lr .025 / wd .05, seeds 0/1, runs 59/60)
the control crosses at 3175/3200 with finals 3.27439/3.27750: ahead of
Muon by 0.0035/0.0023 and 50 steps at both seeds, so its edge does not
depend on COSMOS's lr/wd. COSMOS-r32 at the same hyperparameters (runs
41/42) finished at 3.27899/3.27610.

Scope: rank 32, 124M parameters, one benchmark. Ranks above 32 and
COSMOS without the NorMuon update were tested afterwards (Sec. 2.4):
higher ranks beat the control, and at rank 32 the subspace is not
inert but substitutes for NorMuon.

### 2.2 Subspace upkeep: QR cadence and Sinkhorn iterations (runs 43–54)

Each cell is first obs <=3.28 / final, with the change against the
seed-paired COSMOS-r32 baseline (QR every step, 10 Sinkhorn
iterations; runs 23/25).

| variant | seed 0 | seed 1 |
|---|---|---|
| QR every 2 steps | 3225 / 3.27810 (+25, +0.0018) | 3225 / 3.27916 (+25, +0.0023) |
| QR every 4 steps | 3225 / 3.27864 (+25, +0.0023) | 3200 / 3.27665 (0, −0.0003) |
| QR every 8 steps | 3200 / 3.27794 (0, +0.0016) | 3200 / 3.27748 (0, +0.0006) |
| Sinkhorn 5 iterations | 3175 / 3.27608 (−25, −0.0003) | 3200 / 3.27756 (0, +0.0007) |
| Sinkhorn 3 iterations | 3200 / 3.27764 (0, +0.0013) | 3200 / 3.27727 (0, +0.0004) |
| Sinkhorn 1 iteration | 3225 / 3.27870 (+25, +0.0024) | 3225 / 3.27928 (+25, +0.0024) |

QR cadence: five of six finals sit above baseline, but the shift is
not ordered in k (every-8 is the closest), so +0.001–0.002 is at the
edge of same-seed rerun noise (runs 37/38 moved +0.0001/+0.0008 with
no real change). A refresh every 8 steps starts behind (step 125:
4.77 vs 4.67) and recovers. Sinkhorn: 5 and 3 iterations are within
noise; 1 iteration costs 0.0024 and 25 steps at both seeds.

Step time on one node: QR every 8 steps 398.7 ms against 401.7 ms
(−0.7%); 1 Sinkhorn iteration 401.0 ms (−0.2%). The whole subspace
path costs 1.4% of step time at rank 32 (Sec. 3), which bounds what
any of these knobs can save. The rank-0 control drops both components
and matches the final loss, so at this rank they do not matter for
cost.

### 2.3 Newton-Schulz tolerance: what the probes say

Runs 31–36: COSMOS-r32 holds at 6 NS iterations where Muon loses
0.010–0.013. Candidate explanation (deflation): removing the dominant
directions shrinks the residual's Frobenius norm, so after Frobenius
normalization the remaining singular values start larger and need
fewer iterations.

Offline check of the premise (`runs/deflation_spectra_scratch.py`,
log `runs/deflspec_run.log`; step-2750 checkpoints, saved momentum
buffer as the proxy for the NS input, fp64 singular-value polynomials):
- Deflation removes exactly 32 directions, shrinks the Frobenius norm
  to 0.57–0.69x, and raises the normalized remaining singular values
  1.3–1.6x (smallest nonzero, 10th percentile and mean; every shape).
- On the nonzero support the residual orthogonalizes better than the
  full momentum at 6 iterations: error 0.19 vs 0.34 (MLP fc), 0.16 vs
  0.34 (MLP proj), 0.53 vs 0.64 (attention).
- The earlier reading that the residual orthogonalizes worse was an
  artifact: the metric counted the 32 exactly-zero deflated directions
  as misses (0.2041 = sqrt(32/768) at NS12).

So the premise holds on this proxy. Whether it explains the end-to-end
split is a separate question. `runs/ns_norm_shape_scratch.py` (log
`runs/nsnorm_run.log`) shows that a spectral-norm start would
orthogonalize Muon's momentum at 6 iterations better than deflation
does for COSMOS: share of directions above 0.9 is 98/95/63%
(fc/proj/attention) against 75/79/50%, with Frobenius-normalized Muon
at 57/65/34%. If input orthogonalization were the whole story, Muon
with a spectral-norm start should recover. Runs 55/56 tested that and
do not settle it:
- They are 500-step runs, so their LR cools from step 150 and their
  step-500 loss is not comparable to the 3250-step NS6/NS12 runs. This
  was a design error in the probe.
- At step 125, where the schedules agree: spectral-norm NS6
  4.758/4.752, Frobenius NS6 4.785/4.779, NS12 4.587/4.599 — 13% and
  15% of the gap recovered.
- Against the one same-schedule reference (500-step NS12 Muon, seed 0,
  run20): +0.035 at step 500.

Reading: little early rescue. Either the late-checkpoint proxy does
not describe early training, where NS6 does most of its damage, or
the tolerance comes from something else the COSMOS path has and Muon
lacks: the fixed-norm rescale (Muon applies the raw NS output, which
at 6 iterations is 9–29% short of nominal norm on this proxy) or the
NorMuon update. The rank-0 control has both and no deflation, so the
control at 6 iterations separates these, and a full-length
spectral-norm run gives the missing final-loss comparison. Neither has
been run.

### 2.4 Rank against the control, and NorMuon switched off (runs 70–96)

Seeds 0–2, 8 GPUs, 3250 steps. Every cell runs at lr .04 / wd .025,
not re-tuned per cell. Decision rules were fixed before submission;
numbers from `runs/rank_normuon_stats.py`. "NorMuon off" skips only the
second-moment step on the rectangular matrices (SinkSOAP: its own
`apply_normuon=False`); `research_stack/test_variant_math_cpu.py`
checks each switch against its base script.

Final val loss, mean of seeds 0–2 (first observed step <= 3.28):

| subspace | NorMuon on | NorMuon off |
|---|---|---|
| rank 0 | 3.27620 (3200 3200 3175) | 3.28388 (none) |
| rank 32 | 3.27635 (3200 3200 3175) | 3.28148 (— — 3250) |
| rank 256 | 3.27411 (3175 3175 3150) | not run |
| rank 512 | 3.27098 (3125 3125 3125) | 3.27473 (3175 3175 3175) |
| full (SinkSOAP) | 3.26845 (3100 3125 3100) | 3.26954 (3100 3125 3125) |

Muon at its own lr/wd, seeds 0–2: 3.27895 (3225 3250 3225).

Rank against the control (rule: beats it if control − X ≥ 0.0015 on
average, positive at every seed, paired t one-sided p < 0.05):

| X | control − X per seed | mean | p | 95% CI | verdict |
|---|---|---|---|---|---|
| rank 256 | +0.00222 +0.00230 +0.00175 | +0.0021 | 0.003 | +0.0014 to +0.0028 | beats |
| rank 512 | +0.00536 +0.00576 +0.00454 | +0.0052 | 0.002 | +0.0037 to +0.0068 | beats |
| SinkSOAP | +0.00933 +0.00709 +0.00682 | +0.0078 | 0.005 | +0.0043 to +0.0112 | beats |

The rank-32 null does not extend upward. Quality improves steadily
from rank 256 on, and full rank is best; rank 32 still ties the
control. On 1 GPU, ranks 64 and 128 (single runs) sat at rank 32's
level, which puts the start of the gain between 128 and 256, on thin
evidence.

Subspace and NorMuon together (per-seed differences, seeds 0–2):

| | rank 32 | rank 512 | full |
|---|---|---|---|
| NorMuon's gain, r(off) − r(on) | +0.0051 | +0.0038 | +0.0011 (p = 0.05) |
| subspace gain over rank 0, NorMuon off | +0.0024 | +0.0091 | +0.0143 |
| subspace gain over rank 0, NorMuon on | −0.0002 | +0.0052 | +0.0078 |
| interaction (off − on) | +0.0025 | +0.0039 | +0.0066 |

At rank 0, NorMuon's gain is +0.0077. The interaction is positive at
every seed and rank, which the pre-set rule calls substitutes: each
removes part of the gain the other would give. At full rank, NorMuon
adds about 0.001. Rank 32 is not inert: without NorMuon it gains 0.0024
over rank 0 (p = 0.04, 95% CI −0.0008 to +0.0055).

Sanity gate failed. Rank 0 with NorMuon off was meant to reproduce
Muon at the same lr/wd (runs 39/40). It is 0.0040 / 0.0037 worse at
seeds 0/1 (band ±0.0015) and never reaches 3.28. The two scripts
differ only in the update function, and on CPU the update directions
agree (cosine ≥ 0.999999). The remaining difference is COSMOS's rescale
of every update to the nominal Muon Frobenius norm, where Muon applies
the raw NS output, which falls short of nominal when NS has not fully
converged (Sec. 2.3). Hypothesis, untested: the rescale gives larger
steps at an lr already above Muon's best (Muon itself loses 0.0015 going
from its own lr/wd, .025/.05, to .04/.025). So the NorMuon-off cells are the COSMOS code path
with one switch off, not plain Muon, and run at an lr chosen for the
NorMuon-on version. NorMuon's shrinking gain across ranks 32, 512 and
full does not use the failed cell. All cells share one lr, though, so
the size of the interaction may move under per-cell tuning.

Where the rank is needed (seeds 0/1, descriptive): rank 32 everywhere
with rank 512 on one group of sides. Precondition met (r32 − r512 =
+0.0053 / +0.0058). Share of the rank-512 gain recovered:

| rank 512 on | finals s0 / s1 | share s0 / s1 | mean share | step time vs Muon-cost reference |
|---|---|---|---|---|
| attention matrices (768×768) | 3.27674 / 3.27496 | −0.07 / +0.34 | 0.13 | +7.1% (node 020-2-0) |
| MLP, 768-wide side | 3.27455 / 3.27457 | +0.34 / +0.41 | 0.37 | +3.6% (node 014-23-0) |
| MLP, 3072-wide side | 3.27583 / 3.27499 | +0.10 / +0.33 | 0.21 | +3.5% / +4.0% (both nodes) |
| all sides (uniform rank 512) | 3.27102 / 3.27115 | 1 | 1 | +10.9% / +14.4% |

No group carries the gain alone (pre-set bar: share ≥ 0.5); the shares
sum to 0.72, at the low edge of the 0.7–1.3 band that reads as "gains
add across groups". The 768-wide side gives the most gain for the least
cost: rank 32 itself costs +1.6% on node 014-23-0, so that side adds
about 2 points of step time for 37% of the gain, against 13 points for
uniform rank 512.

Cost and time to target (8 GPUs, same node only; steady step time over
steps 1000–3250, relative to Muon):

| arm | node 020-2-0 (Muon 396 ms) | node 014-23-0 (rank-0 off, 299 ms) | added ms | s to 3.28, node 020-2-0 |
|---|---|---|---|---|
| Muon | — | — | — | 1309, 1310, 1319 |
| rank-0 control | +0.1% | — | ~0 | 1292, 1302 (1331 cold compile) |
| rank 32 | +1.4% | +1.6% | ~5 | 1310 |
| rank 256 | +5.5% | +7.4% | ~21–22 | 1362, 1396 |
| rank 512 | +10.9% | +14.4% | ~43 | 1408 |
| SinkSOAP | +63.8% | +85.9% | ~253–257 | 2014 |

Node 014-23-0 had no Muon run; rank 0 with NorMuon off runs at Muon's
cost there and serves as the reference. The added milliseconds match
across the two nodes; only the rest of the step differs. At this scale,
no subspace setting reaches 3.28 sooner in wall-clock than Muon or the
control. As a rough break-even, suppose the added milliseconds stayed
fixed while the rest of the step grew. They will not exactly, since QR
and eigh costs grow with width. Rank 512 would overtake the control
once the rest of the step takes about 2 s (here 0.3–0.4 s), and
SinkSOAP at about 9 s.

## 3. Per-step cost (measured, 1x Blackwell unless marked)

| method | s/step | note |
|---|---|---|
| Muon | 1.82–1.86 | run1/run5/run16/run17 |
| SinkSOAP (eigh every step) | 3.75 | 96–98% eigh; Sinkhorn ~0.5ms |
| amort SinkSOAP (freq=10/T=1) | 1.95 | ~1.8 normal, ~3.7 refresh steps |
| COSMOS (rank-256 QR) | 1.99–2.04 | run6/run9/run10 |
| COSMOS r32/r64/r128/r512 | 1.89/1.93/1.90/2.15 | rank sweep, seed0 |
| SOAP-QR (eigh once + track) | 1.89 | full eigh at init only |
| Newton-Muon port | 1.96 | run7 |
| 8-GPU Muon / COSMOS-r32 | 0.39–0.42 | same-node steady-state; SinkSOAP 0.64 |
| 8-GPU Muon / rank-0 control / COSMOS-r32 | 0.3961 / 0.3963 / 0.4017 | node `atl1-1-03-020-2-0`, 2026-11-15, steps 1000–3250; 3 / 5 / 3 runs |
| 8-GPU COSMOS-r256 / r512 / SinkSOAP | 0.4179 / 0.4394 / 0.6487 | same node, 2026-11-23; +5.5% / +10.9% / +63.8% vs Muon (Sec. 2.4) |

Cost-model closure: microbenchmark-predicted SinkSOAP overhead (1.88s)
matches the measured run2−run1 delta (1.92s). The 2x SinkSOAP slowdown
is an implementation choice (unamortized eigh), not an inherent cost —
but run8 shows aggressive amortization can erase the quality edge
(Sec. 8). 1-GPU timing flatters QR-SOAP (hides distributed sync) and
punishes unamortized SinkSOAP (shards ÷W on multi-GPU); treat these as
1-GPU bounds. NS count barely moves step time (~2% NS8-vs-NS12 within
one node): Newton-Schulz iterations are a small cost fraction.
At 8 GPUs the rank-0 control runs at Muon's step time (+0.03%) and
COSMOS-r32 at +1.4%. Seconds to reach 3.28 on that node: Muon
1309–1319, control 1292–1302 (leaving out the batch's first job, which
paid a cold compile cache), COSMOS-r32 1310–1320.

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
| 12 | 0 (5-dec) | 7e-5 | ~0.15 (4 runs, high spread) |
| 125–130 | 1.9e-4 | 4.8e-3 | ~0.16 (persists) |

Measured: `torch.linalg.eigh` is bit-identical 20x in-process + 3x
cross-process on CUDA, including rank-deficient 3072 grams
(`research_stack/test_eigh_determinism.py`, job 13848105). fp64
per-step traces show both methods receive last-bit harness flips at
step 1 (consistent with CUDA atomic races in compiled backward);
Muon contracts them below print precision in one step while SinkSOAP
amplifies ~360x. Hypothesized mechanism (unproven): small Gram
eigengaps amplify via Davis-Kahan; SOAP-QR spread via low-rank Grams,
near-tied eigenvalues, and argsort/QR divergence. Decisive tests
pending (tolerance-quantized sort, deterministic rerun, autotune
check).
Practical consequence: ±5e-3 loss jitter maps to ~±70 crossing steps
near target, so single-seed gaps near that floor are inconclusive,
and SOAP-QR needs multi-seed spread before any crossing claim.

## 6. Distributed cost (8-GPU measured + analytical)

Two regimes, with different bottlenecks:
- Data-parallel + optimizer param sharding (this harness). Each rank
  owns whole 2D matrices, so eigh/QR is rank-local compute; per-step
  comm matches Muon's (gradient allreduce + parameter all_gather).
  Measured at 8 GPUs (single node, PCIe-only, no NVLink): Muon and
  COSMOS-r32 both ~0.40 s/step (within 1–3% on one node; runs 22–26
  share `atl1-1-03-020-6-0`); SinkSOAP 0.64 s/step. The 2026-11-15
  batch on `atl1-1-03-020-2-0` puts COSMOS-r32 at +1.4% and the rank-0
  control at +0.03% (Sec. 3).
- Tensor parallelism (single 2D weights sharded across GPUs).
  Assembling each mode's Gram matrix needs a cross-rank AllReduce over
  the sharded dimension, and the eigendecomposition becomes a
  distributed eigensolve — cubic work that parallelizes poorly plus
  fine-grained synchronization. Muon's Newton-Schulz iteration is
  structurally local here (no Gram eigenbasis needed). Untested.
8-GPU wall-clock is node-linked (~30% spread across nodes measured in
the November runs: ~0.30 vs ~0.39 s/step regardless of method), so
step times are compared only within one node. Step-count and loss
comparisons stand across nodes.
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
  Sinkhorn blocks + NorMuon update. Beats Muon by 25–50 steps and
  0.0026 final loss at five paired 8-GPU seeds (r32), and by 50–100
  steps at r256 on 1 GPU (3 paired seeds); quality nearly flat in rank
  (r32 ≈ r256). The rank-0 control attributes the r32 edge to the
  NorMuon update: removing the subspace moves the final loss by
  +0.0003 (within noise) and removes the 1.4% step-time overhead. The
  subspace gives an early lead that is gone by step 2500. The edge is
  not hyperparameter tuning (swap runs; the control also beats Muon at
  Muon's lr/wd). Tolerates NS6 where Muon collapses; cause open
  (Sec. 2.3). Wall-clock flat-to-slower than Muon at measured scales.
- Rank-0 control (NorMuon only): Muon plus the NorMuon second-moment
  update on the rectangular matrices, through the COSMOS code path.
  Matches COSMOS-r32 on final loss at Muon's step time.

## 8. Claim register

- [measured] Profiler closure: 1.88s predicted vs 1.92s observed;
  SinkSOAP overhead ~97% eigh.
- [measured, low-confidence] SinkSOAP crossed ~100 steps before Muon
  (3100 vs 3200); gap is near the ±70-step noise floor, single
  unseeded draws — suggestive, not conclusive.
- [measured] COSMOS-256 edge over Muon, 3 seeds each: 3150/3150/3175
  vs 3225/3250/3225, no overlap.
- [measured] 1-GPU rank sweep (single runs): r32/r64/r128 cross
  3175/3200/3200 vs r256 3150–3175; r512 3125 (final 3.27021).
  Superseded above rank 128 by the 8-GPU result below.
- [measured] 8-GPU, 3 paired seeds: rank 256, rank 512 and SinkSOAP
  beat the rank-0 control by 0.0021 / 0.0052 / 0.0078 final loss
  (p = 0.003 / 0.002 / 0.005), at +5.5% / +10.9% / +63.8% step time.
  None reaches 3.28 sooner than Muon or the control in wall-clock.
- [measured] Subspace and NorMuon substitute: NorMuon's gain is
  0.0051 / 0.0038 / 0.0011 at rank 32 / 512 / full, and the subspace
  gains more with NorMuon off than on at every rank (interaction
  +0.0025 / +0.0039 / +0.0066, positive at every seed). One lr for
  all cells.
- [measured, failed check] Rank 0 with NorMuon off is 0.004 worse than
  Muon at the same lr/wd (seeds 0/1). [hypothesized] COSMOS's rescale
  to the nominal Muon norm is the cause.
- [measured, n=2] Rank 512 on the 768-wide side of the MLP matrices
  alone recovers 37% of the uniform-rank-512 gain at +3.6% step time;
  attention 13%, 3072-wide side 21%.
- [measured] Amortized SinkSOAP (freq=10/T=1): 1.95s/step, NO crossing,
  final 3.29182. T exonerated (run12 T=10 also no-cross); warm-start
  recovers (run13 crosses 3125) — the failure is cold-start staleness.
- [measured] SOAP-QR (seed0, lr=0.002, no warmup): no crossing, 3.34559.
- [measured] SOAP reference recipe (lr=0.0018 + warmup): no crossing,
  3.30265 (run11) — closer, but recipe is not the fix.
- [measured] Same-seed SOAP-QR @875 differs by 0.037 (run3 vs run4A).
- [measured] eigh bit-identical 20x + 3x on CUDA (job 13848105).
- [measured] 8-GPU, 5 paired seeds: COSMOS-r32 beats Muon by 0.0026
  final loss (paired t, one-sided p = 0.0006) and 25–50 steps at every
  seed, at +1.4% step time; SinkSOAP 1.6x/step. PCIe-only node.
- [measured] That edge is the NorMuon update, not the subspace: the
  rank-0 control matches COSMOS-r32 (control − r32 = +0.0003, 95% CI
  −0.0006 to +0.0012) and beats Muon by 0.0023 (p = 0.006) at Muon's
  step time. Rank 32 only (higher ranks: next entries).
- [measured] The subspace gives an early lead over the control (0.10
  at step 125) that decays to zero by step 2500.
- [measured] The rank-0 control at Muon's lr/wd beats Muon at both
  seeds (0.0035/0.0023, 50 steps each).
- [measured] QR refresh every 2/4/8 steps: 0 to +0.0023 final, not
  ordered in k. Sinkhorn 5/3 iterations within noise; 1 iteration
  +0.0024 at both seeds. Step-time savings under 1%.
- [measured] NorMuon on square matrices (runs 37/38): no detectable
  effect (+0.0001/+0.0008, same crossings).
- [measured] Newton-Muon port (v2): no crossing, 3.31761. Footnote only.
- [measured] 150-step Muon warmup then COSMOS (runs 27/28): no
  detectable effect, no harm — all diffs within Muon seed spread.
- [measured] NS8 ≈ NS12 on final loss (all four within 0.002); Muon
  seed 1 misses the crossing at 3.28071. NS6 splits: COSMOS holds (3225/3200, within +0.002
  of NS12), Muon collapses (no crossing, +0.010–0.013). Interaction,
  not uniform.
- [measured] Hyperparameter swap rules out tuning as the full
  explanation: Muon at COSMOS hypers never matches COSMOS (seed 1 is
  worse than Muon at its own hypers).
- [measured, low-confidence] COSMOS ahead-or-tied in all 4
  matched-hyperparameter comparisons, small margins. Paired stats
  (n=4): sign test on crossings 3/3, one-sided p=0.125; paired t on
  finals t=2.03, one-sided p~0.07 — suggestive, not significant.
  Superseded at default hyperparameters by the 5-seed result above.
- [measured, corrected] Offline NS-residual check: the earlier
  "residual orthogonalizes worse" reading was an artifact of counting
  the 32 deflated (exactly zero) directions as misses. On the nonzero
  support the residual orthogonalizes better than the full momentum at
  6/8/12 iterations, and deflation raises the normalized remaining
  singular values 1.3–1.6x (one late checkpoint, buffer proxy).
- [inconclusive] Spectral-norm start for Muon at NS6 (runs 55/56):
  13–15% of the gap recovered at step 125; the step-500 comparison is
  invalid (500-step schedule). Cause of the NS6 split still open.
- [measured] 8-GPU step time is node-linked (~30% spread across nodes);
  wall-clock compared only within one node. Runs 22–26 share one node.
- [hypothesized] SOAP-QR spread mechanism (low-rank Grams, near-tied
  eigenvalues, argsort/QR divergence). Decisive battery pending.
- [hypothesized] Noise-robustness axis (NS contraction vs eigengap
  amplification); supported by fp64 traces, needs tensor-level proof.
- [untested] Rank-0 control at NS6; COSMOS lr/wd tuning per cell;
  ranks between 128 and 256; unequal ranks beyond one-group-at-512;
  larger models.
- [analytical] FSDP-vs-TP distributed-cost split; width scaling table.

## 9. Open questions

- The failed gate: does COSMOS's fixed-norm rescale cost 0.004 at
  lr .04? Rank 0 with NorMuon off at Muon's lr/wd, against runs 22/24,
  would say.
- Per-cell lr: the NorMuon-off cells run at an lr chosen for the
  NorMuon-on version. Does the substitute interaction survive tuning?
- Unequal ranks: the 768-wide side of the MLP matrices gives the most
  gain per unit of cost. How far does a high short-side rank with a low
  long-side rank get towards uniform rank 512?
- Larger scale: the added milliseconds are fixed per step, so a model
  with a slower forward/backward pass shrinks their share. Does the
  rank gain hold, and does it start paying for itself, past 124M
  parameters?
- NS6 tolerance: the rank-0 control at 6 iterations, and
  spectral-norm Muon at full length (Sec. 2.3).
- COSMOS hyperparameters (lr=0.04/wd=0.025) are untuned.
- SOAP-QR spread mechanism: the low-rank-Gram hypothesis needs the
  decisive battery (tolerance-quantized sort, deterministic rerun,
  autotune check).
