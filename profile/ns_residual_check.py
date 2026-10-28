"""NS orthogonalization residual: full momentum (M) vs projected residual (M_res).

Loads optimizer states from on-disk ckpts (no training loop, no dist, no
data): momentum buffers + COSMOS U/O subspaces are IN the ckpt. Runs the
harness Newton-Schulz loop (copied from the variants, parameterized by step
count) at 12/8/6 steps on M and M_res and reports the normalized
orthogonality residual, rectangular-aware:
  m >= n: ||X^T X - I||_F / sqrt(n)   else: ||X X^T - I||_F / sqrt(m)

Approximation (documented): uses the saved momentum buffer as M_hat (true
M_hat = 0.95*buf + 0.05*G needs grads, not stored). Spectra are nearly
identical late in training; the Exp-1 smoke logs the exact in-training
M_hat comparison at steps 50/100/150.

Usage (needs GPU, no torchrun):
  ./run_container.sh python profile/ns_residual_check.py \
    runs/run18_cosmos_r32/checkpoints/ckpt_02750.pt \
    runs/run16_muon_s1/checkpoints/ckpt_02750.pt
Second arg (Muon ckpt) optional.
"""
import sys

import torch

SETUP_MARKER = ("########################################\n"
                "#                Setup                 #\n"
                "########################################\n")


def load_variant(path):
    # Exec head (imports/arch, side-effect-free) plus the def-only optimizer
    # block for cosmos variants (spliced AFTER the Setup marker). Baseline
    # files have no optimizer code past the marker; the tail split is a no-op
    # guard that must still find the dist setup line.
    src = open(path).read()
    assert src.count(SETUP_MARKER) == 1, path
    head, rest = src.split(SETUP_MARKER)
    tail = rest.split("\ndevice = torch.device")[0]
    ns = {}
    exec(compile(head, path, "exec"), ns)
    exec(compile(tail, path, "exec"), ns)
    return ns


def ns_power(G, steps):
    # verbatim harness quintic NS (baseline zeropower_via_newtonschulz5),
    # parameterized by iteration count.
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.float()


def orth_resid(X):
    m, n = X.shape[-2], X.shape[-1]
    if m >= n:
        return ((X.mT @ X - torch.eye(n, device=X.device)).norm() / (n ** 0.5)).item()
    return ((X @ X.mT - torch.eye(m, device=X.device)).norm() / (m ** 0.5)).item()


def load_opt_states(variant_path, ckpt_path, opt_idx, model_ctor_args):
    """Return (params, state_list) with ckpt optimizer states installed."""
    ns = load_variant(variant_path)
    model = ns["GPT"](*model_ctor_args).cuda()
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"])
    params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    opt_cls = ns["COSMOS"] if "COSMOS" in ns else ns["Muon"]
    opt = opt_cls(params) if "COSMOS" not in ns else ns["COSMOS"](
        params, rank_o=32, rank_u=32)
    # state_dict load needs identical group structure; COSMOS/Muon sort
    # deterministically, and the list comp matches the variant exactly.
    opt.load_state_dict(ck["optims"][opt_idx])
    states = []
    for p in opt.param_groups[0]["params"]:
        st = opt.state[p]
        assert st["momentum"].shape == p.shape, "state/param misalignment"
        states.append((p.shape, st))
    return states


def main():
    assert torch.cuda.is_available()
    cpath = sys.argv[1]
    mpath = sys.argv[2] if len(sys.argv) > 2 else None
    ctor = (50304, 12, 768)
    co = load_opt_states("variants/train_gpt_cosmos_r32_seed0.py", cpath, 1, ctor)
    print(f"COSMOS ckpt: {cpath} ({len(co)} 2D params)")
    print(f"{'shape':>14} {'M@12':>10} {'M@8':>10} {'M@6':>10} "
          f"{'Mres@12':>10} {'Mres@8':>10} {'Mres@6':>10}")
    agg_m, agg_r = {12: [], 8: [], 6: []}, {12: [], 8: [], 6: []}
    seen = set()
    for (m, n), st in co:
        M = st["momentum"].float()
        U, O = st["U"].float(), st["O"].float()
        M_U = M @ U
        O_T_M = O.mT @ M
        M_res = M - O @ O_T_M - M_U @ U.mT + O @ (O_T_M @ U) @ U.mT
        row_m, row_r = [], []
        for s in (12, 8, 6):
            row_m.append(orth_resid(ns_power(M, s)))
            row_r.append(orth_resid(ns_power(M_res, s)))
            agg_m[s].append(row_m[-1])
            agg_r[s].append(row_r[-1])
        key = f"{m}x{n}"
        if key not in seen:
            seen.add(key)
            print(f"{key:>14} {row_m[0]:10.2e} {row_m[1]:10.2e} {row_m[2]:10.2e} "
                  f"{row_r[0]:10.2e} {row_r[1]:10.2e} {row_r[2]:10.2e}")
    print("mean-over-params:",
          " ".join(f"M@{s}={sum(agg_m[s]) / len(agg_m[s]):.2e}" for s in (12, 8, 6)),
          " ".join(f"Mr@{s}={sum(agg_r[s]) / len(agg_r[s]):.2e}" for s in (12, 8, 6)))
    if mpath:
        mu = load_opt_states("variants/train_gpt_baseline_seed0.py", mpath, 1, ctor)
        print(f"Muon ckpt: {mpath} ({len(mu)} 2D params)")
        agg = {12: [], 8: [], 6: []}
        for (m, n), st in mu:
            M = st["momentum"].float()
            for s in (12, 8, 6):
                agg[s].append(orth_resid(ns_power(M, s)))
        print("mean-over-params:",
              " ".join(f"M@{s}={sum(agg[s]) / len(agg[s]):.2e}" for s in (12, 8, 6)))


if __name__ == "__main__":
    main()
