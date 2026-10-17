"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from research_stack.ckpt import (CKPT_EVERY_DEFAULT, arm_preemption_handlers,
                                    load_ckpt, parse_resume_argv, save_ckpt,
                                    term_requested)


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
########################################
# MIT-licensed portions (c) 2024 Keller Jordan, via lliu606/Modded-NanoGPT-Private:
# COSMOS optimizer block. See NOTICE.
# COSMOS optimizer (spliced 2026-10-04 from upstream public fork
# github.com/lliu606/Modded-NanoGPT-Private — "Private" is just a name):
# Modded-NanoGPT-Private/records/track_3_optimization/train_gpt_simple.py
# compressed_sinkhorn_coeffs + cosmos_normuon_update VERBATIM;
# COSMOS class verbatim EXCEPT dist calls guarded for uninitialized-dist
# robustness (no-op under torchrun). Harness = OUR seeded Sep baseline
# (3250 steps, seed0, Sep optimizer1 lr 0.7/0.004/0.015) — NOT upstream's
# May harness (3225 steps, unseeded, optimizer1 0.3/1/320/0.01): isolates
# the 2D-optimizer effect vs run5, same protocol as run2's SinkSOAP splice.
# Wiring: lr=0.04, wd=0.025, rank_o=rank_u=256 (upstream track-3 values).
########################################


@torch.compile
def compressed_sinkhorn_coeffs(
    A11,      # [..., r_o, r_u]
    s12,      # [..., r_o]
    s21,      # [..., r_u]
    S22,      # [...] or [..., 1, 1]
    m: int,
    n: int,
    steps: int = 10,
    gamma: float = 1.0,
    eps: float = 1e-10,
):
    """
    Compressed Sinkhorn on energy matrix:

        C = [[A11^2, s12],
             [s21^T, S22]]

    Returns coefficients for update A, not energy:
        rt = x_top^(gamma/2)
        ct = y_top^(gamma/2)
        r0 = x_res^(gamma/2)
        c0 = y_res^(gamma/2)
    """
    B11 = A11.square()

    *batch, ro, ru = A11.shape
    device = A11.device
    dtype = A11.dtype

    S22 = S22.reshape(*batch, 1, 1)

    top = torch.cat(
        [B11, s12.unsqueeze(-1)],
        dim=-1,
    )  # [..., ro, ru + 1]

    bottom = torch.cat(
        [s21.unsqueeze(-2), S22],
        dim=-1,
    )  # [..., 1, ru + 1]

    C = torch.cat([top, bottom], dim=-2).clamp_min(eps)
    # [..., ro + 1, ru + 1]

    # Dimension-proportional marginals.
    a = torch.empty(ro + 1, device=device, dtype=dtype)
    a[:ro] = 1.0 / m
    a[ro] = max(m - ro, 0) / m

    b = torch.empty(ru + 1, device=device, dtype=dtype)
    b[:ru] = 1.0 / n
    b[ru] = max(n - ru, 0) / n

    x = torch.ones(*batch, ro + 1, device=device, dtype=dtype)
    y = torch.ones(*batch, ru + 1, device=device, dtype=dtype)

    for _ in range(steps):
        x = a / ((C @ y.unsqueeze(-1)).squeeze(-1) + eps)
        y = b / ((C.transpose(-2, -1) @ x.unsqueeze(-1)).squeeze(-1) + eps)

    # No coefficient clipping.
    x = x.clamp_min(eps).pow(gamma / 2)
    y = y.clamp_min(eps).pow(gamma / 2)

    rt = x[..., :ro]
    r0 = x[..., ro]

    ct = y[..., :ru]
    c0 = y[..., ru]

    return rt, ct, r0, c0


@torch.compile
def cosmos_normuon_update(
    grad,
    momentum,
    second_momentum,
    U,
    O,
    S,
    R,
    step,
    mu=0.95,
    beta2=0.95,
    eps=1e-10,
    nesterov=True,
):
    # ------------------------------------------------------------------
    # Original Normuon momentum
    # ------------------------------------------------------------------
    momentum.lerp_(grad, 1 - mu)
    M_hat = grad.lerp(momentum, mu) if nesterov else momentum

    G = grad.float()
    M_hat = M_hat.float()

    U_f = U.float()  # n x r_u
    O_f = O.float()  # m x r_o
    S_f = S.float()  # r_u x r_u
    R_f = R.float()  # r_o x r_o

    m = grad.size(-2)
    n = grad.size(-1)

    # ------------------------------------------------------------------
    # COSMOS subspace update, allowing rank_o != rank_u
    # ------------------------------------------------------------------

    # U_t <- QR(beta2 U_{t-1} S_{t-1}
    #           + (1-beta2) G_t^T G_t U_{t-1})
    GU = G @ U_f
    U_new, _ = torch.linalg.qr(
        beta2 * (U_f @ S_f)
        + (1 - beta2) * (G.transpose(-2, -1) @ GU),
        mode="reduced",
    )

    # O_t <- QR(beta2 O_{t-1} R_{t-1}
    #           + (1-beta2) G_t G_t^T O_{t-1})
    GtO = G.transpose(-2, -1) @ O_f
    O_new, _ = torch.linalg.qr(
        beta2 * (O_f @ R_f)
        + (1 - beta2) * (G @ GtO),
        mode="reduced",
    )

    # S_t <- U_t^T (beta2 U S U^T + (1-beta2) G^T G) U_t
    U_old_T_U_new = U_f.transpose(-2, -1) @ U_new
    G_U_new = G @ U_new
    S_new = (
        beta2 * (U_new.transpose(-2, -1) @ U_f @ S_f @ U_old_T_U_new)
        + (1 - beta2) * (G_U_new.transpose(-2, -1) @ G_U_new)
    )

    # R_t <- O_t^T (beta2 O R O^T + (1-beta2) G G^T) O_t
    O_old_T_O_new = O_f.transpose(-2, -1) @ O_new
    Gt_O_new = G.transpose(-2, -1) @ O_new
    R_new = (
        beta2 * (O_new.transpose(-2, -1) @ O_f @ R_f @ O_old_T_O_new)
        + (1 - beta2) * (Gt_O_new.transpose(-2, -1) @ Gt_O_new)
    )


    # ------------------------------------------------------------------
    # Top-left COSMOS branch
    # ------------------------------------------------------------------
    A_core = O_new.transpose(-2, -1) @ M_hat @ U_new  # r_o x r_u

    # ------------------------------------------------------------------
    # Bottom-left branch: (I - OO^T) M U
    # ------------------------------------------------------------------

    # n0 = ||M_hat U U^T||_F = ||M_hat U||_F
    M_U = M_hat @ U_new  # m x r_u
    norm0 = M_U.norm(dim=(-2, -1), keepdim=True)
    

    O_T_M_U = O_new.transpose(-2, -1) @ M_U
    Y_U  = M_U - O_new @ O_T_M_U  # m x r_u

    # ------------------------------------------------------------------
    # Compressed Sinkhorn block statistics
    # ------------------------------------------------------------------

    # O^T M, shape r_o x n
    O_T_M = O_new.transpose(-2, -1) @ M_hat

    # Raw top-left coordinate of momentum
    O_T_M_U = O_T_M @ U_new

    # Top-right:
    # A12 = O^T M (I - U U^T)
    A12_ambient = O_T_M - O_T_M_U @ U_new.transpose(-2, -1)
    s12 = A12_ambient.square().sum(dim=-1)  # r_o

    # Bottom-left:
    # Use preconditioned residual candidate Y_U.
    s21 = Y_U.square().sum(dim=-2)  # r_u

    # Bottom-right:
    # M22 = (I - OO^T) M (I - UU^T)
    M_right_top = M_U @ U_new.transpose(-2, -1)
    M_left_top = O_new @ O_T_M
    M_top_top = O_new @ O_T_M_U @ U_new.transpose(-2, -1)

    M22 = M_hat - M_left_top - M_right_top + M_top_top
    S22 = M22.square().sum(dim=(-2, -1))

    rt, ct, r0, c0 = compressed_sinkhorn_coeffs(
        A11=A_core,
        s12=s12,
        s21=s21,
        S22=S22,
        m=m,
        n=n,
        steps=10,
        gamma=1.0,
        eps=eps,
    )

    # ------------------------------------------------------------------
    # Build balanced four-block update
    # ------------------------------------------------------------------

    # Top-left block
    A11_bal = rt.unsqueeze(-1) * A_core * ct.unsqueeze(-2)
    update_11 = O_new @ A11_bal @ U_new.transpose(-2, -1)

    # Top-right block
    A12_bal = rt.unsqueeze(-1) * A12_ambient * c0.unsqueeze(-1).unsqueeze(-1)
    update_12 = O_new @ A12_bal

    # Bottom-left block
    A21_bal = r0.unsqueeze(-1).unsqueeze(-1) * Y_U * ct.unsqueeze(-2)
    update_21 = A21_bal @ U_new.transpose(-2, -1)

    # Bottom-right block:
    # Muon on M22, rescale back to original M22 norm,
    # then multiply sqrt(x0 y0) = r0 * c0.
    update_22 = zeropower_via_newtonschulz5(M22).float()

    M22_norm = M22.norm(dim=(-2, -1), keepdim=True)
    update_22 = update_22 * (
        M22_norm / update_22.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    )

    update_22 = (
        update_22
        * r0.unsqueeze(-1).unsqueeze(-1)
        * c0.unsqueeze(-1).unsqueeze(-1)
    )

    update = update_11 + update_12 + update_21 + update_22

    # ------------------------------------------------------------------
    # Final Muon-level norm
    #
    # Target:
    #   sqrt(max(1, m/n) * min(m, n))
    # ------------------------------------------------------------------
    muon_norm = (max(1.0, m / n) * min(m, n)) ** 0.5


    update = update * (
        muon_norm / update.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    )

    # ------------------------------------------------------------------
    # Normuon second-order branch only for rectangular matrices
    # ------------------------------------------------------------------
    if m != n:
        norm = update.norm(dim=(-2, -1), keepdim=True)

        v_mean = (
            torch.mean(update * update, dim=-1, keepdim=True)
            if m >= n
            else torch.mean(update * update, dim=-2, keepdim=True)
        )

        second_momentum.lerp_(v_mean, 1 - beta2)

        step_size = 1 / second_momentum.sqrt().clamp_min(eps)
        update.mul_(step_size)

        norm_new = update.norm(dim=(-2, -1), keepdim=True)
        update.mul_(norm / norm_new.clamp_min(eps))

       

    update = update.bfloat16()

    # Write COSMOS states back in-place
    U.copy_(U_new)
    O.copy_(O_new)
    S.copy_(S_new)
    R.copy_(R_new)

    return update


class COSMOS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.02,
        weight_decay=0,
        mu=0.95,
        beta2=0.95,
        rank_o=256,
        rank_u=256,
    ):
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )

        params = sorted(params, key=lambda x: x.size(), reverse=True)

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            mu=mu,
            beta2=beta2,
            rank_o=rank_o,
            rank_u=rank_u,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank_id = dist.get_rank() if dist.is_initialized() else 0

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                world_size - len(params) % world_size
            )

            for base_i in range(0, len(params), world_size):
                if base_i + rank_id < len(params):
                    p = params[base_i + rank_id]

                    if p.grad is None:
                        if world_size > 1:
                            dist.all_gather(
                                params_pad[base_i:base_i + world_size],
                                params_pad[base_i + rank_id],
                            )
                        continue

                    state = self.state[p]

                    if len(state) == 0:
                        m = p.size(-2)
                        n = p.size(-1)

                        rank_o = min(group["rank_o"], m)
                        rank_u = min(group["rank_u"], n)

                        state["momentum"] = torch.zeros_like(p)

                        # Normuon second_momentum shape.
                        # It is only used for m != n, but keeping allocation simple.
                        state["second_momentum"] = (
                            torch.zeros_like(p.data[..., 0:1])
                            if m >= n
                            else torch.zeros_like(p.data[0:1, ...])
                        )

                        # First step: initialize O and U from SVD of G.
                        # O: m x rank_o
                        # U: n x rank_u
                        G0 = p.grad.float()
                        O0, _, Vh0 = torch.linalg.svd(G0, full_matrices=False)

                        state["O"] = O0[..., :, :rank_o].contiguous()
                        state["U"] = Vh0.transpose(-2, -1)[..., :, :rank_u].contiguous()

                        state["S"] = torch.zeros(
                            *p.shape[:-2],
                            rank_u,
                            rank_u,
                            device=p.device,
                            dtype=torch.float32,
                        )

                        state["R"] = torch.zeros(
                            *p.shape[:-2],
                            rank_o,
                            rank_o,
                            device=p.device,
                            dtype=torch.float32,
                        )

                        state["step"] = torch.zeros(
                            (),
                            device=p.device,
                            dtype=torch.float32,
                        )

                    state["step"].add_(1.0)

                    update = cosmos_normuon_update(
                        p.grad,
                        state["momentum"],
                        state["second_momentum"],
                        state["U"],
                        state["O"],
                        state["S"],
                        state["R"],
                        state["step"],
                        mu=group["mu"],
                        beta2=group["beta2"],
                    )

                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

                if world_size > 1:
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank_id],
                    )


device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
       + f" on {torch.cuda.get_device_name(device)} with world_size {dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


resume_path = parse_resume_argv(sys.argv)
CKPT_EVERY = int(os.getenv("CKPT_EVERY", str(CKPT_EVERY_DEFAULT)))
num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1

for _ in range(num_trials):

    # Fixed seed => identical init every run, so optimizer variants are paired
    # comparisons with no init-noise confounding (new paired set; run1/run2 were unseeded).
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)


    ########################################
    #       Init & Optim Hyperparams       #
    ########################################

    # we want to minimize this while still reaching 3.28 val loss
    train_steps = 3250

    # initialize model parameters
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.normal_(mean=1, std=0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

    # create the optimizer(s)
    optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.7),
                        dict(params=[model.proj.weight], lr=0.004),
                        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                       betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
    optimizer2 = COSMOS([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.04, weight_decay=0.025)
    optimizers = [optimizer1, optimizer2]
    assert set(p for opt in optimizers for group in opt.param_groups
               for p in group["params"]) == set(model.parameters())
    init_lrs = []  # re-applied after ckpt load (torch may drop custom group keys)
    for opt in optimizers:
        opt_lrs = []
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
            opt_lrs.append(group["lr"])
        init_lrs.append(opt_lrs)

    # learning rate schedule: stable then decay
    def set_hparams(step, cooldown_frac=0.7):
        progress = step / train_steps
        assert 0 <= progress < 1
        if progress < 1 - cooldown_frac:
            eta = 1.0
        else:
            eta = (1 - progress) / cooldown_frac
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * eta


    ########################################
    #        Training and Validation       #
    ########################################

    train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
    for p in model.parameters():
        dist.broadcast(p.detach(), 0)
    # start the clock
    arm_preemption_handlers()
    start_step, training_time = 0, 0.0
    if resume_path is not None:
        start_step, training_time = load_ckpt(resume_path, model, optimizers)
        for _ in range(start_step):
            next(train_loader)
        print0(f"resumed at step {start_step} from {resume_path}", console=True)
        for opt, opt_lrs in zip(optimizers, init_lrs):
            for group, lr in zip(opt.param_groups, opt_lrs):
                group["initial_lr"] = lr
    last_val_step = start_step
    dist.barrier()
    t0 = time.perf_counter()
    for step in range(start_step, train_steps + 1):

        # --------------- VALIDATION SECTION -----------------
        val_step_freq = 125 if step / train_steps < 0.9 else 25
        if step == train_steps or step % val_step_freq == 0:
            # stop the clock
            dist.barrier()
            time_since_last_val = time.perf_counter() - t0
            step_avg = time_since_last_val / (step - last_val_step) if step > 0 else float("nan")
            last_val_step = step
            training_time += time_since_last_val
            model.eval()
            val_loss = 0
            with torch.no_grad():
                assert len(val_inputs) % mbs == 0
                for i in range(len(val_inputs) // mbs):
                    val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_loss /= val_tokens
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                   + f" step_avg:{1000*step_avg:.2f}ms", console=True)
            model.train()
            # start the clock again
            dist.barrier()
            t0 = time.perf_counter()

        if step == train_steps:
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        # accumulate across microbatches in case we are running with fewer than 8 gpus
        assert len(inputs) % mbs == 0
        for i in range(len(inputs) // mbs):
            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        # set optimization hyperparameters and take a step
        set_hparams(step)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        if (step + 1) % CKPT_EVERY == 0 and step + 1 < train_steps:
            ckpt_path = save_ckpt(step + 1, training_time + (time.perf_counter() - t0),
                                  model, optimizers)
            print0(f"checkpoint: {ckpt_path}", console=True, log=False)
        if term_requested():
            ckpt_path = save_ckpt(step + 1, training_time + (time.perf_counter() - t0),
                                  model, optimizers)
            print0(f"preempted at step {step + 1}, checkpoint saved: {ckpt_path}", console=True)
            dist.destroy_process_group()
            sys.exit(143)
        approx_training_time = training_time + (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
               + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

print0(f"max_memory_allocated: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB", console=True)
dist.destroy_process_group()
