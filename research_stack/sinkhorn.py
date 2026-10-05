"""Sinkhorn-balanced embedding/head optimizer (DeepSeek-V4.1-Flash report).

Transcribes Algorithm 1 (Sec 2.5) + hyperparameters (Sec 4.2.2):
Nesterov momentum (beta=0.95), mask near-zero rows (rho_i <= tau*rho_bar,
tau=1e-3), K=11 (odd) alternating row/col l2-normalizations (rows first
AND last), Delta = sqrt(n)*U, apply with lr*gamma (gamma=0.18, "reutilization
of the AdamW learning rate"). No weight decay. fp32 momentum regardless of
param dtype; all math in fp32, Delta cast to param dtype on apply.

Conventions: params are (m, n) with m >> n; dim0 indexes tokens/classes
(rows), dim1 indexes features (columns). Matches our embedding (50304x768,
bf16) and prediction head (50304x768, fp32). Rows = token index, columns =
hidden feature; the procedure approximately equalizes row- and column-wise
RMS of the update (report Eq. 7).

CPU-safe self-test: ./run_container.sh python -m research_stack.sinkhorn
"""
import torch


def sinkhorn_balance(Ghat, K=11, tau=1e-3, eps=1e-20):
    """Pure function: Nesterov momentum -> balanced update Delta (fp32)."""
    assert Ghat.ndim == 2
    n = Ghat.shape[1]
    X = Ghat.float()
    rho = X.norm(dim=1)  # (m,) row l2 norms
    rho_bar = rho.mean()
    U = X.clone()
    U[rho <= tau * rho_bar] = 0.0  # mask near-zero (absent/stale-token) rows
    for k in range(1, K + 1):
        if k % 2 == 1:
            U = U / (U.norm(dim=1, keepdim=True) + eps)  # rows to unit l2
        else:
            U = U / (U.norm(dim=0, keepdim=True) + eps)  # cols to unit l2
    return (n ** 0.5) * U  # unit row l2 -> unit row RMS


class SinkhornEmbed(torch.optim.Optimizer):
    """torch Optimizer wrapper. Accepts per-group lr via param-group dicts."""

    def __init__(self, params, lr, beta=0.95, K=11, tau=1e-3, eps=1e-20,
                 gamma=0.18, nesterov=True):
        if K % 2 == 0:
            raise ValueError("K must be odd (report Alg. 1 ends on row-norm)")
        defaults = dict(lr=lr, beta=beta, K=K, tau=tau, eps=eps,
                        gamma=gamma, nesterov=nesterov)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2 or p.shape[0] < p.shape[1]:
                    raise ValueError(
                        "SinkhornEmbed expects (tokens, features) 2D with "
                        f"dim0>=dim1, got {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta, K, tau = group["beta"], group["K"], group["tau"]
            eps, gamma, nesterov = group["eps"], group["gamma"], group["nesterov"]
            step_size = group["lr"] * gamma
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(G)  # fp32
                M = state["momentum"]
                M.lerp_(G, 1.0 - beta)  # M <- beta*M + (1-beta)*G
                Ghat = (beta * M + (1.0 - beta) * G) if nesterov else M
                delta = sinkhorn_balance(Ghat, K=K, tau=tau, eps=eps)
                p.add_(delta.to(p.dtype), alpha=-step_size)


def _self_test():
    torch.manual_seed(0)
    m, n = 4096, 64
    G = torch.randn(m, n)
    G[100:200] = 0.0  # simulate absent-token rows (exactly zero grad)
    d = sinkhorn_balance(0.05 * G, K=11, tau=1e-3, eps=1e-20)
    assert torch.isfinite(d).all(), "NaN/Inf in Delta"
    assert (d[100:200] == 0.0).all(), "masked rows must stay exactly 0"
    row_rms = d[:100].pow(2).mean(dim=1).sqrt()
    assert torch.allclose(row_rms, torch.ones_like(row_rms), atol=1e-4), \
        f"row RMS != 1: {row_rms.min().item():.4f}..{row_rms.max().item():.4f}"
    col_rms = d.pow(2).mean(dim=0).sqrt()
    assert (col_rms > 0).all() and torch.isfinite(col_rms).all()
    d2 = sinkhorn_balance(0.05 * G, K=11, tau=1e-3, eps=1e-20)
    assert torch.equal(d, d2), "not a pure function of input"
    print(f"SELFTEST OK: masked->0, rowRMS~1 "
          f"(colRMS mean {col_rms.mean().item():.3f}, no-NaN, deterministic)")


if __name__ == "__main__":
    _self_test()
