"""Newton-Muon optimizer (Du & Su, arXiv:2604.01472) for the Track-3 harness.

Update: W <- W - eta * msgn(G @ (ZZ^T + gamma*I)^{-1}): right-precondition the
RAW gradient by the inverse input-activation second moment, then run the
harness-standard Muon pipeline (EMA momentum + Nesterov + Newton-Schulz +
rect scaling + decoupled wd). Reference repo: zhehangdu/Newton-Muon
(cloned at Newton-Muon-ref/; our port follows their Algorithm 1 + code).

Faithfulness notes (paper + reference repo):
- Refresh cov + factor every k steps (default k=10 HERE vs paper k=32 --
  aligned with our SOAP/SinkSOAP precond cadence for comparability).
- EWMA beta=0.95, relative ridge 0.2*tr(K)/d + 1e-8, K init 1e-3*I.
- Preconditioner applied to RAW grad BEFORE momentum (paper Alg.1 comment).
- MLP contraction (d x 4d input): FOUR d-block covariances (paper rule);
  split activations/gradients into contiguous d-chunks along input dim.
- First (k-1) steps run pure Muon (identity) until first refresh (reference).
- Deviation 1: cached Cholesky FACTOR + per-step triangular solves instead of
  the reference's explicit Cholesky inverse + matmul (identical math, better
  stability). PD fallback: ridge x10 retry (x3, the paper's own diagnostic
  rule), then keep previous factor (reference resets to identity).
- Deviation 2: momentum/NS/wd = HARNESS Muon (EMA mu, NS12, rect scale,
  decoupled wd), per the paper's own protocol ("replace the Muon update",
  keep settings). Reference used heavy-ball + NS5 (their harness flavor).
- Ours: separate q/k/v linears (no fused-QKV splitting needed).
- Accumulation: forward hooks, train-mode + grad-enabled + refresh-pending
  only; fp32 X^T X summed over microbatches, normalized by token count.

Distributed notes: hooks assume locally-available inputs. Under DP the local
Sigma uses the local microbatch (approximation; grads are still allreduced).
Under TP, column-parallel inputs are replicated across ranks, so ZZ^T is exact
with ZERO communication (OUR inference from the paper's §6-flagged
distributed gap — the paper only says distributed eval is future work;
untested here).
Checkpointing: refresh cadence is preserved across resume via per-param
state['_step'] (optimizer attrs are NOT in state_dict).
"""
import torch
import torch.compiler
import torch.distributed as dist

_USE_COMPILED_NS = True


def _ns_raw(G, steps=12):
    # Stock quintic Newton-Schulz, duplicated from the harness baseline
    # (modded-nanogpt-public @ bc3a0c2, train_gpt_simple.py zeropower fn):
    # a=2, b=-1.5, c=0.5, 12 iterations. Separate code object => separate
    # inductor cache entry from harness Muon (one extra one-time compile).
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
    return X


_compiled_ns = torch.compile(_ns_raw)  # lazy: no-op until first call


def zeropower_ns(G, steps=12):
    fn = _compiled_ns if _USE_COMPILED_NS else _ns_raw
    return fn(G, steps)


def _blocks_for(n_out, n_in):
    """Contiguous input-dim blocks (paper's 4d rule, generalized).

    Covariance is always over INPUT features (n_in x n_in). When the input
    dim is a multiple of the output dim (contraction layers like our
    (768,3072) MLP proj), split into output-sized chunks; else one block.
    """
    if n_in > n_out and n_in % n_out == 0:
        nb = n_in // n_out
        return nb, n_in // nb
    return 1, n_in


class NewtonMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.025, momentum=0.95, nesterov=True,
                 weight_decay=0.05, beta=0.95, ridge=0.2, refresh_k=10,
                 init_diag=1e-3, eps=1e-8, ns_steps=12):
        assert isinstance(params, list) and len(params) >= 1
        assert isinstance(params[0], torch.nn.Parameter)
        for p in params:
            if p.ndim != 2:
                raise ValueError(f"NewtonMuon needs 2D params, got {tuple(p.shape)}")
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.beta = beta
        self.ridge = ridge
        self.refresh_k = refresh_k
        self.init_diag = init_diag
        self.eps = eps
        self.ns_steps = ns_steps
        self.global_step = 0
        self._warned_chol = False

    def _ensure_state(self, p):
        st = self.state[p]
        if "momentum_buf" in st:
            return st
        m, n = p.shape
        nb, d = _blocks_for(m, n)
        st["momentum_buf"] = torch.zeros_like(p)
        st["nblocks"] = nb
        st["block_d"] = d
        cov = torch.zeros((nb, d, d), device=p.device, dtype=torch.float32)
        cov.diagonal(dim1=-2, dim2=-1).fill_(self.init_diag)
        st["cov"] = cov
        st["accum"] = torch.zeros_like(cov)
        st["count"] = 0
        st["chol"] = None  # None => identity (pure Muon) until first refresh
        st["_step"] = 0
        return st

    def _completed_steps(self, p):
        st = self.state.get(p)
        if st is not None and "_step" in st:
            return st["_step"]
        return self.global_step

    def _refresh_param(self, p):
        """EWMA update + Cholesky refactor (called on refresh steps)."""
        st = self.state[p]
        nb, d = st["nblocks"], st["block_d"]
        if st["count"] > 0:
            st["cov"].lerp_(st["accum"] / st["count"], 1.0 - self.beta)
            st["accum"].zero_()
            st["count"] = 0
        K = st["cov"].clone()
        tr = K.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        base = tr / d * self.ridge + self.eps  # (nb,1,1) relative ridge
        eye = torch.eye(d, device=K.device, dtype=K.dtype)
        for attempt in range(3):
            Kt = K + base * (10 ** attempt) * eye
            L, info = torch.linalg.cholesky_ex(Kt, upper=False, check_errors=False)
            if bool((info == 0).all()):
                st["chol"] = L
                return
        if not self._warned_chol:
            self._warned_chol = True
            print("NewtonMuon: Cholesky failed after ridge x100; keeping previous factor")
        # else: keep previous st["chol"] (or None => identity)

    @staticmethod
    def _apply_precond(G, chol):
        """G (m,n) right-multiplied by (LL^T)^{-1} via triangular solves."""
        if chol is None:
            return G.float().clone()
        m, n = G.shape
        nb = chol.shape[0]
        d = n // nb
        Gb = G.float().reshape(m, nb, d).permute(1, 0, 2)  # (nb,m,d)
        A = torch.linalg.solve_triangular(chol, Gb.mT, upper=False, left=True)
        B = torch.linalg.solve_triangular(chol, A, upper=False, left=True)
        return B.mT.permute(1, 0, 2).reshape(m, n)

    def _first_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                return self.state.get(p)
        return None

    @torch.no_grad()
    def step(self):
        # Cadence truth lives in per-param state (always preserved by
        # state_dict); group["_global_step"] is only a best-effort mirror.
        st0 = self._first_state()
        if st0 is not None and "_step" in st0:
            self.global_step = st0["_step"]
        do_refresh = ((self.global_step + 1) % self.refresh_k == 0)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        for group in self.param_groups:
            params = group["params"]
            if world_size > 1:
                pad = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1])] * pad
            else:
                params_pad = params
            for base_i in range(0, len(params_pad), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    st = self._ensure_state(p)
                    if do_refresh:
                        self._refresh_param(p)
                    if p.grad is None:
                        continue
                    Gp = self._apply_precond(p.grad, st["chol"])
                    buf = st["momentum_buf"]
                    mu = group["momentum"]
                    buf.lerp_(Gp, 1.0 - mu)
                    upd = (mu * buf + (1.0 - mu) * Gp) if group["nesterov"] else buf
                    upd = zeropower_ns(upd, self.ns_steps)
                    upd *= max(1, p.size(-2) / p.size(-1)) ** 0.5
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(upd, alpha=-group["lr"])
                if world_size > 1:
                    dist.all_gather(params_pad[base_i:base_i + world_size],
                                    params_pad[base_i + rank])
        self.global_step += 1
        for group in self.param_groups:
            group["_global_step"] = self.global_step
            for p in group["params"]:
                if p in self.state:
                    self.state[p]["_step"] = self.global_step


def attach_activation_hooks(model, optimizer):
    """Register forward hooks on this optimizer's Linear modules.

    Accumulates fp32 X^T X (+ token counts) into optimizer state, ONLY on
    refresh-pending steps and only in train mode with grad enabled (val
    forwards must not pollute the training covariance). Returns handles.
    """
    wanted = {id(p) for g in optimizer.param_groups for p in g["params"]}
    handles = []
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear) and id(mod.weight) in wanted:
            if getattr(mod, "_nm_hook_attached", False):
                continue  # idempotent: never double-accumulate
            handles.append(mod.register_forward_hook(_make_hook(optimizer, mod.weight)))
            mod._nm_hook_attached = True
    assert handles, "attach_activation_hooks: no Linear modules matched optimizer params"
    return handles


def _make_hook(optimizer, param):
    # Compiler-disabled BY DESIGN: dynamo must never inline this body (it reads
    # the mutable global_step counter, whose guard would fail and recompile
    # every step). The hook runs eager behind a static boundary break: bounded
    # one-time fragmentation, zero per-step recompiles. Backward (no hooks)
    # compiles whole. See research_stack/test_nm_smoke.sh for the check.
    @torch.compiler.disable
    def hook(mod, inp, out):
        if not mod.training or not torch.is_grad_enabled():
            return
        done = optimizer._completed_steps(param)
        if (done + 1) % optimizer.refresh_k != 0:
            return
        X = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        st = optimizer._ensure_state(param)
        nb, d = st["nblocks"], st["block_d"]
        Xb = X.reshape(-1, nb, d)
        st["accum"] += torch.einsum("tbd,tbe->bde", Xb, Xb)
        st["count"] += X.shape[0]
    return hook


def _self_test():
    global _USE_COMPILED_NS
    _USE_COMPILED_NS = False  # validate math; compile is harness-validated
    torch.manual_seed(0)

    # 1. block rules + hook wiring on a toy MLP
    mlp = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 128),
                              torch.nn.Linear(128, 32))
    params = [mlp[0].weight, mlp[1].weight, mlp[2].weight]
    opt = NewtonMuon(params, lr=0.05, refresh_k=1)  # k=1: accumulate every forward
    h = attach_activation_hooks(mlp, opt)
    assert len(h) == 3, f"expected 3 hooks, got {len(h)}"
    x = torch.randn(64, 32)
    mlp.train()
    mlp(x).sum().backward()
    assert opt.state[params[0]]["nblocks"] == 1
    assert opt.state[params[2]]["nblocks"] == 4  # (32,128): 4 blocks
    assert opt.state[params[0]]["count"] == 64, "hook must accumulate train tokens"
    mlp.eval()
    mlp(x).sum().backward()
    assert opt.state[params[0]]["count"] == 64, "eval forwards must not accumulate"

    # 2. toy least-squares must converge (hooks + cov + cholesky + NS end to end)
    torch.manual_seed(1)
    Xt = torch.randn(256, 8)
    Wt = torch.randn(6, 8)
    Yt = Xt @ Wt.mT + 0.01 * torch.randn(256, 6)
    lin = torch.nn.Linear(8, 6, bias=False)
    opt2 = NewtonMuon([lin.weight], lr=0.05, refresh_k=5)
    attach_activation_hooks(lin, opt2)
    lin.train()
    l0 = None
    for _ in range(100):
        opt2.zero_grad()
        loss = (lin(Xt) - Yt).pow(2).mean()
        if l0 is None:
            l0 = loss.item()
        loss.backward()
        opt2.step()
    l1 = loss.item()
    assert l1 < 0.2 * l0, f"no convergence: {l0:.4f} -> {l1:.4f}"
    assert opt2.state[lin.weight]["chol"] is not None, "never refreshed"

    # 3. PD fallback: indefinite cov -> ridge rescue, then identity degradation
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt3 = NewtonMuon([p], lr=0.01, refresh_k=1, ridge=0.2)
    st = opt3._ensure_state(p)
    st["cov"][0].copy_(torch.eye(4))
    st["cov"][0][0, 1] = st["cov"][0][1, 0] = 2.0  # indefinite 2x2 corner
    st["count"] = 0  # skip EMA so the crafted cov is used as-is
    opt3._refresh_param(p)
    assert st["chol"] is not None and torch.isfinite(st["chol"]).all(), "ridge rescue failed"
    st["cov"][0].fill_(0.0)
    st["cov"][0][0, 1] = st["cov"][0][1, 0] = 100.0  # hopelessly indefinite
    st["chol"] = None
    opt3._refresh_param(p)
    assert st["chol"] is None, "total failure must keep previous factor (None here)"
    g = torch.randn(4, 4)
    assert torch.equal(NewtonMuon._apply_precond(g, None), g.float()), "identity path broken"

    # 4. determinism: same toy twice -> identical
    def run_once(seed):
        torch.manual_seed(seed)
        m = torch.nn.Linear(8, 6, bias=False)
        o = NewtonMuon([m.weight], lr=0.05, refresh_k=5)
        attach_activation_hooks(m, o)
        m.train()
        for _ in range(10):
            o.zero_grad()
            ((m(Xt) - Yt).pow(2).mean()).backward()
            o.step()
        return ((m(Xt) - Yt).pow(2).mean()).item()
    assert run_once(7) == run_once(7), "not a pure function of inputs"
    print("SELFTEST OK: hooks/shapes/blocks, toy convergence "
          f"({l0:.4f}->{l1:.4f}), PD fallback, determinism")


if __name__ == "__main__":
    _self_test()
