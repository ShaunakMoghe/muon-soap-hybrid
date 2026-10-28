"""Optimizer microbenchmarks for the Muon-vs-SOAP tradeoff study.

Loads Muon / SinkSOAP from the ACTUAL variant files (exec everything before
the Setup marker; the training loop is never imported or run), so the
profiled code cannot drift from the shipped code.

Measures, for representative 124M-GPT block shapes:
  1. full optimizer-update time per shape (Muon vs SinkSOAP)
  2. SinkSOAP cost breakdown (eigh L/R, project, sinkhorn, reconstruct, scale)
  3. eigh run-to-run noise floor (sign-invariant subspace distance)
  4. exact optimizer-state memory per shape + whole-model totals (analytic)

Usage (needs GPU, no torchrun): ./run_container.sh python profile/profile_optimizers.py
"""
import time

import torch

SETUP_MARKER = ("########################################\n"
                "#                Setup                 #\n"
                "########################################\n")


def load_optims(path):
    src = open(path).read()
    assert src.count(SETUP_MARKER) == 1, path
    ns = {}
    exec(compile(src.split(SETUP_MARKER)[0], path, "exec"), ns)
    return ns


def bench(fn, reps=20, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


SHAPES = {"attn (768x768)": (768, 768), "mlp-fc (768x3072)": (768, 3072),
          "mlp-proj (3072x768)": (3072, 768)}


def main():
    assert torch.cuda.is_available()
    dev = "cuda"
    mu = load_optims("variants/train_gpt_baseline_seed0.py")
    sk = load_optims("variants/train_gpt_sinksoap_seed0.py")
    so = load_optims("variants/train_gpt_soap_seed0.py")
    co = load_optims("variants/train_gpt_cosmos_r32_seed0.py")
    assert "Muon" in mu and "SinkSOAP" in sk and "SOAP" in so and "COSMOS" in co

    print("== 1. full update ms (mean of reps, post-warmup) ==")
    for name, (m, n) in SHAPES.items():
        g = torch.randn(m, n, device=dev)
        mom = torch.zeros(m, n, device=dev)
        t_mu = bench(lambda: mu["muon_update"](g, mom))
        lg = torch.zeros(m, m, device=dev)
        rg = torch.zeros(n, n, device=dev)
        sm = torch.zeros((m, 1) if m >= n else (1, n), device=dev)
        # burn-in: fill grams/momentum like mid-training state
        for _ in range(3):
            sk["gram_sinkhorn_normuon_update"](g, mom, lg, rg, sm)
        t_sk = bench(lambda: sk["gram_sinkhorn_normuon_update"](g, mom, lg, rg, sm))
        sp = torch.nn.Parameter(torch.randn(m, n, device=dev))
        sopt = so["SOAP"]([sp], lr=0.002, betas=(0.95, 0.95), weight_decay=0,
                          precondition_frequency=10)
        for _ in range(12):  # burn-in: init + EMAs + one QR refresh
            sp.grad = g
            sopt.step()

        def soap_step():
            sp.grad = g
            sopt.step()

        t_so = bench(soap_step)
        print(f"{name:22s} muon={t_mu:8.2f}ms  sinksoap={t_sk:8.2f}ms  soap={t_so:8.2f}ms")

    print("== 2. sinksoap breakdown ms ==")
    for name, (m, n) in SHAPES.items():
        g = torch.randn(m, n, device=dev)
        M = torch.randn(m, n, device=dev).float()
        L = (g @ g.mT).float()
        R = (g.mT @ g).float()
        U = sk["gram_eigenbasis"](L)
        V = sk["gram_eigenbasis"](R)
        A = U.mT @ M @ V
        t_eigh = bench(lambda: sk["gram_eigenbasis"](L)) + bench(lambda: sk["gram_eigenbasis"](R))
        t_proj = bench(lambda: U.mT @ M @ V)
        t_sink = bench(lambda: sk["sinkhorn_energy_balance_rect_no_scale"](A))
        t_recon = bench(lambda: U @ A @ V.mT)
        X = (U @ A @ V.mT).float()
        sm = torch.zeros((m, 1) if m >= n else (1, n), device=dev)
        t_scale = bench(lambda: sk["scale_update_like_muon"](X))
        t_norm = bench(lambda: sk["normuon_postcondition_update"](
            sk["scale_update_like_muon"](X), sm)) - t_scale if m != n else 0.0
        print(f"{name:22s} eigh={t_eigh:7.2f} proj={t_proj:6.2f} sink={t_sink:6.2f} "
              f"recon={t_recon:6.2f} scale={t_scale:5.2f} normuon={t_norm:5.2f}")

    print("== 3. eigh noise floor: rel ||QQ^T - Q0Q0^T||_F over 20 calls ==")
    for name, (m, n) in SHAPES.items():
        g = torch.randn(max(m, n), min(m, n), device=dev)
        G = (g @ g.mT).float()
        Q0 = sk["gram_eigenbasis"](G)
        P0 = Q0 @ Q0.mT
        ds = [((Q @ Q.mT - P0).norm() / P0.norm()).item()
              for Q in (sk["gram_eigenbasis"](G) for _ in range(20))]
        print(f"gram {P0.shape[0]:5d}x{P0.shape[0]:<5d} mean={sum(ds) / len(ds):.2e} "
              f"max={max(ds):.2e}")

    print("== 4. optimizer-state MB (fp32, analytic, exact) ==")
    tot_mu = tot_sk = tot_so = 0.0
    per_layer = [((768, 768), 4), ((768, 3072), 1), ((3072, 768), 1)]
    for (m, n), cnt in per_layer:
        s_mu = m * n * 4 / 1e6
        s_sk = (m * n + m * m + n * n + (m if m >= n else n)) * 4 / 1e6
        s_so = (2 * m * n + 2 * m * m + 2 * n * n) * 4 / 1e6
        tot_mu += 12 * cnt * s_mu
        tot_sk += 12 * cnt * s_sk
        tot_so += 12 * cnt * s_so
        print(f"({m}x{n}) x{cnt}/layer: muon={s_mu:6.2f}MB  sinksoap={s_sk:6.2f}MB  soap={s_so:6.2f}MB")
    print(f"12-layer totals: muon={tot_mu:.0f}MB  sinksoap={tot_sk:.0f}MB  soap={tot_so:.0f}MB")
    print("(excludes AdamW embed/head/1D states, identical for both)")

    print("== 5. sinkhorn-embed (50304x768, K=11): ms/step + state MB ==")
    from research_stack.sinkhorn import SinkhornEmbed
    m, n = 50304, 768
    g = torch.randn(m, n, device=dev)
    ep = torch.nn.Parameter(torch.randn(m, n, device=dev))
    eopt = SinkhornEmbed([ep], lr=0.7)
    for _ in range(3):
        ep.grad = g
        eopt.step()

    def embed_step():
        ep.grad = g
        eopt.step()

    t_em = bench(embed_step, reps=10, warmup=2)
    s_em = m * n * 4 / 1e6
    print(f"embed/head-like: sinkhorn={t_em:8.2f}ms  state(fp32 momentum)={s_em:.0f}MB "
          f"(x2 params = {2 * s_em:.0f}MB; AdamW m+v would be {4 * s_em:.0f}MB)")

    print("== 6. COSMOS-r32 full update + breakdown ms (r=32) ==")
    for name, (m, n) in SHAPES.items():
        r = 32
        g = torch.randn(m, n, device=dev)
        mom = torch.randn(m, n, device=dev)
        sm = torch.zeros((m, 1) if m >= n else (1, n), device=dev)
        U, _ = torch.linalg.qr(torch.randn(n, r, device=dev), mode="reduced")
        O, _ = torch.linalg.qr(torch.randn(m, r, device=dev), mode="reduced")
        S = torch.eye(r, device=dev)
        R = torch.eye(r, device=dev)
        stp = torch.tensor(10.0, device=dev)
        fn = co["cosmos_normuon_update"]
        for _ in range(3):  # burn-in: realistic states + compiled entry
            fn(g, mom, sm, U, O, S, R, stp)
        t_full = bench(lambda: fn(g, mom, sm, U, O, S, R, stp))
        # pieces, replicated from the variant's update body
        G = g.float()
        GU = G @ U
        t_qr = bench(lambda: torch.linalg.qr(
            0.95 * (U @ S) + 0.05 * (G.mT @ GU), mode="reduced"))
        t_qr += bench(lambda: torch.linalg.qr(
            0.95 * (O @ R) + 0.05 * (G @ (G.mT @ O)), mode="reduced"))
        Mh = mom.float()
        t_proj = bench(lambda: O.mT @ Mh @ U) + bench(lambda: Mh @ U)
        A11 = O.mT @ Mh @ U
        s12 = torch.randn(r, device=dev)
        s21 = torch.randn(r, device=dev)
        S22 = torch.tensor(1.0, device=dev)
        t_sink = bench(lambda: co["compressed_sinkhorn_coeffs"](
            A11=A11, s12=s12, s21=s21, S22=S22, m=m, n=n,
            steps=10, gamma=1.0, eps=1e-10))
        M22 = torch.randn(m, n, device=dev)
        t_ns = bench(lambda: co["zeropower_via_newtonschulz5"](M22))
        print(f"{name:22s} full={t_full:8.2f}ms  qr={t_qr:7.2f} proj={t_proj:6.2f} "
              f"sink={t_sink:6.2f} ns22={t_ns:6.2f}")


if __name__ == "__main__":
    main()
