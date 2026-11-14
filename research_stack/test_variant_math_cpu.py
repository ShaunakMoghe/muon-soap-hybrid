"""CPU math checks for optimizer variants (no GPU, no training loop).

Executes the optimizer sections of variant scripts eagerly (torch.compile
stubbed to identity) and asserts:
  1. cosmos_r0: its update equals the verbatim r32 function called at
     rank_o = rank_u = 0, over several steps and all three block
     orientations; the r32 function at rank 32 differs materially (the
     check is not vacuous).
  2. qrk{2,4,8}: with do_qr=True every step, update and O/U/S/R state are
     bit-identical to r32; with do_qr=False the subspace state carries
     over unchanged; the optimizer class refreshes on steps 1, 1+k, ...
  3. sink{5,3,1}: forced back to 10 iterations the update is bit-identical
     to r32; at the file's own count it is finite.
  4. specns6: on a condition-number-30 input, spectral-start NS6 output is
     near-orthogonal while the Frobenius-start NS6 baseline is not.

Usage (repo root, in the container, PYTHONPATH=repo):
    python research_stack/test_variant_math_cpu.py
"""
import torch

torch.compile = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)

# Rank-0 states make QR inputs with zero columns; the factorization is the
# empty matrix. Route those through a trivial answer in case LAPACK rejects
# empty inputs; non-empty calls are passed through untouched.
_native_qr = torch.linalg.qr


def _qr(A, mode="reduced"):
    if A.size(-1) == 0:
        return A, A.new_zeros(*A.shape[:-2], 0, 0)
    return _native_qr(A, mode=mode)


torch.linalg.qr = _qr

SETUP_MARKER = ("########################################\n"
                "#                Setup                 #\n"
                "########################################\n")
V = "variants/train_gpt_"
SHAPES = [(64, 64), (192, 48), (48, 192)]
FN = "cosmos_normuon_update"


def load(path):
    # Same contract as profile/profile_optimizers.py: exec the head
    # (imports/arch/Muon) plus the def-only optimizer block after the Setup
    # marker, stopping before the dist setup line.
    src = open(path).read()
    parts = src.split(SETUP_MARKER)
    assert len(parts) == 2, path
    ns = {}
    exec(compile(parts[0], path, "exec"), ns)
    exec(compile(parts[1].split("\ndevice = torch.device")[0], path, "exec"), ns)
    return ns


def fresh_state(m, n, r, seed=0):
    gen = torch.Generator().manual_seed(seed)
    O0, _, Vh0 = torch.linalg.svd(torch.randn(m, n, generator=gen), full_matrices=False)
    return dict(
        momentum=torch.zeros(m, n),
        second_momentum=torch.zeros(m, 1) if m >= n else torch.zeros(1, n),
        U=Vh0.mT[:, :r].contiguous(),
        O=O0[:, :r].contiguous(),
        S=torch.zeros(r, r),
        R=torch.zeros(r, r),
        step=torch.zeros(()),
    )


def clone_state(s):
    return {k: v.clone() for k, v in s.items()}


def grads(m, n, steps, seed):
    # Low-rank signal plus noise with a decaying scale, like real gradients.
    gen = torch.Generator().manual_seed(seed)
    out = []
    for t in range(steps):
        a = torch.randn(m, 4, generator=gen)
        b = torch.randn(4, n, generator=gen)
        out.append((a @ b + 0.3 * torch.randn(m, n, generator=gen)) * 0.7 ** t)
    return out


def call(fn, g, s, **kw):
    return fn(g.clone(), s["momentum"], s["second_momentum"],
              s["U"], s["O"], s["S"], s["R"], s["step"], **kw)


def rel(a, b):
    return ((a.float() - b.float()).norm() / a.float().norm()).item()


def test_r0_equals_rank0_limit(r32, r0):
    for m, n in SHAPES:
        s_ref = fresh_state(m, n, 0)
        assert s_ref["U"].shape == (n, 0) and s_ref["O"].shape == (m, 0)
        s_r32 = fresh_state(m, n, 32)
        mom = torch.zeros(m, n)
        sm = s_ref["second_momentum"].clone()
        worst, gap32 = 0.0, 0.0
        for g in grads(m, n, 6, seed=m + n):
            u_ref = call(r32[FN], g, s_ref)
            u_new = r0["cosmos_r0_update"](g.clone(), mom, sm)
            u_r32 = call(r32[FN], g, s_r32)
            assert u_ref.dtype == u_new.dtype == torch.bfloat16
            assert u_ref.shape == u_new.shape == (m, n)
            assert torch.equal(s_ref["momentum"], mom), (m, n)
            if m != n:
                assert torch.allclose(s_ref["second_momentum"], sm, rtol=1e-4, atol=0), (m, n)
            else:
                assert float(sm.abs().sum()) == 0.0
            worst = max(worst, rel(u_ref, u_new))
            gap32 = max(gap32, rel(u_r32, u_new))
        print(f"r0 vs verbatim rank-0 {m}x{n}: max rel diff {worst:.2e} (rank-32 differs by {gap32:.2e})")
        assert worst < 2e-3, (m, n, worst)
        assert gap32 > 1e-2, (m, n, gap32)


def test_qrk(r32, qrk, k):
    assert qrk["QR_EVERY"] == k
    for m, n in SHAPES:
        base = fresh_state(m, n, 32)
        s_a, s_b = clone_state(base), clone_state(base)
        for g in grads(m, n, 5, seed=1 + m):
            u_a = call(r32[FN], g, s_a)
            u_b = call(qrk[FN], g, s_b, do_qr=True)
            assert torch.equal(u_a, u_b), (k, m, n)
            for key in s_a:
                assert torch.equal(s_a[key], s_b[key]), (k, m, n, key)
        before = clone_state(s_b)
        u = call(qrk[FN], grads(m, n, 1, seed=99)[0], s_b, do_qr=False)
        assert torch.isfinite(u.float()).all()
        for key in ("U", "O", "S", "R"):
            assert torch.equal(before[key], s_b[key]), (k, m, n, key)
        assert not torch.equal(before["momentum"], s_b["momentum"])
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(64, 64)),
              torch.nn.Parameter(torch.randn(192, 48))]
    opt = qrk["COSMOS"](params, lr=0.02, weight_decay=0.0)
    prev, refreshed = None, []
    for t in range(1, 2 * k + 3):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
        cur = [opt.state[p]["U"].clone() for p in params]
        if prev is not None:
            changed = [not torch.equal(a, b) for a, b in zip(prev, cur)]
            assert all(changed) or not any(changed), (k, t, changed)
            if changed[0]:
                refreshed.append(t)
        prev = cur
    expect = [t for t in range(2, 2 * k + 3) if (t - 1) % k == 0]
    assert refreshed == expect, (k, refreshed, expect)
    print(f"qrk{k}: do_qr=True bit-identical to r32; refresh steps after step 1: {refreshed}")


def test_sink(r32, sink, k):
    assert sink["SINK_STEPS"] == k
    worst = 0.0
    for m, n in SHAPES:
        base = fresh_state(m, n, 32)
        sink["SINK_STEPS"] = 10
        s_a, s_b = clone_state(base), clone_state(base)
        for g in grads(m, n, 4, seed=2 + m):
            assert torch.equal(call(r32[FN], g, s_a), call(sink[FN], g, s_b)), (k, m, n)
        sink["SINK_STEPS"] = k
        s_a, s_b = clone_state(base), clone_state(base)
        for g in grads(m, n, 4, seed=2 + m):
            u_a, u_b = call(r32[FN], g, s_a), call(sink[FN], g, s_b)
            assert torch.isfinite(u_b.float()).all(), (k, m, n)
            worst = max(worst, rel(u_a, u_b))
    if k == 1:
        assert worst > 0, k
    print(f"sink{k}: bit-identical to r32 at 10 iters; at {k} iters max rel diff vs r32 {worst:.2e}")


def test_specns6(spec, frob):
    gen = torch.Generator().manual_seed(0)
    for m, n in SHAPES:
        k = min(m, n)
        q1, _ = _native_qr(torch.randn(m, k, generator=gen))
        q2, _ = _native_qr(torch.randn(n, k, generator=gen))
        G = (q1 * torch.logspace(0, -1.5, k)) @ q2.mT * 37.0
        sv_spec = torch.linalg.svdvals(spec["zeropower_via_newtonschulz5"](G.clone()).float())
        sv_frob = torch.linalg.svdvals(frob["zeropower_via_newtonschulz5"](G.clone()).float())
        print(f"specns6 {m}x{n}: spectral-start NS6 sv in [{sv_spec.min():.3f}, {sv_spec.max():.3f}]; "
              f"Frobenius-start NS6 sv in [{sv_frob.min():.3f}, {sv_frob.max():.3f}]")
        assert torch.isfinite(sv_spec).all()
        assert sv_spec.min() > 0.9 and sv_spec.max() < 1.15, (m, n)
        assert sv_frob.min() < 0.85, (m, n)


def main():
    torch.set_num_threads(4)
    r32 = load(V + "cosmos_r32_seed0.py")
    test_r0_equals_rank0_limit(r32, load(V + "cosmos_r0_seed0.py"))
    for k in (2, 4, 8):
        test_qrk(r32, load(V + f"cosmos_r32_qrk{k}_seed0.py"), k)
    for k in (5, 3, 1):
        test_sink(r32, load(V + f"cosmos_r32_sink{k}_seed0.py"), k)
    test_specns6(load(V + "baseline_specns6_seed0.py"), load(V + "baseline_ns6_seed0.py"))
    print("VARIANT_MATH_OK")


if __name__ == "__main__":
    main()
