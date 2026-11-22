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
  5. Every variant of the 2026-11-22 batch differs from its base script
     only in the lines it is meant to change (seed copies: the two seed
     lines).
  6. nonormuon (cosmos r0 / r32 / r512, sinksoap): with the switch back on
     the update and all state are bit-identical to the base; with it off
     the square case is unchanged, momentum and subspace state are
     unchanged, second_momentum stays zero, the update keeps the Muon-level
     norm, and re-applying the NorMuon step outside the function gives the
     base update back. r0 without NorMuon is Muon's update times a scalar.
  7. rank placement (sq512 / short512 / long512): with all three ranks at
     32 the optimizer is bit-identical to r32; O/U get the intended rank
     on each side for all three matrix orientations.

Usage (repo root, in the container, PYTHONPATH=repo):
    python research_stack/test_variant_math_cpu.py
"""
import difflib

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


def code_changes(base_path, var_path):
    """(removed, added) non-comment lines between two scripts, stripped."""
    a = open(base_path).read().splitlines()
    b = open(var_path).read().splitlines()
    rem, add = [], []
    for line in difflib.unified_diff(a, b, n=0, lineterm=""):
        if line.startswith(("---", "+++", "@@")):
            continue
        body = line[1:].strip()
        if body.startswith("#"):
            continue
        (rem if line[0] == "-" else add).append(body)
    return rem, add


def seed_lines(s):
    return [f"torch.manual_seed({s})", f"torch.cuda.manual_seed_all({s})"]


GATE_OLD, GATE_NEW = ["if m != n:"], ["APPLY_NORMUON = False", "if APPLY_NORMUON and m != n:"]
SINK_OLD = ["lr=0.04, weight_decay=0.025)"]
SINK_NEW = ["lr=0.04, weight_decay=0.025, apply_normuon=False)"]
PLACE_OLD = ['rank_o = min(group["rank_o"], m)', 'rank_u = min(group["rank_u"], n)']
PLACE_NEW = ["if m == n:", "rank_o = rank_u = RANK_SQ", "elif m > n:",
             "rank_o, rank_u = RANK_LONG, RANK_SHORT", "else:",
             "rank_o, rank_u = RANK_SHORT, RANK_LONG",
             "rank_o = min(rank_o, m)", "rank_u = min(rank_u, n)"]
PLACEMENTS = {"sq512": (512, 32, 32), "short512": (32, 512, 32), "long512": (32, 32, 512)}


def test_sources():
    """Each new script = its base plus exactly the intended lines."""
    families = [  # (base seed0, family, seeds, removed, added) for the family's seed0
        ("cosmos_r512", "cosmos_r512", (1, 2), None, None),
        ("sinksoap", "sinksoap", (1, 2), None, None),
        ("cosmos_r0", "cosmos_r0_nonormuon", (0, 1, 2), GATE_OLD, GATE_NEW),
        ("cosmos_r32", "cosmos_r32_nonormuon", (0, 1, 2), GATE_OLD, GATE_NEW),
        ("cosmos_r512", "cosmos_r512_nonormuon", (0, 1, 2), GATE_OLD, GATE_NEW),
        ("sinksoap", "sinksoap_nonormuon", (0, 1, 2), SINK_OLD, SINK_NEW),
    ]
    for name, (sq, short, long_) in PLACEMENTS.items():
        const = [f"RANK_SQ, RANK_SHORT, RANK_LONG = {sq}, {short}, {long_}"]
        families.append(("cosmos_r32", f"cosmos_r32_{name}", (0, 1), PLACE_OLD, const + PLACE_NEW))
    n = 0
    for base, fam, seeds, rem, add in families:
        for s in seeds:
            if s == 0:
                got = code_changes(f"{V}{base}_seed0.py", f"{V}{fam}_seed0.py")
                assert got == (rem, add), (fam, got)
            else:
                got = code_changes(f"{V}{fam}_seed0.py", f"{V}{fam}_seed{s}.py")
                assert got == (seed_lines(0), seed_lines(s)), (fam, s, got)
            n += 1
    print(f"sources: {n} new scripts differ from their base only in the intended lines")


def normuon_post(u, sm, beta2=0.95, eps=1e-10):
    # The step APPLY_NORMUON gates, restated outside the variant code.
    u = u.float()
    m, n = u.shape
    norm = u.norm()
    v = (u * u).mean(dim=-1, keepdim=True) if m >= n else (u * u).mean(dim=-2, keepdim=True)
    sm.lerp_(v, 1 - beta2)
    u = u / sm.sqrt().clamp_min(eps)
    return u * (norm / u.norm().clamp_min(eps))


def call_any(ns, g, s):
    if FN in ns:
        return call(ns[FN], g, s)
    return ns["cosmos_r0_update"](g.clone(), s["momentum"], s["second_momentum"])


def test_nonormuon(base, var, r, tag):
    assert var["APPLY_NORMUON"] is False
    for m, n in SHAPES:
        s0 = fresh_state(m, n, r)
        var["APPLY_NORMUON"] = True
        s_a, s_b = clone_state(s0), clone_state(s0)
        for g in grads(m, n, 4, seed=3 + m):
            assert torch.equal(call_any(base, g, s_a), call_any(var, g, s_b)), (tag, m, n)
            for key in s_a:
                assert torch.equal(s_a[key], s_b[key]), (tag, m, n, key)
        var["APPLY_NORMUON"] = False
        s_a, s_b = clone_state(s0), clone_state(s0)
        sm_ext = s0["second_momentum"].clone()
        target = (max(1.0, m / n) * min(m, n)) ** 0.5
        gap, post = 0.0, 0.0
        for g in grads(m, n, 6, seed=3 + m):
            u_a, u_b = call_any(base, g, s_a), call_any(var, g, s_b)
            assert u_b.dtype == torch.bfloat16 and torch.isfinite(u_b.float()).all(), (tag, m, n)
            for key in ("momentum", "U", "O", "S", "R"):
                assert torch.equal(s_a[key], s_b[key]), (tag, m, n, key)
            assert float(s_b["second_momentum"].abs().sum()) == 0.0, (tag, m, n)
            assert abs(u_b.float().norm().item() / target - 1) < 1e-2, (tag, m, n)
            if m == n:
                assert torch.equal(u_a, u_b), (tag, m, n)
            else:
                gap = max(gap, rel(u_a, u_b))
                post = max(post, rel(u_a, normuon_post(u_b, sm_ext)))
        if m != n:
            print(f"{tag} nonormuon {m}x{n}: differs from base by {gap:.2e}; "
                  f"base recovered by external NorMuon step to {post:.2e}")
            assert gap > 1e-2, (tag, m, n, gap)
            assert post < 1e-2, (tag, m, n, post)
    print(f"{tag} nonormuon: switch on bit-identical to base; off leaves square case, momentum and subspace state unchanged")


def test_r0_nonormuon_vs_muon(var):
    for m, n in SHAPES:
        mom_a, mom_b = torch.zeros(m, n), torch.zeros(m, n)
        sm = torch.zeros(m, 1) if m >= n else torch.zeros(1, n)
        lo, hi, cmin = float("inf"), 0.0, 1.0
        for g in grads(m, n, 6, seed=5 + m):
            u_muon = var["muon_update"](g.clone(), mom_a).float()
            u_r0 = var["cosmos_r0_update"](g.clone(), mom_b, sm).float()
            assert torch.equal(mom_a, mom_b), (m, n)
            cmin = min(cmin, ((u_muon * u_r0).sum() / (u_muon.norm() * u_r0.norm())).item())
            c = (u_r0.norm() / u_muon.norm()).item()
            lo, hi = min(lo, c), max(hi, c)
        print(f"r0 nonormuon vs muon_update {m}x{n}: cosine >= {cmin:.6f}, step scale {lo:.3f}-{hi:.3f}x Muon")
        assert cmin > 0.9999, (m, n, cmin)


def test_sinksoap_nonormuon(base, var):
    fn_a, fn_b = base["gram_sinkhorn_normuon_update"], var["gram_sinkhorn_normuon_update"]
    for m, n in SHAPES:
        def state():
            return dict(momentum=torch.zeros(m, n), left_gram=torch.zeros(m, m),
                        right_gram=torch.zeros(n, n),
                        second_momentum=torch.zeros(m, 1) if m >= n else torch.zeros(1, n))
        s_on, s_chk, s_off = state(), state(), state()
        sm_ext = s_off["second_momentum"].clone()
        target = (max(1.0, m / n) * min(m, n)) ** 0.5
        gap, post = 0.0, 0.0
        for g in grads(m, n, 5, seed=7 + m):
            u_on = fn_a(g.clone(), **s_on)
            assert torch.equal(u_on, fn_b(g.clone(), **s_chk, apply_normuon=True)), (m, n)
            u_off = fn_b(g.clone(), **s_off, apply_normuon=False)
            assert torch.isfinite(u_off).all(), (m, n)
            for key in ("momentum", "left_gram", "right_gram"):
                assert torch.equal(s_on[key], s_off[key]), (m, n, key)
            assert float(s_off["second_momentum"].abs().sum()) == 0.0, (m, n)
            assert abs(u_off.norm().item() / target - 1) < 1e-3, (m, n)
            if m == n:
                assert torch.equal(u_on, u_off), (m, n)
            else:
                gap = max(gap, rel(u_on, u_off))
                post = max(post, rel(u_on, normuon_post(u_off, sm_ext)))
        if m != n:
            print(f"sinksoap nonormuon {m}x{n}: differs from base by {gap:.2e}; "
                  f"base recovered by external NorMuon step to {post:.2e}")
            assert gap > 1e-2 and post < 1e-4, (m, n, gap, post)
    print("sinksoap nonormuon: same function as base; apply_normuon=False leaves square case and Gram state unchanged")


def test_placement(r32, var, name):
    assert (var["RANK_SQ"], var["RANK_SHORT"], var["RANK_LONG"]) == PLACEMENTS[name], name

    def run(ns, steps):
        torch.manual_seed(0)
        params = [torch.nn.Parameter(torch.randn(m, n)) for m, n in SHAPES]
        opt = ns["COSMOS"](params, lr=0.02, weight_decay=0.01)
        gen = torch.Generator().manual_seed(7)
        for _ in range(steps):
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen)
            opt.step()
        return params, opt

    var["RANK_SQ"], var["RANK_SHORT"], var["RANK_LONG"] = 32, 32, 32
    for a, b in zip(run(r32, 4)[0], run(var, 4)[0]):
        assert torch.equal(a, b), (name, tuple(a.shape))
    var["RANK_SQ"], var["RANK_SHORT"], var["RANK_LONG"] = 8, 12, 20
    params, opt = run(var, 5)
    want = {(64, 64): (8, 8), (192, 48): (20, 12), (48, 192): (12, 20)}  # (rank_o, rank_u)
    for p in params:
        m, n = p.shape
        st, (ro, ru) = opt.state[p], want[(m, n)]
        assert st["O"].shape == (m, ro) and st["U"].shape == (n, ru), (name, m, n)
        assert st["R"].shape == (ro, ro) and st["S"].shape == (ru, ru), (name, m, n)
        for q in (st["O"], st["U"]):
            assert (q.mT @ q - torch.eye(q.size(1))).abs().max() < 1e-4, (name, m, n)
        assert torch.isfinite(p).all(), (name, m, n)
    var["RANK_SQ"], var["RANK_SHORT"], var["RANK_LONG"] = PLACEMENTS[name]
    print(f"{name}: all-32 bit-identical to r32; sides get (square, short, long) ranks as set; unequal ranks run finite")


def main():
    torch.set_num_threads(4)
    r32 = load(V + "cosmos_r32_seed0.py")
    test_r0_equals_rank0_limit(r32, load(V + "cosmos_r0_seed0.py"))
    for k in (2, 4, 8):
        test_qrk(r32, load(V + f"cosmos_r32_qrk{k}_seed0.py"), k)
    for k in (5, 3, 1):
        test_sink(r32, load(V + f"cosmos_r32_sink{k}_seed0.py"), k)
    test_specns6(load(V + "baseline_specns6_seed0.py"), load(V + "baseline_ns6_seed0.py"))
    test_sources()
    r0_off = load(V + "cosmos_r0_nonormuon_seed0.py")
    test_nonormuon(load(V + "cosmos_r0_seed0.py"), r0_off, 0, "r0")
    test_r0_nonormuon_vs_muon(r0_off)
    test_nonormuon(r32, load(V + "cosmos_r32_nonormuon_seed0.py"), 32, "r32")
    test_nonormuon(load(V + "cosmos_r512_seed0.py"), load(V + "cosmos_r512_nonormuon_seed0.py"), 40, "r512-code@40")
    test_sinksoap_nonormuon(load(V + "sinksoap_seed0.py"), load(V + "sinksoap_nonormuon_seed0.py"))
    for name in PLACEMENTS:
        test_placement(r32, load(V + f"cosmos_r32_{name}_seed0.py"), name)
    print("VARIANT_MATH_OK")


if __name__ == "__main__":
    main()
