"""NS update-norm / spectrum-shape check (CPU analysis; results in analysis.md Sec. 2.3).

Question: COSMOS tolerates NS6 end-to-end, Muon does not (runs 31/32 vs
35/36). COSMOS rescales its final update to a fixed Frobenius norm
(`update * muon_norm / ||update||_F`); Muon applies the raw NS output.
How much of the nominal (fully orthogonal) update norm does Muon lose at
NS6/NS8, and how non-flat is the output spectrum, for (a) Muon's
Frobenius-normalized start, (b) the spectral-norm start used by the
specns6 probe (runs 55/56), (c) COSMOS's deflated residual M_res?

Per 2D block param, from singular values of the momentum buffer (M_hat
proxy, same documented approximation as deflation_spectra_scratch.py):
  rho_k  = sqrt(mean f_k^2)            update ||.||_F / orthogonal ||.||_F
  pr_k   = (mean f_k^2)^2 / mean f_k^4 flatness of output spectrum (1=flat)
  conv_k = frac of directions with f_k > 0.9
where f_k is the harness quintic NS map applied k times in fp64.
M_res metrics are on the nonzero support only (32 deflated zeros dropped).
Also replays the specns6 power iteration verbatim to check
sigma_est/sigma_1 (NS diverges if the normalized top value exceeds sqrt(2)).
"""
import torch

MUON_CKPTS = {
    "muon_s1": "runs/run16_muon_s1/checkpoints/ckpt_02750.pt",
    "muon_s2": "runs/run17_muon_s2/checkpoints/ckpt_02750.pt",
}
COSMOS_CKPT = "runs/run18_cosmos_r32/checkpoints/ckpt_02750.pt"
ZERO_TOL = 1e-6
ITERS = (12, 8, 6)
SPEC_POWER_ITERS = 10


def ns_poly(s, iters):
    for _ in range(iters):
        s = 2 * s - 1.5 * s ** 3 + 0.5 * s ** 5
    return s


def power_sigma(M):
    # Numerics of variants/train_gpt_baseline_specns6_seed0.py, unbatched.
    X = M.bfloat16()
    if X.size(-2) > X.size(-1):
        X = X.mT
    sv = torch.ones(X.size(-1), 1, dtype=torch.float32)
    sv = sv / (sv.norm() + 1e-7)
    Xf = X.float()
    for _ in range(SPEC_POWER_ITERS):
        u = Xf @ sv
        u = u / (u.norm() + 1e-7)
        sv = Xf.mT @ u
        sv = sv / (sv.norm() + 1e-7)
    return (u.mT @ Xf @ sv).squeeze().clamp_min(1e-7).item()


def shape_metrics(start, mask):
    out = {}
    for it in ITERS:
        f = ns_poly(start.clone(), it)[mask]
        m2 = (f ** 2).mean()
        out[f"rho_{it}"] = m2.sqrt().item()
        out[f"pr_{it}"] = (m2 ** 2 / (f ** 4).mean()).item()
        out[f"conv_{it}"] = (f > 0.9).double().mean().item()
    return out


def add(agg, shp, kind, row):
    agg.setdefault(shp, {}).setdefault(kind, []).append(row)


def main():
    torch.set_num_threads(4)
    agg = {}
    for name, path in MUON_CKPTS.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        st = ck["optims"][1]["state"]
        n2d = 0
        for k in sorted(st, key=lambda x: int(x)):
            M = st[k]["momentum"].float()
            if M.ndim < 2:
                continue
            n2d += 1
            shp = tuple(M.shape)
            sv = torch.linalg.svdvals(M).double()
            frob = sv / (sv.norm() + 1e-30)
            allm = torch.ones_like(frob, dtype=torch.bool)
            add(agg, shp, f"{name} frob", shape_metrics(frob, allm))
            sig = power_sigma(M)
            row = shape_metrics(sv / sig, allm)
            row["est_ratio"] = sig / sv.max().item()
            row["stable_rank"] = ((sv ** 2).sum() / sv.max() ** 2).item()
            add(agg, shp, f"{name} spec", row)
        print(f"{name}: step={ck['step']} 2D params={n2d}", flush=True)
        del ck, st

    ck = torch.load(COSMOS_CKPT, map_location="cpu", weights_only=False)
    st = ck["optims"][1]["state"]
    n2d = 0
    for k in sorted(st, key=lambda x: int(x)):
        s = st[k]
        M = s["momentum"].float()
        if M.ndim < 2:
            continue
        n2d += 1
        O, U = s["O"].float(), s["U"].float()
        Mr = M - O @ (O.mT @ M) - (M @ U) @ U.mT + O @ (O.mT @ (M @ U)) @ U.mT
        shp = tuple(M.shape)
        sv = torch.linalg.svdvals(Mr).double()
        frob = sv / (sv.norm() + 1e-30)
        add(agg, shp, "cosmos M_res frob(support)", shape_metrics(frob, frob > ZERO_TOL))
    print(f"cosmos_r32: step={ck['step']} 2D params={n2d}", flush=True)

    for shp in sorted(agg, key=str):
        print(f"\n== shape {shp} ==")
        for kind in sorted(agg[shp]):
            rows = agg[shp][kind]
            g = lambda f: sum(r[f] for r in rows) / len(rows)
            parts = [f"NS{it}: rho={g(f'rho_{it}'):.3f} pr={g(f'pr_{it}'):.3f} conv={g(f'conv_{it}'):.3f}"
                     for it in ITERS]
            lo = min(r["rho_6"] for r in rows)
            hi = max(r["rho_6"] for r in rows)
            print(f"  {kind:<28} n={len(rows):>2} | " + " | ".join(parts)
                  + f" | rho_6 range [{lo:.3f}, {hi:.3f}]")
            if "est_ratio" in rows[0]:
                er = [r["est_ratio"] for r in rows]
                print(f"  {'':<28}        sigma_est/sigma_1: min={min(er):.4f} mean={sum(er)/len(er):.4f} "
                      f"max={max(er):.4f}; stable rank mean={g('stable_rank'):.1f}")


if __name__ == "__main__":
    main()
