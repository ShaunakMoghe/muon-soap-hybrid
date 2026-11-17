"""Deflation-spectrum check (CPU analysis; results in analysis.md Sec. 2.3).

Why does COSMOS tolerate NS6 end-to-end while Muon collapses?
Hypothesis under test: removing dominant singular components shrinks
the residual's Frobenius norm, so after normalization the remaining
(nonzero) singular values start larger and need fewer NS iterations.

Loads run18 (COSMOS-r32) + run16 (Muon) ckpt_02750 on CPU. M_hat proxy =
saved momentum buffer (same documented approximation as
profile/ns_residual_check.py). Per 2D block param:
  M_res = (I-OO^T) M (I-UU^T), then SVD spectra after Frobenius
  normalization, smallest-nonzero-sigma comparison, and NS
  orthogonalization error on the NONZERO subspace only vs the existing
  metric (which counts ~r exactly-zero deflated directions as misses).
"""
import torch

COSMOS_CKPT = "runs/run18_cosmos_r32/checkpoints/ckpt_02750.pt"
MUON_CKPT = "runs/run16_muon_s1/checkpoints/ckpt_02750.pt"
ZERO_TOL = 1e-6  # relative: normalized-spectrum entries below this = deflated


def ns_poly(s, iters):
    # Per-singular-value map of the harness quintic NS iteration.
    for _ in range(iters):
        s = 2 * s - 1.5 * s ** 3 + 0.5 * s ** 5
    return s


def spec_metrics(spec, iters_list=(12, 8, 6)):
    spec = spec.double()
    fnorm = spec.norm().item()
    n = spec / (fnorm + 1e-30)
    mask = n > ZERO_TOL
    out = {
        "fnorm": fnorm,
        "n_total": len(n),
        "n_zero": int((~mask).sum()),
        "smin_nz": n[mask].min().item() if mask.any() else float("nan"),
        "p10_nz": n[mask].quantile(0.1).item() if mask.any() else float("nan"),
        "mean_nz": n[mask].mean().item() if mask.any() else float("nan"),
    }
    for it in iters_list:
        f = ns_poly(n.clone(), it)
        dev = (f ** 2 - 1) ** 2
        out[f"err_all_{it}"] = dev.mean().sqrt().item()
        out[f"err_nz_{it}"] = dev[mask].mean().sqrt().item() if mask.any() else float("nan")
        tot = dev.sum().item()
        out[f"zero_share_{it}"] = float((~mask).sum()) / tot if tot > 0 else 0.0
    return out


def main():
    torch.set_num_threads(4)
    ck_c = torch.load(COSMOS_CKPT, map_location="cpu", weights_only=False)
    ck_m = torch.load(MUON_CKPT, map_location="cpu", weights_only=False)
    print(f"cosmos step={ck_c['step']} muon step={ck_m['step']}", flush=True)
    cs = ck_c["optims"][1]["state"]
    ms = ck_m["optims"][1]["state"]

    agg = {}  # shape -> {kind -> [metrics]}
    n_c = 0
    for k in sorted(cs, key=lambda x: int(x)):
        st = cs[k]
        M = st["momentum"].float()
        if M.ndim < 2:
            continue
        O, U = st["O"].float(), st["U"].float()
        Mr = M - O @ (O.mT @ M) - (M @ U) @ U.mT + O @ (O.mT @ (M @ U)) @ U.mT
        shp = tuple(M.shape)
        d = agg.setdefault(shp, {"M": [], "M_res": [], "M_muon": []})
        d["M"].append(spec_metrics(torch.linalg.svdvals(M)))
        d["M_res"].append(spec_metrics(torch.linalg.svdvals(Mr)))
        n_c += 1
    n_m = 0
    for k in sorted(ms, key=lambda x: int(x)):
        M = ms[k]["momentum"].float()
        if M.ndim < 2:
            continue
        shp = tuple(M.shape)
        d = agg.setdefault(shp, {"M": [], "M_res": [], "M_muon": []})
        d["M_muon"].append(spec_metrics(torch.linalg.svdvals(M)))
        n_m += 1
    print(f"cosmos 2D params: {n_c}, muon 2D params: {n_m}", flush=True)

    for shp in sorted(agg, key=str):
        d = agg[shp]
        print(f"\n== shape {shp} (n={len(d['M'])}/{len(d['M_res'])}/{len(d['M_muon'])}) ==")
        for kind in ("M", "M_res", "M_muon"):
            rows = d[kind]
            if not rows:
                print(f"  {kind}: none"); continue
            g = lambda f: sum(r[f] for r in rows) / len(rows)
            print(f"  {kind}: fnorm={g('fnorm'):.4g} n_zero={g('n_zero'):.1f}/{rows[0]['n_total']} "
                  f"smin_nz={g('smin_nz'):.4g} p10_nz={g('p10_nz'):.4g} mean_nz={g('mean_nz'):.4g}")
            for it in (12, 8, 6):
                print(f"    NS{it}: err_all={g(f'err_all_{it}'):.4g} err_nz={g(f'err_nz_{it}'):.4g} "
                      f"zero_share_of sumsq={g(f'zero_share_{it}'):.3f}")


if __name__ == "__main__":
    main()
