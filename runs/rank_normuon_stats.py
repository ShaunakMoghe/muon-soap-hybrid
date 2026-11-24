"""Seed-paired statistics for the rank x NorMuon batch (runs 70-96).

Reads logs via parse_results.py; prints only. Decision rules R1-R3 were
written into runs/predictions.md ("Rank / NorMuon factorial") before the
runs were submitted. All factorial cells use lr 0.04 / wd 0.025.
"""
import itertools, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_results import parse

ARMS = {  # seeds 0, 1, 2
    "muon": [22, 24, 61],          # Muon, own hypers (lr .025 / wd .05)
    "r0": [57, 58, 64],            # rank 0 = NorMuon-only control
    "r32": [23, 25, 67],
    "r256": [76, 77, 78],
    "r512": [70, 71, 72],
    "sink": [73, 74, 75],          # SinkSOAP, full rank
    "r0_off": [82, 83, 84],        # NorMuon step switched off
    "r32_off": [79, 80, 81],
    "r512_off": [85, 86, 87],
    "sink_off": [88, 89, 90],
}
MUON_CH = [39, 40]                 # Muon at COSMOS hypers, seeds 0/1 (sanity gate)
PLACE = {"attention": [91, 92], "mlp-768-side": [93, 94], "mlp-3072-side": [95, 96]}
R = {n: parse(n) for n in sorted({n for v in ARMS.values() for n in v} | set(MUON_CH)
                                  | {n for v in PLACE.values() for n in v} | {62, 63, 65, 66, 68, 69})}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def t_sf(t, df, n=200000, hi=200.0):
    """One-sided P(T > t), Student t, by Simpson integration of the pdf."""
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    f = lambda x: c * (1 + x * x / df) ** (-(df + 1) / 2)
    a, b = abs(t), hi
    h = (b - a) / n
    s = f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, n))
    p = s * h / 3
    return p if t >= 0 else 1 - p


def stats(d):
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se if se > 0 else float("inf")
    return m, sd, se, t, t_sf(t, n - 1), (m - T975[n - 1] * se, m + T975[n - 1] * se)


def fin(arm):
    return [R[n]["final"] for n in ARMS[arm]]


def diff(a, b):
    return [x - y for x, y in zip(a, b)]


def show(name, d):
    m, sd, se, t, p, (lo, hi) = stats(d)
    print(f"{name:<30} " + " ".join(f"{x:+.5f}" for x in d)
          + f"  mean {m:+.5f} t {t:+6.2f} p {p:.4f} 95% CI [{lo:+.5f}, {hi:+.5f}]")
    return m, p


print("== arms, seeds 0/1/2: final / crossing / @125 / @500 ==")
for k, runs in ARMS.items():
    f = fin(k)
    print(f"{k:<9} finals " + " ".join(f"{x:.5f}" for x in f) + f"  mean {sum(f)/3:.5f}"
          + "  cross " + " ".join(str(R[n]["cross"]) for n in runs)
          + "  @125 " + " ".join(f"{R[n]['vals'][125]:.3f}" for n in runs)
          + "  @500 " + " ".join(f"{R[n]['vals'][500]:.4f}" for n in runs))

print("\n== R1: d_X = r0(ON) - X(ON); BEATS iff mean >= 0.0015, p < 0.05, d > 0 at all seeds ==")
for x in ("r256", "r512", "sink"):
    d = diff(fin("r0"), fin(x))
    m, p = show(f"r0 - {x}", d)
    v = ("BEATS" if m >= 0.0015 and p < 0.05 and min(d) > 0 else
         "MATCHES" if abs(m) < 0.001 else "UNRESOLVED")
    print(f"{'':<30} -> {v}")

print("\n== R2: subspace x NorMuon (per seed) ==")
gate = [R[a]["final"] - R[b]["final"] for a, b in zip(ARMS["r0_off"][:2], MUON_CH)]
print("gate r0(OFF) - Muon@COSMOS hypers (s0, s1): " + " ".join(f"{x:+.5f}" for x in gate)
      + ("  PASS" if max(abs(x) for x in gate) <= 0.0015 else "  FAIL (band +-0.0015)"))
show("N(0)   = r0(OFF) - r0(ON)", diff(fin("r0_off"), fin("r0")))
for r, on, off in (("32", "r32", "r32_off"), ("512", "r512", "r512_off"), ("full", "sink", "sink_off")):
    s_off = diff(fin("r0_off"), fin(off))
    s_on = diff(fin("r0"), fin(on))
    inter = diff(s_off, s_on)
    show(f"S_off({r}) = r0(OFF) - {off}", s_off)
    show(f"S_on({r})  = r0(ON) - {on}", s_on)
    show(f"N({r})  = {off} - {on}", diff(fin(off), fin(on)))
    mi, _ = show(f"I({r})  = S_off - S_on", inter)
    v = ("SUBSTITUTES" if mi >= 0.0015 and min(inter) > 0 else
         "ADDITIVE" if abs(mi) < 0.001 else
         "REINFORCING" if mi <= -0.0015 and max(inter) < 0 else "UNRESOLVED")
    if r == "32" and abs(sum(s_off) / 3) < 0.001:
        v += " (rank 32 INERT)"
    print(f"{'':<30} -> {v}")

print("\n== R3: placement, seeds 0/1 (rank 32 everywhere, 512 on one group) ==")
gap = [R[a]["final"] - R[b]["final"] for a, b in zip(ARMS["r32"][:2], ARMS["r512"][:2])]
print("precondition r32 - r512 (s0, s1): " + " ".join(f"{x:+.5f}" for x in gap)
      + ("  MET" if min(gap) >= 0.002 else "  NOT MET"))
tot = 0.0
for k, runs in PLACE.items():
    g = [R[a]["final"] - R[p]["final"] for a, p in zip(ARMS["r32"][:2], runs)]
    sh = [x / y for x, y in zip(g, gap)]
    tot += sum(sh) / 2
    carries = sum(sh) / 2 >= 0.5 and min(g) >= 0.001
    print(f"{k:<14} finals " + " ".join(f"{R[n]['final']:.5f}" for n in runs)
          + "  r32-P " + " ".join(f"{x:+.5f}" for x in g) + "  share " + " ".join(f"{x:+.2f}" for x in sh)
          + f"  mean share {sum(sh)/2:+.2f}" + ("  CARRIES" if carries else ""))
print(f"sum of mean shares {tot:.2f}")

print("\n== step time (steady ms, steps 1000-3250) and seconds to 3.28, by node ==")
REF = {"020-2-0": [61, 62, 63], "014-23-0": [82]}   # Muon; r0(OFF) as Muon-cost proxy on 014-23-0
GROUPS = dict(ARMS, **PLACE, muon=[22, 24, 61, 62, 63])
for node, ref in REF.items():
    base = sum(R[n]["steady"] for n in ref) / len(ref)
    print(f"-- node ...{node}: reference {base*1e3:.1f} ms ({'Muon runs' if node == '020-2-0' else 'r0(OFF) run'} {ref})")
    for k, runs in GROUPS.items():
        rs = [R[n] for n in runs if R[n].get("node") and R[n]["node"].endswith(node)]
        if not rs:
            continue
        st = sum(r["steady"] for r in rs) / len(rs)
        print(f"   {k:<14} n={len(rs)} {st*1e3:6.1f} ms ({(st/base-1)*100:+5.1f}%)  s to 3.28: "
              + " ".join(f"{r['cross_time']:.0f}@{r['cross']}" if r["cross"] else "none" for r in rs))
