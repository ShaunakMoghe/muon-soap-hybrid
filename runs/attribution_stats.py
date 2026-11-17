"""Seed-paired statistics for the attribution batch (runs 57-69) and the
cadence / Sinkhorn sweeps (runs 43-54). Reads logs via parse_results.py;
prints only. Decision rule: runs/predictions.md, "Attribution batch".
"""
import itertools, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_results import parse

R = {n: parse(n) for n in list(range(20, 26)) + list(range(29, 70))}
ARMS = {  # seeds 0..4
    "muon": [22, 24, 61, 62, 63],
    "r0": [57, 58, 64, 65, 66],
    "r32": [23, 25, 67, 68, 69],
}
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


def paired(name, a, b, key="final"):
    """Differences a - b per seed; positive = a has the higher value."""
    d = [R[x][key] - R[y][key] for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = m / se
    perm = sum(sum(s * x for s, x in zip(signs, d)) >= sum(d) - 1e-12 for signs in itertools.product((1, -1), repeat=n)) / 2 ** n
    print(f"{name:<34} n={n} diffs " + " ".join(f"{x:+.5f}" for x in d))
    print(f"{'':<34} mean {m:+.5f} sd {sd:.5f} se {se:.5f} t {t:+.2f} one-sided p {t_sf(t, n - 1):.4f} sign-flip p {perm:.4f} 95% CI [{m - T975[n-1]*se:+.5f}, {m + T975[n-1]*se:+.5f}]")
    return d


print("== arms (seed 0..4): final / crossing ==")
for k, runs in ARMS.items():
    f = [R[n]["final"] for n in runs]
    m = sum(f) / 5
    sd = math.sqrt(sum((x - m) ** 2 for x in f) / 4)
    print(f"{k:<5} finals " + " ".join(f"{x:.5f}" for x in f) + f"  mean {m:.5f} sd {sd:.5f}  cross " + " ".join(str(R[n]["cross"]) for n in runs))

print("\n== paired final-loss differences (first minus second) ==")
paired("r0 - r32  (subspace effect)", ARMS["r0"], ARMS["r32"])
paired("muon - r0 (NorMuon-only edge)", ARMS["muon"], ARMS["r0"])
paired("muon - r32 (full COSMOS edge)", ARMS["muon"], ARMS["r32"])
print("\n== paired crossing-step differences ==")
paired("muon - r0 crossing", ARMS["muon"], ARMS["r0"], "cross")
paired("muon - r32 crossing", ARMS["muon"], ARMS["r32"], "cross")
paired("r0 - r32 crossing", ARMS["r0"], ARMS["r32"], "cross")

print("\n== curve gaps r0 - r32, per seed, at checkpoints ==")
for s in (125, 250, 500, 1000, 1500, 2000, 2500, 3000, 3250):
    d = [R[a]["vals"][s] - R[b]["vals"][s] for a, b in zip(ARMS["r0"], ARMS["r32"])]
    d2 = [R[a]["vals"][s] - R[b]["vals"][s] for a, b in zip(ARMS["muon"], ARMS["r32"])]
    print(f"@{s:<5} r0-r32 mean {sum(d)/5:+.5f} [{min(d):+.5f},{max(d):+.5f}]   muon-r32 mean {sum(d2)/5:+.5f} [{min(d2):+.5f},{max(d2):+.5f}]")

print("\n== Muon hypers (lr .025 / wd .05), seeds 0/1 ==")
for lab, runs in (("muon", [22, 24]), ("r0@MH", [59, 60]), ("r32@MH", [41, 42])):
    print(f"{lab:<7} " + "  ".join(f"{R[n]['cross']}/{R[n]['final']:.5f}" for n in runs))
paired("muon - r0@MH", [22, 24], [59, 60])
paired("r32@MH - r0@MH", [41, 42], [59, 60])

print("\n== sweeps vs seed-paired r32 baseline (runs 23/25) and r0 (57/58) ==")
for lab, runs in (("qr k=2", [43, 44]), ("qr k=4", [45, 46]), ("qr k=8", [47, 48]), ("sink 5", [49, 50]), ("sink 3", [51, 52]), ("sink 1", [53, 54])):
    cells = []
    for n, b, z in zip(runs, [23, 25], [57, 58]):
        cells.append(f"{R[n]['cross']}/{R[n]['final']:.5f} (vs r32 {R[n]['cross']-R[b]['cross']:+d}, {R[n]['final']-R[b]['final']:+.5f}; vs r0 {R[n]['final']-R[z]['final']:+.5f}; @125 {R[n]['vals'][125]:.3f}; {R[n]['node'][-8:]} {R[n]['steady']*1e3:.1f}ms)")
    print(f"{lab:<7} " + " | ".join(cells))

print("\n== step time and time-to-3.28, node atl1-1-03-020-2-0 only ==")
for k, runs in (("muon", [61, 62, 63]), ("r0", [57, 58, 64, 65, 66]), ("r0@MH", [59, 60]), ("r32", [67, 68, 69]), ("qr k=8", [47, 48]), ("sink 1", [53, 54])):
    rs = [R[n] for n in runs if R[n]["node"].endswith("020-2-0")]
    print(f"{k:<7} steady ms " + " ".join(f"{r['steady']*1e3:.1f}" for r in rs) + "  wall ms " + " ".join(f"{r['wall']*1e3:.1f}" for r in rs) + "  s to cross " + " ".join(f"{r['cross_time']:.0f}@{r['cross']}" for r in rs) + "  total s " + " ".join(f"{r['train_time']:.0f}" for r in rs))
