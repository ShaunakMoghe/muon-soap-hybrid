"""Parse 8-GPU run logs into one table (stdout only; no files written).

Usage: python3 runs/parse_results.py [run numbers...]   (default 20-69)
Per run: crossing = first validated step with val_loss <= 3.28; final =
val at the last step; wall = final train_time / steps; steady = slope of
train_time over steps 1000..last (375..500 for 500-step runs); node from
the newest runs/sbatch/runN_8gpu_<job>.out; code = logged source equals
the variant script named in runs/sbatch/runN_8gpu.sbatch.
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+|nan|inf) train_time:([\d.]+)s")


def parse(n):
    dirs = glob.glob(f"{ROOT}/runs/run{n}_*/")
    if len(dirs) != 1:
        return None
    d = dirs[0]
    r = {"run": n, "dir": os.path.basename(d.rstrip("/"))}
    txt = open(d + "stdout.log", errors="replace").read()
    uuids = re.findall(r"^logs/([0-9a-f-]+)\.txt$", txt, re.M)
    r["uuid"] = uuids[-1] if uuids else None
    r["n_uuid"] = len(uuids)
    vals = {}
    for line in txt.splitlines():
        m = VAL.match(line)
        if m:
            vals[int(m[1])] = (float(m[3]), float(m[4]))
            r["steps"] = int(m[2])
    r["vals"] = {k: v[0] for k, v in vals.items()}
    last = max(vals)
    r["last"], r["final"], r["train_time"] = last, vals[last][0], vals[last][1]
    r["cross"] = next((s for s in sorted(vals) if vals[s][0] <= 3.28), None)
    r["cross_time"] = vals[r["cross"]][1] if r["cross"] else None
    r["wall"] = vals[last][1] / last
    a = 1000 if last > 1000 else 375
    r["steady"] = (vals[last][1] - vals[a][1]) / (last - a)
    m = re.search(r"max_memory_allocated: ([\d.]+) GB", txt)
    r["mem"] = float(m[1]) if m else None
    r["flags"] = [k for k, pat in (("traceback", "Traceback"), ("recompile", r"(?i)recompil"), ("nanval", r"val_loss:(nan|inf)"), ("cudaerr", "CUDA error")) if re.search(pat, txt)]
    outs = sorted(glob.glob(f"{ROOT}/runs/sbatch/run{n}_8gpu_*.out"), key=lambda p: int(re.search(r"_(\d+)\.out$", p)[1]))
    if outs:
        o = open(outs[-1], errors="replace").read()
        r["job"] = int(re.search(r"_(\d+)\.out$", outs[-1])[1])
        m = re.search(r"^Nodes:\s+(\S+)", o, re.M)
        r["node"] = m[1] if m else None
        r["preflight"] = "GPU_PREFLIGHT_OK 8" in o
        r["done_marker"] = f"RUN{n}_8GPU_DONE" in o
    sb = f"{ROOT}/runs/sbatch/run{n}_8gpu.sbatch"
    if os.path.exists(sb) and r["uuid"]:
        m = re.search(r"\$PROJ/(variants/\S+\.py)", open(sb).read())
        if m:
            r["script"] = m[1]
            src = open(f"{ROOT}/{m[1]}").read()
            r["code_ok"] = open(f"{d}logs/{r['uuid']}.txt", errors="replace").read().startswith(src)
    return r


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--json"]
    runs = [int(a) for a in args] or range(20, 70)
    out = [r for r in map(parse, runs) if r]
    if "--json" in sys.argv:
        print(json.dumps(out))
        sys.exit()
    for r in out:
        v = r["vals"]
        g = lambda s: f"{v[s]:.5f}" if s in v else "   -   "
        print(f"run{r['run']:<3} {r['dir'][len(str(r['run']))+4:]:<22} last {r['last']:>4}/{r.get('steps')} cross {str(r['cross']):>5} final {r['final']:.5f} @125 {g(125)} @500 {g(500)} @2500 {g(2500)} @3000 {g(3000)} "
              f"tt {r['train_time']:8.3f} wall {r['wall']*1e3:5.1f} steady {r['steady']*1e3:5.1f}ms mem {r['mem']} node {str(r.get('node'))[-8:]} job {r.get('job')} "
              f"pre {r.get('preflight')} done {r.get('done_marker')} code {r.get('code_ok')} nuuid {r['n_uuid']} flags {r['flags']}")
