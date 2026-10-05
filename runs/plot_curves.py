"""Plot val_loss vs step for every run with runs/*/stdout.log. CPU-only.

Usage: ./run_container.sh python runs/plot_curves.py   (writes runs/curves.png)
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/storage/project/ps-moereasoning-0/smoghe6"

runs = {}
for log in sorted(glob.glob(os.path.join(PROJ, "runs", "run*", "stdout.log"))):
    pts = []
    nlogs = 0  # only the FIRST run per file (run4's log also holds killed ghost run4b)
    with open(log, errors="ignore") as f:
        for line in f:
            if line.startswith("logs/"):
                nlogs += 1
                if nlogs > 1:
                    break
            m = re.match(r"step:(\d+)/\d+ val_loss:([0-9.]+)", line)
            if m:
                pts.append((int(m.group(1)), float(m.group(2))))
    if pts:
        runs[os.path.basename(os.path.dirname(log))] = pts

for name, pts in sorted(runs.items()):
    xs, ys = zip(*pts)
    plt.plot(xs, ys, marker=".", ms=3, label=name)
plt.axhline(3.28, color="k", ls="--", lw=1, label="target 3.28")
plt.xlabel("step")
plt.ylabel("val_loss")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
out = os.path.join(PROJ, "runs", "curves.png")
plt.savefig(out, dpi=150)
print("wrote", out, sorted(runs))
