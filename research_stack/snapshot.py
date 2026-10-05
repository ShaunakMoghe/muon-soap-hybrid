"""Write run_dir/meta.json: script sha256, torch/CUDA versions, GPU, SLURM job,
git hashes. Run BEFORE launching training (it briefly inits CUDA).

Usage (from $PROJ): ./run_container.sh python -m research_stack.snapshot <run_dir> <script> [cmd...]
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys


def main():
    run_dir, script = sys.argv[1], sys.argv[2]
    import torch
    meta = {
        "script": script,
        "sha256": hashlib.sha256(open(script, "rb").read()).hexdigest(),
        "torch": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cmdline": sys.argv[3:],
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
    for name, path in (("public", "modded-nanogpt-public"),
                       ("private", "Modded-NanoGPT-Private")):
        try:
            out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=15)
            if out.returncode == 0:
                meta[f"git_{name}"] = out.stdout.strip()
        except Exception:
            pass
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {run_dir}/meta.json")


if __name__ == "__main__":
    main()
