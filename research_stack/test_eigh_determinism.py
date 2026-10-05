"""Committed CUDA eigh determinism check (redo of the lost /tmp probe, 2026-10-04).

20x in-process bitwise comparison on full-rank and rank-deficient 3072 grams
(the SOAP/SinkSOAP case); prints result hashes. Cross-process check = run this
script 3x in separate processes and compare hash lines (see eigh_det.sbatch).
"""
import hashlib
import sys

import torch


def h(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, "torch:", torch.__version__)
    torch.manual_seed(0)
    ok = True
    cases = {
        "fullrank-3072-fp64": (lambda: (lambda G: G @ G.T)(
            torch.randn(3072, 3072, dtype=torch.float64, device=dev)))(),
        "rank12-3072-fp64": (lambda: (lambda G: G @ G.T)(
            torch.randn(3072, 12, dtype=torch.float64, device=dev)))(),
    }
    for name, C in cases.items():
        ref = None
        for i in range(20):
            e, v = torch.linalg.eigh(C)
            key = (h(e), h(v))
            if ref is None:
                ref = key
            elif key != ref:
                ok = False
                print(name, "MISMATCH at iter", i)
                break
        else:
            print(name, "20x identical hash=", ref[0], ref[1])
    print("EIGH_DET_OK" if ok else "EIGH_DET_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
