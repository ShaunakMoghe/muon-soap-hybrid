"""CPU-fast ckpt round-trip narrowing test (2026-10-04).

Isolates research_stack/ckpt.py mechanics from the full training resume:
t monuments: weights_only unpickling, custom param-group keys (initial_lr),
optimizer load_state_dict, RNG restore, step/clock round-trip.

SCOPE: uses torch AdamW (fused=False on CPU) + a Muon-shape stand-in is NOT
possible here (Muon class lives in single-file variants, deliberately not
importable). So: if THIS passes and GPU resume still fails, the bug is
Muon/fused-specific and GPU-side debugging continues. If this fails, the bug
is in generic ckpt mechanics and the fix is verifiable here in seconds.
"""
import os
import sys

import torch
import torch.distributed as dist
from torch import nn

sys.path.insert(0, os.environ.get("PROJ", "/storage/project/ps-moereasoning-0/smoghe6"))
from research_stack.ckpt import load_ckpt, save_ckpt  # noqa: E402


def build():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 4))
    o1 = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.8, 0.95),
                           eps=1e-10, weight_decay=0.001, fused=False)
    for g in o1.param_groups:
        g["initial_lr"] = g["lr"]
    return model, [o1]


def main():
    os.chdir("/tmp")
    dist.init_process_group(backend="gloo",
                            init_method=f"file:///tmp/ckpt_repro_{os.getpid()}",
                            rank=0, world_size=1)
    model, optims = build()
    x = torch.randn(8, 16)
    for _ in range(3):  # dirty the state (momentum buffers, step counters)
        optims[0].zero_grad()
        model(x).square().mean().backward()
        optims[0].step()
    ref_params = [p.detach().clone() for p in model.parameters()]
    ref_mom = optims[0].state[model[0].weight]["exp_avg"].clone()
    p = save_ckpt(100, 123.456, model, optims)
    print("saved:", p)

    model2, optims2 = build()  # fresh objects, like a resumed process
    step, tt = load_ckpt(p, model2, optims2, map_location="cpu")
    assert step == 100, step
    assert abs(tt - 123.456) < 1e-9, tt
    for a, b in zip(ref_params, model2.parameters()):
        assert torch.equal(a, b), "param mismatch after load"
    assert torch.equal(ref_mom, optims2[0].state[model2[0].weight]["exp_avg"]), "momentum mismatch"
    assert optims2[0].param_groups[0]["initial_lr"] == 1e-3, "custom group key lost"
    dist.destroy_process_group()
    print("CKPT_ROUNDTRIP_CPU_OK")


if __name__ == "__main__":
    main()
