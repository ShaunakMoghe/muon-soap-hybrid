"""Checkpoint save/resume + preemption handling for single-GPU training runs.

Single-GPU ONLY: Muon-style param sharding spreads optimizer state across
ranks, so rank 0 cannot snapshot everything alone. save/load assert
world_size == 1 (loud failure, never silent corruption).

Checkpoint contents: next 0-based step S, model + optimizer states,
train_time clock, torch/CUDA RNG states. Resume fast-forwards the
deterministic dataloader by exactly S batches.
"""
import os
import signal
import sys

import torch
import torch.distributed as dist

CKPT_EVERY_DEFAULT = 250
KEEP_LAST = 2

_term_requested = False


def _on_term(signum, frame):
    global _term_requested
    _term_requested = True


def arm_preemption_handlers():
    """Catch SIGTERM (SLURM preemption) / SIGINT; the main loop polls
    term_requested() at a safe point (no NCCL in flight) and exits cleanly."""
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)


def term_requested():
    return _term_requested


def ckpt_dir():
    d = os.path.join(os.getcwd(), "checkpoints")
    os.makedirs(d, exist_ok=True)
    return d


def save_ckpt(step, train_time, model, optimizers):
    """Save state to resume AT step (next 0-based step). Rank 0, safe point."""
    assert dist.get_world_size() == 1, "ckpt only supports world_size 1"
    d = ckpt_dir()
    final = os.path.join(d, f"ckpt_{step:05d}.pt")
    tmp = final + ".tmp"
    torch.save({
        "step": step,
        "train_time": train_time,
        "model": model.state_dict(),
        "optims": [o.state_dict() for o in optimizers],
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, tmp)
    os.replace(tmp, final)  # atomic: a kill mid-write never corrupts `final`
    ckpts = sorted(f for f in os.listdir(d)
                   if f.startswith("ckpt_") and f.endswith(".pt"))
    for old in ckpts[:-KEEP_LAST]:
        os.remove(os.path.join(d, old))
    return final


def load_ckpt(path, model, optimizers):
    """Install checkpoint; returns (step, train_time)."""
    assert dist.get_world_size() == 1, "ckpt only supports world_size 1"
    ck = torch.load(path, map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"])
    assert len(ck["optims"]) == len(optimizers)
    for o, s in zip(optimizers, ck["optims"]):
        o.load_state_dict(s)
    torch.set_rng_state(ck["rng"])
    torch.cuda.set_rng_state_all(ck["cuda_rng"])
    return ck["step"], ck.get("train_time", 0.0)


def parse_resume_argv(argv):
    """Pop ['--resume', path] from argv if present; returns path or None."""
    if "--resume" in argv:
        i = argv.index("--resume")
        path = argv[i + 1]
        del argv[i:i + 2]
        return path
    return None
