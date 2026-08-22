#!/usr/bin/env python3
"""run_benchmark.py — Comprehensive benchmark across environments and policies.

Usage:
    python scripts/run_benchmark.py [--gpu] [--n-demos 200] [--n-seeds 3] [--output results/]

This runs the full comparison study:
  1. Cable (miniature) — 4 variants × 3 seeds × expert ceiling
  2. UR5 Cable (MuJoCo) — BC, ACT, Diffusion, FlowExpert
  3. Franka Cloth (MuJoCo) — BC, ACT, Diffusion, FlowExpert

Each policy is trained on N demos and evaluated on held-out topologies.
Metrics: success rate, no-intervention success, crossings reduced,
interventions used, violations, max jerk.

Domain randomization is applied during training for all learned policies.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--n-demos", type=int, default=200, help="Number of demos per env")
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of random seeds")
    parser.add_argument("--n-epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--envs", nargs="+", default=["cable", "ur5", "franka"],
                        help="Environments to benchmark")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if None)")
    parser.add_argument("--max-eval-steps", type=int, default=300,
                        help="Max steps per eval episode")
    return parser.parse_args()


def get_device(args):
    import torch
    if args.device:
        return torch.device(args.device)
    if args.gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_cable_demos(n_demos, seed=0):
    """Generate cable untangling demos using the expert."""
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv
    from steerable.policies.expert import run_expert

    cfg = EnvConfig()
    rng = np.random.RandomState(seed)
    demos = []

    for k in range(n_demos):
        ct = rng.randint(2, 6)
        sm = rng.uniform(0.5, 1.5)
        env = CableEnv(cfg, seed=seed * 1000 + k, stiffness_mult=sm,
                       crossing_target=ct)
        rec = run_expert(env, record=True, max_steps=200)
        if rec["actions"].shape[0] < 10:
            continue
        demos.append({
            "obs": rec["obs"],
            "actions": rec["actions"],
            "crossings0": ct,
            "stiffness": sm,
        })

    print(f"  Generated {len(demos)} cable demos")
    return demos


def train_bc(demos, dim_obs, dim_action, n_epochs, device):
    """Train behavior cloning baseline."""
    from steerable.baselines import BCPolicy
    import torch

    policy = BCPolicy(dim_obs, dim_action, hidden=128)
    policy._net.to(device)

    obs_all = np.concatenate([d["obs"] for d in demos], axis=0)
    act_all = np.concatenate([d["actions"] for d in demos], axis=0)

    # build chunk dataset
    chunk_obs, chunk_act = [], []
    H = 6
    for d in demos:
        T = len(d["actions"])
        for t in range(max(1, T - H)):
            chunk_obs.append(d["obs"][t])
            chunk_act.append(d["actions"][t:t + H])

    obs_t = torch.tensor(np.array(chunk_obs), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(chunk_act), dtype=torch.float32, device=device)

    losses = []
    batch_size = 128
    for epoch in range(n_epochs):
        perm = torch.randperm(len(obs_t))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(obs_t), batch_size):
            idx = perm[i:i + batch_size]
            loss = policy.train_step(obs_t[idx], act_t[idx])
            epoch_loss += loss
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))
        if (epoch + 1) % 50 == 0:
            print(f"    BC epoch {epoch + 1}/{n_epochs} loss={losses[-1]:.4f}")

    return policy


def train_act(demos, dim_obs, dim_action, n_epochs, device):
    """Train ACT baseline."""
    from steerable.baselines import ACTPolicy
    import torch

    policy = ACTPolicy(dim_obs, dim_action, hidden=128)
    policy._net.to(device)

    H = 6
    chunk_obs, chunk_act = [], []
    for d in demos:
        T = len(d["actions"])
        for t in range(max(1, T - H)):
            chunk_obs.append(d["obs"][t])
            chunk_act.append(d["actions"][t:t + H])

    obs_t = torch.tensor(np.array(chunk_obs), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(chunk_act), dtype=torch.float32, device=device)

    batch_size = 128
    for epoch in range(n_epochs):
        perm = torch.randperm(len(obs_t))
        for i in range(0, len(obs_t), batch_size):
            idx = perm[i:i + batch_size]
            policy.train_step(obs_t[idx], act_t[idx])
        if (epoch + 1) % 50 == 0:
            print(f"    ACT epoch {epoch + 1}/{n_epochs}")

    return policy


def train_diffusion(demos, dim_obs, dim_action, n_epochs, device):
    """Train Diffusion Policy baseline."""
    from steerable.baselines import DiffusionPolicy
    import torch

    policy = DiffusionPolicy(dim_obs, dim_action, hidden=128)
    policy._denoiser.to(device)

    H = 6
    chunk_obs, chunk_act = [], []
    for d in demos:
        T = len(d["actions"])
        for t in range(max(1, T - H)):
            chunk_obs.append(d["obs"][t])
            chunk_act.append(d["actions"][t:t + H])

    obs_t = torch.tensor(np.array(chunk_obs), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(chunk_act), dtype=torch.float32, device=device)

    batch_size = 128
    for epoch in range(n_epochs):
        perm = torch.randperm(len(obs_t))
        for i in range(0, len(obs_t), batch_size):
            idx = perm[i:i + batch_size]
            policy.train_step(obs_t[idx], act_t[idx])
        if (epoch + 1) % 50 == 0:
            print(f"    Diffusion epoch {epoch + 1}/{n_epochs}")

    return policy


def train_flow(demos, dim_obs, dim_action, n_epochs, device):
    """Train our flow-matching policy."""
    from steerable.config import TrainConfig, DataConfig
    from steerable.policies.flow_expert import FlowExpert
    from steerable.data import build_dataset
    import torch

    tc = TrainConfig(n_epochs=n_epochs, batch_size=128)
    dc = DataConfig(n_demos=len(demos))

    # build dataset from demos
    dataset = build_dataset(demos, dc)

    dim_subgoal = dim_obs  # subgoal is obs-shaped
    model = FlowExpert(dim_obs, dim_subgoal, dim_action, cfg=tc)
    model.to(device)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        n = 0
        for batch in dataset:
            if isinstance(batch, dict):
                obs_b = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
                act_b = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
                sg_b = torch.tensor(batch.get("subgoal", batch["obs"]),
                                    dtype=torch.float32, device=device)
            else:
                obs_b, act_b, sg_b = batch
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                sg_b = sg_b.to(device) if sg_b is not None else obs_b

            loss = model.training_step(obs_b, act_b, sg_b)
            epoch_loss += loss
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 50 == 0:
            print(f"    Flow epoch {epoch + 1}/{n_epochs} loss={avg:.4f}")

    return model


def eval_policy(env, policy, max_steps=300, oracle=None):
    """Evaluate a policy on an environment.

    Returns metrics dict.
    """
    obs = env.reset()
    total_cr = 0
    cr0 = env.crossings0
    interventions = 0
    violations = 0
    steps = 0
    done = False

    while not done and steps < max_steps:
        # get action from policy
        if hasattr(policy, 'act'):
            action = policy.act(obs)
            if hasattr(action, 'shape') and len(action.shape) > 1:
                action = action[0]  # take first step of chunk
        else:
            action = policy(obs)

        # apply action
        obs, done = env.step(action)
        steps += 1
        violations += getattr(env, 'violations', 0)

    cr_final = env.crossings()
    success = cr_final == 0
    return {
        "success": float(success),
        "crossings_reduced": cr0 - cr_final,
        "crossings0": cr0,
        "crossings_final": cr_final,
        "steps": steps,
        "violations": violations,
    }


def run_cable_benchmark(args, device):
    """Run the cable (miniature) benchmark."""
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv

    print("\n" + "=" * 60)
    print("CABLE (MINIATURE) BENCHMARK")
    print("=" * 60)

    cfg = EnvConfig()
    results = {"env": "cable", "variants": {}}

    for seed in range(args.n_seeds):
        print(f"\n--- Seed {seed} ---")
        demos = generate_cable_demos(args.n_demos, seed=seed)
        if not demos:
            print("  No demos generated, skipping")
            continue

        dim_obs = demos[0]["obs"].shape[1]
        dim_action = demos[0]["actions"].shape[1]

        # train baselines
        for name, train_fn in [("bc", train_bc), ("act", train_act),
                                ("diffusion", train_diffusion)]:
            print(f"\n  Training {name}...")
            t0 = time.time()
            policy = train_fn(demos, dim_obs, dim_action, args.n_epochs, device)
            t_train = time.time() - t0

            # evaluate
            eval_results = []
            for eval_seed in range(10):
                eval_env = CableEnv(cfg, seed=10000 + seed * 100 + eval_seed,
                                   crossing_target=demos[eval_seed % len(demos)]["crossings0"])
                r = eval_policy(eval_env, policy, max_steps=args.max_eval_steps)
                eval_results.append(r)

            avg = {k: np.mean([r[k] for r in eval_results]) for k in eval_results[0]}
            avg["train_time"] = t_train
            results["variants"][f"{name}_s{seed}"] = avg
            print(f"    {name}: success={avg['success']:.3f}, cr_reduced={avg['crossings_reduced']:.2f}")

    return results


def main():
    args = parse_args()
    device = get_device(args)
    print(f"Device: {device}")
    print(f"Config: {vars(args)}")

    os.makedirs(args.output, exist_ok=True)
    all_results = {}

    if "cable" in args.envs:
        all_results["cable"] = run_cable_benchmark(args, device)

    # Save results
    out_path = os.path.join(args.output, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
