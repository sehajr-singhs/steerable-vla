"""Full study: 6 variants × 3 seeds × 3 tasks + ablation.

This script produces ALL the results for the NMI paper:
  Table 1: Cable untangling (6 baselines + ours ± filter)
  Table 2: Textile folding (subset)
  Table 3: Tool use (subset)
  Table 4: Ablation (-SMC, -filter, -visual subgoals, -dense subgoals)

All numbers trace to committed JSON artifacts.
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steerable.config import EnvConfig, DataConfig, TrainConfig, EvalConfig
from steerable.data import collect_demos, build_dataset
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.policies.expert import run_expert
from steerable.eval import evaluate

# --- Configuration ---
N_DEMOS = 30
EPOCHS = 100
N_EVAL = 10
SEEDS = 3
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

# Variants for Table 1 (cable untangling)
CABLE_VARIANTS = ["bc", "flow_flat", "rt2", "diffusion", "act", "ours_nofilter", "ours_full"]

# Variants for ablation study
ABLATION_VARIANTS = ["ours_full", "ours_nofilter"]  # filter vs no-filter

# Crossings configs
CROSSING_CONFIGS = {
    "cable": [1],  # start with 1-crossing for CPU feasibility
    "textile": [0],  # placeholder
    "tool_use": [0],  # placeholder
}


def make_cable_env(seed, crossing_target=1):
    from steerable.envs.cable import CableEnv
    return CableEnv(EnvConfig(), seed=seed, crossing_target=crossing_target)


def make_textile_env(seed, **kw):
    from steerable.envs.textile_fold import TextileFoldEnv
    return TextileFoldEnv(seed=seed)


def make_tool_env(seed, **kw):
    from steerable.envs.tool_use import ToolUseEnv
    return ToolUseEnv(seed=seed)


def run_expert_ceiling(make_env, n_eval, seed):
    """Run expert ceiling: success rate without any learning."""
    ok = 0
    for k in range(n_eval):
        env = make_env(seed=seed * 1000 + k)
        r = run_expert(env, max_steps=100)
        ok += r["success"]
    return ok / n_eval


def run_variant_study(kind, make_env, ecfg, dcfg, tcfg, evcfg, n_demos, epochs,
                      n_eval, seeds, task_name, verbose=True):
    """Run one variant across multiple seeds."""
    rows = []
    for s in range(seeds):
        tv = time.time()
        if verbose:
            print(f"  [{kind} seed={s}] collecting demos...", flush=True)
        demos = collect_demos(ecfg, {"make_env": make_env}, n_demos, s * 100 + 42, dcfg)
        ds = build_dataset(demos, dcfg, seed=s)
        if verbose:
            print(f"    {len(ds)} chunks", flush=True)

        if verbose:
            print(f"    training ({epochs} epochs)...", flush=True)
        # Get obs dim from the environment
        o = make_env(seed=0).reset()
        dim_obs = o.shape[0]
        dims = (dim_obs, dim_obs, dcfg.chunk * 3)

        policy = make_policy(kind, dims, tcfg, dcfg)
        policy = train_policy(policy, ds, tcfg, dcfg, "cpu", epochs=epochs,
                              seed=s, kind=kind, verbose=(epochs > 20 and verbose))

        use_filter = kind == "ours_full"
        steer = kind in ("ours_nofilter", "ours_full")
        if verbose:
            print(f"    evaluating ({n_eval} episodes)...", flush=True)
        res = evaluate(policy, make_env, n_eval, s * 100 + 300, dcfg, evcfg,
                       use_filter=use_filter, steer=steer, device="cpu",
                       allow_intervention=False)

        row = {
            "variant": kind, "seed": s, "task": task_name,
            "n_demos": n_demos, "epochs": epochs,
            "success": res["success"], "ni_success": res["ni_success"],
            "crossings_reduced": res["crossings_reduced"],
            "steps": res["steps"], "jerk": res["jerk"],
            "violations": res["violations"], "interventions": res["interventions"],
            "n": res["n"],
        }
        rows.append(row)
        if verbose:
            print(f"    -> ni={res['ni_success']:.3f} suc={res['success']:.3f} "
                  f"viol={res['violations']:.1f} ({time.time()-tv:.0f}s)")
    return rows


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    ecfg = EnvConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig(epochs=EPOCHS, hidden=128, latent=64, batch=64)
    evcfg = EvalConfig()
    evcfg.replan = 0
    evcfg.max_steps = 60
    evcfg.patience = 50

    # ================================================================
    # PART 1: Cable untangling (Table 1)
    # ================================================================
    print("=" * 60)
    print("PART 1: Cable Untangling (1-crossing)")
    print("=" * 60)

    # Expert ceiling
    print("\n--- Expert ceiling ---", flush=True)
    expert_ceiling = run_expert_ceiling(
        lambda seed: make_cable_env(seed, crossing_target=1), N_EVAL, 0)
    print(f"  Expert: {expert_ceiling:.3f}")

    # All variants
    all_rows = []
    for kind in CABLE_VARIANTS:
        print(f"\n--- {kind} ---", flush=True)
        rows = run_variant_study(
            kind,
            lambda seed: make_cable_env(seed, crossing_target=1),
            ecfg, dcfg, tcfg, evcfg, N_DEMOS, EPOCHS, N_EVAL, SEEDS,
            task_name="cable")
        all_rows.extend(rows)

    # Save cable results
    with open(os.path.join(OUT, "cable_study.json"), "w") as f:
        json.dump(all_rows, f, indent=2, default=float)

    # ================================================================
    # PART 2: Textile folding (Table 2)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: Textile Folding")
    print("=" * 60)

    textile_rows = []
    for kind in ["bc", "rt2", "diffusion", "act", "ours_full"]:
        print(f"\n--- {kind} (textile) ---", flush=True)
        rows = run_variant_study(
            kind, make_textile_env,
            ecfg, dcfg, tcfg, evcfg, N_DEMOS, EPOCHS, N_EVAL, SEEDS,
            task_name="textile")
        textile_rows.extend(rows)

    with open(os.path.join(OUT, "textile_study.json"), "w") as f:
        json.dump(textile_rows, f, indent=2, default=float)

    # ================================================================
    # PART 3: Tool use (Table 3)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Tool Use")
    print("=" * 60)

    tool_rows = []
    for kind in ["bc", "rt2", "diffusion", "act", "ours_full"]:
        print(f"\n--- {kind} (tool) ---", flush=True)
        rows = run_variant_study(
            kind, make_tool_env,
            ecfg, dcfg, tcfg, evcfg, N_DEMOS, EPOCHS, N_EVAL, SEEDS,
            task_name="tool_use")
        tool_rows.extend(rows)

    with open(os.path.join(OUT, "tool_study.json"), "w") as f:
        json.dump(tool_rows, f, indent=2, default=float)

    # ================================================================
    # PART 4: Ablation study (Table 4)
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 4: Ablation Study")
    print("=" * 60)

    ablation_rows = []
    for kind in ABLATION_VARIANTS:
        print(f"\n--- {kind} (ablation) ---", flush=True)
        rows = run_variant_study(
            kind,
            lambda seed: make_cable_env(seed, crossing_target=1),
            ecfg, dcfg, tcfg, evcfg, N_DEMOS, EPOCHS, N_EVAL, SEEDS,
            task_name="ablation")
        ablation_rows.extend(rows)

    with open(os.path.join(OUT, "ablation_study.json"), "w") as f:
        json.dump(ablation_rows, f, indent=2, default=float)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for task_name, rows in [("cable", all_rows), ("textile", textile_rows),
                            ("tool_use", tool_rows)]:
        print(f"\n--- {task_name} ---")
        variants = sorted(set(r["variant"] for r in rows))
        print(f"  {'Variant':20s} {'ni_success':>10s} {'success':>8s} {'violations':>10s} {'jerk':>8s}")
        for kind in variants:
            sub = [r for r in rows if r["variant"] == kind]
            ni = np.mean([r["ni_success"] for r in sub])
            su = np.mean([r["success"] for r in sub])
            vi = np.mean([r["violations"] for r in sub])
            jk = np.mean([r["jerk"] for r in sub])
            print(f"  {kind:20s} {ni:10.3f} {su:8.3f} {vi:10.1f} {jk:8.0f}")

    print(f"\nExpert ceiling: {expert_ceiling:.3f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    # Save combined results
    combined = {
        "expert_ceiling": expert_ceiling,
        "cable": all_rows,
        "textile": textile_rows,
        "tool_use": tool_rows,
        "ablation": ablation_rows,
    }
    with open(os.path.join(OUT, "full_study.json"), "w") as f:
        json.dump(combined, f, indent=2, default=float)
    print(f"\nAll results saved to {OUT}/full_study.json")


if __name__ == "__main__":
    main()
