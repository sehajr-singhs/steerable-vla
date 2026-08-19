"""Run the Steerable VLA study and write results/*.json.

Protocols:
  main     variants {bc, flow_flat, ours_nofilter, ours_full} x seeds,
           trained on train families, evaluated zero-shot on held-out
           families (unseen stiffness + crossing topologies)
  flywheel the data flywheel on ours_full (near_miss vs relabel vs none)
  expert   expert success ceiling on train + held-out families

Every number is written to results/<protocol>_*.json, committed.

Usage:
  PYTHONPATH=src python scripts/run_experiment.py --smoke
  PYTHONPATH=src python scripts/run_experiment.py --protocols main flywheel
  PYTHONPATH=src python scripts/run_experiment.py --protocols all --seeds 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steerable.config import (EnvConfig, DataConfig, TrainConfig, EvalConfig,
                              FlywheelConfig)
from steerable.data import collect_demos, build_dataset
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.policies.expert import run_expert
from steerable.eval import evaluate
from steerable.flywheel.loop import run_flywheel
from steerable.envs.cable import CableEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Train / held-out families -------------------------------------------------
# train:     crossing_target in {2, 3}, stiffness in [0.9, 1.1]
# held-out:  crossing_target in {4},      stiffness in {0.6, 1.5}
TRAIN_CROSS = [2, 3]
TRAIN_STIFF = [0.9, 1.1]
EVAL_CROSS = [4]
EVAL_STIFF = [0.6, 1.5]


def make_env_factory(env_cfg, cross_list, stiff_list):
    def factory(seed):
        ct = cross_list[np.random.RandomState(seed).randint(len(cross_list))]
        sm = stiff_list[np.random.RandomState(seed).randint(len(stiff_list))]
        return CableEnv(env_cfg, seed=seed, stiffness_mult=sm, crossing_target=ct)
    return factory


def dims(env_cfg):
    o = CableEnv(env_cfg).reset().shape[0]
    return o, o, DataConfig().chunk * 3   # (dim_obs, dim_sub, dim_action)


def run_variant(kind, env_cfg, dcfg, tcfg, ecfg, seed, n_demos, epochs, device):
    train_env = make_env_factory(env_cfg, TRAIN_CROSS, TRAIN_STIFF)
    demos = collect_demos(env_cfg, {"make_env": train_env}, n_demos, seed, dcfg)
    dataset = build_dataset(demos, dcfg, seed=seed)
    o, s, a = dims(env_cfg)
    policy = make_policy(kind, (o, s, a), tcfg, dcfg)
    train_policy(policy, dataset, tcfg, dcfg, device, seed=seed, kind=kind,
                 epochs=epochs)
    # zero-shot eval on held-out families
    held = make_env_factory(env_cfg, EVAL_CROSS, EVAL_STIFF)
    use_filter = kind == "ours_full"
    steer = kind in ("ours_nofilter", "ours_full")
    res = evaluate(policy, held, ecfg.n_eval, seed, dcfg, ecfg,
                   use_filter=use_filter, steer=steer, device=device)
    return {"variant": kind, "seed": seed, "n_demos": n_demos, "epochs": epochs,
            "success": res["success"],
            "ni_success": res["ni_success"],
            "crossings_reduced": res["crossings_reduced"],
            "steps": res["steps"], "jerk": res["jerk"],
            "violations": res["violations"],
            "interventions": res["interventions"], "n": res["n"]}


def run_expert_ceiling(env_cfg, ecfg, seeds=3):
    out = {"train": [], "held": []}
    for s in range(seeds):
        for fam_name, fac in [("train", make_env_factory(env_cfg, TRAIN_CROSS, TRAIN_STIFF)),
                              ("held", make_env_factory(env_cfg, EVAL_CROSS, EVAL_STIFF))]:
            succ = []
            for k in range(ecfg.n_eval):
                env = fac(seed=s * 1000 + k)
                r = run_expert(env, max_steps=ecfg.max_steps)
                succ.append(1.0 if r["success"] else 0.0)
            out[fam_name].append({"seed": s, "success": float(np.mean(succ))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--protocols", nargs="+", default=["main", "flywheel", "expert"],
                    choices=["main", "flywheel", "expert", "all"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--n-demos", type=int, default=150)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    if "all" in args.protocols:
        args.protocols = ["main", "flywheel", "expert"]
    os.makedirs(args.out_dir, exist_ok=True)

    env_cfg = EnvConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig()
    ecfg = EvalConfig()
    fcfg = FlywheelConfig()

    if args.smoke:
        args.seeds = 2
        args.n_demos = 12
        tcfg.epochs = 4
        ecfg.n_eval = 6
        ecfg.max_steps = 60
        fcfg.iterations = 2
        fcfg.n_deploy = 8
        fcfg.retrain_epochs = 3
    if args.epochs:
        tcfg.epochs = args.epochs
    if args.n_eval:
        ecfg.n_eval = args.n_eval

    t0 = time.time()
    meta = {"device": args.device, "cuda": torch.cuda.is_available(),
            "torch": torch.__version__, "smoke": args.smoke,
            "seeds": args.seeds, "train_cross": TRAIN_CROSS,
            "train_stiff": TRAIN_STIFF, "eval_cross": EVAL_CROSS,
            "eval_stiff": EVAL_STIFF}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if "expert" in args.protocols:
        out = run_expert_ceiling(env_cfg, ecfg, seeds=args.seeds)
        with open(os.path.join(args.out_dir, "expert.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("expert ceiling:", {k: round(np.mean([x["success"] for x in v]), 3)
                                  for k, v in out.items()})

    if "main" in args.protocols:
        variants = ["bc", "flow_flat", "ours_nofilter", "ours_full"]
        rows = []
        for kind in variants:
            for s in range(args.seeds):
                print(f"[main] {kind} seed={s} ...", flush=True)
                r = run_variant(kind, env_cfg, dcfg, tcfg, ecfg, s,
                                args.n_demos, tcfg.epochs, args.device)
                rows.append(r)
                with open(os.path.join(args.out_dir, f"main_{kind}_s{s}.json"), "w") as f:
                    json.dump(r, f, indent=2, default=float)
        with open(os.path.join(args.out_dir, "main.json"), "w") as f:
            json.dump(rows, f, indent=2, default=float)
        print("[main] done:", [(r["variant"], round(r["ni_success"], 3), round(r["interventions"], 1)) for r in rows])

    if "flywheel" in args.protocols:
        o, s, a = dims(env_cfg)
        for strat in ["none", "near_miss", "relabel"]:
            fw = run_flywheel(
                make_env_factory(env_cfg, TRAIN_CROSS, TRAIN_STIFF),
                lambda: make_policy("ours_full", (o, s, a), tcfg, dcfg),
                (o, s, a), tcfg, dcfg, fcfg, strategy=strat,
                seed=0, device=args.device, n_demos0=args.n_demos)
            with open(os.path.join(args.out_dir, f"flywheel_{strat}.json"), "w") as f:
                json.dump(fw, f, indent=2, default=float)
            print(f"[flywheel] {strat}: {fw['curve']}")

    print(f"total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
