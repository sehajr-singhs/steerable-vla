"""Steerable VLA — full GPU study v12 with Transformer backbone.

Adds the 5.3M-parameter CableTransformer alongside the existing baselines.
"""

import os, sys, time, json

import torch
print("cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)

SRC = "/kaggle/input/steerable-vla-src"
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "src"))

from steerable.config import EnvConfig, DataConfig, TrainConfig, EvalConfig, FlywheelConfig
env_cfg = EnvConfig()
dcfg = DataConfig()
tcfg = TrainConfig()
ecfg = EvalConfig()
fcfg = FlywheelConfig()

tcfg.epochs = 200
tcfg.batch = 128
tcfg.flow_steps = 24
tcfg.n_samples = 8
ecfg.n_eval = 30
ecfg.max_steps = 120
ecfg.patience = 24
ecfg.replan = 1
N_DEMOS = 150
N_SEEDS = 3

from steerable.envs.cable import CableEnv
from steerable.data import collect_demos, build_dataset
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.policies.expert import run_expert
from steerable.eval import evaluate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}", flush=True)

TRAIN_CROSS = [2, 3]
TRAIN_STIFF = [0.9, 1.1]
EVAL_CROSS = [4]
EVAL_STIFF = [0.6, 1.5]

def make_env_factory(cross_list, stiff_list):
    def factory(seed):
        ct = cross_list[torch.randint(len(cross_list), (1,)).item()]
        sm = stiff_list[torch.randint(len(stiff_list), (1,)).item()]
        return CableEnv(env_cfg, seed=seed, stiffness_mult=sm, crossing_target=ct)
    return factory

def dims():
    o = CableEnv(env_cfg).reset().shape[0]
    return o, o, dcfg.chunk * 3

os.makedirs("/kaggle/working/results", exist_ok=True)
os.chdir("/kaggle/working")
t0 = time.time()

# Phase 1: Expert ceiling
print("=" * 60, flush=True)
print("PHASE 1: Expert ceiling", flush=True)
print("=" * 60, flush=True)
train_fac = make_env_factory(TRAIN_CROSS, TRAIN_STIFF)
held_fac = make_env_factory(EVAL_CROSS, EVAL_STIFF)
expert_out = {"train": [], "held": []}
for s in range(N_SEEDS):
    for fam_name, fac in [("train", train_fac), ("held", held_fac)]:
        succ = []
        for k in range(ecfg.n_eval):
            env = fac(seed=s * 1000 + k)
            r = run_expert(env, max_steps=ecfg.max_steps)
            succ.append(1.0 if r["success"] else 0.0)
        expert_out[fam_name].append({"seed": s, "success": float(sum(succ) / len(succ))})
        print(f"  {fam_name} seed={s}: {sum(succ)/len(succ):.3f} ({time.time()-t0:.0f}s)", flush=True)
with open("results/expert.json", "w") as f:
    json.dump(expert_out, f, indent=2)

# Phase 2: Main protocol — all variants including transformer
print("=" * 60, flush=True)
print("PHASE 2: Main protocol", flush=True)
print("=" * 60, flush=True)
o, s, a = dims()
variants = ["bc", "flow_flat", "ours_nofilter", "ours_full", "transformer"]
rows = []
for kind in variants:
    for seed in range(N_SEEDS):
        t_v0 = time.time()
        print(f"[main] {kind} seed={seed} collecting demos...", flush=True)
        demos = collect_demos(env_cfg, {"make_env": train_fac}, N_DEMOS, seed, dcfg)
        dataset = build_dataset(demos, dcfg, seed=seed)
        print(f"  demos={len(demos)} items={len(dataset)} ({time.time()-t_v0:.0f}s)", flush=True)

        policy = make_policy(kind, (o, s, a), tcfg, dcfg)
        if hasattr(policy, '_net'):
            policy._net.to(DEVICE)
        train_policy(policy, dataset, tcfg, dcfg, DEVICE, seed=seed, kind=kind)
        print(f"  trained ({time.time()-t_v0:.0f}s)", flush=True)

        use_filter = kind == "ours_full"
        steer = kind in ("ours_nofilter", "ours_full")
        res = evaluate(policy, held_fac, ecfg.n_eval, seed, dcfg, ecfg,
                       use_filter=use_filter, steer=steer, device=DEVICE)
        row = {"variant": kind, "seed": seed, "n_demos": N_DEMOS,
               "epochs": tcfg.epochs, "success": res["success"],
               "ni_success": res["ni_success"],
               "crossings_reduced": res["crossings_reduced"],
               "steps": res["steps"], "jerk": res["jerk"],
               "violations": res["violations"],
               "interventions": res["interventions"], "n": res["n"]}
        rows.append(row)
        with open(f"results/main_{kind}_s{seed}.json", "w") as f:
            json.dump(row, f, indent=2, default=float)
        print(f"  done: ni={res['ni_success']:.2f} cr={res['crossings_reduced']:.1f} "
              f"intv={res['interventions']:.1f} vio={res['violations']:.0f} "
              f"({time.time()-t_v0:.0f}s)", flush=True)

with open("results/main.json", "w") as f:
    json.dump(rows, f, indent=2, default=float)
print(f"main done total={time.time()-t0:.0f}s", flush=True)

# Summary
meta = {"device": DEVICE, "cuda": torch.cuda.is_available(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "seeds": N_SEEDS, "n_demos": N_DEMOS, "epochs": tcfg.epochs,
        "total_time_s": time.time() - t0}
with open("results/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("=" * 60, flush=True)
print(f"STUDY COMPLETE in {time.time()-t0:.0f}s", flush=True)
for r in rows:
    print(f"  {r['variant']:20s} s{r['seed']}: ni={r['ni_success']:.2f} "
          f"cr={r['crossings_reduced']:.1f} vio={r['violations']:.0f}", flush=True)
print("=" * 60, flush=True)
