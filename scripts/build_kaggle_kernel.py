"""Build + push the Kaggle GPU kernel.

A `kaggle kernels push` only carries the code file to the worker -- auxiliary
files must ride as a Kaggle *dataset* (the pattern proven in this account's
robotic-data-flywheel repo). So:

  1. sync()           stages src/steerable + scripts/run_experiment.py as a
                      dataset dir (kaggle/steerable-pkg)
  2. version_dataset() versions it as sehajrsingh/steerable-vla-src
  3. push()           pushes the kernel (metadata declares the dataset as a
                      source; kernel.py imports from /kaggle/input)

Usage:
  PYTHONPATH=src python scripts/build_kaggle_kernel.py            # build only
  PYTHONPATH=src python scripts/build_kaggle_kernel.py --push     # dataset + push
  PYTHONPATH=src python scripts/build_kaggle_kernel.py --poll     # poll + pull
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
KAG = os.path.join(ROOT, "kaggle")
PKG = os.path.join(KAG, "steerable-pkg")
OUT = os.path.join(ROOT, ".kaggle_output")

KERNEL_ID = "sehajrsingh/steerable-vla-gpu-study"
KERNEL_TITLE = "steerable-vla-gpu-study"
DATASET_ID = "sehajrsingh/steerable-vla-src"

KERNEL_PY = r'''"""Steerable VLA — full GPU study (run on Kaggle GPU).

The study source rides in as a Kaggle dataset (/kaggle/input/steerable-vla-src);
this file just imports and runs it. Results are written to /kaggle/working,
which the harness downloads with `kaggle kernels output`.
"""

import os, sys, time, json

import torch
print("cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)
    print("gpu memory:", round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), "GB", flush=True)

SRC = "/kaggle/input/steerable-vla-src"
print("input dirs:", sorted(os.listdir("/kaggle/input"))[:20], flush=True)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "src"))

# ── Study config (tuned for 6h GPU budget) ──────────────────────────
from steerable.config import (EnvConfig, DataConfig, TrainConfig,
                               EvalConfig, FlywheelConfig)

env_cfg = EnvConfig()
dcfg = DataConfig()
tcfg = TrainConfig()
ecfg = EvalConfig()
fcfg = FlywheelConfig()

# GPU-scale hyperparameters
tcfg.epochs = 200
tcfg.batch = 128
tcfg.flow_steps = 24
tcfg.n_samples = 8
ecfg.n_eval = 30           # enough for stable means, fast enough to complete
ecfg.max_steps = 120
ecfg.patience = 24
fcfg.iterations = 3
fcfg.n_deploy = 20
fcfg.retrain_epochs = 25
N_DEMOS = 150
N_SEEDS = 3

# ── Import and run ───────────────────────────────────────────────────
from steerable.envs.cable import CableEnv
from steerable.data import collect_demos, build_dataset
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.policies.expert import run_expert
from steerable.eval import evaluate
from steerable.flywheel.loop import run_flywheel

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

os.makedirs("results", exist_ok=True)
t0 = time.time()

# ── Phase 1: Expert ceiling ──────────────────────────────────────────
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
print("expert done", flush=True)

# ── Phase 2: Main protocol ───────────────────────────────────────────
print("=" * 60, flush=True)
print("PHASE 2: Main protocol (4 variants x 3 seeds)", flush=True)
print("=" * 60, flush=True)
o, s, a = dims()
variants = ["bc", "flow_flat", "ours_nofilter", "ours_full"]
rows = []
for kind in variants:
    for seed in range(N_SEEDS):
        t_v0 = time.time()
        print(f"[main] {kind} seed={seed} collecting demos...", flush=True)
        demos = collect_demos(env_cfg, {"make_env": train_fac}, N_DEMOS, seed, dcfg)
        dataset = build_dataset(demos, dcfg, seed=seed)
        print(f"  demos={len(demos)} items={len(dataset)} ({time.time()-t_v0:.0f}s)", flush=True)

        policy = make_policy(kind, (o, s, a), tcfg, dcfg)
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

# ── Phase 3: Flywheel ────────────────────────────────────────────────
print("=" * 60, flush=True)
print("PHASE 3: Data flywheel (3 strategies)", flush=True)
print("=" * 60, flush=True)
for strat in ["none", "near_miss", "relabel"]:
    t_f0 = time.time()
    fw = run_flywheel(
        train_fac,
        lambda: make_policy("ours_full", (o, s, a), tcfg, dcfg),
        (o, s, a), tcfg, dcfg, fcfg, strategy=strat,
        seed=0, device=DEVICE, n_demos0=N_DEMOS)
    with open(f"results/flywheel_{strat}.json", "w") as f:
        json.dump(fw, f, indent=2, default=float)
    print(f"  {strat}: curve={fw['curve']} ({time.time()-t_f0:.0f}s)", flush=True)

# ── Summary ──────────────────────────────────────────────────────────
meta = {"device": DEVICE, "cuda": torch.cuda.is_available(),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "seeds": N_SEEDS, "n_demos": N_DEMOS, "epochs": tcfg.epochs,
        "n_eval": ecfg.n_eval,
        "train_cross": TRAIN_CROSS, "train_stiff": TRAIN_STIFF,
        "eval_cross": EVAL_CROSS, "eval_stiff": EVAL_STIFF,
        "total_time_s": time.time() - t0}
with open("results/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("=" * 60, flush=True)
print(f"STUDY COMPLETE in {time.time()-t0:.0f}s", flush=True)
print(f"device: {DEVICE}", flush=True)
for r in rows:
    print(f"  {r['variant']:20s} s{r['seed']}: ni={r['ni_success']:.2f} "
          f"cr={r['crossings_reduced']:.1f} intv={r['interventions']:.1f} "
          f"vio={r['violations']:.0f}", flush=True)
print("=" * 60, flush=True)
'''

METADATA = {
    "id": KERNEL_ID,
    "title": KERNEL_TITLE,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "accelerator": "GPU",
    "enable_internet": "true",
    "competition_sources": [],
    "dataset_sources": [DATASET_ID],
    "model_sources": [],
}


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=kw.pop("cwd", ROOT), **kw)


def sync():
    if os.path.exists(PKG):
        shutil.rmtree(PKG)
    os.makedirs(os.path.join(PKG, "src"))
    os.makedirs(os.path.join(PKG, "scripts"))
    shutil.copytree(os.path.join(ROOT, "src", "steerable"),
                    os.path.join(PKG, "src", "steerable"))
    shutil.copy(os.path.join(ROOT, "scripts", "run_experiment.py"),
                os.path.join(PKG, "scripts", "run_experiment.py"))
    shutil.copy(os.path.join(ROOT, "requirements.txt"),
                os.path.join(PKG, "requirements.txt"))
    with open(os.path.join(PKG, "dataset-metadata.json"), "w") as f:
        json.dump({"id": DATASET_ID, "title": "Steerable VLA study source",
                   "licenses": [{"name": "MIT"}]}, f, indent=2)
    with open(os.path.join(KAG, "kernel.py"), "w") as f:
        f.write(KERNEL_PY)
    with open(os.path.join(KAG, "kernel-metadata.json"), "w") as f:
        json.dump(METADATA, f, indent=2)
    print("dataset staged at", PKG)


def version_dataset():
    marker = os.path.join(KAG, ".dataset-created")
    if os.path.exists(marker):
        cmd = ["kaggle", "datasets", "version", "-p", ".", "--dir-mode", "tar",
               "-m", "Steerable VLA v2: GPU study with proper protocol execution"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", ".", "--dir-mode", "tar"]
    r = run(cmd, cwd=PKG)
    if r.returncode == 0:
        with open(marker, "w") as f:
            f.write(DATASET_ID)


def push():
    run(["kaggle", "kernels", "push", "-p", KAG])


def status(ref=KERNEL_ID):
    p = run(["kaggle", "kernels", "status", ref], capture_output=True, text=True)
    return p.stdout.strip().split(" has status ")[-1].replace('"', "")


def poll(wait=120, timeout=8 * 3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = status()
        print(f"  {st}", flush=True)
        if "COMPLETE" in st or "ERROR" in st or "CANCEL" in st:
            os.makedirs(OUT, exist_ok=True)
            run(["kaggle", "kernels", "output", KERNEL_ID, "-p", OUT])
            return st
        time.sleep(wait)
    print(f"timed out after {timeout}s; run --poll again", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--wait", type=int, default=120)
    args = ap.parse_args()
    if args.poll:
        st = poll(wait=args.wait)
        sys.exit(0 if st and "COMPLETE" in st else 1)
    sync()
    if args.push:
        version_dataset()
        push()


if __name__ == "__main__":
    main()
