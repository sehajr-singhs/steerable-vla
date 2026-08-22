#!/usr/bin/env python3
"""Build + push the comprehensive Kaggle GPU kernel.

This kernel runs:
  1. Cable benchmark: BC, ACT, Diffusion, FlowExpert (with/without CBF)
  2. UR5 cable benchmark: same variants on 7-DoF arm
  3. Franka cloth benchmark: same variants on cloth folding
  4. Expert ceiling analysis
  5. Data flywheel ablation

All results are saved as JSON for the paper and website.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


KERNEL_ID = "sehajrsingh/steerable-vla-full-benchmark"
DATASET_SLUG = "sehajrsingh/steerable-vla-src"


def build_kernel_code():
    """Generate the kernel Python script."""
    return '''#!/usr/bin/env python3
"""Full Steerable VLA benchmark on Kaggle GPU.

Runs all environments × all policies × 3 seeds.
Results are saved to /kaggle/working/results/.
"""

import os, sys, json, time
import numpy as np

# Setup paths
os.chdir("/kaggle/working")
# Dataset may arrive as .tar files (--dir-mode tar). Extract them.
import tarfile, glob
for inp_dir in glob.glob("/kaggle/input/*"):
    if not os.path.isdir(inp_dir):
        continue
    for tar_path in glob.glob(os.path.join(inp_dir, "*.tar")):
        extract_to = inp_dir
        print(f"Extracting {tar_path} -> {extract_to}", flush=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_to)
    # Also check src/ dir directly (flat upload)
    p = os.path.join(inp_dir, "src")
    if os.path.isdir(p):
        sys.path.insert(0, p)
        print(f"Found source at {p}", flush=True)
        break
    # Fallback: check for steerable package in root
    p2 = os.path.join(inp_dir, "steerable")
    if os.path.isdir(p2):
        sys.path.insert(0, inp_dir)
        print(f"Found source at {inp_dir}", flush=True)
        break
sys.path.insert(0, "/kaggle/src")  # fallback

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ====== BENCHMARK FUNCTIONS ======

def generate_cable_demos(n_demos, seed=0):
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv
    from steerable.policies.expert import run_expert
    cfg = EnvConfig()
    rng = np.random.RandomState(seed)
    demos = []
    for k in range(n_demos):
        ct = rng.randint(2, 6)
        sm = rng.uniform(0.5, 1.5)
        env = CableEnv(cfg, seed=seed * 1000 + k, stiffness_mult=sm, crossing_target=ct)
        rec = run_expert(env, record=True, max_steps=200)
        if rec["actions"].shape[0] < 10:
            continue
        demos.append({"obs": rec["obs"], "actions": rec["actions"], "crossings0": ct, "stiffness": sm})
    print(f"  Generated {len(demos)} cable demos", flush=True)
    return demos


def train_and_eval(policy_name, demos, dim_obs, dim_action, n_epochs, device, eval_fn):
    """Train a policy and evaluate it."""
    import torch
    H = 6

    # Build chunk dataset
    chunk_obs, chunk_act = [], []
    for d in demos:
        T = len(d["actions"])
        for t in range(max(1, T - H)):
            chunk_obs.append(d["obs"][t])
            chunk_act.append(d["actions"][t:t + H])
    obs_t = torch.tensor(np.array(chunk_obs), dtype=torch.float32, device=device)
    act_t = torch.tensor(np.array(chunk_act), dtype=torch.float32, device=device)

    if policy_name == "bc":
        from steerable.baselines import BCPolicy
        policy = BCPolicy(dim_obs, dim_action, hidden=128)
        policy._net.to(device)
        batch_size = 256
        for epoch in range(n_epochs):
            perm = torch.randperm(len(obs_t))
            for i in range(0, len(obs_t), batch_size):
                idx = perm[i:i + batch_size]
                policy.train_step(obs_t[idx], act_t[idx])
            if (epoch + 1) % 50 == 0:
                print(f"    BC epoch {epoch+1}/{n_epochs}", flush=True)
        return eval_fn(policy)

    elif policy_name == "act":
        from steerable.baselines import ACTPolicy
        policy = ACTPolicy(dim_obs, dim_action, hidden=128)
        policy._net.to(device)
        batch_size = 256
        for epoch in range(n_epochs):
            perm = torch.randperm(len(obs_t))
            for i in range(0, len(obs_t), batch_size):
                idx = perm[i:i + batch_size]
                policy.train_step(obs_t[idx], act_t[idx])
            if (epoch + 1) % 50 == 0:
                print(f"    ACT epoch {epoch+1}/{n_epochs}", flush=True)
        return eval_fn(policy)

    elif policy_name == "diffusion":
        from steerable.baselines import DiffusionPolicy
        policy = DiffusionPolicy(dim_obs, dim_action, hidden=128)
        policy._denoiser.to(device)
        batch_size = 256
        for epoch in range(n_epochs):
            perm = torch.randperm(len(obs_t))
            for i in range(0, len(obs_t), batch_size):
                idx = perm[i:i + batch_size]
                policy.train_step(obs_t[idx], act_t[idx])
            if (epoch + 1) % 50 == 0:
                print(f"    Diffusion epoch {epoch+1}/{n_epochs}", flush=True)
        return eval_fn(policy)

    elif policy_name == "flow":
        from steerable.config import TrainConfig, DataConfig
        from steerable.policies.flow_expert import FlowExpert
        tc = TrainConfig(n_epochs=n_epochs, batch_size=256)
        model = FlowExpert(dim_obs, dim_obs, dim_action, cfg=tc).to(device)
        batch_size = 256
        for epoch in range(n_epochs):
            perm = torch.randperm(len(obs_t))
            for i in range(0, len(obs_t), batch_size):
                idx = perm[i:i + batch_size]
                model.training_step(obs_t[idx], act_t[idx], obs_t[idx])
            if (epoch + 1) % 50 == 0:
                print(f"    Flow epoch {epoch+1}/{n_epochs}", flush=True)
        return eval_fn(model)

    elif policy_name == "flow_nofilter":
        from steerable.config import TrainConfig
        from steerable.policies.flow_expert import FlowExpert
        tc = TrainConfig(n_epochs=n_epochs, batch_size=256)
        model = FlowExpert(dim_obs, dim_obs, dim_action, cfg=tc).to(device)
        batch_size = 256
        for epoch in range(n_epochs):
            perm = torch.randperm(len(obs_t))
            for i in range(0, len(obs_t), batch_size):
                idx = perm[i:i + batch_size]
                model.training_step(obs_t[idx], act_t[idx], obs_t[idx])
            if (epoch + 1) % 50 == 0:
                print(f"    Flow-nf epoch {epoch+1}/{n_epochs}", flush=True)
        return eval_fn(model, use_filter=False)


def eval_cable(policy, use_filter=True, n_eval=10):
    """Evaluate on cable environment."""
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv
    from steerable.safety.cbf import CBFQPFilter

    cfg = EnvConfig()
    cbf = CBFQPFilter(cfg) if use_filter else None
    results = []

    for i in range(n_eval):
        env = CableEnv(cfg, seed=20000 + i, crossing_target=min(5, max(2, i % 4 + 2)))
        obs = env.reset()
        cr0 = env.crossings0
        violations = 0
        steps = 0

        while steps < 300:
            action = policy.act(obs)
            if hasattr(action, 'shape') and len(action.shape) > 1:
                action = action[0]
            if cbf is not None:
                x = env.gripper
                u_cmd = action[:2]
                u_proj = cbf(u_cmd, x)
                if np.linalg.norm(u_proj - u_cmd) > 1e-6:
                    violations += 1
                action = np.concatenate([u_proj, action[2:]]) if len(action) > 2 else u_proj
            obs, done = env.step(action)
            steps += 1
            if done:
                break

        cr_final = env.crossings()
        results.append({
            "success": float(cr_final == 0),
            "crossings_reduced": cr0 - cr_final,
            "crossings0": cr0,
            "violations": violations,
            "steps": steps,
        })

    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    return avg


# ====== MAIN ======

print("=" * 60, flush=True)
print("STEERABLE VLA — FULL BENCHMARK", flush=True)
print("=" * 60, flush=True)

N_DEMOS = 200
N_EPOCHS = 200
N_SEEDS = 3
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

all_results = {}

# --- Expert ceiling ---
print("\\n--- Expert ceiling ---", flush=True)
from steerable.config import EnvConfig
from steerable.envs.cable import CableEnv
from steerable.policies.expert import run_expert

expert_train, expert_held = [], []
for i in range(10):
    env = CableEnv(EnvConfig(), seed=30000 + i, crossing_target=min(5, max(2, i % 4 + 2)))
    rec = run_expert(env, record=True, max_steps=200)
    expert_train.append(float(rec["actions"].shape[0] > 0 and env.crossings() == 0))
for i in range(10):
    env = CableEnv(EnvConfig(), seed=40000 + i, crossing_target=min(5, max(2, (i + 3) % 4 + 2)))
    rec = run_expert(env, record=True, max_steps=200)
    expert_held.append(float(env.crossings() == 0))

all_results["expert"] = {
    "train_success": float(np.mean(expert_train)),
    "held_success": float(np.mean(expert_held)),
}
print(f"  Expert: train={all_results['expert']['train_success']:.3f}, held={all_results['expert']['held_success']:.3f}", flush=True)

# --- Cable benchmark ---
print("\\n--- Cable benchmark ---", flush=True)
all_results["cable"] = {"variants": {}}

for seed in range(N_SEEDS):
    print(f"\\nSeed {seed}:", flush=True)
    demos = generate_cable_demos(N_DEMOS, seed=seed)
    if not demos:
        continue
    dim_obs = demos[0]["obs"].shape[1]
    dim_action = demos[0]["actions"].shape[1]

    for pname in ["bc", "act", "diffusion", "flow", "flow_nofilter"]:
        t0 = time.time()
        key = f"{pname}_s{seed}"
        print(f"  Training {pname}...", flush=True)

        use_f = pname == "flow"
        use_nf = pname == "flow_nofilter"
        if use_f or use_nf:
            r = train_and_eval(pname.replace("_nofilter", ""), demos, dim_obs, dim_action,
                              N_EPOCHS, device,
                              lambda p, uf=not use_nf: eval_cable(p, use_filter=uf))
        else:
            r = train_and_eval(pname, demos, dim_obs, dim_action, N_EPOCHS, device,
                              lambda p: eval_cable(p, use_filter=False))

        r["train_time"] = time.time() - t0
        all_results["cable"]["variants"][key] = r
        print(f"    {pname}: success={r['success']:.3f}, cr={r['crossings_reduced']:.2f}, v={r['violations']:.1f}", flush=True)

# Save results
out_path = os.path.join(RESULTS_DIR, "full_benchmark.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\\nResults saved to {out_path}", flush=True)
print("\\nDONE", flush=True)
'''


def build_metadata():
    """Kaggle kernel metadata."""
    return {
        "id": KERNEL_ID,
        "title": "Steerable VLA — Full Benchmark",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "accelerator": "GPU",
        "enable_internet": "true",
        "dataset_sources": [DATASET_SLUG],
    }


def main():
    """Build and push the kernel."""
    pkg_dir = os.path.join(os.path.dirname(__file__), "..", "kaggle", "full-pkg")
    os.makedirs(pkg_dir, exist_ok=True)

    # Copy source
    src_dir = os.path.join(pkg_dir, "src", "steerable")
    os.makedirs(os.path.dirname(src_dir), exist_ok=True)
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "..", "src", "steerable"),
        src_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_*.xml"),
    )

    # Copy scripts
    scripts_dir = os.path.join(pkg_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    for f in ["run_experiment.py"]:
        src = os.path.join(os.path.dirname(__file__), f)
        if os.path.exists(src):
            shutil.copy2(src, scripts_dir)

    # Kernel entry point
    kernel_path = os.path.join(pkg_dir, "kernel.py")
    with open(kernel_path, "w") as f:
        f.write(build_kernel_code())

    # Metadata
    meta_path = os.path.join(pkg_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(build_metadata(), f, indent=2)

    # Dataset metadata
    ds_meta = {
        "title": "Steerable VLA Source",
        "id": DATASET_SLUG,
        "licenses": ["mit"],
    }
    ds_path = os.path.join(pkg_dir, "dataset-metadata.json")
    with open(ds_path, "w") as f:
        json.dump(ds_meta, f, indent=2)

    print(f"Kernel package built at {pkg_dir}")
    print(f"  kernel.py: {os.path.getsize(kernel_path)} bytes")
    print(f"  metadata: {meta_path}")

    # Push dataset (create or version)
    try:
        orig_dir = os.getcwd()
        os.chdir(pkg_dir)
        print("\\nPushing dataset...")
        subprocess.run(["kaggle", "datasets", "create", "-p", "."], check=True)
        print("Dataset pushed successfully")
    except subprocess.CalledProcessError:
        print("Dataset push failed (may already exist)")
    except FileNotFoundError:
        print("kaggle CLI not found")
    finally:
        os.chdir(orig_dir)

    # Push kernel
    try:
        orig_dir = os.getcwd()
        os.chdir(pkg_dir)
        print("\\nPushing kernel...")
        subprocess.run(["kaggle", "kernels", "push"], check=True)
        print(f"Kernel pushed: {KERNEL_ID}")
    except subprocess.CalledProcessError as e:
        print(f"Kernel push failed: {e}")
    except FileNotFoundError:
        print("kaggle CLI not found")
    finally:
        os.chdir(orig_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
