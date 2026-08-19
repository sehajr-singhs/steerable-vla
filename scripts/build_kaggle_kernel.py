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

import os, sys, time

import torch
print("cuda available:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), flush=True)

SRC = "/kaggle/input/steerable-vla-src"
print("input dirs:", sorted(os.listdir("/kaggle/input"))[:20], flush=True)
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(SRC, "src"))

from scripts.run_experiment import main as run_main

# Trim the flywheel so the whole study fits comfortably in the GPU quota
# (the flywheel mechanics are identical at smaller deployment scale).
from steerable.config import FlywheelConfig
FlywheelConfig.n_deploy = 25
FlywheelConfig.iterations = 3
FlywheelConfig.retrain_epochs = 20

# run_main() parses sys.argv; on Kaggle it runs with default settings
# (protocols all, seeds=3, n_demos=150, epochs=120) and writes to
# /kaggle/working/results because cwd is /kaggle/working. The device is
# resolved inside run_experiment (cuda when available).
t0 = time.time()
try:
    run_main()
except Exception:
    import traceback
    traceback.print_exc()
    raise
print("total GPU study time: %.1fs" % (time.time() - t0))
'''

METADATA = {
    "id": KERNEL_ID,
    "title": KERNEL_TITLE,
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
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
    # `kaggle datasets version` on a NOT-YET-EXISTING dataset 403s on this
    # account; `create` works. Track creation with a local marker file and
    # create once, version afterwards. Windows CLI mangles absolute upload
    # paths, so always run from inside the package dir with a relative path.
    marker = os.path.join(KAG, ".dataset-created")
    if os.path.exists(marker):
        cmd = ["kaggle", "datasets", "version", "-p", ".", "--dir-mode", "tar",
               "-m", "Steerable VLA study source: env, flow expert, SMC, CBF filter, flywheel, harness"]
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
