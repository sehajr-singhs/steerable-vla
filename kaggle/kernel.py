"""Steerable VLA — full GPU study (run on Kaggle GPU).

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
