"""Run the full study on Lightning AI (GPU cloud).

Three ways to use it:

1. Lightning Studio (recommended): clone the repo into a Studio machine,
   open a terminal, and run

       PYTHONPATH=src python scripts/run_lightning.py --seeds 3

   The Studio machine auto-provisions a GPU when you pick a GPU template;
   torch.cuda.is_available() resolves inside the script.

2. Lightning Fabric / `lightning run` (single GPU):

       lightning run script scripts/run_lightning.py -- --seeds 3

3. Any CUDA box (Colab, your own GPU): just run it, it auto-detects cuda.

Output: results/*.json written to the working directory, identical schema
to the Kaggle kernel and the local CPU run.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from scripts.run_experiment import main as run_main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-demos", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--protocols", nargs="+", default=["main", "flywheel", "expert"])
    args = ap.parse_args()

    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    # run_main() builds its own argparse; mirror the settings via sys.argv.
    sys.argv = ["run_experiment.py",
                "--seeds", str(args.seeds),
                "--n-demos", str(args.n_demos),
                "--out-dir", args.out_dir,
                "--protocols"] + args.protocols
    if args.epochs:
        sys.argv += ["--epochs", str(args.epochs)]
    if args.n_eval:
        sys.argv += ["--n-eval", str(args.n_eval)]
    run_main()


if __name__ == "__main__":
    main()
