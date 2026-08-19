"""Sync the committed code + results into hf-space/ for the HF Space push.

Usage:
  PYTHONPATH=src python scripts/sync_hf_space.py
  # then: cd hf-space && huggingface-cli upload --repo-type space <space-id> .
"""

import os
import shutil

ROOT = os.path.join(os.path.dirname(__file__), "..")
SPACE = os.path.join(ROOT, "hf-space")
SRC = os.path.join(ROOT, "src", "steerable")
RESULTS = os.path.join(ROOT, "results")


def main():
    # code (the dashboard's live demo imports steerable)
    dst = os.path.join(SPACE, "steerable")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)

    # results (committed JSON, rendered by the dashboard)
    dst = os.path.join(SPACE, "results")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(RESULTS, dst)

    print("synced steerable/ + results/ into", SPACE)
    print("push with:  cd hf-space && huggingface-cli upload "
          "--repo-type space <owner>/<space> .")


if __name__ == "__main__":
    main()
