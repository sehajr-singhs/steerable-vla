"""Loaders for real robotics datasets.

Each loader returns a unified interface:
  demo = {
      "observations": np.ndarray (T, obs_dim),
      "actions": np.ndarray (T, action_dim),
      "episode_id": int,
      "language": str,           # task description
  }

The loaders handle downloading, caching, and frame conversion so the
training loop is dataset-agnostic.

Supported datasets:
  - DROID: large-scale bimanual manipulation (7-DoF + gripper)
  - BridgeData V2: tabletop manipulation with WidowX arms
  - Open X-Embodiment subsets: RT-X format parquet files

Install: pip install droid datasets h5py
"""

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Optional


# ── Unified demo interface ───────────────────────────────────────────

def normalize_actions(actions, method="minmax"):
    """Normalize actions to [-1, 1] using min-max scaling."""
    if method == "minmax":
        lo = actions.min(axis=0)
        hi = actions.max(axis=0)
        mid = (lo + hi) / 2
        scale = (hi - lo) / 2 + 1e-8
        return (actions - mid) / scale, {"mid": mid, "scale": scale}
    elif method == "zscore":
        mu = actions.mean(axis=0)
        std = actions.std(axis=0) + 1e-8
        return (actions - mu) / std, {"mu": mu, "std": std}
    return actions, {}


def denormalize_actions(norm_actions, stats, method="minmax"):
    """Reverse normalization."""
    if method == "minmax":
        return norm_actions * stats["scale"] + stats["mid"]
    elif method == "zscore":
        return norm_actions * stats["std"] + stats["mu"]
    return norm_actions


# ── BridgeData V2 ────────────────────────────────────────────────────

def load_bridgedata_v2(
    data_dir: str,
    tasks: Optional[List[str]] = None,
    max_episodes: int = 200,
    max_episode_len: int = 300,
    action_dim: int = 5,
    image_size: int = 224,
) -> List[Dict]:
    """Load BridgeData V2 episodes.

    BridgeData V2 contains ~60K episodes of WidowX tabletop manipulation
    across 109 language-conditioned tasks. Episodes are stored as HDF5 files
    with keys: observations/images (T, H, W, 3), observations/qpos (T, 7),
    actions (T, 5), language (str).

    Download: https://huggingface.co/datasets/bridge-data/bridgedata_v2

    Args:
        data_dir: path to the directory containing .hdf5 files
        tasks: filter to specific task names (None = all)
        max_episodes: maximum number of episodes to load
        max_episode_len: truncate episodes longer than this
        action_dim: 5 (dx, dy, dz, wrist_rot, gripper)
    """
    try:
        import h5py
    except ImportError:
        print("WARNING: h5py not installed. Install with: pip install h5py")
        return []

    demos = []
    if not os.path.exists(data_dir):
        print(f"BridgeData V2 directory not found: {data_dir}")
        print("Download from: https://huggingface.co/datasets/bridge-data/bridgedata_v2")
        return []

    hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])

    for fname in hdf5_files[:max_episodes]:
        fpath = os.path.join(data_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                lang = f.attrs.get("language", f.attrs.get("task", ""))
                if tasks and lang not in tasks:
                    continue

                obs_imgs = f["observations"]["images"][:]
                qpos = f["observations"]["qpos"][:]
                acts = f["actions"][:]

                T = min(len(acts), max_episode_len)
                if T < 10:
                    continue

                # convert to unified format
                obs = np.concatenate([
                    qpos[:T],  # (T, 7) joint positions
                    obs_imgs[:T].mean(axis=(1, 2, 3)).reshape(-1, 1) / 255.0  # brightness feature
                ], axis=-1).astype(np.float32)

                demos.append({
                    "observations": obs,
                    "actions": acts[:T, :action_dim].astype(np.float32),
                    "episode_id": len(demos),
                    "language": str(lang),
                    "source": "bridgedata_v2",
                })
        except Exception as e:
            continue

    print(f"Loaded {len(demos)} episodes from BridgeData V2")
    return demos


# ── DROID ────────────────────────────────────────────────────────────

def load_droid(
    data_dir: str,
    tasks: Optional[List[str]] = None,
    max_episodes: int = 200,
    max_episode_len: int = 300,
    action_dim: int = 7,
) -> List[Dict]:
    """Load DROID (Distributed Robot Interaction Dataset) episodes.

    DROID contains 76K episodes of bimanual manipulation with a Franka
    Panda across 255 tasks. Episodes are stored as individual .npy files
    or parquet tables.

    Download: https://huggingface.co/datasets/droid-official/DROID

    Args:
        data_dir: path to the DROID data directory
        tasks: filter to specific task descriptions
        max_episodes: maximum episodes to load
        max_episode_len: truncate long episodes
        action_dim: 7 (6-DoF end-effector delta + gripper)
    """
    demos = []

    if not os.path.exists(data_dir):
        print(f"DROID directory not found: {data_dir}")
        print("Download from: https://huggingface.co/datasets/droid-official/DROID")
        return []

    # DROID can be stored as .npy episode files or parquet
    npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])

    if npy_files:
        for fname in npy_files[:max_episodes]:
            fpath = os.path.join(data_dir, fname)
            try:
                ep = np.load(fpath, allow_pickle=True).item()
                lang = ep.get("language", ep.get("task", ""))
                if tasks and lang not in tasks:
                    continue

                obs = ep.get("observations", ep.get("obs"))
                acts = ep.get("actions", ep.get("act"))
                if obs is None or acts is None:
                    continue

                T = min(len(acts), max_episode_len)
                if T < 10:
                    continue

                demos.append({
                    "observations": obs[:T].astype(np.float32),
                    "actions": acts[:T, :action_dim].astype(np.float32),
                    "episode_id": len(demos),
                    "language": str(lang),
                    "source": "droid",
                })
            except Exception:
                continue

    if parquet_files:
        try:
            import pandas as pd
            for fname in parquet_files[:max_episodes]:
                fpath = os.path.join(data_dir, fname)
                df = pd.read_parquet(fpath)
                for _, row in df.iterrows():
                    obs = np.array(row["observations"], dtype=np.float32)
                    acts = np.array(row["actions"], dtype=np.float32)
                    lang = row.get("language", row.get("task", ""))
                    T = min(len(acts), max_episode_len)
                    if T >= 10:
                        demos.append({
                            "observations": obs[:T],
                            "actions": acts[:T, :action_dim],
                            "episode_id": len(demos),
                            "language": str(lang),
                            "source": "droid",
                        })
                    if len(demos) >= max_episodes:
                        break
        except ImportError:
            print("WARNING: pandas not installed for parquet loading")

    print(f"Loaded {len(demos)} episodes from DROID")
    return demos


# ── Open X-Embodiment (RT-X format) ─────────────────────────────────

def load_open_x_embodiment(
    data_dir: str,
    datasets: Optional[List[str]] = None,
    max_episodes: int = 200,
    max_episode_len: int = 300,
    action_dim: int = 7,
) -> List[Dict]:
    """Load Open X-Embodiment subsets (RT-X format).

    OXE is the cross-embodiment dataset used by RT-X / Octo / OpenVLA.
    It contains 22 robot embodiments across 500K+ episodes, stored as
    parquet tables with images and proprioception.

    Download: https://huggingface.co/datasets/omega-health/open-x-embodiment

    Args:
        data_dir: path to OXE parquet files
        datasets: filter to specific dataset names (e.g., "kuka_bridge")
        max_episodes: maximum episodes to load
        max_episode_len: truncate long episodes
        action_dim: typically 7 for end-effector control
    """
    demos = []

    if not os.path.exists(data_dir):
        print(f"OXE directory not found: {data_dir}")
        print("Download from: https://huggingface.co/datasets/omega-health/open-x-embodiment")
        return []

    try:
        import pandas as pd
    except ImportError:
        print("WARNING: pandas not installed. pip install pandas pyarrow")
        return []

    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])

    count = 0
    for fname in parquet_files:
        if count >= max_episodes:
            break
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                if count >= max_episodes:
                    break
                src = row.get("dataset_name", row.get("source", ""))
                if datasets and src not in datasets:
                    continue

                obs = np.array(row.get("observations", row.get("obs", [])),
                               dtype=np.float32)
                acts = np.array(row.get("actions", row.get("act", [])),
                                dtype=np.float32)
                lang = row.get("language_instruction", row.get("task", ""))

                T = min(len(acts), max_episode_len)
                if T < 10:
                    continue

                demos.append({
                    "observations": obs[:T],
                    "actions": acts[:T, :action_dim],
                    "episode_id": count,
                    "language": str(lang),
                    "source": f"oxe/{src}",
                })
                count += 1
        except Exception:
            continue

    print(f"Loaded {len(demos)} episodes from Open X-Embodiment")
    return demos


# ── Procedural cable dataset (synthetic) ─────────────────────────────

def generate_cable_demos(
    n_episodes: int = 200,
    n_beads: int = 33,
    crossing_range: tuple = (2, 5),
    stiffness_range: tuple = (0.5, 1.5),
    seed: int = 0,
) -> List[Dict]:
    """Generate cable untangling demonstrations procedurally.

    Uses the miniature CableEnv to generate demonstrations at scale.
    This is the same data the miniature study uses, but exposed through
    the same interface as the real-dataset loaders for pipeline uniformity.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from steerable.config import EnvConfig
    from steerable.envs.cable import CableEnv
    from steerable.policies.expert import run_expert

    cfg = EnvConfig(n_nodes=n_beads)
    rng = np.random.RandomState(seed)
    demos = []

    for k in range(n_episodes):
        ct = rng.randint(crossing_range[0], crossing_range[1] + 1)
        sm = rng.uniform(*stiffness_range)
        env = CableEnv(cfg, seed=seed * 1000 + k, stiffness_mult=sm,
                       crossing_target=ct)
        rec = run_expert(env, record=True, max_steps=200)
        if rec["actions"].shape[0] < 10:
            continue
        demos.append({
            "observations": rec["obs"],
            "actions": rec["actions"],
            "episode_id": k,
            "language": f"untangle cable with {ct} crossings, stiffness {sm:.2f}",
            "source": "procedural_cable",
            "crossings0": ct,
            "stiffness": sm,
        })

    print(f"Generated {len(demos)} procedural cable demos")
    return demos
