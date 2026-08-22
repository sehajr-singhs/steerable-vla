"""Data pipeline: expert demonstrations -> subgoal segmentation -> chunks.

The toy analog of the proposal's pipeline:
  * expert demos  = teleoperation episodes
  * milestones    = VLM-extracted keyframes (state snapshots at the moment a
                    crossing is resolved -> the visual subgoal)
  * subgoal noise = synthetic subgoal coverage (conditioning distribution
                    denser than the demonstration distribution)
  * steering      = synthetic steering supervision: random nudges on the first
                    chunk step, relabeled as steering signals for the SMC layer
"""

import numpy as np

from .config import DataConfig
from .policies.expert import run_expert


def collect_demos(env_cfg, expert_kwargs, n_demos, seed, cfg: DataConfig):
    """Collect n_demos expert episodes on a train family; return list of dicts."""
    rng = np.random.RandomState(seed)
    demos = []
    for k in range(n_demos):
        env = expert_kwargs["make_env"](seed=seed * 1000 + k)
        rec = run_expert(env, record=True)
        if rec["actions"].shape[0] < cfg.chunk + 2:
            continue
        demos.append(rec)
    return demos


def subgoal_from_state(obs_dim, n_nodes, cable_len, nodes, gripper_w, n_cross):
    """Subgoal vector: goal cable nodes + first-crossing point, obs-dim.

    Layout (mirrors the observation): the straight, crossing-free cable in
    the node slots, and in the tail a 4-vector encoding where the high-level
    planner says to act next -- the first crossing point relative to the
    gripper (normalized by the workspace), its distance, and the remaining
    crossing count. The crossing point is the miniature's stand-in for a
    VLM-extracted visual subgoal: the flow expert is conditioned on *where*
    to intervene, never on the raw geometry required to find it.
    """
    sg = np.zeros(obs_dim, dtype=np.float32)
    tt = np.linspace(-cable_len / 2, cable_len / 2, n_nodes)
    sg[: 2 * n_nodes] = np.stack([tt, np.zeros(n_nodes)], axis=1).flatten()
    if nodes is not None and gripper_w is not None:
        from .envs.geometry import first_crossing_point
        p = first_crossing_point(nodes)
        if p is not None:
            rel = p - np.asarray(gripper_w, dtype=float)
            sg[-4] = float(rel[0]) / 2.0
            sg[-3] = float(rel[1]) / 2.0
            sg[-2] = float(np.linalg.norm(rel)) / 2.0
    sg[-1] = float(n_cross)
    return sg


def chunk_episode(rec, cfg: DataConfig, rng):
    """Split an episode into (obs, subgoal, chunk, steer_nudge) training items.

    Subgoal: goal-state conditioning (straight cable) plus the first-crossing
    point relative to the gripper -- the high-level "act here" signal -- with
    synthetic noise for conditioning coverage. Steering: with prob steer_prob,
    a random nudge u on the first chunk step; the model must absorb it (its
    effect is visible in the loss because the demonstration chunk is shifted
    by u).
    """
    obs, acts = rec["obs"], rec["actions"]
    items = []
    n = len(acts)
    obs_dim = obs.shape[1] if obs.ndim == 2 else obs[0].shape[0]
    n_nodes = max(1, (obs_dim - 4) // 2)
    for t in range(0, n - cfg.chunk, cfg.stride):
        chunk = acts[t:t + cfg.chunk].copy()
        if chunk.shape[0] < cfg.chunk:
            break
        # reconstruct the cable + gripper in world frame from the recorded obs
        nodes = obs[t][: 2 * n_nodes].reshape(n_nodes, 2)
        gripper_w = obs[t][2 * n_nodes: 2 * n_nodes + 2] + nodes.mean(axis=0)
        n_cross = obs[t][-2]
        subgoal = subgoal_from_state(obs_dim, n_nodes, 1.6, nodes, gripper_w, n_cross)
        subgoal = subgoal + rng.normal(0, cfg.subgoal_noise, size=subgoal.shape)
        nudge = np.zeros(3, dtype=np.float32)
        if rng.rand() < cfg.steer_prob:
            nudge = rng.normal(0, cfg.steer_mag, size=3).astype(np.float32)
            nudge[2] = 0.0
            # the demonstrated chunk absorbs the correction on its first step
            chunk[0, :2] += nudge[:2]
        items.append((obs[t].astype(np.float32), subgoal.astype(np.float32),
                      chunk.flatten().astype(np.float32), nudge))
    return items


def build_dataset(demos, cfg: DataConfig, seed=0):
    rng = np.random.RandomState(seed)
    items = []
    for rec in demos:
        items.extend(chunk_episode(rec, cfg, rng))
    return items


def make_episode_start(env_cfg, seed, crossing_target, stiffness_mult):
    """Build a fresh env at a given family/seed (used by eval + flywheel)."""
    from .envs.cable import CableEnv
    return CableEnv(env_cfg, seed=seed, stiffness_mult=stiffness_mult,
                    crossing_target=crossing_target)
