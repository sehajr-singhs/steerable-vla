"""Sanity tests: env physics, expert, flow training, filter, flywheel.

Run: PYTHONPATH=src python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from steerable.config import EnvConfig, DataConfig, TrainConfig, EvalConfig
from steerable.envs.cable import CableEnv, count_crossings
from steerable.policies.expert import run_expert
from steerable.policies.flow_expert import make_policy, train_policy
from steerable.data import collect_demos, build_dataset
from steerable.safety.cbf import CBFQPFilter
from steerable.eval import evaluate


def test_crossing_counting():
    x = np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5]])
    # segments (0-1) and (2-3) cross
    hits, _ = count_crossings(x)
    assert hits == 0  # only 3 segments: adjacent -> no proper crossings
    x = np.array([[-1, 0], [0, 0.5], [1, 0], [0, -0.5], [-1, 0.4], [1, 0.4]])
    hits, _ = count_crossings(x)
    assert hits >= 1


def test_env_reset_realizes_crossing_target():
    env = CableEnv(seed=0, crossing_target=2)
    assert env.crossings() == 2


def test_expert_untangles_train_family():
    env = CableEnv(seed=0, crossing_target=2, stiffness_mult=1.0)
    r = run_expert(env, max_steps=200)
    assert r["success"], r


def test_flow_training_loss_decreases():
    import torch
    torch.manual_seed(0)
    env_cfg = EnvConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig()
    env = CableEnv(env_cfg, seed=0, crossing_target=2)
    from steerable.policies.expert import run_expert as re
    rec = re(env, record=True)
    from steerable.data import chunk_episode
    items = chunk_episode(rec, dcfg, np.random.RandomState(0))
    assert len(items) > 0
    o, s, a = items[0][0].shape[0], items[0][1].shape[0], dcfg.chunk * 3
    policy = make_policy("ours_full", (o, s, a), tcfg, dcfg)
    obs = torch.as_tensor([it[0] for it in items])
    sub = torch.as_tensor([it[1] for it in items])
    ch = torch.as_tensor([it[2] for it in items])
    nu = torch.as_tensor([it[3] for it in items])
    rng = np.random.RandomState(0)
    l0 = float(policy.cfm_loss(obs, sub, ch, nu, rng, dcfg.steer_prob))
    for _ in range(30):
        loss = policy.cfm_loss(obs, sub, ch, nu, rng, dcfg.steer_prob)
        loss.backward()
        for p in policy.parameters():
            p.data -= 1e-2 * p.grad
        p.grad = None
    l1 = float(policy.cfm_loss(obs, sub, ch, nu, rng, dcfg.steer_prob))
    assert l1 < l0, (l0, l1)


def test_cbf_filter_projects_outside_bounds():
    env = CableEnv(seed=0)
    f = CBFQPFilter(env.cfg)
    # command that would drive the gripper far outside the right edge
    u = f(np.array([5.0, 0.0, 0.0]), np.array([0.8, 0.0]))
    assert abs(u[0]) <= env.cfg.v_gripper


def test_eval_harness_runs():
    env_cfg = EnvConfig()
    dcfg = DataConfig()
    tcfg = TrainConfig()
    from steerable.policies.expert import run_expert as re
    env = CableEnv(env_cfg, seed=0, crossing_target=2)
    rec = re(env, record=True)
    items = __import__("steerable.data", fromlist=["chunk_episode"]).chunk_episode(
        rec, dcfg, np.random.RandomState(0))
    o, s, a = items[0][0].shape[0], items[0][1].shape[0], dcfg.chunk * 3
    policy = make_policy("ours_full", (o, s, a), tcfg, dcfg)

    def fac(seed):
        return CableEnv(env_cfg, seed=seed, crossing_target=2, stiffness_mult=1.0)
    res = evaluate(policy, fac, 4, 0, dcfg, EvalConfig(), device="cpu")
    assert res["n"] == 4
    assert 0.0 <= res["success"] <= 1.0
