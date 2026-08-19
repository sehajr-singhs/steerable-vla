"""The data flywheel (DataFly thesis, applied to the VLA setting).

Loop: train -> deploy n rollouts -> score -> curate near-misses -> relabel
failures with the oracle -> merge into the dataset -> retrain.

Curation strategies implemented:
  * none           keep nothing new (control)
  * near_miss      keep failures that made >= near_miss_frac progress
  * relabel        relabel every failure with the oracle (DAgger-style)

Prior controlled work showed the *strategy* decides whether the loop
compounds; this loop re-measures that in the flow-VLA setting.
"""

import numpy as np

from ..config import FlywheelConfig, DataConfig
from ..data import build_dataset, chunk_episode
from ..policies.flow_expert import train_policy
from ..policies.expert import run_expert


def _score(ep):
    """Progress signal in [0, 1] from crossings0 to 0."""
    c0 = max(1, ep["crossings0"])
    return 1.0 - ep["crossings_final"] / c0


def run_flywheel(make_env, make_policy, dims, tcfg, dcfg, fcfg: FlywheelConfig,
                 strategy="near_miss", seed=0, device="cpu", n_demos0=40):
    """Returns {iterations: [success...], metrics per iteration}."""
    import numpy as np

    rng = np.random.RandomState(seed)

    demos = []
    # initial dataset: expert demos on train family
    for k in range(n_demos0):
        env = make_env(seed=seed * 1000 + k + 5000)
        rec = run_expert(env, record=True)
        if rec["actions"].shape[0] >= dcfg.chunk + 2:
            demos.append(rec)

    policy = make_policy()
    dataset = build_dataset(demos, dcfg, seed=seed)
    train_policy(policy, dataset, tcfg, dcfg, device, kind="ours_full", seed=seed)

    curve = []
    detail = []
    for it in range(fcfg.iterations):
        # deploy
        deploys = []
        for k in range(fcfg.n_deploy):
            env = make_env(seed=seed * 1000 + k + 20000 + it * 1000)
            ep = run_episode_local(policy, env, tcfg, dcfg, device)
            deploys.append(ep)
        # score + curate
        successes = [e for e in deploys if e["success"]]
        fails = [e for e in deploys if not e["success"]]
        progress = [(e, _score(e)) for e in fails]
        kept = []
        if strategy == "near_miss":
            kept = [e for e, p in progress if p >= fcfg.near_miss_frac]
        elif strategy == "relabel":
            kept = fails
        # relabel kept failures with the oracle FROM THE FAILURE STATE
        # (DAgger-style): the deployment env is still in the tangled terminal
        # state, so the expert completes the task from exactly where the
        # policy got stuck -- not from a fresh re-seed of the same config.
        for e in kept:
            env = e["env"]
            rec = run_expert(env, record=True)
            if rec["actions"].shape[0] >= dcfg.chunk + 2:
                demos.append(rec)
        dataset = build_dataset(demos, dcfg, seed=seed)
        train_policy(policy, dataset, tcfg, dcfg, device, kind="ours_full",
                     seed=seed, epochs=fcfg.retrain_epochs)
        succ = len(successes) / len(deploys) if deploys else 0.0
        curve.append(round(succ, 4))
        detail.append({
            "iter": it, "deployed": len(deploys), "success": succ,
            "n_failures": len(fails), "curated": len(kept),
            "dataset_episodes": len(demos)})
    return {"strategy": strategy, "curve": curve, "detail": detail,
            "n_demos0": n_demos0}


def run_episode_local(policy, env, tcfg, dcfg, device):
    """Deployment-time episode used by the flywheel.

    Attaches the env itself so curation can relabel from the failure's
    terminal state (the env object is still tangled after a failed deploy).
    """
    from ..eval import run_episode, _subgoal_for
    ep = run_episode(policy, env, max_steps=120, patience=24, chunk=dcfg.chunk,
                     steer=True, use_filter=True, device=device,
                     subgoal_fn=_subgoal_for, allow_intervention=False)
    ep["env"] = env
    return ep
