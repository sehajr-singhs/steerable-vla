"""Pre-registered evaluation harness.

Runs a policy over a held-out family (crossing target / stiffness held out
from every training source) with a fixed seed grid. Metrics per episode:
success, crossings reduced, steps, max jerk, safety violations, interventions.
Interventions: if the policy stalls (no crossing reduction within `patience`
steps) an oracle assist performs one expert maneuver — counted, then the
policy continues. This is the deployment-cost metric of the study.
"""

import numpy as np

from .envs.cable import CableEnv
from .policies.expert import run_expert


def _subgoal_for(env, obs):
    """Toy subgoal at inference: goal cable + first-crossing point.

    The crossing point is extracted from the environment state by the
    high-level planner (the miniature's stand-in for a VLM), exactly as in
    training (data.subgoal_from_state). The flow expert only ever sees
    the resulting conditioning vector.
    """
    from .data import subgoal_from_state
    n = env.cfg.n_nodes
    return subgoal_from_state(obs.shape[0], n, env.cfg.cable_len,
                              env.x, env.gripper, env.crossings() / max(1, env.crossings0))


def run_episode(policy, env: CableEnv, max_steps, patience, chunk, steer,
                use_filter, device, subgoal_fn=None, allow_intervention=True,
                act_fn=None, replan=1, refresh_subgoal=True):
    """One episode. Returns the metric dict.

    receding-horizon execution: the policy emits an action chunk of length
    `chunk`, but only the first `replan` steps are executed before the
    policy is re-queried. On every re-query the high-level planner re-issues
    the subgoal from the current scene state (the steerable conditioning
    loop of the paper) when refresh_subgoal is True; otherwise the initial
    subgoal is kept (open-loop conditioning). replan == chunk recovers
    full open-loop chunk execution.
    """
    act_fn = act_fn or policy.act
    obs = env._obs()
    subgoal = (subgoal_fn or _subgoal_for)(env, obs)
    done = False
    steps = 0
    interventions = 0
    stall = 0
    best = env.crossings()
    ever_ok = False          # policy reached crossings==0 on its own
    last_nudge = None if steer is None else np.zeros(3, dtype=np.float32)
    filter_ = None
    if use_filter:
        from .safety.cbf import CBFQPFilter
        filter_ = CBFQPFilter(env.cfg)

    horizon = max(1, replan) if replan else chunk
    while not done and steps < max_steps:
        a = act_fn(obs, subgoal, nudge=last_nudge)
        nudge = None if steer is None else np.zeros(3, dtype=np.float32)
        for k in range(horizon):
            if done or steps >= max_steps:
                break
            a0 = a[3 * k: 3 * k + 3]
            if use_filter:
                u = filter_(a0, env.gripper)
                a0 = np.concatenate([u, a0[2:3]])
            obs, done = env.step(a0)
            steps += 1
            if env.crossings() == 0:
                ever_ok = True

            # intervention logic: no progress for `patience` steps -> assist
            cr = env.crossings()
            if cr < best:
                best = cr
                stall = 0
            else:
                stall += 1
            if (allow_intervention and stall >= patience
                    and env.crossings() > 0 and interventions < 3):
                # teleoperator takes over for one maneuver, then hands back
                run_expert(env, max_steps=patience)
                interventions += 1
                stall = 0
                best = env.crossings()
                obs = env._obs()
            if env.zero_streak >= env.cfg.hold_steps:
                done = True
        if horizon < chunk and not done:
            # receding horizon: re-observe and (optionally) re-reason
            obs = env._obs()
            if refresh_subgoal:
                subgoal = (subgoal_fn or _subgoal_for)(env, obs)

    success = bool(env.crossings() == 0)
    ni_success = bool(ever_ok or (interventions == 0 and success))
    return {
        "success": success,
        "ni_success": ni_success,
        "crossings0": int(env.crossings0),
        "crossings_final": int(env.crossings()),
        "crossings_reduced": int(env.crossings0 - env.crossings()),
        "steps": int(steps),
        "jerk": float(env.max_jerk()),
        "violations": int(env.violations),
        "interventions": int(interventions),
    }


def evaluate(policy, make_env, n_episodes, seed, dcfg, ecfg, use_filter=True,
             steer=True, device="cpu", subgoal_fn=None, allow_intervention=True,
             act_fn=None, refresh_subgoal=True):
    """Evaluate on a held-out family: n_episodes fresh starts, mean metrics."""
    import numpy as np

    rows = []
    for k in range(n_episodes):
        env = make_env(seed=seed * 1000 + k)
        rows.append(run_episode(policy, env, ecfg.max_steps, ecfg.patience,
                                dcfg.chunk, steer, use_filter, device,
                                subgoal_fn, allow_intervention, act_fn,
                                replan=getattr(ecfg, "replan", 1),
                                refresh_subgoal=refresh_subgoal))
    return {
        "n": len(rows),
        "success": float(np.mean([r["success"] for r in rows])),
        "ni_success": float(np.mean([r["ni_success"] for r in rows])),
        "crossings_reduced": float(np.mean([r["crossings_reduced"] for r in rows])),
        "steps": float(np.mean([r["steps"] for r in rows])),
        "jerk": float(np.mean([r["jerk"] for r in rows])),
        "violations": float(np.mean([r["violations"] for r in rows])),
        "interventions": float(np.mean([r["interventions"] for r in rows])),
        "rows": rows,
    }
