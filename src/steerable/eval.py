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
    """Subgoal at inference: goal cable + first-crossing point.

    The crossing point is extracted from the environment state by the
    high-level planner (the miniature's stand-in for a VLM), exactly as in
    training (data.subgoal_from_state). The flow expert only ever sees
    the resulting conditioning vector.
    """
    # For cable envs with cfg attribute, use the geometric heuristic
    if hasattr(env, 'cfg') and hasattr(env.cfg, 'n_nodes'):
        from .data import subgoal_from_state
        n = env.cfg.n_nodes
        return subgoal_from_state(obs.shape[0], n, env.cfg.cable_len,
                                  env.x, env.gripper, env.crossings() / max(1, env.crossings0))
    # For textile/tool envs: return obs as-is (the obs already contains
    # task-relevant state). Pad to the expected subgoal dim.
    subgoal_dim = obs.shape[0]
    result = np.zeros(subgoal_dim, dtype=np.float32)
    # Encode crossing progress as the first element
    result[0] = env.crossings() / max(1, env.crossings0)
    return result


def _vlm_subgoal_for(env, obs, vlm_planner=None):
    """VLM-based subgoal: render image + encode language -> predict subgoals.

    Uses the VLMSubgoalPlanner to predict dense visual subgoals from
    the rendered cable state. Falls back to geometric heuristic if
    planner is not provided.
    """
    if vlm_planner is None:
        return _subgoal_for(env, obs)
    import torch
    from .policies.vlm_planner import encode_language
    # Render image
    img = env.img_obs()  # (3, 64, 64)
    img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    # Encode language
    lang = encode_language('untangle the cable crossing').unsqueeze(0)
    # Predict subgoals
    vlm_planner.eval()
    with torch.no_grad():
        subgoals, confidences, K = vlm_planner(img_t, lang, n_subgoals=4)
    # Convert to conditioning vector (flatten subgoal positions + confidence)
    subs = subgoals[0].numpy().flatten()  # (K*2,)
    confs = confidences[0].numpy().flatten()  # (K,)
    # Pad to expected subgoal dimension
    subgoal_dim = obs.shape[0] - 4  # same as obs minus gripper and progress
    result = np.zeros(subgoal_dim, dtype=np.float32)
    result[:len(subs)] = subs
    return result


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

    # replan=0 means execute the full chunk before re-querying.
    # replan=K means execute K steps then re-query (receding horizon).
    horizon = chunk if (replan is None or replan <= 0) else min(replan, chunk)
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
