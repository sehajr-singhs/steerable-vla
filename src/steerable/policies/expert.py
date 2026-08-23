"""Scripted oracle — the "teleoperator" that produces demonstrations.

Strategy per crossing: take the return strand (the later of the two crossing
segments), try BOTH of its endpoints, and for each try pulling the grabbed
node toward-and-through the crossing plus a set of fallback directions (up,
down, sideways, diagonals). Every attempt starts from the pre-maneuver
snapshot (and the recorded trajectory is rolled back with it); the first
attempt that reduces the crossing count is kept. This makes the oracle a
robust local search rather than a fixed handcrafted move.

The oracle is also the *relabeler* in the flywheel loop: given a failed
episode's start state, it produces a fresh expert trajectory from that state.
"""

import numpy as np

from ..envs.geometry import seg_intersect


def _clamp(env, p):
    """Keep a target inside the gripper workspace (the oracle respects it)."""
    x_lo, x_hi, y_lo, y_hi = env.cfg.bounds
    m = 0.02
    return np.clip(np.asarray(p, dtype=float),
                   [x_lo + m, y_lo + m], [x_hi - m, y_hi - m])


def _first_crossing(x):
    n = len(x)
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            ok, p = seg_intersect(x[i], x[i + 1], x[j], x[j + 1])
            if ok:
                return i, j, p
    return None


def _reachable(env, p):
    x_lo, x_hi, y_lo, y_hi = env.cfg.bounds
    m = 0.03
    return (x_lo + m <= p[0] <= x_hi - m and y_lo + m <= p[1] <= y_hi - m)


def _candidates(env, j, p):
    """(node, pull_dir) candidates for the return strand of a crossing."""
    dirs = [np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
            np.array([-1.0, 1.0]) / np.sqrt(2)]
    out = []
    for idx in (j, j + 1):
        if not (0 < idx < len(env.x) - 1) or not _reachable(env, env.x[idx]):
            continue
        pos = env.x[idx]
        toward = np.asarray(p, dtype=float) - pos
        n = np.linalg.norm(toward)
        toward = toward / n if n > 1e-6 else np.array([0.0, 1.0])
        for d in [toward, -toward] + dirs:
            out.append((idx, d))
    return out


def run_expert(env, max_steps=250, record=False):
    """Run the oracle to completion. Optionally record (obs, action, subgoal)."""
    actions = []
    obs = [env._obs().copy()]
    milestones = []          # subgoal keyframes: snapshots at crossing decreases
    crossing_hist = [env.crossings()]
    done = False
    failed = set()           # crossings that resisted one full candidate pass
    sim = 0                  # total simulated steps (incl. rolled-back attempts)
    SIM_BUDGET = 5000

    def act(a):
        nonlocal sim
        sim += 1
        env.step(a)
        actions.append([float(a[0]), float(a[1]), float(a[2])])
        obs.append(env._obs().copy())

    def rollback(snap_x, snap_gripper, n_actions):
        env.x[:] = snap_x
        env.gripper = snap_gripper.copy()
        env.holding = None
        del actions[n_actions:]
        del obs[n_actions + 1:]

    def pick_crossing():
        """First crossing not in the failed set (skips stuck ones)."""
        n = len(env.x)
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                ok, p = seg_intersect(env.x[i], env.x[i + 1], env.x[j], env.x[j + 1])
                if ok and (round(p[0], 1), round(p[1], 1)) not in failed:
                    return i, j, p
        return _first_crossing(env.x)

    for _ in range(max_steps):
        cr = env.crossings()
        if cr == 0:
            act([0.0, 0.0, 0.0])         # hold still to confirm
            if env.zero_streak >= env.cfg.hold_steps:
                done = True
                break
            continue

        hit = pick_crossing()
        if hit is None:
            dx = 0.08 * (0.5 - env.rng.rand())
            dy = 0.08 * (0.5 - env.rng.rand())
            act([dx, dy, 0.0])
            continue

        i, j, p = hit
        # exhaustive (node, direction) candidates on the return strand
        base_dirs = [np.array([0.0, 1.0]), np.array([0.0, -1.0]),
                     np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
                     np.array([1.0, 1.0]) / np.sqrt(2),
                     np.array([-1.0, 1.0]) / np.sqrt(2),
                     np.array([1.0, -1.0]) / np.sqrt(2),
                     np.array([-1.0, -1.0]) / np.sqrt(2)]
        dirs = []
        # Try BOTH segments of the crossing (not just the return strand)
        for idx in (i, i + 1, j, j + 1):
            if not (0 < idx < len(env.x) - 1) or not _reachable(env, env.x[idx]):
                continue
            pos = env.x[idx]
            toward = np.asarray(p, dtype=float) - pos
            nrm = np.linalg.norm(toward)
            toward = toward / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
            dirs.extend((idx, d) for d in [toward, -toward] + base_dirs)
        if not dirs:
            failed.add((round(p[0], 1), round(p[1], 1)))
            continue
        node = dirs[0][0]

        cleared = False
        for node, cand in dirs:
            if cleared:
                break
            snap_x = env.x.copy()
            snap_gripper = env.gripper.copy()
            n_actions = len(actions)
            node_pos = _clamp(env, env.x[node])
            # teleport the gripper to the node (recorded as a single move)
            step_vec = _clamp(env, node_pos) - env.gripper
            act([step_vec[0], step_vec[1], 0.0])
            act([0.0, 0.0, 1.0])          # grab
            target = _clamp(env, node_pos + cand * 0.5)
            for _ in range(8):
                if env.crossings() < cr:
                    cleared = True
                    break
                dd = target - env.gripper
                dist = np.linalg.norm(dd)
                if dist < 0.02:
                    target = _clamp(env, env.gripper + cand * 0.15)
                    dd = target - env.gripper
                    dist = np.linalg.norm(dd)
                step_vec = dd / (dist + 1e-9) * min(env.cfg.v_gripper, dist)
                act([step_vec[0], step_vec[1], 1.0])
            act([0.0, 0.0, 0.0])           # release
            if not cleared:
                rollback(snap_x, snap_gripper, n_actions)

        if not cleared:
            failed.add((round(p[0], 1), round(p[1], 1)))
        after = env.crossings()
        if after < cr:
            # Crossings decreased: clear failed set so remaining crossings
            # get fresh retry attempts (the cable reconfigured).
            failed.clear()
            milestones.append(env.snapshot())
        crossing_hist.append(after)
        if len(actions) >= max_steps or sim >= SIM_BUDGET:
            break

    if not record:
        return {"success": bool(done), "steps": len(actions),
                "crossings_final": env.crossings(),
                "violations": env.violations}
    return {"success": bool(done), "steps": len(actions),
            "crossings_final": env.crossings(), "violations": env.violations,
            "actions": np.asarray(actions, dtype=np.float32),
            "obs": np.asarray(obs, dtype=np.float32),
            "milestones": milestones, "crossing_hist": crossing_hist}
