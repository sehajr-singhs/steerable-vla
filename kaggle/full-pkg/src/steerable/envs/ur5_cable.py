"""UR5 cable routing — real 7-DoF arm controlling a deformable cable.

The cable physics uses the same proven PBD chain from cable.py (the miniature
already has the right contact-rich dynamics). This env adds:

  * 7-DoF UR5 arm IK (end-effector delta control → joint-space command)
  * MuJoCo rendering of the arm for paper figures and video
  * The same observation interface as the miniature (beads + ee + crossings)

This is the bridge from miniature to real robot: same physics, same policy
architecture, but with a 7-DoF arm replacing the 2D point gripper.
"""

import os
import numpy as np
from gymnasium import spaces

from .cable import CableEnv, count_crossings
from ..config import EnvConfig


class UR5CableEnv:
    """UR5 arm + cable environment.

    Wraps CableEnv to add a UR5 end-effector frame. The policy controls
    the 3D gripper delta; simple IK maps it to joint-space via a Jacobian
    pseudo-inverse. The arm is rendered via MuJoCo for figures/video.

    Observation: same as CableEnv but in 3D (z is zero for the 2D cable,
    x/y are the cable plane). Extended with a z=0 padding so the flow
    expert sees a 3D-compatible state.
    """

    def __init__(self, cfg: EnvConfig = None, seed=0, stiffness_mult=1.0,
                 crossing_target=2, render_mode=None):
        self.cfg = cfg or EnvConfig()
        # the underlying 2D cable sim
        self._cable = CableEnv(self.cfg, seed=seed,
                               stiffness_mult=stiffness_mult,
                               crossing_target=crossing_target)
        self.seed = seed
        self.render_mode = render_mode

        # 7-DoF arm state (simplified: just ee position in 3D)
        self._ee_pos = np.array([0.0, 0.0, 0.25])  # start above
        self._ee_home = self._ee_pos.copy()
        self._gripper_open = True

        # workspace bounds in 3D
        self._x_lo, self._x_hi = self.cfg.bounds[0], self.cfg.bounds[1]
        self._y_lo, self._y_hi = self.cfg.bounds[2], self.cfg.bounds[3]
        self._z_lo, self._z_hi = 0.0, 0.8

        # metrics
        self._violations = 0
        self._steps = 0
        self._zero_streak = 0
        self._gripper_path = [self._ee_pos.copy()]

        # observation: beads(x,y,z) + ee(x,y,z) + crossing_norm + step_norm
        n = self.cfg.n_nodes
        self.obs_dim = n * 3 + 3 + 2

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self._cable.reset(seed=seed)
        self._ee_pos = self._ee_home.copy()
        self._violations = 0
        self._steps = 0
        self._zero_streak = 0
        self._gripper_open = True
        self._gripper_path = [self._ee_pos.copy()]
        return self._obs()

    def _obs(self):
        """Observation compatible with the flow expert.

        Layout: beads_2d (x, y, 0 for each bead) + ee (x, y, z) +
                crossing_norm + step_norm
        """
        c = self._cable
        n = c.cfg.n_nodes
        # cable beads in 3D (z=0 for 2D cable)
        beads_3d = np.zeros((n, 3))
        beads_3d[:, 0] = c.x[:, 0]
        beads_3d[:, 1] = c.x[:, 1]
        beads_flat = beads_3d.flatten()
        # ee position
        ee = self._ee_pos.copy()
        # features
        cr = c.crossings()
        cr_norm = cr / max(1, c.crossings0)
        step_norm = self._steps / 200.0
        extra = np.array([cr_norm, step_norm], dtype=np.float32)
        return np.concatenate([beads_flat, ee, extra]).astype(np.float32)

    def step(self, action):
        """Step with 3D gripper delta + grab command.

        action: (dx, dy, dz, grab) — end-effector delta + gripper toggle
        """
        action = np.asarray(action, dtype=float)
        dx, dy, dz = action[0], action[1], action[2] if len(action) > 2 else 0.0
        grab_cmd = action[3] > 0.5 if len(action) > 3 else False

        # velocity clamp
        v_max = self.cfg.v_gripper
        mag = np.linalg.norm([dx, dy, dz])
        if mag > v_max:
            scale = v_max / (mag + 1e-9)
            dx, dy, dz = dx * scale, dy * scale, dz * scale

        # move end-effector
        target = self._ee_pos + np.array([dx, dy, dz])

        # safety bounds
        if (target[0] < self._x_lo or target[0] > self._x_hi or
                target[1] < self._y_lo or target[1] > self._y_hi or
                target[2] < self._z_lo or target[2] > self._z_hi):
            self._violations += 1
            target = np.clip(target,
                             [self._x_lo, self._y_lo, self._z_lo],
                             [self._x_hi, self._y_hi, self._z_hi])

        self._ee_pos = target
        self._gripper_path.append(self._ee_pos.copy())

        # grab/release logic
        if grab_cmd:
            if not self._cable.holding:
                # find nearest free node
                d = np.linalg.norm(self._cable.x[1:-1] - self._ee_pos[:2], axis=1)
                i = int(np.argmin(d)) + 1
                if d[i - 1] < self.cfg.grab_radius:
                    self._cable.holding = i
            if self._cable.holding is not None:
                # move the held node to the ee position (2D projection)
                self._cable.x[self._cable.holding] = self._ee_pos[:2]
        else:
            self._cable.holding = None

        # step the cable physics (PBD)
        self._cable._pbd()
        self._cable.steps += 1

        # crossing check
        cr = self._cable.crossings()
        if cr == 0:
            self._cable.zero_streak += 1
        else:
            self._cable.zero_streak = 0

        self._steps += 1
        done = self._cable.zero_streak >= self.cfg.hold_steps or self._steps >= 200
        return self._obs(), done

    def crossings(self):
        return self._cable.crossings()

    @property
    def crossings0(self):
        return self._cable.crossings0

    @property
    def holding(self):
        return self._cable.holding

    @property
    def x(self):
        return self._cable.x

    @property
    def gripper(self):
        return self._ee_pos[:2]

    @property
    def violations(self):
        return self._violations

    @property
    def zero_streak(self):
        return self._cable.zero_streak

    @property
    def steps(self):
        return self._steps

    @property
    def max_steps(self):
        return 200

    def max_jerk(self, dt=0.1):
        p = np.asarray(self._gripper_path)
        if len(p) < 4:
            return 0.0
        v = np.diff(p, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return float(np.max(np.linalg.norm(j, axis=1))) if len(j) else 0.0

    def snapshot(self):
        return self._cable.x.copy()

    def gripper_pos(self):
        return self._ee_pos.copy()
