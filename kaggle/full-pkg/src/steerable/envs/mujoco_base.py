"""Base class for MuJoCo robot-arm environments in the Steerable VLA study.

Every MuJoCo env inherits from this to share: action-space normalization,
observation framing, gripper state tracking, safety bounds, and the
crossing-counting interface that the expert / policy / eval harness expect.

Subclasses implement: _load_model(), _apply_action(), _compute_reward(),
and _is_done().
"""

import os
import numpy as np

import mujoco
import gymnasium as gym


class MujocoBaseEnv(gym.Env):
    """Abstract MuJoCo env with a gripper endpoint, workspace bounds, and
    safety-violation counting.

    The concrete env loads an XML model, exposes a (dx, dy, dz, grab) or
    (7-DoF delta + grab) action, and returns observations that are compatible
    with the flow-matching expert (flattened node-like observation + gripper
    state + scalar features).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path, action_dim=4, obs_dim=None,
                 bounds=None, v_max=0.05, force_max=50.0,
                 grab_radius=0.05, hold_steps=10,
                 ctrl_dt=0.02, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.render_mode = render_mode

        self.action_dim = action_dim
        self._v_max = v_max
        self._force_max = force_max
        self._grab_radius = grab_radius
        self._hold_steps = hold_steps
        self._ctrl_dt = ctrl_dt

        # workspace bounds: (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)
        self._bounds = np.array(bounds or [-0.5, 0.5, -0.5, 0.5, 0.0, 0.8])
        assert len(self._bounds) == 6

        # state
        self._gripper_pos = np.zeros(3)
        self._holding = False
        self._held_body = None
        self._violations = 0
        self._steps = 0
        self._zero_streak = 0
        self._gripper_path = []

        # observation space: subclasses set self.observation_space
        if obs_dim is not None:
            self.observation_space = gym.spaces.Box(
                -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._violations = 0
        self._steps = 0
        self._zero_streak = 0
        self._holding = False
        self._held_body = None
        self._gripper_path = []
        self._gripper_pos = self._home_pos()
        self._apply_custom_reset(seed)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        # velocity clamp
        vel = action[:3]
        mag = np.linalg.norm(vel)
        if mag > self._v_max:
            vel = vel / mag * self._v_max
            action = np.concatenate([vel, action[3:]])

        # move gripper target
        target = self._gripper_pos + action[:3]
        # safety bounds check
        lo = self._bounds[::2]
        hi = self._bounds[1::2]
        if np.any(target < lo) or np.any(target > hi):
            self._violations += 1
            target = np.clip(target, lo, hi)

        self._gripper_pos = target
        self._gripper_path.append(self._gripper_pos.copy())

        # grab logic
        grab_cmd = action[3] > 0.5 if len(action) > 3 else False
        self._handle_grab(grab_cmd)

        # step MuJoCo
        self._apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self._steps += 1

        # task-specific done check
        done = self._is_done()
        obs = self._get_obs()
        return obs, done

    # ------------------------------------------------------------------
    # Override points for subclasses
    # ------------------------------------------------------------------

    def _home_pos(self):
        """Gripper home position at reset."""
        return np.array([0.0, 0.0, 0.6])

    def _apply_custom_reset(self, seed):
        """Task-specific reset logic (randomize objects etc.)."""
        pass

    def _apply_action(self, action):
        """Write MuJoCo control signals from the action vector."""
        pass

    def _is_done(self):
        """Return True when the episode should end."""
        return self._steps >= 500

    def _get_obs(self):
        """Return the observation vector. Must match training dimensions."""
        raise NotImplementedError

    def _handle_grab(self, grab_cmd):
        """Manage gripper state (open/close, which body is held)."""
        pass

    def _reward(self):
        """Dense reward signal. Subclasses implement task-specific shaping."""
        return 0.0

    # ------------------------------------------------------------------
    # Safety / metrics
    # ------------------------------------------------------------------

    def violations(self):
        return self._violations

    def max_jerk(self, dt=0.1):
        p = np.asarray(self._gripper_path)
        if len(p) < 4:
            return 0.0
        v = np.diff(p, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return float(np.max(np.linalg.norm(j, axis=1))) if len(j) else 0.0

    def gripper_pos(self):
        return self._gripper_pos.copy()

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            if self.render_mode == "human":
                self.renderer.render()
            return self.renderer.render()
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
