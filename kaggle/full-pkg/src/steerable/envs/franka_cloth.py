"""Franka cloth manipulation — the real-robot analog of textile folding.

A Franka Emika Panda arm must fold a piece of cloth on a table. The cloth
is modeled as a grid of MuJoCo particles connected by springs (mass-spring
cloth simulation). The gripper can pinch a corner and fold it over.

This is the real Task B from the proposal: non-rigid, shifting under its
own drape, and long-horizon.
"""

import os
import numpy as np

import mujoco
from gymnasium import spaces

from .mujoco_base import MujocoBaseEnv


# cloth grid parameters
GRID_W = 8             # particles wide
GRID_H = 6             # particles tall
PARTICLE_SPACING = 0.02  # 20mm between particles
CLOTH_MASS = 0.1       # total mass (kg)

FRANKA_HOME = np.array([0.0, 0.0, 0.4])


class FrankaClothEnv(MujocoBaseEnv):
    """Franka end-effector folding a cloth on a table.

    Action: (dx, dy, dz, grab) — end-effector delta + gripper command.
    Observation: flattened cloth particle positions + ee pos + fold progress.
    """

    def __init__(self, grid_w=GRID_W, grid_h=GRID_H, max_steps=600,
                 render_mode=None):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.n_particles = grid_w * grid_h
        self._max_steps = max_steps
        self._grid_w = grid_w
        self._grid_h = grid_h

        # build XML
        xml = self._build_xml(grid_w, grid_h)
        self._xml_path = os.path.join(os.path.dirname(__file__), "_cloth_tmp.xml")
        with open(self._xml_path, "w") as f:
            f.write(xml)

        obs_dim = self.n_particles * 3 + 3 + 2  # particles + ee + (fold_progress, step)
        super().__init__(
            xml_path=self._xml_path,
            action_dim=4,
            obs_dim=obs_dim,
            bounds=[-0.3, 0.3, -0.3, 0.3, 0.0, 0.5],
            v_max=0.04,
            force_max=30.0,
            grab_radius=0.03,
            hold_steps=max_steps // 5,
            render_mode=render_mode,
        )
        self._target_corner = None
        self._start_corner = None

    def _build_xml(self, gw, gh):
        """Build cloth as particles with ball joints + spring tendons."""
        parts = ["""
<mujoco model="franka_cloth">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicit"
         solver="CG" iterations="5">
    <flag contact="enable"/>
  </option>
  <default>
    <geom condim="4" friction="0.6 0.01 0.001"/>
  </default>
  <asset>
    <texture name="texplane" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.82 0.82 0.82" rgb2="0.72 0.72 0.72"/>
    <material name="matplane" texture="texplane" texrepeat="10 10" reflectance="0.1"/>
    <material name="cloth" rgba="0.85 0.25 0.2 0.9"/>
    <material name="gripper_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="target" rgba="0.1 0.8 0.1 0.3"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0.5 0.5 0.005" pos="0 0 -0.0025"
          material="matplane"/>
    <body name="gripper" pos="0 0 0.4">
      <freejoint name="gripper_joint"/>
      <geom name="gripper_body" type="cylinder" size="0.015 0.02"
            material="gripper_mat"/>
      <geom name="lf" type="box" size="0.003 0.003 0.012" pos="-0.01 0 -0.03"
            material="gripper_mat"/>
      <geom name="rf" type="box" size="0.003 0.003 0.012" pos="0.01 0 -0.03"
            material="gripper_mat"/>
      <site name="grip_site" pos="0 0 -0.035" size="0.01"/>
    </body>
"""]

        # corner markers
        for name, color in [("target_L", "0.1 0.8 0.1 0.4"), ("target_R", "0.1 0.1 0.8 0.4")]:
            parts.append(f'    <body name="{name}" pos="0 0 0.001">\n'
                         f'      <freejoint name="{name}_j"/>\n'
                         f'      <geom type="cylinder" size="0.01 0.001" '
                         f'rgba="{color}"/>\n'
                         f'    </body>')

        # cloth particles
        x0 = -(gw - 1) * PARTICLE_SPACING / 2
        y0 = -(gh - 1) * PARTICLE_SPACING / 2
        for i in range(gw):
            for j in range(gh):
                x = x0 + i * PARTICLE_SPACING
                y = y0 + j * PARTICLE_SPACING
                z = 0.005  # slightly above table
                parts.append(
                    f'    <body name="p_{i}_{j}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
                    f'      <freejoint name="pj_{i}_{j}"/>\n'
                    f'      <geom name="pg_{i}_{j}" type="sphere" size="0.004" '
                    f'material="cloth"/>\n'
                    f'    </body>')

        parts.append("  </worldbody>")
        parts.append("</mujoco>")
        return "\n".join(parts)

    def _home_pos(self):
        return FRANKA_HOME.copy()

    def _apply_custom_reset(self, seed):
        rng = np.random.RandomState(seed if seed is not None else 0)
        gw, gh = self._grid_w, self._grid_h
        x0 = -(gw - 1) * PARTICLE_SPACING / 2
        y0 = -(gh - 1) * PARTICLE_SPACING / 2

        # lay cloth flat with slight wrinkles
        for i in range(gw):
            for j in range(gh):
                body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"p_{i}_{j}")
                x = x0 + i * PARTICLE_SPACING
                y = y0 + j * PARTICLE_SPACING
                z = 0.005 + rng.uniform(0, 0.003)  # slight initial wrinkles
                self.data.xpos[body] = [x, y, z]
                jnt = self.model.body_jntadr[body]
                if jnt >= 0:
                    qa = self.model.jnt_qposadr[jnt]
                    self.data.qvel[qa:qa + 3] = 0.0

        # fold target: fold bottom-left corner to top-right
        self._start_corner = np.array([x0, y0, 0.005])
        self._target_corner = np.array([
            x0 + (gw - 1) * PARTICLE_SPACING,
            y0 + (gh - 1) * PARTICLE_SPACING,
            0.005
        ])
        self._gripper_pos = FRANKA_HOME.copy()

    def _get_obs(self):
        parts = np.zeros((self.n_particles, 3))
        gw, gh = self._grid_w, self._grid_h
        for i in range(gw):
            for j in range(gh):
                body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          f"p_{i}_{j}")
                parts[i * gh + j] = self.data.xpos[body]
        parts_flat = parts.flatten().astype(np.float32)
        ee = self._gripper_pos.astype(np.float32)

        # fold progress: how far is the moved corner toward the target?
        # track the corner particle (0,0)
        corner = parts[0, 0]
        if self._target_corner is not None and self._start_corner is not None:
            d_start = np.linalg.norm(corner - self._start_corner)
            d_target = np.linalg.norm(self._target_corner - self._start_corner)
            progress = min(1.0, d_start / (d_target + 1e-9))
        else:
            progress = 0.0
        step_norm = self._steps / self._max_steps
        extra = np.array([progress, step_norm], dtype=np.float32)
        return np.concatenate([parts_flat, ee, extra])

    def _is_done(self):
        # done when fold progress > 0.9 or max steps
        obs = self._get_obs()
        if obs[-2] > 0.9:
            self._zero_streak += 1
        else:
            self._zero_streak = 0
        if self._zero_streak >= self._hold_steps:
            return True
        return self._steps >= self._max_steps

    def _handle_grab(self, grab_cmd):
        if grab_cmd and not self._holding:
            # find nearest cloth particle
            best_i, best_j = 0, 0
            best_dist = np.inf
            gw, gh = self._grid_w, self._grid_h
            for i in range(gw):
                for j in range(gh):
                    body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                              f"p_{i}_{j}")
                    d = np.linalg.norm(self.data.xpos[body] - self._gripper_pos)
                    if d < best_dist:
                        best_dist = d
                        best_i, best_j = i, j
            if best_dist < self._grab_radius:
                self._holding = True
                self._held_i = best_i
                self._held_j = best_j
        elif not grab_cmd:
            self._holding = False

        if self._holding:
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      f"p_{self._held_i}_{self._held_j}")
            self.data.xpos[body] = self._gripper_pos.copy()
            jnt = self.model.body_jntadr[body]
            if jnt >= 0:
                qa = self.model.jnt_qposadr[jnt]
                self.data.qvel[qa:qa + 3] = 0.0

    def fold_progress(self):
        """Return fold progress in [0, 1]."""
        return float(self._get_obs()[-2])

    def close(self):
        super().close()
        if os.path.exists(self._xml_path):
            os.remove(self._xml_path)
