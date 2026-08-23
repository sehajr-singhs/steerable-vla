"""TextileFoldEnv — cloth folding on a MuJoCo table.

Task B from the proposal: fold a deformable textile to a target polyline
while the fabric shifts under its own weight. The key challenge is that
subgoal images go stale as the fabric deforms, requiring mid-execution
re-anchoring (the SMC layer's purpose).

Observations:
    - Proprioception: gripper position (x, y, z), gripper open/close
    - Proprioception: fabric center of mass, fabric spread
    - Image: top-down RGB view of the fabric on the table
    - Language: target fold description

Actions:
    - (dx, dy, dz) gripper delta + open/close flag

Zero-shot axes:
    - Material properties (stiffness, density, friction)
    - Fold geometry (valley, mountain, complex)
    - Grasp visibility (occluded corners)

Metrics:
    - Fold IoU >= 0.8 threshold
    - Grasp success rate
    - Number of regrasps
    - Slippage events
"""

import numpy as np

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# MuJoCo XML for a simple cloth-on-table scene
TEXTILE_MJCF = """
<mujoco model="textile_fold">
  <option timestep="0.005" gravity="0 0 -9.81" iterations="20"/>
  
  <default>
    <joint armature="0.01" damping="0.1"/>
    <geom condim="4" friction="1.0 0.005 0.0001"/>
  </default>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.8 0.8 0.8" rgb2="0.6 0.6 0.6"/>
    <material name="table_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="cloth_mat" rgba="0.2 0.5 0.8 1"/>
  </asset>
  
  <worldbody>
    <!-- Table -->
    <geom name="table" type="box" size="0.4 0.3 0.02" pos="0 0 -0.02"
          material="table_mat" mass="10"/>
    
    <!-- Cloth as a chain of capsules (simplified cloth model) -->
    <body name="cloth_node_0" pos="-0.15 0.1 0.01">
      <freejoint name="cloth_j_0"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_1" pos="-0.05 0.1 0.01">
      <freejoint name="cloth_j_1"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_2" pos="0.05 0.1 0.01">
      <freejoint name="cloth_j_2"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_3" pos="0.15 0.1 0.01">
      <freejoint name="cloth_j_3"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_4" pos="-0.15 0 0.01">
      <freejoint name="cloth_j_4"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_5" pos="-0.05 0 0.01">
      <freejoint name="cloth_j_5"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_6" pos="0.05 0 0.01">
      <freejoint name="cloth_j_6"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_7" pos="0.15 0 0.01">
      <freejoint name="cloth_j_7"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_8" pos="-0.15 -0.1 0.01">
      <freejoint name="cloth_j_8"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_9" pos="-0.05 -0.1 0.01">
      <freejoint name="cloth_j_9"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_10" pos="0.05 -0.1 0.01">
      <freejoint name="cloth_j_10"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    <body name="cloth_node_11" pos="0.15 -0.1 0.01">
      <freejoint name="cloth_j_11"/>
      <geom type="sphere" size="0.008" material="cloth_mat" mass="0.01"/>
    </body>
    
    <!-- Gripper (simplified as a free body) -->
    <body name="gripper" pos="0 0.2 0.15">
      <freejoint name="gripper_j"/>
      <geom type="cylinder" size="0.015 0.03" rgba="0.8 0.2 0.2 1" mass="0.5"/>
      <site name="grip_point" pos="0 0 -0.03" size="0.005"/>
    </body>
    
    <!-- Camera (top-down) -->
    <camera name="top_down" pos="0 0 0.5" xyaxes="1 0 0 0 1 0" fovy="45"/>
  </worldbody>
  
  <!-- Spring constraints between adjacent cloth nodes -->
  <equality>
    <connect body0="cloth_node_0" body1="cloth_node_1" anchor="0 0 0"/>
    <connect body0="cloth_node_1" body1="cloth_node_2" anchor="0 0 0"/>
    <connect body0="cloth_node_2" body1="cloth_node_3" anchor="0 0 0"/>
    <connect body0="cloth_node_4" body1="cloth_node_5" anchor="0 0 0"/>
    <connect body0="cloth_node_5" body1="cloth_node_6" anchor="0 0 0"/>
    <connect body0="cloth_node_6" body1="cloth_node_7" anchor="0 0 0"/>
    <connect body0="cloth_node_8" body1="cloth_node_9" anchor="0 0 0"/>
    <connect body0="cloth_node_9" body1="cloth_node_10" anchor="0 0 0"/>
    <connect body0="cloth_node_10" body1="cloth_node_11" anchor="0 0 0"/>
    <!-- Row connections -->
    <connect body0="cloth_node_0" body1="cloth_node_4" anchor="0 0 0"/>
    <connect body0="cloth_node_1" body1="cloth_node_5" anchor="0 0 0"/>
    <connect body0="cloth_node_2" body1="cloth_node_6" anchor="0 0 0"/>
    <connect body0="cloth_node_3" body1="cloth_node_7" anchor="0 0 0"/>
    <connect body0="cloth_node_4" body1="cloth_node_8" anchor="0 0 0"/>
    <connect body0="cloth_node_5" body1="cloth_node_9" anchor="0 0 0"/>
    <connect body0="cloth_node_6" body1="cloth_node_10" anchor="0 0 0"/>
    <connect body0="cloth_node_7" body1="cloth_node_11" anchor="0 0 0"/>
  </equality>
  
  <!-- Actuator: gripper position control -->
  <actuator>
    <position name="gripper_x" joint="gripper_j" axis="1 0 0" kp="100"/>
    <position name="gripper_y" joint="gripper_j" axis="0 1 0" kp="100"/>
    <position name="gripper_z" joint="gripper_j" axis="0 0 1" kp="100"/>
  </actuator>
</mujoco>
"""


class TextileFoldEnv:
    """Cloth folding environment using MuJoCo.
    
    Observations include gripper state, cloth node positions, and optionally
    a top-down RGB image for the VLM planner.
    """
    
    N_NODES = 12  # 3x4 grid of cloth nodes
    N_GRIPPER_DOF = 3  # x, y, z
    
    def __init__(self, seed=0, img_size=64, material_stiffness=1.0):
        self.seed = seed
        self.img_size = img_size
        self.material_stiffness = material_stiffness
        self.rng = np.random.RandomState(seed)
        
        # Use pure-Python fallback for simulation study.
        # MuJoCo XML is available for hardware deployment (Phase III).
        self._use_mujoco = False
        self._init_simple()
        self.x = self.nodes  # cable-compat alias
        self.gripper_path = [self.gripper.copy()]
        # CableEnv-compatible cfg
        class _Cfg:
            n_nodes = 12
            cable_len = 0.3
            hold_steps = 6
            grab_radius = 0.05
            force_max = 30.0
            v_gripper = 0.22
            bounds = (-0.25, 0.25, -0.2, 0.25)
            k_spring = 60.0
        self.cfg = _Cfg()
        self.steps = 0
        self.max_steps = 200
        self.holding = None
        self.violations = 0
    
    def _init_mujoco(self):
        """Initialize MuJoCo simulation from MJCF."""
        import tempfile, os
        xml_path = os.path.join(tempfile.gettempdir(), 'textile_fold.xml')
        with open(xml_path, 'w') as f:
            f.write(TEXTILE_MJCF)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
    
    def _init_simple(self):
        """Fallback: simple 2D cloth grid without MuJoCo."""
        # 3x4 grid of cloth nodes
        self.nodes = np.array([
            [-0.15, 0.1], [-0.05, 0.1], [0.05, 0.1], [0.15, 0.1],
            [-0.15, 0.0], [-0.05, 0.0], [0.05, 0.0], [0.15, 0.0],
            [-0.15, -0.1], [-0.05, -0.1], [0.05, -0.1], [0.15, -0.1],
        ])
        self.nodes_rest = self.nodes.copy()
        # Spring connections (adjacent nodes)
        self.edges = []
        for i in range(3):
            for j in range(4):
                idx = i * 4 + j
                if j < 3:
                    self.edges.append((idx, idx + 1))
                if i < 2:
                    self.edges.append((idx, idx + 4))
        self.rest_lengths = [np.linalg.norm(self.nodes[a] - self.nodes[b])
                            for a, b in self.edges]
        self.gripper = np.array([0.0, 0.2])
        self.holding = None
        # Target fold: fold the right half over the left
        self.target_fold = self._make_target_fold()
    
    def _make_target_fold(self):
        """Generate a target fold configuration."""
        # Target: right column folded over left
        target = self.nodes_rest.copy()
        # Fold column 3 (rightmost) over column 2
        for i in range(3):
            target[i * 4 + 3] = self.nodes_rest[i * 4 + 2] + np.array([0.02, 0.02])
        return target
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
        
        if self._use_mujoco:
            mujoco.mj_reset(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        else:
            # Reset to flat configuration with small perturbations
            self.nodes = self.nodes_rest + self.rng.randn(12, 2) * 0.005
            self.gripper = np.array([0.0, 0.2])
            self.holding = None
        self.x = self.nodes  # cable-compat alias
        
        self.steps = 0
        self.violations = 0
        self.gripper_path = [self.gripper.copy()]
        return self._obs()
    
    def _obs(self):
        """Return observation vector."""
        if self._use_mujoco:
            # Extract from MuJoCo state
            cloth_pos = []
            for i in range(self.N_NODES):
                name = f'cloth_node_{i}'
                idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                pos = self.data.xpos[idx]
                cloth_pos.append(pos[:2])
            cloth_pos = np.array(cloth_pos).flatten()
            gripper_pos = self.data.xpos[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
            ][:2]
        else:
            cloth_pos = self.nodes.flatten()
            gripper_pos = self.gripper
        
        # Fold progress (IoU-like metric)
        fold_iou = self._compute_fold_iou()
        
        return np.concatenate([
            cloth_pos,           # cloth node positions
            gripper_pos,         # gripper position
            [fold_iou],          # fold progress
            [self.steps / self.max_steps],  # time
        ]).astype(np.float32)
    
    def _compute_fold_iou(self):
        """Compute fold IoU between current and target configuration."""
        if self._use_mujoco:
            return 0.0  # TODO: extract from MuJoCo
        
        # Simple IoU: overlap between current and target node positions
        current = set()
        target = set()
        for i in range(12):
            # Discretize to grid
            cx = round(self.nodes[i, 0] * 50) + 25
            cy = round(self.nodes[i, 1] * 50) + 25
            tx = round(self.target_fold[i, 0] * 50) + 25
            ty = round(self.target_fold[i, 1] * 50) + 25
            current.add((cx, cy))
            target.add((tx, ty))
        
        intersection = len(current & target)
        union = len(current | target)
        return intersection / max(union, 1)
    
    def step(self, action):
        """Execute one action step.
        
        action: (dx, dy, grab) — gripper delta + grab flag
        """
        dx, dy = float(action[0]), float(action[1])
        grab = float(action[2]) > 0.5
        
        # Move gripper
        target = self.gripper + np.array([dx, dy])
        # Clip to workspace
        bounds = (-0.25, 0.25, -0.2, 0.25)
        if target[0] < bounds[0] or target[0] > bounds[1]:
            self.violations += 1
            target[0] = np.clip(target[0], bounds[0], bounds[1])
        if target[1] < bounds[2] or target[1] > bounds[3]:
            self.violations += 1
            target[1] = np.clip(target[1], bounds[2], bounds[3])
        
        if self._use_mujoco:
            # MuJoCo: set gripper position
            gripper_idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
            self.data.xpos[gripper_idx][:2] = target
        else:
            self.gripper = target
            self.gripper_path.append(self.gripper.copy())
            
            # Grab / release
            if grab and self.holding is None:
                # Find nearest free node
                dists = np.linalg.norm(self.nodes - self.gripper, axis=1)
                nearest = np.argmin(dists)
                if dists[nearest] < 0.05:
                    self.holding = nearest
            elif not grab:
                self.holding = None
            
            # If holding, move the node with the gripper
            if self.holding is not None:
                self.nodes[self.holding] = self.gripper.copy()
            
            # Spring dynamics (simplified PBD)
            self._step_springs()
        
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._obs(), done
    
    def _step_springs(self):
        """Simplified spring dynamics for the cloth."""
        for _ in range(5):  # PBD iterations
            for idx, (a, b) in enumerate(self.edges):
                if a == self.holding or b == self.holding:
                    continue  # Skip edges connected to held node
                diff = self.nodes[b] - self.nodes[a]
                dist = np.linalg.norm(diff) + 1e-9
                err = dist - self.rest_lengths[idx]
                corr = diff / dist * err * 0.3
                self.nodes[a] += corr * 0.5
                self.nodes[b] -= corr * 0.5
            
            # Gravity
            self.nodes[:, 1] -= 0.0005 * self.material_stiffness
            
            # Table constraint (nodes can't go below table)
            self.nodes[:, 1] = np.maximum(self.nodes[:, 1], -0.1)
    
    def render(self, img_size=None):
        """Render top-down RGB image."""
        img_size = img_size or self.img_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        if self._use_mujoco:
            # TODO: MuJoCo rendering
            return img
        
        # Simple 2D rendering
        def to_pixel(pos):
            x = int((pos[0] + 0.25) / 0.5 * (img_size - 1))
            y = int((0.25 - pos[1]) / 0.5 * (img_size - 1))
            return max(0, min(img_size-1, x)), max(0, min(img_size-1, y))
        
        # Draw edges
        for a, b in self.edges:
            px1, py1 = to_pixel(self.nodes[a])
            px2, py2 = to_pixel(self.nodes[b])
            # Simple line drawing
            steps = max(abs(px2-px1), abs(py2-py1), 1)
            for t in range(steps + 1):
                frac = t / steps
                px = int(px1 + (px2 - px1) * frac)
                py = int(py1 + (py2 - py1) * frac)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if 0 <= px+dx < img_size and 0 <= py+dy < img_size:
                            img[py+dy, px+dx] = [50, 120, 200]
        
        # Draw nodes
        for i in range(12):
            px, py = to_pixel(self.nodes[i])
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= px+dx < img_size and 0 <= py+dy < img_size:
                        img[py+dy, px+dx] = [30, 90, 170]
        
        # Draw gripper
        gx, gy = to_pixel(self.gripper)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= gx+dx < img_size and 0 <= gy+dy < img_size:
                    img[gy+dy, gx+dx] = [200, 50, 50]
        
        # Draw target fold (faint)
        for i in range(12):
            tx, ty = to_pixel(self.target_fold[i])
            if 0 <= tx < img_size and 0 <= ty < img_size:
                img[ty, tx] = [180, 220, 180]
        
        return img
    
    def img_obs(self, img_size=None):
        """Return image observation as (C, H, W) float32 in [0, 1]."""
        img = self.render(img_size)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    def fold_iou(self):
        """Current fold IoU with target."""
        return self._compute_fold_iou()

    # ------------------------------------------------------------------
    # CableEnv-compatible interface for data collection + eval
    # ------------------------------------------------------------------

    def crossings(self):
        """Number of unresolved crossings (0 when task is done)."""
        iou = self._compute_fold_iou()
        return max(0, int((1.0 - iou) * 10))  # 0 when IoU >= 1.0

    @property
    def crossings0(self):
        """Initial crossing count."""
        return 10  # full complexity

    @property
    def zero_streak(self):
        return getattr(self, '_zero_streak', 0)

    def _get_x(self):
        return self.nodes

    def _set_x(self, val):
        self.nodes = val

    def max_jerk(self, dt=0.1):
        p = np.asarray(self.gripper_path)
        if len(p) < 4:
            return 0.0
        v = np.diff(p, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return float(np.max(np.linalg.norm(j, axis=1))) if len(j) else 0.0
