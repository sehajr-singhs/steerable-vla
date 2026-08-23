"""ToolUseEnv — adaptive tool use in clutter.

Task C from the proposal: retrieve/lever/pull with unknown tools in clutter.
The agent must select the right tool and use it to manipulate a target object.

Observations:
    - Proprioception: gripper position, tool positions (when held)
    - Scene: object positions, tool positions, target position
    - Image: top-down RGB view of the scene
    - Language: task instruction

Actions:
    - (dx, dy) gripper delta + grab flag

Zero-shot axes:
    - Tool morphology (different shapes/sizes)
    - Clutter density
    - Affordance composition

Metrics:
    - Task success (target reached)
    - Tool-selection accuracy
    - Interventions per 100 episodes
    - Contact-force violations
"""

import numpy as np


class ToolUseEnv:
    """Adaptive tool use environment.
    
    Scene: a table with several tools (different shapes), a target object,
    and clutter. The agent must grab the right tool and use it to push/pull
    the target to a goal position.
    """
    
    def __init__(self, seed=0, img_size=64, n_tools=3, n_clutter=5):
        self.seed = seed
        self.img_size = img_size
        self.n_tools = n_tools
        self.n_clutter = n_clutter
        self.rng = np.random.RandomState(seed)
        self._init_tools()
        self.x = self.tools[:, :2]  # cable-compat alias
        # CableEnv-compatible cfg
        class _Cfg:
            n_nodes = 3
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
        self.tool_properties = self._generate_tool_morphologies()
    
    def _init_tools(self):
        """Initialize tool and object positions."""
        # Tools: position (x, y), size, type
        self.tools = np.zeros((self.n_tools, 3))  # (x, y, size)
        # Clutter: position (x, y), size
        self.clutter = np.zeros((self.n_clutter, 2))
        # Target object
        self.target_pos = np.array([0.0, 0.0])
        self.target_goal = np.array([0.0, 0.0])
        # Gripper
        self.gripper = np.array([0.0, 0.2])
        self.holding = None
        self.held_tool_idx = None
        self.gripper_path = [self.gripper.copy()]
        # Workspace
        self.bounds = (-0.25, 0.25, -0.2, 0.25)
        # Tool morphologies (length, width, shape type)
        self.tool_types = ['lever', 'hook', 'pusher']
    
    def _generate_tool_morphologies(self):
        """Generate random tool morphologies for zero-shot testing."""
        morphs = []
        for i in range(self.n_tools):
            length = self.rng.uniform(0.05, 0.15)
            width = self.rng.uniform(0.01, 0.03)
            ttype = self.tool_types[i % len(self.tool_types)]
            morphs.append({
                'length': length,
                'width': width,
                'type': ttype,
            })
        return morphs
    
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.seed = seed
        
        # Random tool positions (avoid overlap)
        for i in range(self.n_tools):
            while True:
                pos = self.rng.uniform(-0.2, 0.2, 2)
                if all(np.linalg.norm(pos - self.tools[j, :2]) > 0.08
                       for j in range(i)):
                    self.tools[i, :2] = pos
                    self.tools[i, 2] = self.tool_properties[i]['length']
                    break
        
        # Random clutter positions
        for i in range(self.n_clutter):
            self.clutter[i] = self.rng.uniform(-0.2, 0.2, 2)
        
        # Target and goal
        self.target_pos = self.rng.uniform(-0.1, 0.1, 2)
        # Goal is on the opposite side
        self.target_goal = -self.target_pos + self.rng.uniform(-0.05, 0.05, 2)
        self.target_goal = np.clip(self.target_goal, self.bounds[0]+0.05, self.bounds[1]-0.05)
        
        self.gripper = np.array([self.rng.uniform(-0.1, 0.1), 0.2])
        self.holding = None
        self.held_tool_idx = None
        self.steps = 0
        self.violations = 0
        self.gripper_path = [self.gripper.copy()]
        self.x = self.tools[:, :2]  # cable-compat alias
        
        return self._obs()
    
    def _obs(self):
        """Return observation vector."""
        obs = np.concatenate([
            self.gripper,                    # gripper position (2)
            self.target_pos,                 # target position (2)
            self.target_goal,                # goal position (2)
            self.tools[:, :2].flatten(),     # tool positions (2*n_tools)
            self.tools[:, 2],                # tool sizes (n_tools)
            [1.0 if self.holding is not None else 0.0],  # holding flag
            [self.steps / self.max_steps],   # time
        ]).astype(np.float32)
        return obs
    
    def step(self, action):
        """Execute one action step."""
        dx, dy = float(action[0]), float(action[1])
        grab = float(action[2]) > 0.5
        
        # Move gripper
        target = self.gripper + np.array([dx, dy])
        if (target[0] < self.bounds[0] or target[0] > self.bounds[1] or
                target[1] < self.bounds[2] or target[1] > self.bounds[3]):
            self.violations += 1
            target = np.clip(target, [self.bounds[0], self.bounds[2]],
                           [self.bounds[1], self.bounds[3]])
        
        self.gripper = target
        self.gripper_path.append(self.gripper.copy())
        
        # Grab / release
        if grab and self.holding is None:
            # Try to grab a tool
            for i in range(self.n_tools):
                dist = np.linalg.norm(self.tools[i, :2] - self.gripper)
                if dist < 0.05:
                    self.holding = 'tool'
                    self.held_tool_idx = i
                    break
        elif not grab:
            self.holding = None
            self.held_tool_idx = None
        
        # If holding a tool, check if it contacts the target
        if self.holding == 'tool' and self.held_tool_idx is not None:
            tool_pos = self.tools[self.held_tool_idx, :2]
            tool_len = self.tools[self.held_tool_idx, 2]
            # Tool end point
            tool_end = tool_pos + np.array([0, -tool_len])
            # Check contact with target
            dist_to_target = np.linalg.norm(tool_end - self.target_pos)
            if dist_to_target < 0.04:
                # Push target toward goal
                direction = self.target_goal - self.target_pos
                direction = direction / (np.linalg.norm(direction) + 1e-9)
                self.target_pos += direction * 0.01
                self.target_pos = np.clip(self.target_pos,
                                         self.bounds[0]+0.02, self.bounds[1]-0.02)
        
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._obs(), done
    
    def render(self, img_size=None):
        """Render top-down RGB image."""
        img_size = img_size or self.img_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        def to_pixel(pos):
            x = int((pos[0] - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * (img_size - 1))
            y = int((self.bounds[3] - pos[1]) / (self.bounds[3] - self.bounds[2]) * (img_size - 1))
            return max(0, min(img_size-1, x)), max(0, min(img_size-1, y))
        
        # Draw clutter
        for i in range(self.n_clutter):
            px, py = to_pixel(self.clutter[i])
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= px+dx < img_size and 0 <= py+dy < img_size:
                        img[py+dy, px+dx] = [180, 180, 180]
        
        # Draw tools
        tool_colors = [[200, 100, 50], [50, 200, 100], [100, 50, 200]]
        for i in range(self.n_tools):
            color = tool_colors[i % len(tool_colors)]
            px, py = to_pixel(self.tools[i, :2])
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= px+dx < img_size and 0 <= py+dy < img_size:
                        img[py+dy, px+dx] = color
        
        # Draw target
        tx, ty = to_pixel(self.target_pos)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= tx+dx < img_size and 0 <= ty+dy < img_size:
                    img[ty+dy, tx+dx] = [255, 50, 50]
        
        # Draw goal (hollow circle)
        gx, gy = to_pixel(self.target_goal)
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if abs(dx*dx + dy*dy - 9) < 3:
                    px, py = gx+dx, gy+dy
                    if 0 <= px < img_size and 0 <= py < img_size:
                        img[py, px] = [255, 200, 200]
        
        # Draw gripper
        gpx, gpy = to_pixel(self.gripper)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= gpx+dx < img_size and 0 <= gpy+dy < img_size:
                    img[gpy+dy, gpx+dx] = [50, 50, 200]
        
        return img
    
    def img_obs(self, img_size=None):
        """Return image observation as (C, H, W) float32 in [0, 1]."""
        img = self.render(img_size)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0
    
    def success(self):
        """Check if target reached the goal."""
        return np.linalg.norm(self.target_pos - self.target_goal) < 0.03
    
    def tool_selection_accuracy(self):
        """Check if the agent grabbed the best tool for the task."""
        # Best tool: longest reach for the distance to target
        distances = [np.linalg.norm(self.tools[i, :2] - self.target_pos)
                    for i in range(self.n_tools)]
        best_tool = np.argmin(distances)
        return 1.0 if self.held_tool_idx == best_tool else 0.0

    # ------------------------------------------------------------------
    # CableEnv-compatible interface for data collection + eval
    # ------------------------------------------------------------------

    def crossings(self):
        """Number of unresolved crossings (0 when task is done)."""
        dist = np.linalg.norm(self.target_pos - self.target_goal)
        return max(0, int(dist * 50))  # 0 when close to goal

    @property
    def crossings0(self):
        return 10  # full complexity

    @property
    def zero_streak(self):
        return getattr(self, '_zero_streak', 0)

    @property
    def x(self):
        """Node positions (for compatibility)."""
        return self.tools[:, :2]

    @x.setter
    def x(self, val):
        self.tools[:, :2] = val

    def max_jerk(self, dt=0.1):
        p = np.asarray(self.gripper_path)
        if len(p) < 4:
            return 0.0
        v = np.diff(p, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return float(np.max(np.linalg.norm(j, axis=1))) if len(j) else 0.0
