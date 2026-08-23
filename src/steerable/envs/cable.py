"""CableEnv — a 2D mass-spring cable with pinned ends and a point gripper.

The "cable untangling" task at miniature scale. Nodes form a chain pinned at
both ends; crossings are counted as proper intersections of non-adjacent
segments. A point gripper can grab a node (within radius) and drag it; the
holding force is estimated from the spring response, and exceeding force_max
makes the gripper slip (a safety-violation event).

The environment is deliberately procedural: an initial configuration family is
a (stiffness_multiplier, crossing_target) pair, and reset() rejection-samples
a random polyline between the pins that realizes the crossing target. Held-out
families are the zero-shot axes of the study.
"""

import numpy as np

from ..config import EnvConfig
from .geometry import count_crossings
from .generate import generate_config


class CableEnv:
    """Pinned mass-spring cable with a point gripper.

    Actions per step: (dx, dy) gripper target delta (clamped to v_gripper),
    grab flag in {0, 1}. Returns obs, done.
    """

    def __init__(self, cfg: EnvConfig = None, seed=0, stiffness_mult=1.0,
                 crossing_target=2):
        self.cfg = cfg or EnvConfig()
        self.rng = np.random.RandomState(seed)
        self.stiffness_mult = stiffness_mult
        self.crossing_target = crossing_target
        self.reset()

    # ------------------------------------------------------------------
    # reset / state
    # ------------------------------------------------------------------

    def reset(self, seed=None, stiffness_mult=None, crossing_target=None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if stiffness_mult is not None:
            self.stiffness_mult = stiffness_mult
        if crossing_target is not None:
            self.crossing_target = crossing_target
        c = self.cfg
        self.x = generate_config(self.rng, c.n_nodes, c.cable_len, self.crossing_target)
        self.v = np.zeros_like(self.x)
        # fixed segment lengths (the cable at rest): PBD constraint targets
        d = np.diff(self.x, axis=0)
        self.rest = np.linalg.norm(d, axis=1) + 1e-9
        self.gripper = np.array([0.0, 0.55])          # start above the cable
        self.holding = None                            # grabbed node index or None
        self.violations = 0
        self.steps = 0
        self.crossings0, _ = count_crossings(self.x)
        self.zero_streak = 0
        self.force_hist = []
        self.gripper_path = [self.gripper.copy()]
        return self._obs()

    def _obs(self):
        c = self.cfg
        x = self.x - np.array([0.0, 0.0])
        # flatten node positions relative to the cable midpoint
        mid = np.array([0.0, 0.0])
        nodes = (self.x - mid).flatten()
        # gripper relative to cable centroid
        g = self.gripper - self.x.mean(axis=0)
        crossings, _ = count_crossings(self.x)
        sub = np.array([crossings / max(1, self.crossings0),
                        self.steps / 200.0])
        return np.concatenate([nodes, g, sub]).astype(np.float32)

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def _pbd(self, iters=None):
        """Localized position-based dynamics around the gripper-held node.

        A real cable holds its shape at rest (no internal forces -> tangles
        persist) but flexes at its hinges when the gripper pulls. Constraints
        are enforced only within a radius of the held node, so a pull bends
        the local strand WITHOUT whipping distant units into new crossings.

        We run two passes: first a forward pass that pulls toward rest
        length, then a relaxation pass that blends toward the previous
        configuration. This prevents the chaotic post-pull rebound that
        causes re-tangling.
        """
        c = self.cfg
        iters = iters or c.pbd_iters
        x = self.x
        n = len(x)
        if self.holding is None:
            return
        lo = max(0, self.holding - c.pbd_radius)
        hi = min(n - 2, self.holding + c.pbd_radius)
        # Save pre-PBD positions for relaxation
        x_prev = x.copy()
        for _ in range(iters):
            for e in range(lo, hi):
                d = x[e + 1] - x[e]
                dl = np.linalg.norm(d) + 1e-9
                err = dl - self.rest[e]
                corr = d / dl * (err * 0.5)
                if e > 0:
                    x[e] += corr
                if e + 1 < n - 1:
                    x[e + 1] -= corr
            x[self.holding] = self.gripper
        # Damping: blend toward previous positions to reduce oscillation
        damping = 0.15
        for idx in range(lo, hi + 1):
            if idx != self.holding and idx < n:
                x[idx] = x[idx] * (1 - damping) + x_prev[idx] * damping

    def step(self, action):
        c = self.cfg
        dx, dy, grab = float(action[0]), float(action[1]), float(action[2] > 0.5)
        # velocity clamp (actuator limit)
        mag = np.hypot(dx, dy)
        if mag > c.v_gripper:
            dx, dy = dx / mag * c.v_gripper, dy / mag * c.v_gripper
        # gripper dynamics: move to the commanded target; leaving the safe box
        # is a safety violation (the CBF filter exists precisely to prevent it)
        target = self.gripper + np.array([dx, dy])
        if (target[0] < c.bounds[0] or target[0] > c.bounds[1]
                or target[1] < c.bounds[2] or target[1] > c.bounds[3]):
            self.violations += 1
            target = np.clip(target, [c.bounds[0], c.bounds[2]],
                             [c.bounds[1], c.bounds[3]])
        self.gripper = target
        self.gripper_path.append(self.gripper.copy())

        if grab:
            if self.holding is None:
                # try to grab the free node nearest the gripper
                d = np.linalg.norm(self.x[1:-1] - self.gripper, axis=1)
                i = int(np.argmin(d)) + 1
                if d[i - 1] < c.grab_radius:
                    self.holding = i
            if self.holding is not None:
                held_prev = self.x[self.holding].copy()
                self.x[self.holding] = self.gripper
                disp = np.linalg.norm(self.gripper - held_prev)
                force = c.k_spring * self.stiffness_mult * disp
                self.force_hist.append(force)
                if force > c.force_max:
                    # slip: release and flag a safety violation
                    self.holding = None
                    self.violations += 1
        else:
            if self.holding is not None:
                self.holding = None

        self._pbd()
        self.steps += 1

        crossings, _ = count_crossings(self.x)
        if crossings == 0:
            self.zero_streak += 1
        else:
            self.zero_streak = 0
        done = self.zero_streak >= c.hold_steps
        return self._obs(), done

    # ------------------------------------------------------------------
    # metrics / inspection
    # ------------------------------------------------------------------

    def crossings(self):
        return count_crossings(self.x)[0]

    def max_jerk(self, dt=0.1):
        p = np.asarray(self.gripper_path)
        if len(p) < 4:
            return 0.0
        v = np.diff(p, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        return float(np.max(np.linalg.norm(j, axis=1))) if len(j) else 0.0

    def snapshot(self):
        return self.x.copy()

    # ------------------------------------------------------------------
    # image observations (for VLM planner)
    # ------------------------------------------------------------------

    def render(self, img_size=64):
        """Render the cable state as an RGB image.
        
        Returns a (img_size, img_size, 3) uint8 array suitable for
        the VLM planner's image encoder.
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from io import BytesIO
            import PIL.Image
        except ImportError:
            # Fallback: encode the state as a simple pixel pattern
            return self._render_fallback(img_size)
        
        fig, ax = plt.subplots(1, 1, figsize=(1, 1), dpi=img_size)
        ax.set_xlim(self.cfg.bounds[0], self.cfg.bounds[1])
        ax.set_ylim(self.cfg.bounds[2], self.cfg.bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw cable segments
        for i in range(len(self.x) - 1):
            ax.plot(self.x[i:i+2, 0], self.x[i:i+2, 1],
                    'b-', linewidth=2, solid_capstyle='round')
        
        # Draw gripper
        color = 'red' if self.holding is not None else 'gray'
        ax.plot(self.gripper[0], self.gripper[1], 'o',
                color=color, markersize=4)
        
        # Draw pin points
        ax.plot(self.x[0, 0], self.x[0, 1], 's', color='black', markersize=3)
        ax.plot(self.x[-1, 0], self.x[-1, 1], 's', color='black', markersize=3)
        
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = PIL.Image.open(buf).convert('RGB').resize((img_size, img_size))
        import numpy as np
        return np.array(img)
    
    def _render_fallback(self, img_size=64):
        """Fallback renderer: encode state as pixel pattern without matplotlib."""
        import numpy as np
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # Map cable nodes to pixel coordinates
        for i in range(len(self.x) - 1):
            x1 = int((self.x[i, 0] - self.cfg.bounds[0]) / (self.cfg.bounds[1] - self.cfg.bounds[0]) * (img_size - 1))
            y1 = int((self.x[i, 1] - self.cfg.bounds[2]) / (self.cfg.bounds[3] - self.cfg.bounds[2]) * (img_size - 1))
            x2 = int((self.x[i+1, 0] - self.cfg.bounds[0]) / (self.cfg.bounds[1] - self.cfg.bounds[0]) * (img_size - 1))
            y2 = int((self.x[i+1, 1] - self.cfg.bounds[2]) / (self.cfg.bounds[3] - self.cfg.bounds[2]) * (img_size - 1))
            y1, y2 = img_size - 1 - y1, img_size - 1 - y2  # flip y
            # Bresenham line
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx - dy
            while True:
                if 0 <= x1 < img_size and 0 <= y1 < img_size:
                    img[y1, x1] = [50, 100, 200]
                if x1 == x2 and y1 == y2:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        # Gripper
        gx = int((self.gripper[0] - self.cfg.bounds[0]) / (self.cfg.bounds[1] - self.cfg.bounds[0]) * (img_size - 1))
        gy = int((self.gripper[1] - self.cfg.bounds[2]) / (self.cfg.bounds[3] - self.cfg.bounds[2]) * (img_size - 1))
        gy = img_size - 1 - gy
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                px, py = gx + dx, gy + dy
                if 0 <= px < img_size and 0 <= py < img_size:
                    img[py, px] = [200, 50, 50]
        # Pins
        for px, py in [(0, 0), (img_size-1, 0)]:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= px+dx < img_size and 0 <= py+dy < img_size:
                        img[py+dy, px+dx] = [0, 0, 0]
        return img
    
    def img_obs(self, img_size=64):
        """Return image observation as (C, H, W) float32 tensor in [0, 1]."""
        import numpy as np
        img = self.render(img_size)
        # (H, W, 3) -> (3, H, W) and normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img
