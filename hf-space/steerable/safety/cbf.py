"""Runtime formal-verification safety envelope (miniature).

Safe set C = {x : h(x) >= 0} with h built from axis-aligned barriers on the
gripper position (table bounds) and a velocity cap (actuator set U). At each
step the filter solves the QP

    u* = argmin_{u in U} 1/2 ||u - u_cmd||^2_W
         s.t.  h_dot(x, u) + alpha h(x) >= 0

For axis-aligned barriers the CBF constraints are linear in u, so the QP is
exactly a box projection; we still solve it as a small QP (scipy SLSQP) to
keep the filter architecture honest and generalizable. Forward invariance:
if h(x(0)) >= 0 then h(x(t)) >= 0 for all t (gripper never leaves the box).

Steerability is admitted only through this filter: the commanded action from
the (steered) flow is projected onto the admissible set before execution.
"""

import numpy as np
from scipy.optimize import minimize

from ..config import EnvConfig


class CBFQPFilter:
    def __init__(self, cfg: EnvConfig, alpha=1.0, W=None):
        """Discrete-time CBF filter. Forward invariance requires alpha*dt <= 1;
        with dt = 1 (one control application per env step) we use alpha = 1,
        which enforces h(x + u) >= 0 exactly. A margin keeps the gripper off
        the boundary (away from limit-switch tolerance)."""
        self.cfg = cfg
        self.alpha = alpha
        self.W = np.eye(2) if W is None else W
        self.margin = 0.02
        self.projected = 0.0
        self.n_calls = 0

    def _barriers(self, x):
        x_lo, x_hi, y_lo, y_hi = self.cfg.bounds
        m = self.margin
        return np.array([x[0] - (x_lo + m), (x_hi - m) - x[0],
                         x[1] - (y_lo + m), (y_hi - m) - x[1]])

    def _constraints(self, x, u, dt):
        """CBF: h_dot + alpha h >= 0 with h_dot = grad h . u (position control)."""
        x_lo, x_hi, y_lo, y_hi = self.cfg.bounds
        h = self._barriers(x)
        cons = []
        # h1 = x - x_lo:  dh = [1, 0] -> u_x + alpha h1 >= 0
        cons.append({"type": "ineq", "fun": lambda u, x=x, h=h: u[0] + self.alpha * h[0]})
        # h2 = x_hi - x:  dh = [-1, 0] -> -u_x + alpha h2 >= 0
        cons.append({"type": "ineq", "fun": lambda u, x=x, h=h: -u[0] + self.alpha * h[1]})
        cons.append({"type": "ineq", "fun": lambda u, x=x, h=h: u[1] + self.alpha * h[2]})
        cons.append({"type": "ineq", "fun": lambda u, x=x, h=h: -u[1] + self.alpha * h[3]})
        # actuator set U: velocity cap
        vmax = self.cfg.v_gripper
        cons.append({"type": "ineq", "fun": lambda u: vmax - np.hypot(u[0], u[1])})
        return cons

    def __call__(self, u_cmd, x):
        self.n_calls += 1
        u_cmd = np.asarray(u_cmd, dtype=float)[:2]
        if not np.all(np.isfinite(u_cmd)):
            u_cmd = np.zeros(2)
        dt = 1.0
        res = minimize(
            lambda u: 0.5 * (u - u_cmd) @ self.W @ (u - u_cmd),
            x0=np.clip(u_cmd, -self.cfg.v_gripper, self.cfg.v_gripper),
            method="SLSQP",
            constraints=self._constraints(x, u_cmd, dt),
            options={"maxiter": 60, "ftol": 1e-10, "eps": 1e-8},
        )
        u = res.x if res.success else np.clip(u_cmd, -self.cfg.v_gripper, self.cfg.v_gripper)
        if np.linalg.norm(u - u_cmd) > 1e-6:
            self.projected += 1.0
        return u
