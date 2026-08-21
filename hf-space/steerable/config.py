"""Central hyperparameters for the Steerable VLA miniature study."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    n_nodes: int = 33              # fixed across families (see envs/generate.py)
    cable_len: float = 1.6         # distance between pins
    k_spring: float = 60.0
    k_bend: float = 6.0
    damp: float = 0.12
    friction: float = 0.25
    gravity: float = 0.12
    dt: float = 0.02
    pbd_iters: int = 8             # position-based dynamics iterations per step
    pbd_radius: int = 1            # constraint-localization radius around the held node
    grab_radius: float = 0.16
    force_max: float = 30.0        # gripper holding force cap (slip -> violation)
    v_gripper: float = 0.22        # gripper velocity cap per action step
    table_y: float = -0.30         # floor plane (nodes cannot pass below)
    bounds: tuple = (-0.95, 0.95, -0.30, 0.95)  # gripper x_lo x_hi y_lo y_hi
    hold_steps: int = 6            # steps to hold a resolved crossing


@dataclass
class DataConfig:
    chunk: int = 4                 # action chunk length H
    stride: int = 2                # chunking stride
    subgoal_noise: float = 0.03    # synthetic subgoal coverage augmentation
    steer_prob: float = 0.5        # prob of synthetic steering supervision
    steer_mag: float = 0.10        # nudge magnitude on first chunk step


@dataclass
class TrainConfig:
    epochs: int = 120
    batch: int = 128
    lr: float = 1e-3
    hidden: int = 256
    latent: int = 128
    flow_steps: int = 24           # Euler steps at inference
    n_samples: int = 8             # flow samples averaged at inference
    seed: int = 0


@dataclass
class EvalConfig:
    n_eval: int = 60
    max_steps: int = 120
    patience: int = 24             # oracle intervention patience
    hold_ok: int = 12              # steps crossings must stay 0 for success
    replan: int = 1                # chunk steps executed per re-query (receding horizon)


@dataclass
class FlywheelConfig:
    iterations: int = 4
    n_deploy: int = 40             # rollouts per iteration
    near_miss_frac: float = 0.5    # min crossings-reduced fraction to curate
    retrain_epochs: int = 30
