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
    pbd_iters: int = 12            # position-based dynamics iterations per step
    pbd_radius: int = 3            # constraint-localization radius around the held node
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
    epochs: int = 500
    batch: int = 64
    lr: float = 3e-4
    lr_min: float = 1e-5          # cosine schedule floor
    hidden: int = 512
    latent: int = 256
    flow_steps: int = 24           # Euler steps at inference
    n_samples: int = 8             # flow samples averaged at inference
    grad_clip: float = 1.0
    curriculum: bool = True        # ramp crossing_target from 1 to cfg value
    curriculum_warmup: int = 50    # epochs before crossing_target increases
    seed: int = 0


@dataclass
class EvalConfig:
    n_eval: int = 30
    max_steps: int = 200
    patience: int = 30             # oracle intervention patience
    hold_ok: int = 6               # steps crossings must stay 0 for success
    replan: int = 0                # 0=full chunk, K=K steps per re-query (receding horizon)


@dataclass
class FlywheelConfig:
    iterations: int = 4
    n_deploy: int = 40             # rollouts per iteration
    near_miss_frac: float = 0.5    # min crossings-reduced fraction to curate
    retrain_epochs: int = 30
