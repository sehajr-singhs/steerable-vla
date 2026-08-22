"""Baseline policies for comparison.

Implements:
  - BC: standard behavior cloning (MLP, no flow matching)
  - ACT: Action Chunking Transformer (Zhao et al. 2023)
  - DiffusionPolicy: diffusion-based action generation (Chi et al. 2023)
  - ImitatorPolicy: oracle-based imitation (simplified)
"""

from .bc_policy import BCPolicy
from .act_policy import ACTPolicy
from .diffusion_policy import DiffusionPolicy

__all__ = ["BCPolicy", "ACTPolicy", "DiffusionPolicy"]
