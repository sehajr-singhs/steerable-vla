"""Data pipeline: expert demonstrations -> subgoal segmentation -> chunks.

Includes both the original miniature-scale pipeline and the new real-dataset
loaders (BridgeData V2, DROID, Open X-Embodiment, procedural cable).
"""

# re-export everything from the original pipeline
from ..data_original import (
    collect_demos,
    build_dataset,
    subgoal_from_state,
    chunk_episode,
    make_episode_start,
)

# new real-dataset loaders
from .real_datasets import (
    load_bridgedata_v2,
    load_droid,
    load_open_x_embodiment,
    generate_cable_demos,
    normalize_actions,
    denormalize_actions,
)

__all__ = [
    # original pipeline
    "collect_demos",
    "build_dataset",
    "subgoal_from_state",
    "chunk_episode",
    "make_episode_start",
    # real dataset loaders
    "load_bridgedata_v2",
    "load_droid",
    "load_open_x_embodiment",
    "generate_cable_demos",
    "normalize_actions",
    "denormalize_actions",
]
