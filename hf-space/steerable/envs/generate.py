"""Constructive configuration generator for the cable environment.

Random polylines between two pins essentially never self-intersect at the
node counts we can afford, so crossings are built explicitly. One "loop unit"
(7 points, 6 segments) contains exactly ONE proper crossing — an out-and-back
"X" between segment 0 (p0->p1) and segment 3 (p3->p4):

    p0=(0,0)  p1=(0.7,0.6)  p2=(0.7,-0.1)  p3=(0.1,-0.1)
    p4=(0.1,0.7)  p5=(0.9,0.7)  p6=(0.9,0)

s0 crosses s3 at (0.1, 6/70) and no other non-adjacent pair intersects.
Units are chained with 0.2 gaps (connectors at y=0, sharing only vertices,
which are excluded by the interior-intersection definition of a crossing);
remaining nodes are a straight extension to the right pin, so the node count
is IDENTICAL across crossing targets (fixed observation dimension across
train and held-out families). Family variance = per-unit amplitude + small
per-node jitter, both preserving the crossing count.
"""

import numpy as np

from .geometry import count_crossings

N_NODES = 33          # fixed across all families
UNIT = np.array([     # one unit, exactly one crossing (s0 x s3)
    [0.0, 0.0], [0.7, 0.6], [0.7, -0.1], [0.1, -0.1],
    [0.1, 0.7], [0.9, 0.7], [0.9, 0.0]], dtype=float)
UNIT_SPAN = 0.9
GAP = 0.6
STEP = UNIT_SPAN + GAP
PAD_EXT = 1.2          # straight extension beyond the last unit (any k)


def build_chain(k, amps=None, rng=None):
    """k chained units (k crossings) + straight padding to N_NODES."""
    pts = []
    for j in range(k):
        unit = UNIT.copy()
        if amps is not None:
            unit[:, 1] *= amps[j]
        pts.extend((unit + np.array([STEP * j, 0.0])).tolist())
    pts = np.array(pts, dtype=float)
    n_pad = N_NODES - len(pts)
    if n_pad > 0:
        last = pts[-1]
        right = np.array([(k - 1) * STEP + UNIT_SPAN + PAD_EXT, 0.0])
        pad = np.linspace(last, right, n_pad + 2)[1:-1]
        pts = np.concatenate([pts, pad], axis=0)
    return pts[:N_NODES]


def generate_config(rng, n_nodes, cable_len, target_cross, max_tries=40):
    """Return an n_nodes polyline between pins with exactly target_cross crossings."""
    assert n_nodes == N_NODES, f"node count must be fixed at {N_NODES}"
    for _ in range(max_tries):
        amps = rng.uniform(0.7, 1.3, size=target_cross)
        x = build_chain(target_cross, amps=amps, rng=rng)
        x = x + rng.normal(0, 0.015, size=x.shape)
        hits, _ = count_crossings(x)
        if hits == target_cross:
            break
    # normalize: center on origin, span cable_len end to end (y scaled taller)
    lo, hi = x[:, 0].min(), x[:, 0].max()
    span = hi - lo + 1e-9
    x[:, 0] = (x[:, 0] - lo) / span * cable_len - cable_len / 2
    x[:, 1] = x[:, 1] * (cable_len / span) * 1.8
    x[0] = np.array([-cable_len / 2, 0.0])
    x[-1] = np.array([cable_len / 2, 0.0])
    return x
