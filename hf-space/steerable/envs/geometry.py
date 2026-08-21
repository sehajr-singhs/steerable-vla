"""Segment-intersection geometry shared by the env and the generator."""

import numpy as np


def orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_seg(a, b, p, eps=1e-9):
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def seg_intersect(a, b, c, d):
    """Proper intersection of segments ab and cd -> (bool, point).

    Only intersections in the relative interior of both segments count;
    endpoint touching / vertex sharing does not (a vertex touch is not a
    crossing of a polyline).
    """
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)

    def _interior(p):
        return (on_seg(a, b, p) and on_seg(c, d, p)
                and np.linalg.norm(p - a) > 1e-8
                and np.linalg.norm(p - b) > 1e-8
                and np.linalg.norm(p - c) > 1e-8
                and np.linalg.norm(p - d) > 1e-8)

    if o1 == 0 and on_seg(a, b, c):
        return _interior(np.asarray(c, dtype=float)), np.asarray(c, dtype=float)
    if o2 == 0 and on_seg(a, b, d):
        return _interior(np.asarray(d, dtype=float)), np.asarray(d, dtype=float)
    if o3 == 0 and on_seg(c, d, a):
        return _interior(np.asarray(a, dtype=float)), np.asarray(a, dtype=float)
    if o4 == 0 and on_seg(c, d, b):
        return _interior(np.asarray(b, dtype=float)), np.asarray(b, dtype=float)
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        denom = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        if abs(denom) < 1e-12:
            return False, np.zeros(2)
        t = ((c[0] - a[0]) * (d[1] - c[1]) - (c[1] - a[1]) * (d[0] - c[0])) / denom
        p = np.asarray(a, dtype=float) + t * (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
        return True, p
    return False, np.zeros(2)


def _pair_indices(n):
    """(i, j) index pairs for all non-adjacent segment pairs."""
    I, J = [], []
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            I.append(i)
            J.append(j)
    return np.asarray(I, dtype=np.int64), np.asarray(J, dtype=np.int64)


_PAIR_CACHE = {}


def count_crossings(x):
    """Count crossings between non-adjacent segments (vectorized bulk test).

    A crossing is a proper, strictly-interior intersection of two non-adjacent
    segments; vertex sharing / endpoint touching does not count. The bulk
    orientation test is fully vectorized; intersection points are computed
    only for the (few) actual hits.
    """
    n = len(x)
    if n < 4:
        return 0, []
    if n not in _PAIR_CACHE:
        _PAIR_CACHE[n] = _pair_indices(n)
    I, J = _PAIR_CACHE[n]
    a = x[I]
    b = x[I + 1]
    c = x[J]
    d = x[J + 1]
    o1 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    o2 = (b[:, 0] - a[:, 0]) * (d[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (d[:, 0] - a[:, 0])
    o3 = (d[:, 0] - c[:, 0]) * (a[:, 1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0])
    o4 = (d[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0])
    mask = (((o1 > 0) != (o2 > 0)) & ((o3 > 0) != (o4 > 0))
            & (o1 != 0) & (o2 != 0) & (o3 != 0) & (o4 != 0))
    idx = np.nonzero(mask)[0]
    pts = []
    if len(idx):
        # compute the intersection point for actual hits only
        aa = a[idx]
        bb = b[idx]
        cc = c[idx]
        dd = d[idx]
        denom = ((bb[:, 0] - aa[:, 0]) * (dd[:, 1] - cc[:, 1])
                 - (bb[:, 1] - aa[:, 1]) * (dd[:, 0] - cc[:, 0]))
        t = ((cc[:, 0] - aa[:, 0]) * (dd[:, 1] - cc[:, 1])
             - (cc[:, 1] - aa[:, 1]) * (dd[:, 0] - cc[:, 0])) / (denom + 1e-12)
        p = aa + t[:, None] * (bb - aa)
        pts = [pp for pp in p]
    return len(idx), pts


def first_crossing_point(x):
    """The first (by segment order) proper crossing point, or None.

    This is the toy's "VLM subgoal extractor": given the cable state it
    returns the point where the next manipulation should happen, expressed
    in the same frame as x. The learned policy never computes this itself;
    it is the high-level signal that steers the low-level flow expert.
    """
    _, pts = count_crossings(x)
    if not pts:
        return None
    return np.asarray(pts[0], dtype=float).copy()
