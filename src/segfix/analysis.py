"""Segmentation-quality analysis: spatial neighbour queries.

UI-agnostic, like :mod:`~segfix.operations`.  Point-set distances are
approximated by sampling each tree down to a few thousand points and querying
a KD-tree — exact enough for "does this touch?" questions at forest scale,
and fast enough to run per keypress.
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud

_SAMPLE_CAP = 3000


def _sample(rng, idx: np.ndarray, cap: int = _SAMPLE_CAP) -> np.ndarray:
    if idx.size <= cap:
        return idx
    return rng.choice(idx, cap, replace=False)


def _kdtree(points):
    """A cKDTree built for speed rather than for a tidy tree.

    ``balanced_tree``/``compact_nodes`` make cKDTree find the median at every
    split and shrink each node to its points — worth it for a tree queried
    many times, but everything here builds a tree, asks it one thing and
    throws it away. Turning both off measured 2.5x faster to build (59ms
    against 24ms on 120k points) for queries that are no slower in practice.
    """
    from scipy.spatial import cKDTree

    return cKDTree(points, balanced_tree=False, compact_nodes=False)


def point_spacing(coords: np.ndarray, rng=None) -> float:
    """Median nearest-neighbour distance over a sample — the cloud's typical
    point pitch.  Used to pick the gap a region grow is allowed to bridge.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return 0.01
    rng = rng or np.random.default_rng(0)
    sample = coords if n <= 20000 else coords[rng.choice(n, 20000, replace=False)]
    d, _ = _kdtree(coords).query(sample, k=2, workers=-1)
    return float(np.median(d[:, 1])) or 0.01


#: Above this many points, connectivity is computed on one point per
#: (eps / :data:`_VOXEL_DIVISOR`) voxel rather than on every point. Below it
#: the exact pass already answers in well under a tenth of a second, and
#: nothing is gained by approximating.
_VOXEL_ABOVE = 20_000

#: How much finer than ``eps`` the voxels are. At 4 a voxel's diagonal is
#: ``eps * sqrt(3) / 4``, about 0.43 eps.
_VOXEL_DIVISOR = 4


def _components_exact(coords: np.ndarray, eps: float) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(coords)
    pairs = _kdtree(coords).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    data = np.ones(len(pairs), dtype=np.int8)
    graph = coo_matrix(
        (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    return connected_components(graph, directed=False)[1]


def _voxel_groups(coords: np.ndarray, size: float):
    """``(voxel id per point, one representative point per voxel)``."""
    origin = coords.min(axis=0)
    cell = np.floor((coords - origin) / size).astype(np.int64)
    dims = cell.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0**62:
        return None  # pragma: no cover - a vast extent against a tiny gap
    flat = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    return inverse, first


def connected_components_within(coords: np.ndarray, eps: float) -> np.ndarray:
    """Label every point 0..k-1 by which ``eps``-connected blob it belongs to.

    Runs on the given points only (call it per tree, not per plot). Points
    within ``eps`` of each other are in the same blob.

    The work is the number of point pairs within ``eps``, which grows with
    the cube of the gap: on a 120k-point tree that is 4M pairs at one point
    spacing but 113M at sixteen — 1.8GB for the pair array alone, and the
    cluster tool's gap slider goes that far. So past :data:`_VOXEL_ABOVE`
    points the graph is built over one representative per voxel of side
    ``eps / _VOXEL_DIVISOR`` instead, and each point takes its voxel's
    answer. Collapsing a voxel is sound in the direction that matters: its
    diagonal is 0.43 eps, so its points are within ``eps`` of one another and
    genuinely are one blob. What the reduction can miss is a link between two
    points in different voxels whose representatives are further apart than
    ``eps``; measured against the exact pass on a 120k-point tree that moved
    no points at all at 1x, 8x and 16x the spacing, and 24 of 107,236 at 4x.
    In exchange the cost stops following the gap: 22.2s to 1.0s at 16x, and
    the representative count falls as fast as the pair count would have
    climbed.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n > _VOXEL_ABOVE and eps > 0:
        grouped = _voxel_groups(coords, eps / _VOXEL_DIVISOR)
        if grouped is not None:
            inverse, first = grouped
            if len(first) < n:  # only worth it if it actually collapsed
                return _components_exact(coords[first], eps)[inverse]
    return _components_exact(coords, eps)


def connected_patch(
    coords: np.ndarray, labels: np.ndarray, seed: int, eps: float
) -> np.ndarray:
    """Indices of the ``eps``-connected blob of the seed's own tree that the
    seed sits in — i.e. one physically continuous lump of a single tree ID.
    """
    seed = int(seed)
    same = np.flatnonzero(labels == labels[seed])
    if same.size <= 1:
        return same
    comp = connected_components_within(coords[same], eps)
    local = int(np.searchsorted(same, seed))
    return same[comp == comp[local]]


def neighbours_by_points(
    cloud: PointCloud, tid: int, reach: float, rng=None
) -> set[int]:
    """IDs of trees whose points come within ``reach`` metres of tree
    ``tid``'s points.

    Bounding-box tests massively over-count neighbours in a closed canopy
    (a tall tree's box spans its whole crown), so boxes are only used as a
    prefilter; candidates are confirmed by sampled point-to-point distance.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    labels, coords = cloud.labels, cloud.coords
    mine = np.flatnonzero(labels == tid)
    if mine.size == 0:
        return set()
    lo = coords[mine].min(axis=0) - reach
    hi = coords[mine].max(axis=0) + reach
    # density.in_box rather than the chained expression this used to spell
    # out: same per-axis test, but into two reused buffers instead of eleven
    # (N,) temporaries, which matters because this runs over the whole
    # catalogue every time a tree is opened.
    from .density import in_box as _in_box

    in_box = _in_box(coords, lo, hi)
    in_box[mine] = False
    # Everything below works off the in-box points only, so the per-candidate
    # loop no longer rescans all N labels once per candidate.
    box_idx = np.flatnonzero(in_box)
    box_labels = labels[box_idx]
    cand = np.unique(box_labels)
    cand = cand[(cand != UNASSIGNED) & (cand != NOISE) & (cand != tid)]
    if not cand.size:
        return set()
    kd = cKDTree(coords[_sample(rng, mine)])
    out: set[int] = set()
    for t in cand:
        theirs = box_idx[box_labels == t]
        d, _ = kd.query(
            coords[_sample(rng, theirs)], k=1, distance_upper_bound=reach
        )
        if np.isfinite(d).any():
            out.add(int(t))
    return out
