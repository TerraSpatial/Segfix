# segfix — fix instance segmentation of tree point clouds.
# Copyright (C) 2026 Tim Devereux, The University of Queensland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Opening a cloud must not cost several times what it keeps.

A 480M-point plot (21GB of LAS) was OOM-ing a 64GB machine: the open path
decoded the whole cloud to float64 coordinates, widened every label to
int64, and built the voxel keys as a (3, N) int64 array — all live at once,
about 110 bytes a point against the ~30 it ends up holding. Everything that
walks the whole cloud now goes chunk by chunk into the array it is filling.

These tests pin both halves of that: the chunked passes give exactly the
same answer as decoding in one bite (checked with a deliberately tiny chunk,
so a small test cloud still crosses several boundaries), and the peak stays
near what the session actually keeps.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from segfix import io, treecatalog
from segfix.model import PointCloud
from segfix.treecatalog import open_catalog


def _cloud(tmp_path, n, *, dense=False, big_coords=False, name="plot.ply"):
    rng = np.random.default_rng(0)
    side = (n * 5e-6) ** (1 / 3) if dense else 20.0
    coords = rng.random((n, 3)) * side
    if big_coords:
        coords += np.array([500000.0, 7000000.0, 0.0])
    cloud = PointCloud(coords=coords.astype(np.float32),
                       labels=rng.integers(0, 8, n).astype(np.int32))
    path = str(tmp_path / name)
    io.save(cloud, path)
    return path


# -- the chunked passes agree with decoding in one bite -----------------------
@pytest.fixture
def tiny_chunks(monkeypatch):
    """Chunk every whole-cloud pass at 1000 points, so a small test cloud
    still crosses plenty of boundaries."""
    monkeypatch.setattr(treecatalog, "_CHUNK", 1000)


def test_chunked_coordinates_match_a_whole_cloud_decode(tmp_path, tiny_chunks):
    path = _cloud(tmp_path, 5000)
    cat = open_catalog(path)
    whole = cat._decode_coords_raw(cat._mm).astype(np.float32)
    np.testing.assert_array_equal(cat.coords, whole)


def test_chunked_coordinates_match_with_a_global_shift(tmp_path, tiny_chunks):
    """The shift is applied in float64 before the cast, chunk or no chunk —
    that precision is the whole point of shifting."""
    path = _cloud(tmp_path, 5000, big_coords=True, name="utm.ply")
    cat = open_catalog(path, shift_prompt=lambda mins, maxs, suggested: suggested)
    assert cat.global_shift is not None
    whole = (cat._decode_coords_raw(cat._mm) + cat.global_shift).astype(np.float32)
    np.testing.assert_array_equal(cat.coords, whole)


def test_the_bounds_that_decide_the_shift_are_the_cloud_s_own(tmp_path, tiny_chunks):
    seen = {}

    def prompt(mins, maxs, suggested):
        seen["mins"], seen["maxs"] = mins, maxs
        return None  # keep original coordinates

    path = _cloud(tmp_path, 5000, big_coords=True, name="utm.ply")
    cat = open_catalog(path, shift_prompt=prompt)
    raw = cat._decode_coords_raw(cat._mm)
    np.testing.assert_allclose(seen["mins"], raw.min(axis=0))
    np.testing.assert_allclose(seen["maxs"], raw.max(axis=0))


def test_chunked_labels_match_a_whole_cloud_decode(tmp_path, tiny_chunks):
    path = _cloud(tmp_path, 5000)
    cat = open_catalog(path)
    whole, _colors = cat._decode_labels(cat._mm)
    np.testing.assert_array_equal(cat.labels, np.asarray(whole).astype(np.int32))


def test_the_label_sort_key_is_chunked_without_changing_the_order(tiny_chunks):
    rng = np.random.default_rng(1)
    labels = rng.integers(-1, 40, 9999).astype(np.int32)
    order = treecatalog._label_order(labels)
    np.testing.assert_array_equal(labels[order],
                                  labels[np.argsort(labels, kind="stable")])
    np.testing.assert_array_equal(order, np.argsort(labels, kind="stable"))


# -- what it costs ------------------------------------------------------------
def _anonymous_bytes():
    """Memory that isn't backed by a file — what an OOM actually counts. The
    memory-mapped cloud is file-backed and reclaimable, so it doesn't."""
    with open("/proc/self/smaps_rollup") as fh:
        for line in fh:
            if line.startswith("Anonymous:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no Anonymous line")


#: Peak anonymous bytes per point that opening a cloud may cost. The session
#: keeps about 30 (float32 coords, labels, the save baseline, the sort
#: order); measured peak is ~56, and the code this guards against was ~110.
_PEAK_BUDGET = 80


@pytest.mark.skipif(not os.path.exists("/proc/self/smaps_rollup"),
                    reason="needs Linux /proc")
def test_opening_a_dense_cloud_stays_within_its_memory_budget(tmp_path):
    n = 2_000_000
    path = _cloud(tmp_path, n, dense=True)

    peak = [0]
    stop = [False]

    def watch():
        while not stop[0]:
            peak[0] = max(peak[0], _anonymous_bytes())
            time.sleep(0.005)

    base = _anonymous_bytes()
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    try:
        cat = open_catalog(path, density_prompt=lambda s, n_pts, sug, kept: sug)
    finally:
        stop[0] = True
        thread.join()
    assert cat.is_decimated  # the expensive path: decimation on top of the open
    per_point = (peak[0] - base) / n
    assert per_point < _PEAK_BUDGET, f"{per_point:.0f} bytes/point at peak"
