"""Number keys move the selection into a neighbouring tree.

Sending a patch to the tree next door is the commonest edit in the loop,
and it was the only one that needed the mouse: the neighbour buttons are in
the panel, and the hand holding the mouse has just finished drawing a
lasso. 1-5 are the same buttons, in the order they are shown.

Real panel on a real (offscreen) vispy canvas; skipped where no canvas can
be created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel():
    """Three touching trees: 1 in the middle, 2 and 3 either side."""
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    rng = np.random.default_rng(0)
    blocks, labels = [], []
    for k, label in enumerate((1, 2, 3)):
        blocks.append(rng.random((40, 3)) * 0.4 + np.array([k * 0.3, 0, 0]))
        labels.append(np.full(40, label))
    cloud = PointCloud(coords=np.concatenate(blocks).astype(np.float32),
                       labels=np.concatenate(labels).astype(np.int32))
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    said = []
    view.on_status = said.append  # `status` is write-only: it forwards here
    yield p, seg, cloud, said
    p.deleteLater()


def _select(seg, label, count=10):
    idx = np.flatnonzero(seg.cloud.labels == label)[:count]
    seg.view.selected = set(idx.tolist())
    return idx


def test_the_keys_follow_the_buttons_shown(panel):
    p, seg, cloud, _said = panel
    assert p._neighbour_ids == [2, 3]  # what keys 1 and 2 mean here

    idx = _select(seg, 1)
    p.send_to_nth_neighbour(2)  # the second button: tree 3
    assert set(cloud.labels[idx]) == {3}


def test_the_first_key_sends_to_the_first_neighbour(panel):
    p, seg, cloud, _said = panel
    idx = _select(seg, 1)
    p.send_to_nth_neighbour(1)
    assert set(cloud.labels[idx]) == {2}


def test_a_key_past_the_last_neighbour_says_so_and_changes_nothing(panel):
    p, seg, cloud, said = panel
    idx = _select(seg, 1)
    before = cloud.labels.copy()
    p.send_to_nth_neighbour(4)
    assert "Only 2 neighbouring trees" in said[-1]
    np.testing.assert_array_equal(cloud.labels, before)
    assert set(cloud.labels[idx]) == {1}


def test_with_no_neighbours_the_keys_say_so(panel):
    p, seg, cloud, said = panel
    p._neighbour_ids = []
    _select(seg, 1)
    p.send_to_nth_neighbour(1)
    assert "No neighbouring trees" in said[-1]
