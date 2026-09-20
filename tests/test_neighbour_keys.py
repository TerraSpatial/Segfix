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


def _buttons(grid):
    return [grid.itemAt(i).widget() for i in range(grid.count())]


def test_each_keyed_button_wears_the_key_that_presses_it(panel):
    """The key is a drawn keycap in the icon, not a second number in the
    text: the button already carries the tree id."""
    p, _seg, _cloud, _said = panel
    keyed = _buttons(p.keyed_grid)

    assert [b.text().strip() for b in keyed] == ["2", "3"]
    assert [b.property("segfix_key") for b in keyed] == [1, 2]
    assert not any(b.icon().isNull() for b in keyed)
    assert ", key 1" in keyed[0].toolTip()


def test_the_keyed_buttons_never_scroll_out_of_sight(panel):
    """A seventh tree still gets its button, and still works with the
    mouse; it just has no key to advertise."""
    p, seg, _cloud, _said = panel
    from segfix.model import PointCloud
    from segfix.widgets import NEIGHBOUR_KEYS

    # One tree in the middle, touching six others.
    rng = np.random.default_rng(1)
    middle = rng.random((60, 3)) * 0.5
    blocks, labels = [middle], [np.full(60, 1)]
    for k in range(NEIGHBOUR_KEYS + 1):
        angle = 2 * np.pi * k / (NEIGHBOUR_KEYS + 1)
        shift = np.array([np.cos(angle), np.sin(angle), 0.0]) * 0.4
        blocks.append(rng.random((60, 3)) * 0.5 + shift)
        labels.append(np.full(60, k + 2))
    crowded = PointCloud(coords=np.concatenate(blocks).astype(np.float32),
                         labels=np.concatenate(labels).astype(np.int32))
    p.c.view.load_cloud(crowded)
    seg.set_cloud(crowded)
    p._set_current(1, fly=False)

    keyed, extra = _buttons(p.keyed_grid), _buttons(p.neighbour_grid)
    # The five a key reaches stay put; the rest go in the scrolling box.
    assert len(keyed) == NEIGHBOUR_KEYS and len(extra) == 1
    assert [b.property("segfix_key") for b in keyed] == [1, 2, 3, 4, 5]
    assert extra[0].property("segfix_key") is None
    assert "key" not in extra[0].toolTip()
    assert not p.neighbour_scroll.isHidden()  # there is an overflow to scroll


def test_no_scroll_box_when_every_neighbour_has_a_key(panel):
    p, _seg, _cloud, _said = panel
    assert len(p._neighbour_ids) == 2
    assert p.neighbour_scroll.isHidden()


def test_the_keys_reach_the_nearest_trees_not_the_lowest_numbered(panel):
    """Sorted by id, the keys landed on whichever trees happened to have the
    lowest numbers; in a closed canopy that is nobody's idea of the right
    five."""
    p, seg, _cloud, _said = panel
    from segfix.model import PointCloud

    rng = np.random.default_rng(2)
    blocks = [rng.random((60, 3)) * 0.4]
    labels = [np.full(60, 105)]
    # id order and distance order deliberately disagree
    for nid, gap in ((247, 0.1), (102, 0.5), (319, 0.2), (103, 0.8)):
        blocks.append(rng.random((40, 3)) * 0.2 + np.array([0.4 + gap, 0, 0]))
        labels.append(np.full(40, nid))
    crowded = PointCloud(coords=np.concatenate(blocks).astype(np.float32),
                         labels=np.concatenate(labels).astype(np.int32))
    p.c.view.load_cloud(crowded)
    seg.set_cloud(crowded)
    p.focus_margin.setValue(2.0)
    p._set_current(105, fly=False)

    assert p._neighbour_ids == [247, 319, 102, 103]
    assert "m away, key 1" in p._neighbour_btns[0].toolTip()


def test_what_needs_a_selection_is_dead_without_one(panel):
    """The panel used to offer every button with nothing selected, and
    answer a click with a status line."""
    p, seg, cloud, _said = panel
    assert not p.add_btn.isEnabled()
    assert not p.split_btn.isEnabled()
    assert not any(b.isEnabled() for b in p._neighbour_btns)

    _select(seg, 1)
    p._update_selection()
    assert p.add_btn.isEnabled()
    assert p.split_btn.isEnabled()
    assert all(b.isEnabled() for b in p._neighbour_btns)

    seg.view.selected = set()
    p._update_selection()
    assert not p.add_btn.isEnabled()


def test_unassign_and_noise_stay_live_with_nothing_selected(panel):
    """With no selection those act on the whole current tree — that is how a
    bush gets dismissed in one key — so they must not be greyed out with the
    rest."""
    from segfix.model import NOISE

    p, seg, cloud, said = panel
    assert seg.selected_indices().size == 0
    was_tree_1 = np.flatnonzero(cloud.labels == 1)
    assert was_tree_1.size

    p.on_noise()  # no selection: the whole current tree

    np.testing.assert_array_equal(cloud.labels[was_tree_1], NOISE)
    assert any("noise" in m.lower() for m in said)
