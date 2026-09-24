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

"""The View group's three toggles behave like toggles.

"Show unassigned", "Hide others" and "Fade others" all turn something in
the view on and off and stay that way until turned off again. The two
"others" ones used to be push buttons, which showed none of that: with
every neighbour hidden the button looked exactly as it did with none
hidden. They are checkboxes now, so what they must do is keep telling the
truth — including when the same state is reached one tree at a time from
the table's Hide/Fade columns, or when the queue moves on and "others"
means a different set of trees.

Builds the real panel on a real (offscreen) vispy canvas; skipped where no
canvas can be created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QCheckBox  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel():
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    rng = np.random.default_rng(0)
    coords = rng.random((60, 3)).astype(np.float32)
    labels = np.repeat([1, 2, 0], 20).astype(np.int32)  # tree 1, tree 2, ground
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    cloud = PointCloud(coords=coords, labels=labels)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    yield p, view
    p.deleteLater()


def test_all_three_view_toggles_are_the_same_kind_of_control(panel):
    p, _ = panel
    for box in (p.show_unassigned, p.hide_others_cb, p.fade_others_cb):
        assert isinstance(box, QCheckBox), box.text()


def test_the_key_and_the_toggle_are_one_control(panel):
    """G moves the checkbox, so the two can't disagree."""
    p, view = panel
    p.on_hide_neighbours()
    assert p.hide_others_cb.isChecked()
    assert not view.shown[20:40].any()  # tree 2 gone
    p.on_hide_neighbours()
    assert not p.hide_others_cb.isChecked()
    assert view.shown[20:40].all()


def test_fade_others_is_the_same(panel):
    p, _ = panel
    p.on_fade_neighbours()
    assert p.fade_others_cb.isChecked()
    assert p.c.faded_ids == {2}
    p.on_fade_neighbours()
    assert not p.fade_others_cb.isChecked()
    assert p.c.faded_ids == set()


def test_clicking_the_checkbox_does_the_same_as_the_key(panel):
    p, view = panel
    p.hide_others_cb.setChecked(True)
    assert p.hidden_ids == {2}
    assert not view.shown[20:40].any()
    p.hide_others_cb.setChecked(False)
    assert p.hidden_ids == set()


def test_hiding_every_other_tree_by_hand_checks_the_toggle(panel):
    """The table's eye column reaches the same state, so the toggle has to
    follow it rather than remember its own clicks."""
    p, _ = panel
    p.hidden_ids = {2}
    p._refresh_tree_table()
    assert p.hide_others_cb.isChecked()


def test_the_toggle_follows_the_queue(panel):
    """"Others" means others than the current tree: with tree 2 hidden the
    toggle is on for tree 1, and off once tree 2 is the one under review."""
    p, _ = panel
    p.hidden_ids = {2}
    p._refresh_tree_table()
    assert p.hide_others_cb.isChecked()
    p._set_current(2, fly=False)
    assert not p.hide_others_cb.isChecked()


def test_with_no_tree_under_review_the_toggle_stays_off(panel):
    p, view = panel
    said = []
    view.on_status = said.append
    p._set_current(None, fly=False)
    p.hide_others_cb.setChecked(True)
    assert not p.hide_others_cb.isChecked()
    assert p.hidden_ids == set()
    assert "No tree under review" in said[-1]
