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

"""Invert selection stays inside what's on screen.

Only shown points can be selected in the first place — the lasso and the
cluster tool both intersect their result with ``view.shown`` — so inverting
must too. The bug this guards against is the obvious implementation,
``~selected``: inside a cross section that quietly hands back the whole
cloud standing behind the slab, and the next A or D would move all of it.

Builds the real panel on a real (offscreen) vispy canvas; skipped where no
canvas can be created.
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


def selected(view):
    return set(np.flatnonzero(view.selected_mask).tolist())


# -- the plain case ----------------------------------------------------------
def test_inverting_swaps_selected_for_unselected(panel):
    p, view = panel
    view.select(np.arange(0, 20))
    p.on_invert_selection()
    assert selected(view) == set(range(20, 60))


def test_inverting_twice_gives_the_selection_back(panel):
    p, view = panel
    view.select(np.array([3, 7, 41]))
    p.on_invert_selection()
    p.on_invert_selection()
    assert selected(view) == {3, 7, 41}


def test_inverting_nothing_selects_everything_shown(panel):
    """Which is the "select all visible" the section tools otherwise had no
    way to ask for."""
    p, view = panel
    view.select(np.empty(0, dtype=np.int64))
    p.on_invert_selection()
    assert selected(view) == set(range(60))


def test_inverting_everything_clears_the_selection(panel):
    p, view = panel
    view.select(np.arange(60))
    p.on_invert_selection()
    assert selected(view) == set()


# -- bounded by what's shown -------------------------------------------------
def test_a_section_bounds_the_inversion(panel):
    """The whole point: with a slab on, inverting can't reach the cloud
    behind it."""
    p, view = panel
    slab = np.zeros(60, dtype=bool)
    slab[10:30] = True
    view.shown = slab
    view.select(np.arange(10, 20))
    p.on_invert_selection()
    assert selected(view) == set(range(20, 30))


def test_hidden_trees_are_never_picked_up(panel):
    p, view = panel
    p.hidden_ids = {2}
    p._apply_visibility()
    view.select(np.empty(0, dtype=np.int64))
    p.on_invert_selection()
    # Tree 2's points (20-39) are hidden, so they stay out of it.
    assert selected(view) == set(range(20)) | set(range(40, 60))


def test_hidden_unassigned_points_are_never_picked_up(panel):
    p, view = panel
    p.show_unassigned.setChecked(False)
    view.select(np.arange(0, 10))
    p.on_invert_selection()
    assert selected(view) == set(range(10, 40))  # the ground (40-59) stays out


def test_a_selection_of_hidden_points_still_ends_up_inside_the_view(panel):
    """A selection made before a section was switched on can hold points the
    slab now hides. Inverting has to land inside the slab regardless."""
    p, view = panel
    view.select(np.arange(0, 60))
    slab = np.zeros(60, dtype=bool)
    slab[50:] = True
    view.shown = slab
    p.on_invert_selection()
    assert selected(view) == set()
    p.on_invert_selection()
    assert selected(view) == set(range(50, 60))


# -- the status line ---------------------------------------------------------
def test_the_status_line_says_when_the_view_bounded_it(panel):
    p, view = panel
    said = []
    view.on_status = said.append
    view.select(np.arange(0, 20))
    p.on_invert_selection()
    assert "40" in said[-1] and "shown" not in said[-1]
    slab = np.zeros(60, dtype=bool)
    slab[:30] = True
    view.shown = slab
    p.on_invert_selection()
    assert "shown points" in said[-1]
