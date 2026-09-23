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

"""Switching between Move, Lasso, Lasso tree, Cluster and Lasso section.

Exactly one of them is on at a time, and the one that is on owns the mouse:
in Move the camera orbits on a drag, and under any selection tool it must
not. Both halves are checked here on a real (offscreen) panel, because the
bug these cover was invisible in the button states -- the right button lit
up while the drag still rotated the view.
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

    view.canvas.size = (800, 800)
    rng = np.random.default_rng(0)
    cloud = PointCloud(coords=rng.random((50, 3)).astype(np.float32),
                       labels=np.ones(50, np.int32))
    view.load_cloud(cloud)
    seg = SegFixController(view, cloud)
    p = SegFixWidget(seg)
    seg.set_cloud(cloud)
    yield p, view
    p.deleteLater()


def _tools(p):
    """The mode buttons, by the name the panel calls each mode."""
    return {
        "move": p.move_btn,
        "lasso": p.lasso_btn,
        "tree": p.tree_lasso_btn,
        "cluster": p.cluster_btn,
        "section": p.section_draw_btn,
    }


def _checked(p):
    return {name for name, btn in _tools(p).items() if btn.isChecked()}


def _camera_drags(view):
    """Is the camera's own handler still on the viewbox, i.e. would a drag
    orbit the view? This is what 'Move is on too' actually means -- the flag
    CloudView keeps is not the thing vispy reads (see set_camera_interactive).
    """
    return any(
        isinstance(cb, tuple) and cb[1] == "viewbox_mouse_event"
        for cb in view.view.events.mouse_move.callbacks
    )


SELECTION_TOOLS = ("lasso", "tree", "cluster", "section")


def test_move_is_the_only_mode_that_drags_the_camera(panel):
    p, view = panel
    assert _checked(p) == {"move"}
    assert _camera_drags(view)


@pytest.mark.parametrize("tool", SELECTION_TOOLS)
def test_arming_a_tool_takes_the_mouse_off_the_camera(panel, tool):
    p, view = panel
    _tools(p)[tool].toggle()
    assert _checked(p) == {tool}
    assert not _camera_drags(view)


@pytest.mark.parametrize("second", SELECTION_TOOLS)
@pytest.mark.parametrize("first", SELECTION_TOOLS)
def test_switching_straight_between_tools_keeps_the_mouse(panel, first, second):
    """The reported bug: going from Cluster to Lasso tree left the drag
    rotating the view. Each handler armed the new tool and only then stood
    the old one down, and standing a tool down hands the mouse back to the
    camera -- so the disarm undid the arm.
    """
    if first == second:
        pytest.skip("not a switch")
    p, view = panel
    _tools(p)[first].toggle()
    _tools(p)[second].toggle()
    assert _checked(p) == {second}
    assert not _camera_drags(view)


@pytest.mark.parametrize("tool", SELECTION_TOOLS)
def test_leaving_a_tool_gives_the_camera_back(panel, tool):
    p, view = panel
    _tools(p)[tool].toggle()
    _tools(p)[tool].toggle()
    assert _checked(p) == {"move"}
    assert _camera_drags(view)


@pytest.mark.parametrize("tool", SELECTION_TOOLS)
def test_escape_gives_the_camera_back_from_any_tool(panel, tool):
    p, view = panel
    _tools(p)[tool].toggle()
    p.on_move_mode()                      # what Esc is bound to
    assert _checked(p) == {"move"}
    assert _camera_drags(view)


def test_the_status_line_describes_the_tool_switched_to(panel):
    """The same ordering, seen from the status bar: Cluster's "Cluster off"
    used to land *after* the lasso had armed and announced itself, so the
    line described the tool the user had just left.

    Plain Lasso rather than Lasso tree, because Lasso tree with no tree
    under review posts a message of its own afterwards either way.
    """
    p, view = panel
    said = []
    view.on_status = said.append

    p.cluster_btn.toggle()
    p.lasso_btn.toggle()
    assert "Lasso" in said[-1], said
