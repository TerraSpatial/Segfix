"""The two things a completed lasso spends its time on: projecting the cloud
to canvas pixels, and recording which points came back.

Both were rewritten for speed — the projection collapses vispy's nested
transform chain into one matrix, and the selection is a boolean mask rather
than a set of indices — so these pin the behaviour that had to survive it.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.cloudview import CloudView, selection_mask  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _view(n=400, seed=0, base=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    coords = (rng.random((n, 3)) * [40.0, 40.0, 25.0] + base).astype(np.float32)
    view = CloudView()
    view.canvas.size = (1200, 900)
    view._coords = coords
    view._shown = np.ones(n, dtype=bool)
    view._face_color = np.ones((n, 4), np.float32)
    view._selected = np.zeros(n, dtype=bool)
    view.reset_view()
    return view, coords


def _through_the_chain(view, coords):
    """What project_to_canvas did before the matrix was collapsed."""
    tr = view.markers.get_transform(map_from="visual", map_to="canvas")
    mapped = np.asarray(tr.map(np.asarray(coords, dtype=np.float64)))
    w = mapped[:, 3]
    valid = w > 0
    w_safe = np.where(valid, w, 1.0)
    return mapped[:, :2] / w_safe[:, None], mapped[:, 2] / w_safe, valid


# -- projection --------------------------------------------------------------
@pytest.mark.parametrize("name", ["3d", "top", "front", "back", "left", "right"])
@pytest.mark.parametrize("base", [(0.0, 0.0, 0.0), (400000.0, 6000000.0, 0.0)])
def test_projection_matches_the_transform_chain_it_replaced(name, base):
    """Including a cloud left in UTM coordinates, where recovering the
    matrix around the world origin instead of the camera's pivot cancelled
    away most of its significant digits — a two-thirds-of-a-pixel error,
    which is enough to move a lasso's edge."""
    view, coords = _view(base=base)
    view.set_view(name)
    ref_xy, _, ref_valid = _through_the_chain(view, coords)

    xy, valid = view.project_to_canvas(coords)

    # Well under a thousandth of a pixel: both sides are float64 arithmetic
    # in a different order, and nothing here is exact to begin with.
    assert np.abs(xy - ref_xy).max() < 1e-4
    assert np.array_equal(valid, ref_valid)


def test_projection_still_follows_a_perspective_camera():
    """The orthographic shortcut is chosen from the matrix, not from the
    camera, so switching to a perspective fov must still project correctly —
    within a fraction of a pixel, which is where the divide's rounding lands.
    """
    view, coords = _view()
    view.view.camera.fov = 45.0
    ref_xy, _, ref_valid = _through_the_chain(view, coords)

    xy, valid = view.project_to_canvas(coords)

    assert np.array_equal(valid, ref_valid)
    assert np.abs(xy - ref_xy).max() < 1e-3


def test_orthographic_projection_reports_every_point_in_front():
    view, coords = _view()
    _, valid = view.project_to_canvas(coords)
    assert valid.all()


def test_depth_is_only_computed_when_asked_for():
    view, coords = _view()
    _, ref_depth, _ = _through_the_chain(view, coords)

    xy, depth, valid = view._project(coords, with_depth=True)
    assert np.allclose(depth, ref_depth)
    assert view._project(coords, with_depth=False)[1] is None


def test_projection_of_an_empty_cloud():
    view, _ = _view()
    xy, valid = view.project_to_canvas(np.empty((0, 3), np.float32))
    assert xy.shape == (0, 2) and valid.shape == (0,)


# -- selection ---------------------------------------------------------------
def test_selection_mask_from_indices_a_set_and_a_mask():
    from_indices = selection_mask(np.array([1, 3]), 5)
    assert from_indices.tolist() == [False, True, False, True, False]
    assert selection_mask({1, 3}, 5).tolist() == from_indices.tolist()
    assert selection_mask([1, 3], 5).tolist() == from_indices.tolist()
    assert selection_mask(from_indices, 5).tolist() == from_indices.tolist()


def test_selection_mask_replaces_unless_told_to_add():
    base = selection_mask([0, 1], 5)
    assert selection_mask([3], 5).tolist() == [False] * 3 + [True, False]
    added = selection_mask([3], 5, base)
    assert added.tolist() == [True, True, False, True, False]
    # The base it was given is never modified.
    assert base.tolist() == [True, True, False, False, False]


def test_selection_mask_of_nothing_is_empty_not_everything():
    assert not selection_mask([], 4).any()
    assert not selection_mask(np.empty(0, np.int64), 4).any()
    assert not selection_mask(set(), 4).any()


def test_select_returns_the_count_and_round_trips_through_selected():
    view, _ = _view(n=10)
    assert view.select([2, 5, 7]) == 3
    assert view.selected == {2, 5, 7}
    assert view.selected_mask.tolist() == [i in (2, 5, 7) for i in range(10)]
    # Adding keeps what was there; replacing does not.
    assert view.select([1], additive=True) == 4
    assert view.selected == {1, 2, 5, 7}
    assert view.select([9]) == 1
    assert view.selected == {9}


def test_assigning_selected_still_takes_a_set():
    """The old set-based property is what the panel and the tests use to
    clear or preset a selection; it has to keep working over the mask."""
    view, _ = _view(n=10)
    view.selected = {4, 6}
    assert view.selected_mask.sum() == 2
    view.selected = set()
    assert not view.selected_mask.any()


def test_selection_change_notifies_the_panel():
    view, _ = _view(n=10)
    seen = []
    view.on_selection_changed = lambda: seen.append(view.selected)
    view.select([3])
    view.selected = set()
    assert seen == [{3}, set()]


# -- who gets the mouse ------------------------------------------------------
def _camera_is_listening(view):
    """Whether the camera is still subscribed to the ViewBox's mouse events.

    vispy keeps bound methods as ``(weakref, name)`` pairs, so the callback
    is matched by what it points at rather than by identity.
    """
    camera = view.view.camera
    listening = []
    for name in view._CAMERA_MOUSE_EVENTS:
        for callback in getattr(view.view.events, name)._callbacks:
            if (isinstance(callback, tuple) and callback[0]() is camera
                    and callback[1] == "viewbox_mouse_event"):
                listening.append(name)
    return listening


def test_arming_a_tool_takes_the_mouse_off_the_camera():
    """`camera.interactive = False` reads like it does this, but in vispy
    0.16 nothing ever reads that flag — so a drag begun on top of the cloud
    orbited the view mid-lasso. Taking the camera off the ViewBox's mouse
    events is what actually stops it."""
    view, _ = _view()
    assert _camera_is_listening(view) == list(view._CAMERA_MOUSE_EVENTS)

    view.set_camera_interactive(False)
    assert _camera_is_listening(view) == []

    view.set_camera_interactive(True)
    assert _camera_is_listening(view) == list(view._CAMERA_MOUSE_EVENTS)


def test_handing_the_mouse_over_twice_does_not_double_subscribe():
    """Every tool calls this on arm and on disarm, and LassoTool.reassert
    calls it again for good measure; a duplicate subscription would make the
    camera act on each drag twice."""
    view, _ = _view()
    for _ in range(3):
        view.set_camera_interactive(False)
    assert _camera_is_listening(view) == []
    for _ in range(3):
        view.set_camera_interactive(True)
    assert _camera_is_listening(view) == list(view._CAMERA_MOUSE_EVENTS)


def test_the_wheel_and_the_right_drag_go_quiet_too():
    """The docstring promises rotation, pan *and* zoom are off while a tool
    is armed. No selection tool ever looked at the wheel or button 2, so
    those only stop if the camera itself stops listening."""
    view, _ = _view()
    view.set_camera_interactive(False)
    quiet = set(view._CAMERA_MOUSE_EVENTS) - set(_camera_is_listening(view))
    assert {"mouse_wheel", "mouse_move", "mouse_press"} <= quiet


def test_the_double_click_guard_still_reads_the_flag():
    """_on_double_click checks camera.interactive as our own flag; vispy
    ignoring it does not stop us using it."""
    view, _ = _view()
    view.set_camera_interactive(False)
    assert view.view.camera.interactive is False
    view.set_camera_interactive(True)
    assert view.view.camera.interactive is True
