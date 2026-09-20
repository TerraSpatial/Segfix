"""The cluster tool's adjustable gap, and its live re-apply.

One tree in two blobs, 5cm apart, sampled every 1cm: at the default gap
(4x spacing = 4cm) a click takes one blob; loosen it past 5cm and the same
click takes both. The view is a stand-in, as in test_lasso.py -- the
controller and ClusterTool only need its coords/selection/status, never GL.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import types

import numpy as np
import pytest

pytest.importorskip("qtpy")

from segfix.cloudview import selection_mask  # noqa: E402
from segfix.model import PointCloud  # noqa: E402
from segfix.widgets import (  # noqa: E402
    CLUSTER_GAP_FACTORS,
    DEFAULT_CLUSTER_GAP_FACTOR,
    SegFixController,
)

_SPACING = 0.01
_BLOB = 20  # points per blob
_A = np.column_stack([np.arange(_BLOB) * _SPACING, np.zeros(_BLOB), np.zeros(_BLOB)])
# Blob B starts 5cm past A's last point: bridged at 6x spacing, not at 4x.
_B = _A + [(_BLOB - 1) * _SPACING + 0.05, 0.0, 0.0]
_OTHER = np.array([[5.0, 5.0, 5.0]])  # a different tree, far away

A_IDX = set(range(_BLOB))
B_IDX = set(range(_BLOB, 2 * _BLOB))
OTHER_IDX = 2 * _BLOB


class _FakeSignal:
    """Enough of a vispy event for ClusterTool._connect to hook and unhook."""

    def __init__(self):
        self.callbacks = []

    def connect(self, cb):
        self.callbacks.append(cb)

    def disconnect(self, cb):
        if cb not in self.callbacks:
            raise ValueError(cb)
        self.callbacks.remove(cb)


class _FakeView:
    """Stand-in for CloudView: the selection API the controller and
    ClusterTool actually touch, over the same mask helper the real view
    uses, and no GL anywhere.
    """

    def __init__(self, coords):
        self.canvas = types.SimpleNamespace(
            events=types.SimpleNamespace(
                mouse_press=_FakeSignal(), mouse_release=_FakeSignal()
            )
        )
        self.native = None
        self.coords = coords
        self.shown = np.ones(len(coords), dtype=bool)
        self.status = ""
        self._mask = np.zeros(len(coords), dtype=bool)

    def pick_point(self, xy):
        return 0  # every click lands on blob A's first point

    def set_camera_interactive(self, on):
        self.camera_interactive = on

    @property
    def selected(self):
        return set(np.flatnonzero(self._mask).tolist())

    @selected.setter
    def selected(self, indices):
        self.select(indices)

    @property
    def selected_mask(self):
        return self._mask

    def select(self, indices, additive=False):
        self._mask = selection_mask(
            indices, len(self.coords), self._mask if additive else None
        )
        return int(np.count_nonzero(self._mask))


def _controller(factor=4.0):
    """A controller over the two-blob tree. ``factor`` pins the gap (4x by
    default: at the 1x default even blob A falls apart, since a median
    spacing only links about half the pairs); ``None`` leaves the default."""
    coords = np.vstack([_A, _B, _OTHER]).astype(np.float32)
    labels = np.array([1] * (2 * _BLOB) + [2], dtype=np.int32)
    cloud = PointCloud(coords=coords, labels=labels)
    view = _FakeView(coords)
    ctrl = SegFixController(view, cloud)
    # Off unless a test asks for it: the tests below count labellings and
    # assert what is in the cache, and a worker filling the rest of the
    # ladder behind them would make both nondeterministic. The prefetch has
    # its own tests, which turn it back on and join the worker.
    ctrl.cluster_prefetch = False
    # Bypass set_armed(): it would hook real canvas events.
    ctrl.cluster._armed = True
    if factor is not None:
        ctrl.set_cluster_gap_factor(factor)
    return ctrl, view


def _click(ctrl, additive=False, xy=(10.0, 10.0)):
    ctrl.cluster._select_at(xy, additive)


def _step_looser(ctrl):
    """What the panel wires on_repeat to (step_cluster_gap(1) moving the
    slider), minus the slider: one notch up CLUSTER_GAP_FACTORS."""
    def step():
        i = CLUSTER_GAP_FACTORS.index(ctrl.cluster_gap_factor)
        ctrl.set_cluster_gap_factor(
            CLUSTER_GAP_FACTORS[min(i + 1, len(CLUSTER_GAP_FACTORS) - 1)]
        )
    return step


def test_the_default_gap_is_one_point_spacing():
    ctrl, _ = _controller(factor=None)
    assert ctrl.cluster_gap_factor == DEFAULT_CLUSTER_GAP_FACTOR == 1.0
    assert ctrl.cluster_gap == pytest.approx(_SPACING, rel=0.05)


def test_the_default_first_click_is_a_small_seed():
    """What 1x means in practice: linking only pairs no further apart than
    the *median* spacing leaves most of a blob unconnected, so the first
    click takes a sliver and the repeat clicks do the growing."""
    ctrl, view = _controller(factor=None)
    _click(ctrl)
    assert 0 < len(view.selected) < _BLOB


def test_at_4x_a_click_takes_one_blob():
    ctrl, view = _controller()
    _click(ctrl)
    assert view.selected == A_IDX


def test_loosening_the_gap_grows_the_last_click_live():
    ctrl, view = _controller()
    _click(ctrl)

    assert ctrl.set_cluster_gap_factor(6.0) is True
    # No second click: the patch already on screen took in blob B.
    assert view.selected == A_IDX | B_IDX


def test_tightening_again_really_shrinks_it():
    """reapply() rebuilds from the pre-click selection, not the current
    one -- otherwise a patch could grow but never shrink back."""
    ctrl, view = _controller()
    _click(ctrl)
    ctrl.set_cluster_gap_factor(6.0)
    ctrl.set_cluster_gap_factor(3.0)
    assert view.selected == A_IDX


def test_reapply_keeps_what_a_shift_click_was_added_to():
    ctrl, view = _controller()
    view.selected = {OTHER_IDX}
    _click(ctrl, additive=True)
    assert view.selected == A_IDX | {OTHER_IDX}

    ctrl.set_cluster_gap_factor(6.0)
    assert view.selected == A_IDX | B_IDX | {OTHER_IDX}
    ctrl.set_cluster_gap_factor(4.0)
    assert view.selected == A_IDX | {OTHER_IDX}


def test_changing_the_gap_with_no_click_leaves_the_selection_alone():
    ctrl, view = _controller()
    view.selected = {3, 4}
    assert ctrl.set_cluster_gap_factor(8.0) is False
    assert view.selected == {3, 4}
    assert ctrl.cluster_gap == pytest.approx(8 * _SPACING, rel=0.05)


def test_setting_the_same_gap_is_a_no_op():
    ctrl, view = _controller()
    _click(ctrl)
    view.selected = {7}  # if it re-ran, this would be overwritten
    assert ctrl.set_cluster_gap_factor(4.0) is False
    assert view.selected == {7}


def test_the_gap_setting_survives_loading_another_cloud():
    """It's a preference, like the lasso mode -- but the measured spacing
    belongs to the old points and must be re-measured."""
    ctrl, view = _controller()
    ctrl.set_cluster_gap_factor(8.0)
    _ = ctrl.cluster_gap  # measure on the first cloud

    coarse = (np.vstack([_A, _B, _OTHER]) * 3).astype(np.float32)  # 3cm spacing
    view.coords = coarse
    # set_cloud disarms and re-arms an armed tool through real canvas events,
    # which the stand-in view doesn't have; arming isn't what's under test.
    ctrl.cluster._armed = False
    ctrl.set_cloud(PointCloud(coords=coarse, labels=ctrl.cloud.labels.copy()))

    assert ctrl.cluster_gap_factor == 8.0
    assert ctrl.cluster_gap == pytest.approx(8 * 3 * _SPACING, rel=0.05)


def test_clicking_the_same_spot_again_loosens_the_gap_a_step():
    """A repeat click is the slider's move, not a level of its own: the gap
    goes up one notch (4x -> 6x) and the patch grows to bridge blob B."""
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    _click(ctrl)
    assert view.selected == A_IDX

    _click(ctrl)
    assert ctrl.cluster_gap_factor == 6.0
    assert view.selected == A_IDX | B_IDX


def test_a_repeat_click_keeps_its_seed_and_does_not_pick_again():
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    picks = []
    view.pick_point = lambda xy: picks.append(xy) or 0

    _click(ctrl)
    _click(ctrl)
    _click(ctrl)
    assert len(picks) == 1  # only the first click picked a point
    assert ctrl.cluster_gap_factor == 8.0  # 4 -> 6 -> 8


def test_a_repeat_click_keeps_what_a_shift_click_was_added_to():
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    view.selected = {OTHER_IDX}
    _click(ctrl, additive=True)
    _click(ctrl)  # a plain repeat click still just loosens
    assert view.selected == A_IDX | B_IDX | {OTHER_IDX}


def test_a_click_somewhere_else_starts_afresh_instead_of_loosening():
    ctrl, view = _controller()
    stepped = []
    ctrl.cluster.on_repeat = lambda: stepped.append(True)
    _click(ctrl)
    _click(ctrl, xy=(500.0, 500.0))  # far past CHAIN_PIXELS
    assert stepped == []
    assert ctrl.cluster_gap_factor == 4.0


def test_reset_goes_back_to_the_default_and_keeps_the_selection():
    """Switching Cluster off: the gap resets, but the patch the clicks grew
    stays selected for A/N/U -- it must not be re-run at 1x and shrunk."""
    ctrl, view = _controller()
    ctrl.cluster.on_repeat = _step_looser(ctrl)
    _click(ctrl)
    _click(ctrl)  # 4x -> 6x: both blobs
    assert view.selected == A_IDX | B_IDX

    ctrl.reset_cluster_gap()
    assert ctrl.cluster_gap_factor == DEFAULT_CLUSTER_GAP_FACTOR
    assert view.selected == A_IDX | B_IDX


def test_reset_ends_the_click_sequence():
    """After a reset the same spot is a fresh first click at the default,
    not a repeat that loosens from where the old sequence left off."""
    ctrl, view = _controller()
    stepped = []
    ctrl.cluster.on_repeat = lambda: stepped.append(True)
    _click(ctrl)
    ctrl.reset_cluster_gap()
    _click(ctrl)
    assert stepped == []
    assert len(view.selected) < _BLOB  # a 1x seed, not blob A


def test_gap_steps_are_ordered_and_include_the_default():
    assert list(CLUSTER_GAP_FACTORS) == sorted(CLUSTER_GAP_FACTORS)
    assert DEFAULT_CLUSTER_GAP_FACTOR in CLUSTER_GAP_FACTORS


def test_the_component_cache_survives_a_gap_change(monkeypatch):
    """Stepping the gap up and back must not recompute a labelling it has
    already built: the workflow is "click again to loosen", and each step
    used to wipe the cache and redo the whole tree's blobs."""
    from segfix import analysis

    ctrl, _ = _controller()
    calls = []
    real = analysis.connected_components_within

    def counted(coords, eps):
        calls.append(float(eps))
        return real(coords, eps)

    monkeypatch.setattr(analysis, "connected_components_within", counted)

    _click(ctrl)
    assert len(calls) == 1
    _step_looser(ctrl)()                    # a new gap: one more computation
    assert len(calls) == 2
    ctrl.set_cluster_gap_factor(4.0)        # back to the gap of the first click
    assert len(calls) == 2                  # answered from the cache
    assert len(ctrl._cluster_cc) == 2       # one entry per (tree, gap)


def test_an_edit_retires_the_component_cache_for_the_tree_it_changed():
    """The other half of the workflow is edits, and the cache has to notice.

    At 6x a click on tree 1 takes both blobs. Split blob B off as its own
    tree and click tree 1 again: the cached entry describes the tree as it
    was, and handing it back selected blob B too -- points that now belong
    to tree 3, which the next A would have pulled straight back in.
    """
    from segfix import operations as ops

    ctrl, view = _controller(factor=6.0)
    _click(ctrl)
    assert view.selected == A_IDX | B_IDX

    ops.create_new(ctrl.cloud, np.array(sorted(B_IDX)))
    view.select([])
    ctrl.cluster.end_chain()                # a fresh click, not a repeat
    _click(ctrl)
    assert view.selected == A_IDX           # blob B is somebody else's now


def test_an_undo_brings_the_cached_labelling_back():
    """Undo restores the very label array an entry was built for, so the
    entry is good again -- the check is against the membership, not a
    "something changed" flag that could only ever throw the entry away."""
    from segfix import operations as ops

    ctrl, view = _controller(factor=6.0)
    _click(ctrl)
    ops.create_new(ctrl.cloud, np.array(sorted(B_IDX)))
    ctrl.cloud.undo()
    view.select([])
    ctrl.cluster.end_chain()
    _click(ctrl)
    assert view.selected == A_IDX | B_IDX


def test_an_edit_leaves_other_trees_cached(monkeypatch):
    """Only the edited tree's entries are retired: a plot is a queue of
    trees and an edit on one must not make the next click on another pay
    for the whole labelling again."""
    from segfix import analysis
    from segfix import operations as ops

    ctrl, view = _controller(factor=6.0)
    _click(ctrl)                            # caches (tree 1, 6x)

    calls = []
    real = analysis.connected_components_within
    monkeypatch.setattr(
        analysis, "connected_components_within",
        lambda coords, eps: (calls.append(float(eps)), real(coords, eps))[1],
    )
    # Edit a *different* tree: tree 2's lone point becomes noise.
    ops.mark_noise(ctrl.cloud, np.array([OTHER_IDX]))
    view.select([])
    ctrl.cluster.end_chain()
    _click(ctrl)                            # tree 1 again, untouched
    assert calls == []                      # still answered from the cache


# -- the gap-ladder prefetch --------------------------------------------------
def _prefetching(factor=4.0):
    """A controller with the prefetch on, and a way to wait for it."""
    ctrl, view = _controller(factor=factor)
    ctrl.cluster_prefetch = True
    return ctrl, view


def _joined(ctrl, timeout=10.0):
    assert ctrl._prefetch, "a click should have started a prefetch"
    assert ctrl.join_cluster_prefetch(timeout), "prefetch did not finish"


def test_a_click_warms_the_rest_of_the_gap_ladder():
    """One click, and every gap the next click could step to is already
    computed -- which is the whole point: the steps stop costing anything."""
    ctrl, _ = _prefetching()
    _click(ctrl)
    _joined(ctrl)

    warmed = {gap for label, gap in ctrl._cluster_cc if label == 1}
    assert len(warmed) == len(CLUSTER_GAP_FACTORS)


def test_the_warmed_ladder_is_what_computing_it_would_have_given():
    """The worker is handed a snapshot and must come back with exactly the
    answer the synchronous path does, gap for gap."""
    hot, view = _prefetching()
    _click(hot)
    _joined(hot)

    cold, _ = _controller(factor=4.0)        # prefetch off
    _click(cold)                             # measures the spacing
    for factor in CLUSTER_GAP_FACTORS:
        _, want = cold._cluster_component(1, cold._cluster_spacing * factor)
        _, got = hot._cluster_component(1, hot._cluster_spacing * factor)
        assert np.array_equal(got, want), f"differs at {factor}x"


def test_the_prefetch_steps_loosen_first():
    """Order matters: "click again to loosen" is the move, so the gaps above
    the click are the ones that must already be there."""
    ctrl, _ = _prefetching(factor=1.0)
    _click(ctrl)
    _joined(ctrl)
    assert (1, ctrl._cluster_spacing * 1.5) in ctrl._cluster_cc


def test_a_stepped_gap_is_answered_from_the_warmed_cache(monkeypatch):
    """What the user feels: after the first click, stepping the gap does no
    work on the calling thread at all."""
    from segfix import analysis

    ctrl, _ = _prefetching(factor=1.0)
    _click(ctrl)
    _joined(ctrl)

    def refuse(coords, eps):
        raise AssertionError(f"recomputed {eps} on the GUI thread")

    monkeypatch.setattr(analysis, "connected_components_within", refuse)
    for _ in range(len(CLUSTER_GAP_FACTORS) - 1):
        _step_looser(ctrl)()                 # every step a cache hit


def test_an_edit_during_the_prefetch_is_not_overruled():
    """The worker computes against a snapshot, so an edit that lands while
    it runs leaves it holding a labelling of a tree that no longer exists.
    That entry must not be handed to a click."""
    from segfix import operations as ops

    ctrl, view = _prefetching(factor=6.0)
    _click(ctrl)
    _joined(ctrl)                            # ladder warmed for tree 1 as it was

    ops.create_new(ctrl.cloud, np.array(sorted(B_IDX)))
    view.select([])
    ctrl.cluster.end_chain()
    _click(ctrl)
    assert view.selected == A_IDX            # not the warmed pre-edit blob


def test_a_new_cloud_stands_the_prefetch_down():
    ctrl, _ = _prefetching()
    _click(ctrl)
    coords = np.vstack([_A, _B, _OTHER]).astype(np.float32)
    ctrl.set_cloud(PointCloud(coords=coords, labels=np.ones(len(coords), np.int32)))
    assert ctrl._prefetch_label is None      # the next click starts afresh
    assert ctrl.join_cluster_prefetch(10.0)  # and these wind down
    assert ctrl._cluster_cc == {}            # nothing from the old points


def test_the_cache_stops_growing():
    """The prefetch fills nine entries a tree, so the cache needs a ceiling
    or a plot walked tree by tree climbs without limit."""
    from segfix.widgets import CLUSTER_CACHE_ENTRIES

    ctrl, _ = _controller()
    for label in range(200):                 # far more trees than the cap
        for factor in CLUSTER_GAP_FACTORS:
            ctrl._cache_cluster((label, factor), (np.empty(0, np.int64),) * 2)
    assert len(ctrl._cluster_cc) == CLUSTER_CACHE_ENTRIES


def test_a_superseded_prefetch_winds_down():
    """Clicking another tree supersedes the batch on the last one. The old
    workers check between gaps, so they linger for one labelling and then
    go -- they must not pile up over a review."""
    import threading

    ctrl, view = _prefetching()
    _click(ctrl)
    ctrl._start_cluster_prefetch(2, np.array([OTHER_IDX]))   # too small to run
    ctrl._start_cluster_prefetch(1, np.flatnonzero(ctrl.cloud.labels == 1))
    assert ctrl.join_cluster_prefetch(10.0)

    ctrl._stop_cluster_prefetch()
    for _ in range(200):
        if not [t for t in threading.enumerate() if "prefetch" in t.name]:
            break
        time.sleep(0.05)
    assert not [t for t in threading.enumerate() if "prefetch" in t.name]


def test_a_warm_ladder_starts_no_more_workers():
    """Every gap step calls back into the prefetch. Once the ladder is warm
    there is nothing for it to do, and it must not keep spawning threads to
    discover that."""
    ctrl, _ = _prefetching(factor=1.0)
    _click(ctrl)
    _joined(ctrl)

    started = ctrl._prefetch
    for _ in range(len(CLUSTER_GAP_FACTORS) - 1):
        _step_looser(ctrl)()
    assert ctrl._prefetch is started          # the same finished batch
