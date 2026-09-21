"""File > Open Project (issue #3).

It crashed on every use from v0.4.0 on: the handler called the loaded
cloud's ``can_undo`` — a property — as a method, and PyQt answers an
exception in a slot by aborting, so the window just vanished. Behind that
sat a quieter bug: the loaded tree's undo stack only covers the tree on
screen, so edits made on trees visited earlier were never offered a save.
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from segfix import app, startup_ui  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _panel(can_undo):
    cloud = types.SimpleNamespace(can_undo=can_undo)  # a property: a bool
    return types.SimpleNamespace(c=types.SimpleNamespace(cloud=cloud),
                                 on_save=lambda: None)


@pytest.fixture
def asked(monkeypatch):
    """Record the save prompt, and cancel the project picker so nothing is
    re-executed."""
    seen = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: seen.append(a[2]) or QMessageBox.StandardButton.Cancel,
    )
    monkeypatch.setattr(startup_ui, "choose_project", lambda: None)
    monkeypatch.setattr(os, "execv",
                        lambda *a: pytest.fail("cancelled: nothing to exec"))
    return seen


@pytest.mark.parametrize("can_undo", [True, False])
def test_open_project_no_longer_raises(asked, can_undo):
    """The bug itself: this raised TypeError on the first line, every time."""
    app._open_project(None, _panel(can_undo))
    assert bool(asked) is can_undo


def test_edits_on_an_earlier_tree_still_offer_a_save(asked):
    """The loaded tree has nothing to undo, but the session does."""
    scene = types.SimpleNamespace(has_unsaved_edits=lambda: True)
    app._open_project(None, _panel(can_undo=False), scene)
    assert asked == ["Save changes to the current project first?"]


def test_nothing_unsaved_asks_nothing(asked):
    scene = types.SimpleNamespace(has_unsaved_edits=lambda: False)
    app._open_project(None, _panel(can_undo=True), scene)
    assert asked == []


def test_the_scene_knows_about_every_visited_tree(tmp_path):
    """Edit one tree, move to another: the loaded cloud's undo stack is
    empty again, but the edit is still unsaved — until it is saved."""
    from segfix import io
    from segfix.model import PointCloud
    from segfix.scene_ui import SceneController
    from segfix.treecatalog import open_catalog

    rng = np.random.default_rng(0)
    coords = np.concatenate([rng.random((50, 3)) + [k * 10, 0, 0] for k in range(2)])
    labels = np.repeat([1, 2], 50).astype(np.int32)
    path = str(tmp_path / "plot.ply")
    io.save(PointCloud(coords=coords.astype(np.float32), labels=labels), path)

    class _Seg:
        cloud = None
        on_save_override = None

        def set_cloud(self, cloud, focus=None):
            self.cloud = cloud

    view = types.SimpleNamespace(load_cloud=lambda *a, **k: None,
                                 reset_view=lambda: None,
                                 native=types.SimpleNamespace(window=lambda: None))
    scene = SceneController(view, open_catalog(path), _Seg())
    scene.load_tree(1)
    assert not scene.has_unsaved_edits()

    scene.seg.cloud.labels[:10] = -1          # an edit on tree 1
    scene.load_tree(2)                        # ...and move on
    assert scene.seg.cloud.can_undo is False  # what the old check looked at
    assert scene.has_unsaved_edits()          # what actually matters

    scene.catalog.save()
    assert not scene.has_unsaved_edits()


# -- the safety net ------------------------------------------------------------
def test_an_uncaught_error_is_shown_not_fatal(monkeypatch, capsys):
    """PyQt aborts on an exception in a slot unless sys.excepthook is ours;
    with it, the user sees what happened and the session survives."""
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append(
        (self.text(), self.detailedText())))
    try:
        raise TypeError("'bool' object is not callable")
    except TypeError as exc:
        app._report_uncaught(type(exc), exc, exc.__traceback__)

    [(text, details)] = shown
    assert "TypeError" in text and "still running" in text
    assert "Traceback" in details
    assert "TypeError" in capsys.readouterr().err  # and in the terminal


def test_main_installs_the_safety_net(monkeypatch):
    import sys

    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    monkeypatch.setattr(app, "choose_project", None, raising=False)
    from segfix import startup_ui as su

    monkeypatch.setattr(su, "choose_project", lambda: None)  # cancel at once
    monkeypatch.delenv("SEGFIX_OPEN", raising=False)
    assert app.main([]) == 0
    assert sys.excepthook is app._report_uncaught
