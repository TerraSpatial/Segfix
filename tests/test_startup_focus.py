"""Enter opens the project you were last working on.

The startup dialog is unskippable — every session goes through it — and the
overwhelmingly common answer is "reopen the top one". That should cost one
key, which means the list holds focus and Open is the button Enter presses.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication, QDialog  # noqa: E402

from segfix import registry, startup_ui, update  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def config(tmp_path, monkeypatch):
    """An empty config home, so the registry starts with nothing in it."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(update, "check_for_update", lambda: None)
    return tmp_path


def _dialog():
    dlg = startup_ui.StartupDialog()
    dlg.show()
    QApplication.processEvents()
    return dlg


def test_the_recent_list_has_focus_and_open_is_the_default(config):
    cloud = config / "plot.las"
    cloud.write_bytes(b"")
    registry.add_entry(str(cloud), kind="file")

    dlg = _dialog()
    try:
        # focusWidget(), not hasFocus(): the offscreen platform never makes
        # the window active, so nothing in it reports holding focus.
        assert dlg.focusWidget() is dlg.list
        assert dlg.open_btn.isDefault() and dlg.open_btn.isEnabled()
        assert dlg.list.currentRow() == 0  # the most recent, preselected
    finally:
        dlg.done(QDialog.Rejected)


def test_enter_opens_the_preselected_project(config):
    cloud = config / "plot.las"
    cloud.write_bytes(b"")
    registry.add_entry(str(cloud), kind="file")

    dlg = _dialog()
    QTest.keyClick(dlg, Qt.Key.Key_Return)

    assert dlg.result() == QDialog.Accepted
    assert dlg.open_path == str(cloud)


def test_enter_opens_whichever_row_the_arrows_moved_to(config):
    older, newer = config / "older.las", config / "newer.las"
    for path in (older, newer):
        path.write_bytes(b"")
        registry.add_entry(str(path), kind="file")

    dlg = _dialog()
    assert dlg.list.count() == 2
    QTest.keyClick(dlg.list, Qt.Key.Key_Down)  # to the older project
    QTest.keyClick(dlg, Qt.Key.Key_Return)

    assert dlg.result() == QDialog.Accepted
    assert dlg.open_path == str(older)


def test_with_nothing_to_reopen_new_project_takes_the_key(config):
    """An empty list has nothing for Open to do, so it neither holds the
    key nor pretends to be pressable."""
    dlg = _dialog()
    try:
        assert dlg.list.count() == 0
        assert not dlg.open_btn.isEnabled()
        assert dlg.new_btn.isDefault()
        assert dlg.focusWidget() is dlg.new_btn
    finally:
        dlg.done(QDialog.Rejected)


def test_no_other_button_answers_enter(config):
    """Whichever button last had focus would otherwise answer the key."""
    cloud = config / "plot.las"
    cloud.write_bytes(b"")
    registry.add_entry(str(cloud), kind="file")

    dlg = _dialog()
    try:
        for button in (dlg.new_btn, dlg.cancel_btn, dlg.update_btn):
            assert not button.autoDefault(), button.text()
    finally:
        dlg.done(QDialog.Rejected)
