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

"""The startup dialog's Update button on Windows: hand the install to the
helper and close segfix, rather than running pip against its own locked
files. Offscreen Qt; the platform check, the prompt and the helper launch
are faked."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")

from qtpy.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

from segfix import startup_ui, update  # noqa: E402
from segfix.update import UpdateStatus  # noqa: E402


@pytest.fixture
def dialog(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(update, "check_for_update", lambda: None)
    monkeypatch.setattr(update, "must_close_to_update", lambda: True)
    monkeypatch.setattr(
        update, "apply_update",
        lambda status: pytest.fail("pip must not run while segfix is open"),
    )
    app = QApplication.instance() or QApplication([])
    quits = []
    monkeypatch.setattr(app, "quit", lambda: quits.append(True))
    dlg = startup_ui.StartupDialog()
    dlg._update_status = UpdateStatus(source="pypi", latest="1.0.3")
    yield dlg, quits
    dlg.done(QDialog.Rejected)


def _answer(monkeypatch, button):
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: button)


def test_update_hands_over_to_the_helper_and_closes_segfix(dialog, monkeypatch):
    dlg, quits = dialog
    scheduled = []
    monkeypatch.setattr(update, "schedule_update_after_exit", scheduled.append)
    _answer(monkeypatch, QMessageBox.StandardButton.Ok)
    dlg._apply_update()
    assert scheduled == [dlg._update_status]
    assert dlg.result() == QDialog.Rejected and quits == [True]


def test_cancel_leaves_segfix_open(dialog, monkeypatch):
    dlg, quits = dialog
    monkeypatch.setattr(
        update, "schedule_update_after_exit",
        lambda status: pytest.fail("cancelled: nothing should be scheduled"),
    )
    _answer(monkeypatch, QMessageBox.StandardButton.Cancel)
    dlg._apply_update()
    assert quits == []
