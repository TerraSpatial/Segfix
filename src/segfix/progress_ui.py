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

"""The progress window the long operations run behind — opening a cloud,
saving one, importing one, exporting trees.

Opening a big plot is tens of seconds of numpy: decoding coordinates and
labels, measuring density, decimating, sorting the label index (about 22
seconds all told for a 39-million-point PLY). Behind a single status-bar
line and a window that painted nothing, that read as a crash.

:func:`run_with_progress` runs the work on a worker thread and leaves the
GUI thread in an event loop, which is what keeps Windows from declaring the
window "(Not Responding)": that verdict is passed on a top-level window
whose message queue has gone unpumped for about five seconds, and every
phase of an open used to be one blocking call on the GUI thread. numpy,
scipy and laspy drop the GIL for the big operations, and CPython hands it
over every few milliseconds regardless, so the bar really does keep moving
rather than merely looking alive.

The work touches no Qt or GL object, which is what makes it safe to move —
except for the prompts an open raises from inside itself (the global-shift
and downsample questions), which ``run_with_progress`` marshals back to the
GUI thread through ``ask``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from qtpy.QtCore import QEventLoop, QObject, Qt, Signal
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QLabel,
    QProgressBar,
    QVBoxLayout,
)

_BAR_STEPS = 1000  # bar resolution; fractions are 0..1


class ProgressWindow(QDialog):
    """Title, a line of detail, and a bar. No cancel button.

    Cancelling is deliberately not offered. The window does stay clickable
    now that the work is on a worker thread, but each phase is still one
    numpy call with nowhere to check a flag, so a Cancel could not act until
    the phase it interrupted had finished of its own accord — a button that
    does nothing for twelve seconds is worse than no button.
    """

    def __init__(self, title: str, detail: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        # No close button: there is nothing sensible to do with a half-built
        # catalog, and the window closes itself the moment the work is done.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.detail_label = QLabel(detail)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: gray;")
        layout.addWidget(self.detail_label)

        self.stage_label = QLabel("Starting…")
        layout.addWidget(self.stage_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, _BAR_STEPS)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.setMinimumWidth(420)

    def show_progress(self, message: str, fraction: float) -> None:
        """Show ``message`` at ``fraction`` (0..1), and leave the painting to
        the event loop.

        What :func:`run_with_progress` connects the worker's reports to: it
        keeps a loop running on the GUI thread, so a queued update is drawn
        without anyone pumping by hand — and pumping from inside a slot that
        loop is already delivering would be re-entering it.
        """
        self.stage_label.setText(message)
        self.bar.setValue(int(max(0.0, min(1.0, fraction)) * _BAR_STEPS))

    def report(self, message: str, fraction: float) -> None:
        """Show ``message`` at ``fraction`` (0..1) and paint it now.

        For a caller blocking the GUI thread itself: control does not come
        back to the event loop until the whole thing is finished, so anything
        merely queued would never be drawn. See :func:`segfix.viewer.busy`,
        which does the same for the status bar. Work that can run on a
        thread should go through :func:`run_with_progress` instead.
        """
        self.show_progress(message, fraction)
        QApplication.processEvents()

    # Called by prompts that have to interrupt the work to ask a question
    # (the global-shift and downsample dialogs both run from inside the
    # load): a progress bar frozen at 10% behind a question looks stuck.
    def pause(self) -> None:
        self.hide()
        QApplication.processEvents()

    def resume(self) -> None:
        self.show()
        QApplication.processEvents()


@contextmanager
def progress_window(parent, title: str, detail: str = ""):
    """Show a :class:`ProgressWindow` for the duration of the block.

    Yields the window itself, whose ``report(message, fraction)`` matches
    :data:`segfix.treecatalog.ProgressFn` — pass it straight to
    ``open_catalog`` or ``save``. Closes on the way out, including when the
    work raises, so a failed load can't leave the bar on screen in front of
    the error dialog.
    """
    win = ProgressWindow(title, detail, parent)
    win.show()
    QApplication.processEvents()
    try:
        yield win
    finally:
        win.close()
        QApplication.processEvents()


class _Call:
    """Something the worker needs run on the GUI thread, and its answer."""

    __slots__ = ("fn", "args", "done", "result", "error")

    def __init__(self, fn, args):
        self.fn = fn
        self.args = args
        self.done = threading.Event()
        self.result = None
        self.error = None


class _Bridge(QObject):
    """Carries the worker's messages to the GUI thread.

    Built on the GUI thread, so every emit from the worker crosses the thread
    boundary as a queued connection and is delivered by the event loop in
    :func:`run_with_progress` — which is the point of the whole arrangement.
    """

    progressed = Signal(str, float)
    called = Signal(object)
    finished = Signal()


def run_with_progress(parent, title: str, detail: str, work):
    """Run ``work(report, ask)`` on a worker thread, behind a progress window.

    ``report(message, fraction)`` moves the bar — the same signature as
    :data:`segfix.treecatalog.ProgressFn`, so it passes straight to
    ``open_catalog``, ``save`` and friends.

    ``ask(fn, *args)`` runs ``fn(*args)`` on the GUI thread and returns what
    it returned, blocking the worker until it does. That is how a question
    raised from inside the work — the global-shift and downsample prompts an
    open asks partway through — gets a real dialog: Qt widgets may only be
    touched from the thread that owns them. The bar hides for the duration,
    since one frozen behind a question looks stuck.

    Returns whatever ``work`` returned. An exception from ``work`` is
    re-raised here, on the GUI thread, so callers keep their ordinary
    ``try``/``except`` around it.
    """
    if QApplication.instance() is None:
        # No application, so no window to keep responsive and no loop to run
        # one in — a headless caller, or a test with the dialogs faked out.
        # Run the work inline: driving a QEventLoop with nothing underneath
        # it aborts the process rather than raising.
        return work(lambda message, fraction: None, lambda fn, *args: fn(*args))

    outcome = {}
    bridge = _Bridge()

    def serve(call: _Call) -> None:
        win.pause()
        try:
            call.result = call.fn(*call.args)
        except BaseException as exc:  # noqa: BLE001 - handed back to the worker
            call.error = exc
        finally:
            win.resume()
            call.done.set()

    def report(message, fraction) -> None:
        bridge.progressed.emit(str(message), float(fraction))

    def ask(fn, *args):
        call = _Call(fn, args)
        bridge.called.emit(call)
        # Queued, so this waits for the GUI thread's loop to get to it.
        call.done.wait()
        if call.error is not None:
            raise call.error
        return call.result

    def run() -> None:
        try:
            outcome["result"] = work(report, ask)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc
        finally:
            bridge.finished.emit()

    with progress_window(parent, title, detail) as win:
        loop = QEventLoop()
        bridge.progressed.connect(win.show_progress)
        bridge.called.connect(serve)
        bridge.finished.connect(loop.quit)
        worker = threading.Thread(target=run, name="segfix-work", daemon=True)
        worker.start()
        # `finished` is posted as an event, so a worker that beats us to this
        # line still wakes the loop rather than stranding it.
        loop.exec()
        worker.join()

    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")
