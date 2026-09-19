"""ProgressWindow: the bar the long open and save drive.

Like the other Qt tests here, this runs on PyQt6's bundled "offscreen"
platform plugin, with no real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.progress_ui import ProgressWindow, progress_window  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def test_report_moves_the_bar_and_names_the_stage():
    win = ProgressWindow("Opening cloud", "plot.las")
    win.show()

    win.report("Downsampling…", 0.25)
    assert win.stage_label.text() == "Downsampling…"
    quarter = win.bar.value()
    assert quarter == pytest.approx(win.bar.maximum() * 0.25, abs=1)

    win.report("Indexing trees…", 0.8)
    assert win.bar.value() > quarter
    win.close()


def test_report_clamps_a_fraction_outside_the_bar():
    """A phase weight that doesn't quite add up must not throw or wrap the
    bar round to empty."""
    win = ProgressWindow("Opening cloud")
    win.report("over", 1.4)
    assert win.bar.value() == win.bar.maximum()
    win.report("under", -0.2)
    assert win.bar.value() == win.bar.minimum()
    win.close()


def test_pause_hides_the_window_so_a_prompt_can_be_answered():
    win = ProgressWindow("Opening cloud")
    win.show()
    assert win.isVisible()

    win.pause()
    assert not win.isVisible()
    win.resume()
    assert win.isVisible()
    win.close()


def test_context_manager_closes_the_window_even_when_the_work_raises():
    """A failed load pops an error dialog; the bar must not still be sitting
    in front of it."""
    captured = {}
    with pytest.raises(ValueError):
        with progress_window(None, "Opening cloud", "broken.las") as win:
            captured["win"] = win
            win.report("Reading coordinates…", 0.1)
            raise ValueError("bad file")

    assert not captured["win"].isVisible()


def test_an_import_drives_a_real_window_through_to_full(tmp_path):
    """The end-to-end shape of a first-time open: create_workspace's report
    is handed straight to a live ProgressWindow, so the two signatures have
    to match and the bar has to land on full."""
    from segfix import workspace

    source = tmp_path / "plot.ply"
    source.write_bytes(b"z" * (workspace._COPY_CHUNK * 2 + 5))

    stages = []
    with progress_window(None, "Importing cloud", source.name) as win:
        original_report = win.report

        def spy(message, fraction):
            original_report(message, fraction)
            stages.append((win.stage_label.text(), win.bar.value()))

        win.report = spy
        workspace.create_workspace(source, tmp_path / "proj", report=win.report)

    assert [text for text, _ in stages] == ["Copying"] * 3
    values = [value for _, value in stages]
    assert values[0] < win.bar.maximum()
    assert values[-1] == win.bar.maximum()
    assert values == sorted(values)


# -- run_with_progress: the work on a worker thread --------------------------
def test_the_work_runs_and_its_return_value_comes_back():
    from segfix.progress_ui import run_with_progress

    result = run_with_progress(
        None, "Working", "detail", lambda report, ask: 6 * 7
    )
    assert result == 42


def test_reports_from_the_worker_reach_the_window():
    from segfix.progress_ui import run_with_progress

    seen = []
    real = ProgressWindow.show_progress

    def spy(self, message, fraction):
        real(self, message, fraction)
        seen.append((self.stage_label.text(), self.bar.value()))

    ProgressWindow.show_progress = spy
    try:
        def work(report, ask):
            for i, name in enumerate(("Reading", "Indexing", "Done"), 1):
                report(name, i / 3)
            return None
        run_with_progress(None, "Working", "detail", work)
    finally:
        ProgressWindow.show_progress = real

    assert [text for text, _ in seen] == ["Reading", "Indexing", "Done"]
    assert [value for _, value in seen] == sorted(value for _, value in seen)


def test_the_gui_thread_keeps_running_while_the_worker_is_busy():
    """The whole point: Windows calls a window "(Not Responding)" when its
    message queue goes unpumped for about five seconds, so the event loop
    has to keep turning over for the length of the work."""
    import time

    from qtpy.QtCore import QTimer

    from segfix.progress_ui import run_with_progress

    ticks = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    try:
        run_with_progress(
            None, "Working", "detail", lambda report, ask: time.sleep(0.5)
        )
    finally:
        timer.stop()

    # A frozen GUI thread would have delivered none of these.
    assert len(ticks) > 10


def test_a_question_from_the_worker_is_answered_on_the_gui_thread():
    import threading

    from segfix.progress_ui import run_with_progress

    gui_thread = threading.current_thread()
    ran_on = {}

    def dialog(value):
        ran_on["thread"] = threading.current_thread()
        return value * 2

    def work(report, ask):
        ran_on["worker"] = threading.current_thread()
        return ask(dialog, 21)

    assert run_with_progress(None, "Working", "", work) == 42
    assert ran_on["thread"] is gui_thread
    assert ran_on["worker"] is not gui_thread


def test_the_window_steps_aside_while_a_question_is_up():
    from segfix.progress_ui import run_with_progress

    visible = []
    real = ProgressWindow.pause

    def work(report, ask):
        return ask(lambda: None)

    def dialog_probe(self):
        visible.append(self.isVisible())
        real(self)

    ProgressWindow.pause = dialog_probe
    try:
        run_with_progress(None, "Working", "", work)
    finally:
        ProgressWindow.pause = real
    assert visible == [True]  # shown before the question, hidden for it


def test_an_error_in_the_work_is_raised_on_the_gui_thread():
    from segfix.progress_ui import run_with_progress

    def work(report, ask):
        raise ValueError("bad cloud")

    with pytest.raises(ValueError, match="bad cloud"):
        run_with_progress(None, "Working", "", work)


def test_an_error_from_a_question_reaches_the_worker():
    from segfix.progress_ui import run_with_progress

    def boom():
        raise KeyError("no dialog")

    def work(report, ask):
        try:
            ask(boom)
        except KeyError:
            return "worker saw it"
        return "swallowed"

    assert run_with_progress(None, "Working", "", work) == "worker saw it"


def test_the_window_closes_even_when_the_work_raises():
    from segfix.progress_ui import run_with_progress

    alive = []
    real = ProgressWindow.close

    def spy(self):
        alive.append(self)
        return real(self)

    ProgressWindow.close = spy
    try:
        with pytest.raises(RuntimeError):
            run_with_progress(
                None, "Working", "",
                lambda report, ask: (_ for _ in ()).throw(RuntimeError("x")),
            )
    finally:
        ProgressWindow.close = real
    assert len(alive) == 1 and not alive[0].isVisible()
