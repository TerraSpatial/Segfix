"""File > Open Project restarts segfix on the chosen project.

Issue #3: on Windows it closed the window and reopened nothing, with

    File "<string>", line 1
        import
        ^
    SyntaxError: invalid syntax

``os.execv`` passes its arguments to the new process unquoted on Windows,
so ``-c "import sys; from segfix.app import main; ..."`` arrived as
``-c import``. These pin the fix: a quoting Popen there, ``execv`` only
where it really replaces the process, and a frozen build relaunched as
itself.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from segfix import app, update


@pytest.fixture
def argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["segfix", "--point-size", "4"])


def test_an_interpreter_is_told_to_run_segfix(argv, monkeypatch):
    monkeypatch.setattr(update, "is_frozen", lambda: False)
    assert app.relaunch_command() == [
        sys.executable, "-c", app._RELAUNCH_CODE, "--point-size", "4",
    ]


def test_a_frozen_build_is_relaunched_as_itself(argv, monkeypatch):
    """There sys.executable *is* segfix.exe, which would choke on -c."""
    monkeypatch.setattr(update, "is_frozen", lambda: True)
    monkeypatch.setattr(sys, "executable", r"C:\\Program Files\\segfix\\segfix.exe")
    assert app.relaunch_command() == [
        r"C:\\Program Files\\segfix\\segfix.exe", "--point-size", "4",
    ]


def test_the_relaunch_code_really_starts_segfix():
    """Run it for real (with --help, so no window): a typo in the -c string
    would be this bug all over again, on every platform."""
    done = subprocess.run(
        [sys.executable, "-c", app._RELAUNCH_CODE, "--help"],
        capture_output=True, text=True, timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "usage: segfix" in done.stdout


def test_what_went_wrong_on_windows():
    """The code has spaces in it. Joined unquoted — what os.execv does on
    Windows — the new Python sees ``-c import``; quoted per argument, as
    Popen does, it sees the whole line."""
    cmd = [sys.executable, "-c", app._RELAUNCH_CODE]
    assert " ".join(cmd).split()[1:3] == ["-c", "import"]   # the bug
    assert f'"{app._RELAUNCH_CODE}"' in subprocess.list2cmdline(cmd)


def test_windows_starts_a_new_segfix_and_quits_this_one(argv, monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(update, "is_frozen", lambda: False)
    started, quit_called = [], []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    monkeypatch.setattr(os, "execv",
                        lambda *a: pytest.fail("no execv on Windows"))

    class _App:
        def quit(self):
            quit_called.append(True)

    from qtpy.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: _App()))
    app._relaunch()

    # A list, not a joined string: Popen quotes each argument itself.
    assert started == [app.relaunch_command()]
    assert quit_called == [True]


def test_elsewhere_the_process_is_replaced_in_place(argv, monkeypatch):
    import os

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(update, "is_frozen", lambda: False)
    replaced = []
    monkeypatch.setattr(os, "execv", lambda path, args: replaced.append((path, args)))
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: pytest.fail("execv replaces in place"))
    app._relaunch()
    cmd = app.relaunch_command()
    assert replaced == [(cmd[0], cmd)]
