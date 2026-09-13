"""The Windows update helper: wait for segfix to close, tidy up, run pip.

Runs everywhere: the waiting and the tidy-up are plain file and process
checks, and the "update commands" here are harmless Python one-liners.
"""

from __future__ import annotations

import os
import subprocess
import sys

from segfix import _update_helper as helper


def test_a_running_process_is_seen_and_an_exited_one_is_not():
    assert helper.process_running(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not helper.process_running(child.pid)
    assert helper.wait_for_exit(child.pid, timeout=1)


def test_waiting_for_a_process_that_stays_up_times_out():
    assert not helper.wait_for_exit(os.getpid(), timeout=0.3, poll=0.05)


def test_free_and_missing_files_count_as_writable(tmp_path):
    exe = tmp_path / "segfix.exe"
    exe.write_bytes(b"MZ")
    assert helper.wait_until_writable([exe, tmp_path / "gone.exe"], timeout=1)
    assert exe.read_bytes() == b"MZ"  # checked, not modified


def test_only_pips_segfix_stash_is_removed(tmp_path):
    (tmp_path / "~egfix").mkdir()
    (tmp_path / "~egfix" / "app.py").write_text("x")
    (tmp_path / "~egfix-1.0.1.dist-info").mkdir()
    (tmp_path / "segfix").mkdir()
    (tmp_path / "~umpy").mkdir()  # another package's stash: not ours to touch
    removed = helper.remove_leftovers([tmp_path])
    assert removed == ["~egfix", "~egfix-1.0.1.dist-info"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["segfix", "~umpy"]


def _job(tmp_path, *codes):
    segfix = subprocess.Popen([sys.executable, "-c", "pass"])
    segfix.wait()  # "segfix" has already closed
    return {
        "pid": segfix.pid,
        "commands": [
            [[sys.executable, "-c",
              f"open('ran_{i}', 'w').close(); raise SystemExit({code})"],
             str(tmp_path)]
            for i, code in enumerate(codes)
        ],
        "locked": [],
        "site_dirs": [],
        "done_message": "segfix 9.9.9 is installed.",
    }


def test_the_commands_run_in_order_once_segfix_has_gone(tmp_path, capsys):
    assert helper.run(_job(tmp_path, 0, 0), exit_timeout=5) == 0
    assert (tmp_path / "ran_0").exists() and (tmp_path / "ran_1").exists()
    assert "segfix 9.9.9 is installed." in capsys.readouterr().out


def test_a_failing_command_stops_the_update(tmp_path, capsys):
    assert helper.run(_job(tmp_path, 3, 0), exit_timeout=5) == 3
    assert not (tmp_path / "ran_1").exists()
    out = capsys.readouterr().out
    assert "failed (exit code 3)" in out and "is installed" not in out
