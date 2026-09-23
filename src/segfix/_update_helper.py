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

"""Finish a segfix update once segfix itself has closed (Windows).

Windows won't let pip replace ``Scripts\\segfix.exe`` while segfix is running
from it: the upgrade fails with ``WinError 32`` part way through and leaves a
half-removed copy behind (pip's ``~egfix`` stash folders in site-packages). So
on Windows the Update button doesn't run pip itself. It copies this script to
a temporary folder, starts it in its own console window, and closes segfix;
this script waits for segfix to exit, then runs the update where the user can
watch it.

Standard library only, and run from its temporary copy: it must not import
the package it is about to replace.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LEFTOVER_PREFIX = "~egfix"  # pip's stash of a half-uninstalled "segfix"


def process_running(pid: int) -> bool:
    """Whether process ``pid`` is still running."""
    if sys.platform == "win32":
        import ctypes

        synchronize, wait_timeout = 0x00100000, 0x102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, pid)
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_exit(pid: int, timeout: float, poll: float = 0.25) -> bool:
    """Wait up to ``timeout`` seconds for ``pid`` to exit; True if it did."""
    end = time.monotonic() + timeout
    while process_running(pid):
        if time.monotonic() >= end:
            return False
        time.sleep(poll)
    return True


def _writable(path: Path) -> bool:
    try:
        with open(path, "ab"):
            return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def wait_until_writable(paths, timeout: float, poll: float = 0.5) -> bool:
    """Wait up to ``timeout`` seconds until every file in ``paths`` can be
    opened for writing; True once they all can.

    Python exiting isn't quite enough on its own: ``segfix.exe`` is a small
    launcher that starts Python and holds its own file open until Python has
    gone, so this waits for the file itself.
    """
    pending = [Path(p) for p in paths]
    end = time.monotonic() + timeout
    while True:
        pending = [p for p in pending if not _writable(p)]
        if not pending:
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(poll)


def remove_leftovers(site_dirs) -> list[str]:
    """Delete pip's ``~egfix*`` stash folders left in site-packages by an
    interrupted upgrade (pip warns "Ignoring invalid distribution ~egfix" for
    each one until they're gone). Returns the names removed."""
    removed = []
    for site in site_dirs:
        for path in sorted(Path(site).glob(LEFTOVER_PREFIX + "*")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            if not path.exists():
                removed.append(path.name)
    return removed


def run(job: dict, exit_timeout: float = 120.0) -> int:
    """Wait for segfix to close, tidy up, then run the update commands in
    order, stopping at the first that fails. Returns its exit code (0 when
    they all succeed)."""
    print("Waiting for segfix to close...", flush=True)
    closed = wait_for_exit(job["pid"], exit_timeout)
    closed = wait_until_writable(job.get("locked", []), exit_timeout) and closed
    if not closed:
        print("segfix still seems to be running; trying anyway.", flush=True)
    for name in remove_leftovers(job.get("site_dirs", [])):
        print(f"Removed a leftover from an earlier update: {name}", flush=True)
    for cmd, cwd in job["commands"]:
        print("\n> " + " ".join(cmd), flush=True)
        code = subprocess.run(cmd, cwd=cwd).returncode
        if code:
            print(f"\nThe update failed (exit code {code}).", flush=True)
            return code
    print("\n" + job["done_message"], flush=True)
    return 0


def main(argv: list[str]) -> int:
    job_file = Path(argv[0])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    try:
        code = run(job)
    finally:
        # The temporary folder holding this script and its job file.
        shutil.rmtree(job_file.parent, ignore_errors=True)
    try:
        input("\nPress Enter to close this window.")
    except EOFError:
        pass
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
