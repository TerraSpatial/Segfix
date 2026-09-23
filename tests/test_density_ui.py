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

"""DownsampleDialog: the two answers it can give back to the catalog.

Like :mod:`tests.test_shift_ui`, this needs a QApplication, which PyQt6's
bundled "offscreen" platform plugin provides with no real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.density import DEFAULT_VOXEL  # noqa: E402
from segfix.density_ui import DownsampleDialog  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _dialog():
    dlg = DownsampleDialog(0.006, 128_000_000, DEFAULT_VOXEL)
    dlg.show()
    QApplication.processEvents()
    return dlg


def test_downsample_button_returns_the_edited_voxel_size():
    dlg = _dialog()
    assert dlg.spin.value() == pytest.approx(DEFAULT_VOXEL)
    dlg.spin.setValue(0.05)
    dlg.downsample_btn.click()
    assert dlg.voxel() == pytest.approx(0.05)


def test_keep_full_resolution_declines_downsampling():
    dlg = _dialog()
    dlg.keep_btn.click()
    assert dlg.voxel() is None


# -- what the chosen size would keep ------------------------------------------
def test_the_estimate_shows_and_follows_the_voxel_size():
    """A number, not a guess: the point of showing it is that it changes as
    you try sizes, so you can see which one is worth taking."""
    asked = []

    def kept_fraction(voxel):
        asked.append(voxel)
        return {0.03: 0.85, 0.05: 0.42}.get(round(voxel, 3), 0.5)

    dlg = DownsampleDialog(0.017, 480_000_000, 0.03, kept_fraction)
    dlg.show()
    QApplication.processEvents()
    assert asked == [pytest.approx(0.03)]
    assert dlg.estimate.text() == "keeps about 85% of the points (408,000,000)"

    dlg.spin.setValue(0.05)
    assert dlg.estimate.text() == "keeps about 42% of the points (201,600,000)"


def test_no_estimate_is_shown_when_it_cannot_be_measured():
    dlg = DownsampleDialog(0.006, 1_000, DEFAULT_VOXEL, lambda voxel: None)
    dlg.show()
    QApplication.processEvents()
    assert dlg.estimate.text() == ""


def test_a_failing_measurement_never_blocks_the_prompt():
    """The dialog's job is the question, not the measurement."""
    def explode(voxel):
        raise MemoryError("not now")

    dlg = DownsampleDialog(0.006, 1_000, DEFAULT_VOXEL, explode)
    dlg.show()
    QApplication.processEvents()
    assert dlg.estimate.text() == ""
    dlg.downsample_btn.click()
    assert dlg.voxel() == pytest.approx(DEFAULT_VOXEL)


def test_the_dialog_works_without_any_estimate_at_all():
    dlg = _dialog()  # no kept_fraction: the old three-argument shape
    assert dlg.estimate.text() == ""
    dlg.keep_btn.click()
    assert dlg.voxel() is None
