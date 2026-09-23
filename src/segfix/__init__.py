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

"""segfix — fix instance segmentation of tree point clouds."""

from importlib.metadata import PackageNotFoundError, version

from .model import NOISE, UNASSIGNED, PointCloud

__all__ = ["PointCloud", "UNASSIGNED", "NOISE"]

try:
    # Single source of truth is pyproject.toml; read it back off the
    # installed distribution rather than repeating the number here, where
    # it silently goes stale the first time a release bumps only one of them.
    __version__ = version("segfix")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"
