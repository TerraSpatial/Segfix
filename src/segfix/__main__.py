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

"""``python -m segfix``.

The same :func:`segfix.app.main` the ``segfix`` console script runs, reached
through the interpreter instead. Useful where that script is not on PATH —
a pip ``--user`` install puts it somewhere Windows usually has not been told
about — and on a machine that declines to run an executable it does not
recognise, where ``pythonw.exe -m segfix`` gets there without one.
"""

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
