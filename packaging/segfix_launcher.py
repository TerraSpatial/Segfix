"""Entry point for the frozen Windows build.

PyInstaller freezes a script, not a console-script entry point, so this is
the ``segfix = segfix.app:main`` line from pyproject.toml spelled out.
"""

import sys

from segfix.app import main

if __name__ == "__main__":
    sys.exit(main())
