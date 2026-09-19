"""``python -m segfix``.

The ``segfix`` console script pip generates is a small .exe, and a locked-down
Windows install may refuse to run one it has not seen before. Going through
the interpreter instead launches the same :func:`segfix.app.main` without any
new binary — which is what the shortcut the PowerShell installer creates
points at (``pythonw.exe -m segfix``).
"""

import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
