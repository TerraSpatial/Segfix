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
