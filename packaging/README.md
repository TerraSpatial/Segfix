# Packaging segfix for Windows

Three ways to put segfix on someone else's machine, and what each is for.

| | Needs Python? | Needs admin? | Survives a locked-down machine? |
|---|---|---|---|
| `install-segfix.ps1` | yes, 3.10–3.12 | no | **usually** — adds no new binary |
| `segfix-<v>-setup.exe` | no | no (per-user) | only if unknown .exe files are allowed |
| `segfix-<v>-portable.zip` | no | no | only if unknown .exe files are allowed |

## Why three

A managed Windows laptop commonly refuses to run a binary it has not seen
before, and an unsigned installer trips SmartScreen regardless. Both the
installer and the portable zip are unknown binaries, so on a strict machine
neither will start. The PowerShell route exists for exactly that case: it
installs the package into a Python that is already there and trusted, and
makes a Start Menu shortcut to **that interpreter** (`pythonw.exe -m
segfix`). It never introduces an executable Windows has to make a decision
about — which is also why `segfix/__main__.py` exists, so the shortcut need
not use the small `segfix.exe` pip generates in `Scripts\`.

Signing the installer would fix SmartScreen, but needs a code-signing
certificate (paid, with identity verification). The project has none, so the
warning is expected: **More info → Run anyway**.

## Building

```powershell
pip install pyinstaller
pyinstaller packaging\segfix.spec --noconfirm     # -> dist\segfix\
iscc /DAppVersion=1.0.6 packaging\segfix.iss      # -> dist\segfix-1.0.6-setup.exe
```

The spec anchors its paths to its own folder, so it builds the same from the
repo root or from `packaging\`. Inno Setup resolves its paths against the
`.iss` either way.

Regenerate `assets/icons/segfix.ico` (and the rest of the icon set) from the
drawing code with:

```powershell
python scripts\export_icons.py
```

## Verifying

Nobody should have to take "it builds" for "it works", and the maintainer
has no unlocked Windows machine to check on. `.github/workflows/installer.yml`
does it instead, on a clean runner, for every change under `packaging/` and
on every tag:

1. freezes the app and checks the bundle still runs after 20 seconds —
   a windowed PyInstaller build has no stdout and exits 0 whatever it is
   asked, so an exit code proves nothing, but staying alive means PyQt6,
   vispy, scipy and laspy all imported;
2. builds the installer, **installs it silently**, runs the installed copy,
   then **uninstalls it** and checks nothing is left behind;
3. runs `install-segfix.ps1` against a fresh venv and checks the shortcut it
   leaves points at `pythonw.exe -m segfix` and that it launches.

Run it by hand from the Actions tab (workflow_dispatch) without tagging
anything.
