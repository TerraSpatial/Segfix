# Packaging segfix for Windows

Two ways to put segfix on someone else's machine. Both bundle their own
Python, Qt, numpy and scipy, so the recipient needs nothing installed.

| | Needs admin? | What it leaves behind |
|---|---|---|
| `segfix-<v>-setup.exe` | no, per-user by default | Start Menu entry and an uninstaller |
| `segfix-<v>-portable.zip` | no | just the folder you unzipped |

The installer is unsigned, so Windows shows a SmartScreen warning the first
time: **More info → Run anyway**. Silencing it needs a code-signing
certificate (paid, with identity verification), which the project does not
have. The portable zip sidesteps the installer but is still an unsigned
executable, so a strict machine may refuse both — there the fallback is
`pip install segfix` into a Python the user already has.

Qt's software OpenGL fallback (`opengl32sw.dll`) is bundled, which is what
a recipient on a VM or a remote desktop session will end up rendering with.

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
   then **uninstalls it** and checks nothing is left behind.

Run it by hand from the Actions tab (workflow_dispatch) without tagging
anything.
