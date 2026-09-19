# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows build.

    pyinstaller packaging/segfix.spec --noconfirm

Produces ``dist/segfix/`` — a folder build, not a one-file exe. One-file
unpacks the whole of Qt, scipy and numpy to a temp folder on every launch,
which on a cold start is slower than the point cloud it is about to open;
the folder build starts immediately and is what the installer lays down
under Program Files anyway.

Three things need helping along:

* **vispy** picks its backend by importing it at runtime, so nothing static
  sees ``vispy.app.backends._pyqt6`` and it has to be named. Its GLSL
  shaders are data files, not modules, so they need collecting too.
* **laspy** loads its LAZ backend the same way — ``lazrs`` is the one the
  project depends on.
* **segfix's own metadata** is what ``importlib.metadata.version("segfix")``
  reads for the version in the title bar; without it the app calls itself
  0.0.0.dev0.
"""

import os

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Relative paths in a spec resolve against the directory PyInstaller was
# invoked from, not the spec's own — so `../src` means different things when
# run from the repo root and from packaging/. SPECPATH is the spec's folder,
# injected by PyInstaller, and anchoring to it makes the build work from
# either.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821

datas = collect_data_files("vispy") + copy_metadata("segfix")

hiddenimports = [
    "vispy.app.backends._pyqt6",
    "vispy.glsl",
    "lazrs",
    "scipy.spatial",
    "scipy.sparse.csgraph",
]

# Nothing here is imported by segfix; they arrive as transitive suggestions
# and cost tens of megabytes each.
excludes = [
    "tkinter", "matplotlib", "IPython", "pytest", "pandas", "PIL",
    "PyQt6.QtWebEngineCore", "PyQt6.QtWebEngineWidgets", "PyQt6.QtBluetooth",
    "PyQt6.QtMultimedia", "PyQt6.QtQuick", "PyQt6.QtQml", "PyQt6.Qt3DCore",
]

a = Analysis(
    [os.path.join(SPECPATH, "segfix_launcher.py")],  # noqa: F821
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="segfix",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # a GUI app: no console window behind it
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(ROOT, "assets", "icons", "segfix.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="segfix",
)
