<#
.SYNOPSIS
    Install or update segfix for the current user, without admin rights and
    without introducing a new executable.

.DESCRIPTION
    The .exe installer (segfix-<version>-setup.exe) is the easier route where
    it is allowed. On a managed machine it often is not: an unsigned
    installer trips SmartScreen, and an application allowlist can refuse any
    binary it has not seen — including the small segfix.exe that pip
    generates in Scripts\.

    This takes the other road. It installs segfix into a Python you already
    have, and makes a Start Menu shortcut to that interpreter's pythonw.exe
    with "-m segfix". Nothing new is written that Windows has to decide
    whether to trust: no installer, no service, no registry class, and no
    executable that was not already on the machine and already permitted.

    Everything lands under the current user. Admin is never requested.

.PARAMETER Python
    The interpreter to install into. Defaults to the one running this
    script's conda env if segfix is already there, otherwise whichever
    python.exe is first on PATH.

.PARAMETER Version
    A specific version to pin, e.g. "1.0.6". Defaults to the newest on PyPI.

.PARAMETER Source
    What to hand pip instead of the name "segfix" - a checkout to install
    from ("." in a clone), or a wheel. Mostly for testing a release before
    it is published; -Version is ignored when this is given.

.PARAMETER NoShortcut
    Install the package but do not create the Start Menu shortcut.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File packaging\install-segfix.ps1

.EXAMPLE
    # Into a specific conda environment
    .\install-segfix.ps1 -Python "$env:USERPROFILE\.conda\envs\segfix\python.exe"
#>
[CmdletBinding()]
param(
    [string] $Python,
    [string] $Version,
    [string] $Source,
    [switch] $NoShortcut
)

$ErrorActionPreference = "Stop"

function Find-Python {
    if ($Python) {
        if (-not (Test-Path $Python)) { throw "No interpreter at $Python" }
        return (Resolve-Path $Python).Path
    }
    $candidate = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($candidate) { return $candidate.Source }
    throw @"
No python.exe on PATH. segfix needs a Python 3.10-3.12 to run in; a
miniforge or Miniconda install is the usual one, and neither needs admin.
Pass one explicitly with -Python <path to python.exe>.
"@
}

$exe = Find-Python
Write-Host "Using $exe"

$check = & $exe -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($LASTEXITCODE -ne 0) { throw "Could not run $exe" }
Write-Host "  Python $check"
if ([version]$check -lt [version]"3.10" -or [version]$check -ge [version]"3.13") {
    throw "segfix needs Python 3.10-3.12; this one is $check"
}

$spec = if ($Source) { $Source }
        elseif ($Version) { "segfix==$Version" }
        else { "segfix" }
Write-Host "Installing $spec (user-level, no admin)..."
# --user keeps it out of a shared install; harmless and ignored inside a
# conda env or a venv, which is where most people will point this.
& $exe -m pip install --upgrade --user $spec
if ($LASTEXITCODE -ne 0) {
    Write-Host "  --user was refused; installing into the environment itself"
    & $exe -m pip install --upgrade $spec
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

$installed = & $exe -c "import segfix; print(segfix.__version__)"
Write-Host "segfix $installed is installed."

if ($NoShortcut) { exit 0 }

# pythonw.exe runs it without a console window behind the GUI. It sits
# beside python.exe in every CPython layout worth supporting.
$pythonw = Join-Path (Split-Path $exe) "pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = $exe }

$menu = [Environment]::GetFolderPath("Programs")
$link = Join-Path $menu "segfix.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($link)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "-m segfix"
$shortcut.WorkingDirectory = Split-Path $exe
$shortcut.Description = "Fix instance segmentation of tree point clouds"
$icon = Join-Path $PSScriptRoot "..\assets\icons\segfix.ico"
if (Test-Path $icon) { $shortcut.IconLocation = (Resolve-Path $icon).Path }
$shortcut.Save()

Write-Host ""
Write-Host "Start Menu shortcut created: $link"
Write-Host "It runs: $pythonw -m segfix"
Write-Host ""
Write-Host "To remove: delete that shortcut and run"
Write-Host "  `"$exe`" -m pip uninstall segfix"
