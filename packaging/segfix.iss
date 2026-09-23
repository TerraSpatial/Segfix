; segfix — fix instance segmentation of tree point clouds.
; Copyright (C) 2026 Tim Devereux, The University of Queensland
;
; This program is free software: you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation, either version 3 of the License, or
; (at your option) any later version.
;
; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.
;
; You should have received a copy of the GNU General Public License
; along with this program.  If not, see <https://www.gnu.org/licenses/>.
;
; SPDX-License-Identifier: GPL-3.0-or-later

; Inno Setup script for the segfix Windows installer.
;
;   pyinstaller packaging/segfix.spec --noconfirm
;   iscc /DAppVersion=1.0.6 packaging\segfix.iss
;
; Takes the folder PyInstaller produced in dist\segfix and wraps it in an
; installer with a Start Menu entry and a working uninstaller. GitHub's
; windows runners ship Inno Setup, so CI needs nothing extra (see
; .github/workflows/installer.yml).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "segfix"
#define AppPublisher "Tim Devereux"
#define AppURL "https://github.com/TerraSpatial/Segfix"
#define AppExe "segfix.exe"

[Setup]
AppId={{8F2C1A94-3B6E-4D77-9C0A-5E1F2B7A4D63}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user by default so a forester on a managed laptop can install without
; calling IT; the dialog still offers all-users to anyone with admin.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=segfix-{#AppVersion}-setup
SetupIconFile=..\assets\icons\segfix.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The bundled Python, Qt and scipy are 64-bit. Spelled "x64" rather than
; the newer "x64compatible": the latter is an error in Inno Setup 6.2, which
; is what some CI images still ship, while "x64" is accepted by every 6.x.
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Everything PyInstaller collected, including the _internal folder.
Source: "..\dist\segfix\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; No file associations. .las/.laz/.ply belong to whatever the user already
; uses to view clouds (CloudCompare, usually), and quietly taking them over
; would be a hostile thing for a review tool to do.
