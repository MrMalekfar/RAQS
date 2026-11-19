[Setup]
AppId=RAQS.ResearchWorkbench
AppName=RAQS Research Workbench
AppVersion={#ReleaseVersion}
DefaultDirName={localappdata}\Programs\RAQS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputBaseFilename=RAQS-Setup
Compression=lzma2/fast
SolidCompression=yes
InfoAfterFile={#PayloadDir}\Third-party notices.txt
UninstallDisplayName=RAQS Research Workbench

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}\{#ReleaseVersion}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\RAQS Research Workbench"; Filename: "{app}\{#ReleaseVersion}\runtime\pythonw.exe"; Parameters: "-m raqs_gui"; WorkingDir: "{app}\{#ReleaseVersion}"
