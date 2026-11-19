"""Stage an isolated Windows runtime and compile the RAQS installer."""

import argparse
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build_payload(payload):
    """Copy the app and put its dependencies beside a private CPython."""
    runtime = payload / "runtime"
    runtime.mkdir(parents=True)
    base = Path(sys.base_prefix)
    for name in ("python.exe", "pythonw.exe", "python3.dll", "python311.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll", "LICENSE.txt"):
        shutil.copy2(base / name, runtime)
    for name in ("DLLs", "Lib"):
        shutil.copytree(base / name, runtime / name,
                        ignore=shutil.ignore_patterns("site-packages", "__pycache__"))
    if (base / "tcl").is_dir():
        shutil.copytree(base / "tcl", runtime / "tcl")
    (runtime / "python311._pth").write_text(
        ".\nDLLs\nLib\nLib\\site-packages\n..\nimport site\n", encoding="ascii"
    )
    packages = runtime / "Lib" / "site-packages"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--no-compile",
        "--target", str(packages), "-r", str(ROOT / "requirements-gui.txt")
    ], check=True)
    # Use the Microsoft runtime shipped in the shiboken6 wheel for native modules.
    for dll in (packages / "shiboken6").glob("msvcp140*.dll"):
        shutil.copy2(dll, runtime)

    files = subprocess.check_output([
        "git", "-C", str(ROOT), "ls-files", "-z", "--",
        "raqs_gui", "rydberg_aoa", "examples", "cli.py", "main.py", "LICENSE"
    ]).decode("utf-8").split("\0")
    for name in filter(None, files):
        target = payload / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(HERE / "Third-party notices.txt", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="Installer version, for example 1.0.0")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", args.version):
        parser.error("version must look like 1.0.0 or 1.0.0-rc.1")
    if sys.platform != "win32" or platform.machine() != "AMD64" or sys.version_info[:2] != (3, 11):
        parser.error("build with Windows x64 Python 3.11")
    compiler = shutil.which("ISCC.exe") or (
        Path(os.environ["ProgramFiles(x86)"]) / "Inno Setup 6" / "ISCC.exe"
    )
    if not Path(compiler).is_file():
        parser.error("install Inno Setup 6.3 or later and make ISCC.exe available on PATH")

    output = ROOT / "outputs" / "windows-installer"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="build-", dir=output) as temporary:
        scratch = Path(temporary)
        payload = scratch / "payload"
        build_payload(payload)
        subprocess.run([
            compiler, f"/DPayloadDir={payload}", f"/DReleaseVersion={args.version}",
            f"/O{scratch}", HERE / "raqs.iss"
        ], check=True)
        installer = output / "RAQS-Setup.exe"
        (scratch / installer.name).replace(installer)
    print(installer)


if __name__ == "__main__":
    main()
