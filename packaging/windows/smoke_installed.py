"""Check the installed GUI, bundled default profile, and isolated worker."""

from pathlib import Path
import sys
import time
import traceback


def check_install():
    from PySide6.QtCore import QLibraryInfo
    from PySide6.QtWidgets import QApplication
    from raqs_gui.window import MainWindow, ROOT

    assert sys.flags.isolated and Path(sys.executable).resolve().parent == ROOT / "runtime"
    plugins = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
    assert (plugins / "platforms" / "qwindows.dll").is_file(), plugins
    app = QApplication([])
    window = MainWindow()
    window.show()
    assert not Path(window.output.text()).is_relative_to(ROOT)
    assert window.sensing_button.isEnabled(), window.notice.text()
    assert window.raw == (ROOT / "examples" / "lab006" / "experimental_near_resonant.json").read_bytes()
    window.output.setText(str(Path(sys.argv[1]) / "calculation output"))
    window.sensing_button.click()
    deadline = time.monotonic() + 180
    while (window.active or window.reading) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    if window.active:
        window.stop_run()
    assert not window.active and not window.reading, "Worker exceeded the test deadline"
    assert window.result_directory is not None, window.notice.text() + "\n" + window.log.toPlainText()
    for name in ("manifest.json", "run_report.json", "simulation_arrays.npz"):
        assert (window.result_directory / name).is_file(), name
    window.close()


if __name__ == "__main__":
    log = Path(sys.argv[1]) / "smoke.log"
    try:
        check_install()
    except Exception:
        log.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    log.write_text("Installed RAQS GUI and its sensing worker passed.\n", encoding="utf-8")
