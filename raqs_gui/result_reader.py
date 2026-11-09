"""Verify potentially large saved artifacts without blocking the UI thread."""
from PySide6.QtCore import QObject, QRunnable, Signal
from .jobs import read_result


class ReadSignals(QObject):
    complete = Signal(int, object, object)
    failed = Signal(int, str)


class ReadResult(QRunnable):
    def __init__(self, generation, directory):
        super().__init__()
        self.generation = generation
        self.directory = directory
        self.signals = ReadSignals()

    def run(self):
        try:
            report, images = read_result(self.directory)
            self.signals.complete.emit(self.generation, report, images)
        except Exception as exc:
            self.signals.failed.emit(self.generation, str(exc))
