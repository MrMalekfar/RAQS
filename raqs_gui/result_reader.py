"""Verify potentially large saved artifacts without blocking the UI thread."""
from PySide6.QtCore import QObject, QRunnable, Signal
from .jobs import read_result


class ReadSignals(QObject):
    complete = Signal(int, object)
    failed = Signal(int, str)


class ReadResult(QRunnable):
    def __init__(self, generation, directory):
        super().__init__()
        self.generation = generation
        self.directory = directory
        self.signals = ReadSignals()

    def run(self):
        try:
            result = read_result(self.directory, capture=True)
            self.signals.complete.emit(self.generation, result)
        except Exception as exc:
            self.signals.failed.emit(self.generation, str(exc))
