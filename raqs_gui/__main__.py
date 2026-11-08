"""Launch from the repository root with python -m raqs_gui."""
import sys


def main():
    from PySide6.QtWidgets import QApplication
    from .window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('RAQS Research Workbench')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
