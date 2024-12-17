from PyQt5.QtWidgets import QApplication
from app.MainWindow import MainWindow
import pytest

@pytest.fixture
def app(qtbot):
    test_app = QApplication([])
    window = MainWindow()
    qtbot.addWidget(window)
    return window

def test_main_window_title(app):
    assert app.windowTitle() == "CommAnalyzer"
