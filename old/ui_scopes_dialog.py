# ui_scopes_dialog.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame
from PyQt6.QtCore import pyqtSignal

class ScopesDialog(QWidget):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Scopes")
        self.setMinimumSize(400, 700)
        self.setStyleSheet("background-color: #1e1e1e;")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Apenas um único frame para o output combinado do mpv
        self.combined_scope_frame = QFrame()
        self.combined_scope_frame.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.combined_scope_frame)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)
