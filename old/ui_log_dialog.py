# ui_log_dialog.py
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPlainTextEdit,
                             QPushButton, QDialogButtonBox, QFileDialog)
from PyQt6.QtCore import Qt

class LogDialog(QDialog):
    def __init__(self, log_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("App Log")
        self.setMinimumSize(700, 500)

        main_layout = QVBoxLayout(self)

        # Usamos QPlainTextEdit, que é otimizado para texto simples e longo
        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlainText(log_text)
        self.log_display.setStyleSheet("font-family: Consolas, Courier New, monospace;")

        main_layout.addWidget(self.log_display)

        button_box = QDialogButtonBox()
        self.save_button = button_box.addButton("Save as TXT...", QDialogButtonBox.ButtonRole.ActionRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)

        main_layout.addWidget(button_box)

        self.save_button.clicked.connect(self.save_log_to_file)
        close_button.clicked.connect(self.accept)

    def save_log_to_file(self):
        """Abre um diálogo para salvar o conteúdo do log num ficheiro de texto."""
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Log", "", "Text Files (*.txt)")
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.log_display.toPlainText())
            except Exception as e:
                print(f"Erro ao salvar o log: {e}") # Este print também será capturado!
