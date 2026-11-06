# ui_batch_export_dialog.py
from PyQt6.QtWidgets import (QDialog, QListWidgetItem, QListWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QGroupBox, QDialogButtonBox, QProgressBar, QLabel)
from PyQt6.QtCore import pyqtSignal
from ui_export_dialog import ExportOptionsWidget
from PyQt6.QtCore import Qt
import os
from preset_manager import PresetWidget

class BatchExportDialog(QDialog):
    export_requested = pyqtSignal()

    def __init__(self, parent=None, detected_encoders=None, selected_gpu=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Export")
        self.setMinimumSize(800, 600)

        self.setAcceptDrops(True)

        main_layout = QVBoxLayout(self)
        top_panel_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.file_list = QListWidget()

        # --- ALTERAÇÃO AQUI: Permite a seleção de múltiplos itens ---
        # Use Ctrl+Click para selecionar vários ou Shift+Click para um intervalo.
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        left_layout.addWidget(self.file_list)

        button_layout = QHBoxLayout()
        self.add_video_button = QPushButton("Add Videos...")
        self.add_sequence_button = QPushButton("Add Sequence...")
        self.remove_button = QPushButton("Remove Selected") # Texto atualizado
        button_layout.addWidget(self.add_video_button)
        button_layout.addWidget(self.add_sequence_button)
        button_layout.addWidget(self.remove_button)
        left_layout.addLayout(button_layout)

        right_groupbox = QGroupBox("Export options")
        right_group_layout = QVBoxLayout(right_groupbox)

        self.options_widget = ExportOptionsWidget(source_resolution=(0,0),
                                                  detected_encoders=detected_encoders,
                                                  selected_gpu=selected_gpu)


        # 2. Cria o widget de presets.
        apply_func = lambda settings: self.options_widget.apply_settings(settings)
        self.preset_widget = PresetWidget(get_settings_func=self.options_widget.get_settings)
        self.preset_widget.preset_loaded.connect(apply_func)

        # 3. Adiciona ao layout na ordem visual desejada.
        right_group_layout.addWidget(self.preset_widget)
        right_group_layout.addWidget(self.options_widget)

        # --- NOVA LINHA ADICIONADA AQUI ---
        # Adiciona um espaçador que ocupa todo o espaço vertical livre, empurrando os widgets para cima.
        right_group_layout.addStretch(1)

        top_panel_layout.addLayout(left_layout, 2)
        top_panel_layout.addWidget(right_groupbox, 1)
        main_layout.addLayout(top_panel_layout)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready to start exporting.")
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)

        button_box = QDialogButtonBox()
        self.export_button = button_box.addButton("Export...", QDialogButtonBox.ButtonRole.ActionRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)

        self.export_button.clicked.connect(self.export_requested.emit)
        close_button.clicked.connect(self.reject)

        main_layout.addWidget(button_box)

    def get_batch_list(self):
        """Retorna os dados completos de cada item na lista."""
        return [self.file_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.file_list.count())]
    def get_settings(self):
        return self.options_widget.get_settings()

    # --- NOVA FUNÇÃO ---
    def find_item_by_text(self, text):
        """Encontra um item na lista com base no seu texto."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == text:
                return item
        return None

    def dragEnterEvent(self, event):
        """Verifica se o arquivo arrastado é válido."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """Lida com os arquivos soltos na janela."""
        urls = event.mimeData().urls()
        if not urls:
            return

        video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.mxf', '.webm')
        image_extensions = ('.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tga', '.bmp')

        for url in urls:
            filepath = url.toLocalFile()
            if os.path.exists(filepath):
                item_data = None
                if filepath.lower().endswith(video_extensions):
                    item_data = {"path": filepath, "is_sequence": False}
                elif filepath.lower().endswith(image_extensions):
                    item_data = {"path": filepath, "is_sequence": True}

                if item_data:
                    item = QListWidgetItem(filepath)
                    item.setData(Qt.ItemDataRole.UserRole, item_data)
                    self.file_list.addItem(item)
