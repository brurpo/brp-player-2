# ui_effects_dialog.py
import os
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QGroupBox,
                             QDialogButtonBox, QLabel, QComboBox, QWidget, QFrame, QCheckBox, QVBoxLayout, QFormLayout, QStyle, QToolButton, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon

DialogStyle = """
    QListWidget::item:selected { background-color: #007bff; }
    QListWidget::item:selected QWidget { background-color: transparent; }
    QListWidget::item:selected QLabel { color: white; background-color: transparent; }
    QListWidget::item:selected QCheckBox { color: white; }
    QListWidget::item:selected:!active { background-color: #555; }

    /* --- ESTILO ATUALIZADO PARA PARECER UMA BARRA --- */
    QToolButton#categoryTitleButton {
        background-color: #3c3f41;
        border: 1px solid #555;
        padding: 5px;
        text-align: center;
        font-weight: bold;
    }

    QToolButton#categoryTitleButton:hover {
        background-color: #4f5355;
    }

    QWidget#categoryContent {
        background-color: #2f2f2f;
        border: 1px solid #555;
        border-top: none; /* Cria uma aparência contínua */
        padding: 10px;
    }

    QScrollArea {
        border: none;
    }
"""

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(self)
        self.toggle_button.setObjectName("categoryTitleButton")
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.content_area = QWidget()
        self.content_area.setObjectName("categoryContent")
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_layout = QFormLayout(self.content_area)
        self.content_layout.setContentsMargins(10, 0, 10, 0)

        self.toggle_animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.toggle_animation.setDuration(200)
        self.toggle_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        self.toggle_button.toggled.connect(self.on_toggled)

    @pyqtSlot(bool)
    def on_toggled(self, checked):
        content_height = self.content_area.sizeHint().height()
        self.toggle_animation.setStartValue(self.content_area.maximumHeight())
        self.toggle_animation.setEndValue(content_height if checked else 0)
        self.toggle_animation.start()

    def addRow(self, label, widget):
        self.content_layout.addRow(label, widget)

class EffectsDialog(QDialog):
    delete_requested = pyqtSignal()
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filters and Effects")
        self.setMinimumSize(800, 800)
        self.setStyleSheet(DialogStyle)
        self.setAcceptDrops(True)

        main_layout = QHBoxLayout(self)

        # --- PAINEL ESQUERDO ---
        left_layout = QVBoxLayout()
        add_shader_layout = QHBoxLayout()
        self.shader_combo = QComboBox()
        self.add_shader_button = QPushButton("Add")
        self.add_shader_button.setAutoDefault(False)
        self.open_shader_folder_button = QPushButton()
        self.open_shader_folder_button.setToolTip("Open local shaders folder")
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.open_shader_folder_button.setIcon(icon)
        self.open_shader_folder_button.setFixedSize(28, 28)
        self.open_shader_folder_button.setAutoDefault(False)
        add_shader_layout.addWidget(QLabel("Add Filter:"))
        add_shader_layout.addWidget(self.shader_combo, 1)
        add_shader_layout.addWidget(self.add_shader_button)
        add_shader_layout.addWidget(self.open_shader_folder_button)
        left_layout.addLayout(add_shader_layout)
        self.shader_list = QListWidget()
        self.shader_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        left_layout.addWidget(self.shader_list)
        list_controls_layout = QHBoxLayout()
        self.load_preset_button = QPushButton("Load Preset...")
        self.save_preset_button = QPushButton("Save Preset...")
        self.clear_button = QPushButton("Clear")
        self.remove_shader_button = QPushButton("Remove Selected")
        list_controls_layout.addWidget(self.load_preset_button)
        list_controls_layout.addWidget(self.save_preset_button)
        list_controls_layout.addWidget(self.clear_button)
        list_controls_layout.addStretch()
        list_controls_layout.addWidget(self.remove_shader_button)
        left_layout.addLayout(list_controls_layout)

        # --- PAINEL DIREITO (com ScrollArea) ---
        right_groupbox = QGroupBox("Parameters:")
        right_groupbox_layout = QVBoxLayout(right_groupbox)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_content_widget = QWidget()
        self.params_layout = QVBoxLayout(scroll_content_widget)
        self.params_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidget(scroll_content_widget)
        right_groupbox_layout.addWidget(scroll_area)
        self.recreate_params_container()

        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(right_groupbox, 1)

    def recreate_params_container(self):
        self.no_params_label = QLabel("Select an effect from the list")
        self.no_params_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.params_widget_container = QWidget()
        self.params_widget_layout = QFormLayout(self.params_widget_container)

        self.params_layout.addWidget(self.no_params_label)
        self.params_layout.addWidget(self.params_widget_container)

    # --- MÉTODO keyPressEvent REMOVIDO E SUBSTITUÍDO POR QAction ---
    def keyPressEvent(self, event):
        # Apenas a lógica de 'Delete' permanece aqui. Ctrl+Z/Y são tratados por QAction.
        if event.key() == Qt.Key.Key_Delete and self.shader_list.currentItem():
            self.delete_requested.emit()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.glsl'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        local_files = [url.toLocalFile() for url in urls if url.isLocalFile() and url.toLocalFile().lower().endswith('.glsl')]
        if local_files:
            self.files_dropped.emit(local_files)
            event.acceptProposedAction()
        else:
            event.ignore()
