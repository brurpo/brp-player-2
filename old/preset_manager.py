# preset_manager.py

import os
import json
import subprocess
import sys
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QComboBox, QPushButton, QInputDialog, QMessageBox, QStyle, QSizePolicy)
from PyQt6.QtCore import pyqtSignal

class PresetManager:
    """Lida com a lógica de ficheiros para guardar, carregar e apagar presets em JSON."""
    def __init__(self, preset_type='export'):
        # Constrói o caminho para a pasta de presets dentro de %LOCALAPPDATA%/BRP_Player/presets
        base_path = os.path.join(os.getenv('LOCALAPPDATA'), "BRP_Player")
        self.preset_dir = os.path.join(base_path, 'presets', preset_type)
        # Cria a pasta se ela não existir
        os.makedirs(self.preset_dir, exist_ok=True)
        print(f"Pasta de presets: {self.preset_dir}")

    def get_presets(self):
        """Retorna um dicionário de presets existentes {nome_preset: caminho_ficheiro}."""
        presets = {}
        for filename in os.listdir(self.preset_dir):
            if filename.endswith('.json'):
                preset_name = os.path.splitext(filename)[0]
                presets[preset_name] = os.path.join(self.preset_dir, filename)
        return presets

    def save_preset(self, name, settings):
        """Guarda as configurações num ficheiro JSON."""
        if not name: return False
        filepath = os.path.join(self.preset_dir, f"{name}.json")
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Erro ao guardar o preset '{name}': {e}")
            return False

    def load_preset(self, name):
        """Carrega as configurações de um ficheiro JSON."""
        filepath = os.path.join(self.preset_dir, f"{name}.json")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar o preset '{name}': {e}")
            return None

    def delete_preset(self, name):
        """Apaga um ficheiro de preset."""
        filepath = os.path.join(self.preset_dir, f"{name}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception as e:
            print(f"Erro ao apagar o preset '{name}': {e}")
        return False

class PresetWidget(QWidget):
    """Widget de UI reutilizável com o dropdown e os botões de preset."""

    # Sinal emitido quando um preset é carregado
    preset_loaded = pyqtSignal(dict)

    def __init__(self, get_settings_func, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.get_settings_func = get_settings_func
        self.manager = PresetManager('export')

        # --- LÓGICA DE LAYOUT CORRIGIDA E MAIS ROBUSTA ---
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5) # Adiciona um pequeno espaço entre os itens

        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(150)

        self.add_button = QPushButton("+")
        self.add_button.setToolTip("Guardar preset atual")
        self.add_button.setFixedSize(28, 28)

        self.remove_button = QPushButton("-")
        self.remove_button.setToolTip("Apagar preset selecionado")
        self.remove_button.setFixedSize(28, 28)

        self.open_folder_button = QPushButton()
        self.open_folder_button.setToolTip("Open preset folder")
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        self.open_folder_button.setIcon(icon)
        self.open_folder_button.setFixedSize(28, 28)

        layout.addWidget(self.preset_combo)  # item no índice 0
        layout.addWidget(self.add_button)     # item no índice 1
        layout.addWidget(self.remove_button)  # item no índice 2
        layout.addWidget(self.open_folder_button)

        # Diz explicitamente ao layout para dar todo o espaço extra ao item 0 (o QComboBox)
        layout.setStretch(0, 1)

        self.setLayout(layout)
        # --- FIM DA CORREÇÃO ---

        self.add_button.clicked.connect(self.save_current_preset)
        self.remove_button.clicked.connect(self.delete_selected_preset)
        self.open_folder_button.clicked.connect(self.open_preset_folder)
        self.preset_combo.activated.connect(self.load_selected_preset)

        self.populate_presets()

    def open_preset_folder(self):
        """Abre a pasta de presets no explorador de arquivos do sistema."""
        path = self.manager.preset_dir
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin": # macOS
            subprocess.run(["open", path])
        else: # Linux
            subprocess.run(["xdg-open", path])

    def populate_presets(self):
        """Carrega os nomes dos presets para o QComboBox."""
        self.preset_combo.clear()
        self.preset_combo.addItem("--Presets--")
        presets = self.manager.get_presets()
        for name in sorted(presets.keys()):
            self.preset_combo.addItem(name)
        self.update_remove_button_state()

    def save_current_preset(self):
        """Pede um nome e guarda as configurações atuais como um novo preset."""
        current_settings = self.get_settings_func()
        if not current_settings:
            QMessageBox.warning(self, "Aviso", "Não há configurações para guardar.")
            return

        text, ok = QInputDialog.getText(self, 'Guardar Preset', 'Nome do Preset:')
        if ok and text:
            if self.manager.save_preset(text, current_settings):
                self.populate_presets()
                self.preset_combo.setCurrentText(text) # Seleciona o preset recém-criado
            else:
                QMessageBox.critical(self, "Erro", f"Não foi possível guardar o preset '{text}'.")

    def delete_selected_preset(self):
        """Apaga o preset atualmente selecionado no QComboBox."""
        preset_name = self.preset_combo.currentText()
        if preset_name == "--Presets--": return

        reply = QMessageBox.question(self, 'Confirmar', f"Tem a certeza que quer apagar o preset '{preset_name}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            if self.manager.delete_preset(preset_name):
                self.populate_presets()
            else:
                QMessageBox.critical(self, "Erro", f"Não foi possível apagar o preset '{preset_name}'.")

    def load_selected_preset(self, index):
        """Carrega as configurações do preset selecionado e emite um sinal."""
        preset_name = self.preset_combo.itemText(index)
        if preset_name == "--Presets--":
            self.update_remove_button_state()
            return

        settings = self.manager.load_preset(preset_name)
        if settings:
            self.preset_loaded.emit(settings) # Emite o sinal com as configurações

        self.update_remove_button_state()

    def update_remove_button_state(self):
        """Ativa ou desativa o botão de apagar."""
        is_preset_selected = self.preset_combo.currentText() != "--Presets--"
        self.remove_button.setEnabled(is_preset_selected)
