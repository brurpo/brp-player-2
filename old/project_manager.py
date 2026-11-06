import sys
import os
import subprocess
import tempfile
from datetime import datetime
import multiprocessing
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QDialog, QLineEdit,
                             QLabel, QFormLayout, QMessageBox, QFrame, QScrollArea,
                             QInputDialog, QDateEdit, QCheckBox, QDialogButtonBox, QSpinBox,
                             QCompleter, QProgressDialog,QListWidget,QFileDialog)
from PyQt6.QtGui import QFont, QPalette, QColor, QCursor, QIcon, QPixmap, QMovie
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve, pyqtSlot, QObject, QThread, pyqtSignal, QDate, QEvent, QSize, QUrl

# --- ALTERAÇÃO APLICADA AQUI ---
# Importa a classe para tocar efeitos sonoros
from PyQt6.QtMultimedia import QSoundEffect

basepath= os.path.dirname(os.path.abspath(__file__))
# --- CONSTANTES GLOBAIS ---
USE_MULTIPROCESSING = False
PROJECT_CATEGORIES = []
SUBPROJECT_CATEGORIES_CONFIG = [
    {"display": "Editoria", "aliases": ["Editoria"]},
    {"display": "Pacote Grafico", "aliases": ["Pacote Grafico", "Pacote_Grafico"]},
    {"display": "Abertura", "aliases": ["Abertura"]},
    {"display": "Chamadas", "aliases": ["Chamadas"]},
]
ICON_FILE = basepath+"/"+"icon.ico"
# --- ALTERAÇÃO APLICADA AQUI ---
# Define o caminho para o arquivo de som
SOUND_FILE = basepath+"/"+"search_sound.wav"
ROOT_PROJECT_PATHS = ["Z:\\1_CV_PROJETOS\\", "Z:\\2_CV_REGIONAIS\\"]
HEADER_LOGO_SEARCH_DEPTH = 3
SUBPROJECT_DEFAULT_FOLDERS = { "ELEMENTOS": ["AUDIO", "FONTS", "IMAGENS", "MOVS", "SEQUENCIAS", "RENDERS3D", "PRERENDERS", "REFERENCIAS"], "PROJETOS": ["AE", "AI", "PR", "PSD", "MAYA", "BLENDER"], "SAIDA": ["LOGO"], "PREVIEW": [] }

APP_BUTTONS_CONFIG = [
    {"label": "B", "tooltip": "Clique: Abrir pasta BLENDER\nShift+Clique: Procurar .blend", "aliases": ["BLENDER", "Blend", "3D"], "extensions": [".blend"]},
    {"label": "Ai", "tooltip": "Clique: Abrir pasta AI\nShift+Clique: Procurar .ai, .svg, .eps", "aliases": ["AI", "Illustrator"], "extensions": [".ai", ".svg", ".eps"]},
    {"label": "Ps", "tooltip": "Clique: Abrir pasta PSD\nShift+Clique: Procurar .psd, .psb", "aliases": ["PSD", "Photoshop", "PS"], "extensions": [".psd", ".psb"]},
    {"label": "Ae", "tooltip": "Clique: Abrir pasta AE\nShift+Clique: Procurar .aep", "aliases": ["AE", "After", "AfterEffects", "After Effects"], "extensions": [".aep"]},
    {"label": "Pr", "tooltip": "Clique: Abrir pasta PR\nShift+Clique: Procurar .prproj", "aliases": ["PR", "Premiere"], "extensions": [".prproj"]},
    {"label": "M", "tooltip": "Clique: Abrir pasta MAYA\nShift+Clique: Procurar .mb, .ma", "aliases": ["MAYA", "3D"], "extensions": [".mb", ".ma"]},
    {"label": "🔎", "tooltip": "Busca avançada neste subprojeto", "type": "search"}
]

FILENAME_SEARCH_CONFIG = [ {"label": "Logo", "search_terms": ["logo"]}, {"label": "BG", "search_terms": ["bg"]}, {"label": "Abertura", "search_terms": ["abertura", "vinheta", "vht"]}, {"label": "Pesquisar...", "type": "custom", "tooltip": "Abrir uma caixa para pesquisar um termo personalizado"} ]
ALIAS_TO_CATEGORY_MAP = {alias.lower(): config['display'] for config in SUBPROJECT_CATEGORIES_CONFIG for alias in config['aliases']}
# --- FIM DAS CONSTANTES ---

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def is_year_folder(folder_name):
    for year in range(2000, 2051):
        if str(year) in folder_name:
            return True
    return False

def _is_in_date_range(path, date_range):
    try:
        mod_date = datetime.fromtimestamp(os.path.getmtime(path)).date()
        return date_range[0] <= mod_date <= date_range[1]
    except OSError:
        return False

def _perform_search(base_path, search_params):
    found_folders = set()
    filter_text = search_params.get("filter_text", "").lower()
    search_term = search_params.get("term", "").lower()
    date_range = search_params.get("date_range")
    depth_limit = search_params.get("depth", 0)
    initial_depth = base_path.rstrip(os.sep).count(os.sep)

    for root, dirs, files in os.walk(base_path):
        current_depth = root.rstrip(os.sep).count(os.sep) - initial_depth + 1
        if depth_limit > 0 and current_depth > depth_limit:
            dirs[:] = []
            continue
        if filter_text == 'pasta':
            for dir_name in dirs:
                if search_term in dir_name.lower():
                    full_path = os.path.join(root, dir_name)
                    if date_range and not _is_in_date_range(full_path, date_range): continue
                    found_folders.add(full_path)
        elif filter_text:
            extensions = [f".{ext.strip()}" for ext in filter_text.split(',')]
            for file in files:
                file_lower = file.lower()
                if file_lower.endswith(tuple(extensions)) and (not search_term or search_term in file_lower):
                    full_path = os.path.join(root, file)
                    if date_range and not _is_in_date_range(full_path, date_range): continue
                    found_folders.add(root)
        else:
            for file in files:
                if search_term in file.lower():
                    full_path = os.path.join(root, file)
                    if date_range and not _is_in_date_range(full_path, date_range): continue
                    found_folders.add(root)
            for dir_name in dirs:
                if search_term in dir_name.lower():
                    full_path = os.path.join(root, dir_name)
                    if date_range and not _is_in_date_range(full_path, date_range): continue
                    found_folders.add(full_path)
    return found_folders

class SearchResultsDialog(QDialog):
    """Uma janela para exibir resultados de busca clicáveis."""
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Resultados da Busca")
        self.setMinimumSize(700, 500)

        # Guarda a lista completa de resultados para a filtragem
        self.all_results = sorted(list(results))

        self.layout = QVBoxLayout(self)

        # --- BARRA DE FILTRO ADICIONADA AQUI ---
        self.filter_input = QLineEdit(self)
        self.filter_input.setPlaceholderText("Filtrar resultados...")
        self.filter_input.textChanged.connect(self.filter_list)

        self.list_widget = QListWidget(self)
        self.list_widget.addItems(self.all_results)

        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2e2e2e;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-radius: 4px; /* Cantos arredondados para cada item */
            }
            QListWidget::item:hover {
                background-color: #455364; /* Azul-acinzentado para hover */
            }
            QListWidget::item:selected {
                background-color: #0078d7; /* Azul vibrante para seleção */
                color: white;
            }
        """)

        self.status_label = QLabel(f"{len(self.all_results)} item(s) encontrado(s):")

        self.layout.addWidget(self.filter_input)
        self.layout.addWidget(self.status_label)
        self.layout.addWidget(self.list_widget)

        self.list_widget.itemDoubleClicked.connect(self.on_item_clicked)
    def filter_list(self, text):
        """Filtra os itens na lista com base no texto digitado."""
        self.list_widget.clear()

        if not text:
            self.list_widget.addItems(self.all_results)
            self.status_label.setText(f"{len(self.all_results)} item(s) encontrado(s):")
            return

        filtered_results = [item for item in self.all_results if text.lower() in item.lower()]
        self.list_widget.addItems(filtered_results)
        self.status_label.setText(f"{len(filtered_results)} de {len(self.all_results)} item(s) exibido(s):")

    def on_item_clicked(self, item):
        path = item.text()
        open_in_explorer(path)

class WaitingDialog(QDialog):
    """Uma janela de 'Aguarde' com uma animação GIF e um botão de cancelar."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Busca")
        self.setModal(True)
        self.setFixedSize(350, 100)

        self.layout = QHBoxLayout(self)

        self.spinner_label = QLabel(self)
        self.spinner_movie = QMovie(basepath+"/"+"spinner.gif")

        spinner_size = 16
        self.spinner_label.setFixedSize(spinner_size, spinner_size)

        if self.spinner_movie.isValid():
            self.spinner_movie.setScaledSize(QSize(spinner_size, spinner_size))
            self.spinner_label.setMovie(self.spinner_movie)
            self.spinner_movie.start()
        else:
            self.spinner_label.setText("⏳")
            self.spinner_label.setFont(QFont("Arial", 20))
            self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            print("Aviso: spinner.gif não encontrado ou inválido.")

        self.message_label = QLabel("Buscando, por favor aguarde...", self)

        self.cancel_button = QPushButton("Cancelar", self)
        self.cancel_button.clicked.connect(self.reject)

        self.layout.addWidget(self.spinner_label)
        self.layout.addWidget(self.message_label, 1)
        self.layout.addWidget(self.cancel_button)

class SearchWorker(QObject):
    """Worker que executa a busca de arquivos em uma thread separada."""
    finished = pyqtSignal(set)

    def __init__(self, base_path, search_params):
        super().__init__()
        self.base_path = base_path
        self.search_params = search_params
        self._is_cancelled = False

    @pyqtSlot()
    def run(self):
        found_items = set()
        try:
            filter_text = self.search_params.get("filter_text", "").lower()
            search_term = self.search_params.get("term", "").lower()
            date_range = self.search_params.get("date_range")
            depth_limit = self.search_params.get("depth", 0)

            initial_depth = self.base_path.rstrip(os.sep).count(os.sep)

            for root, dirs, files in os.walk(self.base_path):
                if self._is_cancelled:
                    print("Busca cancelada pelo usuário.")
                    break

                current_depth = root.rstrip(os.sep).count(os.sep) - initial_depth + 1
                if depth_limit > 0 and current_depth > depth_limit:
                    dirs[:] = []
                    continue

                # ... (a lógica interna de busca permanece a mesma) ...
                if filter_text == 'pasta':
                    for dir_name in dirs:
                        if search_term in dir_name.lower():
                            full_path = os.path.join(root, dir_name)
                            if date_range and not _is_in_date_range(full_path, date_range): continue
                            found_items.add(full_path)
                elif filter_text:
                    extensions = [f".{ext.strip()}" for ext in filter_text.split(',')]
                    for file in files:
                        if file.lower().endswith(tuple(extensions)) and (not search_term or search_term in file.lower()):
                            full_path = os.path.join(root, file)
                            if date_range and not _is_in_date_range(full_path, date_range): continue
                            found_items.add(full_path)
                else:
                    for file in files:
                        if search_term in file.lower():
                            full_path = os.path.join(root, file)
                            if date_range and not _is_in_date_range(full_path, date_range): continue
                            found_items.add(full_path)
                    for dir_name in dirs:
                        if search_term in dir_name.lower():
                            full_path = os.path.join(root, dir_name)
                            if date_range and not _is_in_date_range(full_path, date_range): continue
                            found_items.add(full_path)
        finally:
            self.finished.emit(found_items)

    def cancel(self):
        self._is_cancelled = True

def open_in_explorer(path):
    """Abre o explorador de arquivos no local do path, selecionando o item."""
    if not os.path.exists(path):
        print(f"Caminho não encontrado: {path}")
        return
    try:
        if sys.platform == 'win32':
            subprocess.run(['explorer', '/select,', os.path.normpath(path)])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-R', path])
        else:
            # Tenta abrir o diretório pai, pois nem todos os gerenciadores de arquivos do Linux
            # suportam a seleção de um arquivo específico.
            subprocess.Popen(['xdg-open', os.path.dirname(path)])
    except Exception as e:
        print(f"Não foi possível abrir o explorador: {e}")

class SearchWorker(QObject):
    """Worker que executa a busca de arquivos em uma thread separada."""
    finished = pyqtSignal(set)

    def __init__(self, base_path, search_params):
        super().__init__()
        self.base_path = base_path
        self.search_params = search_params
        self._is_cancelled = False

    @pyqtSlot()
    def run(self):
        found_items = set()
        filter_text = self.search_params.get("filter_text", "").lower()
        search_term = self.search_params.get("term", "").lower()
        date_range = self.search_params.get("date_range")
        depth_limit = self.search_params.get("depth", 0)

        initial_depth = self.base_path.rstrip(os.sep).count(os.sep)

        for root, dirs, files in os.walk(self.base_path):
            if self._is_cancelled:
                break

            current_depth = root.rstrip(os.sep).count(os.sep) - initial_depth + 1
            if depth_limit > 0 and current_depth > depth_limit:
                dirs[:] = []
                continue

            # --- LÓGICA DE BUSCA RESTAURADA AQUI ---
            if filter_text == 'pasta':
                for dir_name in dirs:
                    if search_term in dir_name.lower():
                        full_path = os.path.join(root, dir_name)
                        if date_range and not _is_in_date_range(full_path, date_range): continue
                        found_items.add(full_path)
            elif filter_text:
                extensions = [f".{ext.strip()}" for ext in filter_text.split(',')]
                for file in files:
                    if file.lower().endswith(tuple(extensions)) and (not search_term or search_term in file.lower()):
                        full_path = os.path.join(root, file)
                        if date_range and not _is_in_date_range(full_path, date_range): continue
                        found_items.add(full_path)
            else:
                for file in files:
                    if search_term in file.lower():
                        full_path = os.path.join(root, file)
                        if date_range and not _is_in_date_range(full_path, date_range): continue
                        found_items.add(full_path)
                for dir_name in dirs:
                    if search_term in dir_name.lower():
                        full_path = os.path.join(root, dir_name)
                        if date_range and not _is_in_date_range(full_path, date_range): continue
                        found_items.add(full_path)

        self.finished.emit(found_items)

    def cancel(self):
        self._is_cancelled = True

class SearchDialog(QDialog):
    def __init__(self, parent=None, predefined_terms=None):
        super().__init__(parent)
        self.setWindowTitle("Busca Avançada no Projeto")
        self.setModal(True)
        if os.path.exists(ICON_FILE): self.setWindowIcon(QIcon(ICON_FILE))
        layout = QVBoxLayout(self)
        formLayout = QFormLayout()
        self.terms_edit = QLineEdit(self)
        if predefined_terms:
            self.terms_edit.setText(", ".join(predefined_terms))
        formLayout.addRow("Procurar por Termo:", self.terms_edit)
        self.filter_edit = QLineEdit(self)
        self.filter_edit.setPlaceholderText("Deixe em branco para busca geral")
        formLayout.addRow("Filtrar por (pasta / ext,ext):", self.filter_edit)
        self.depth_spinbox = QSpinBox(self)
        self.depth_spinbox.setMinimum(0)
        self.depth_spinbox.setToolTip("Limite de profundidade da busca.\n0 = sem limite (padrão).\n1 = apenas na pasta raiz.")
        formLayout.addRow("Profundidade da Busca:", self.depth_spinbox)
        self.date_filter_check = QCheckBox("Filtrar por data de modificação", self)
        formLayout.addRow(self.date_filter_check)
        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
        self.start_date_edit.setEnabled(False)
        formLayout.addRow("De:", self.start_date_edit)
        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setEnabled(False)
        formLayout.addRow("Até:", self.end_date_edit)
        self.date_filter_check.toggled.connect(self.start_date_edit.setEnabled)
        self.date_filter_check.toggled.connect(self.end_date_edit.setEnabled)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addLayout(formLayout)
        layout.addWidget(buttons)
    def get_search_parameters(self):
        return { "term": self.terms_edit.text().strip(), "filter_text": self.filter_edit.text().strip(), "use_date": self.date_filter_check.isChecked(), "start_date": self.start_date_edit.date().toPyDate(), "end_date": self.end_date_edit.date().toPyDate(), "depth": self.depth_spinbox.value() }

class CollapsibleSection(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.animation = None
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet(""" QPushButton { font-size: 13pt; font-weight: bold; color: #999; border: none; text-align: center; padding: 8px; background-color: #3c3c3c; } QPushButton:checked { background-color: #4c4c4c; } """)
        self.toggle_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(15, 5, 15, 5)
        self.content_layout.setSpacing(5)
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)
        self.toggle_button.toggled.connect(self.toggle)
        self.content_area.setMaximumHeight(0)
        self.toggle_button.setChecked(False)
    @pyqtSlot(bool)
    def toggle(self, checked):
        self.content_area.setVisible(True)
        start_height = self.content_area.height()
        end_height = self.content_layout.sizeHint().height() if checked else 0
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setStartValue(start_height)
        self.animation.setEndValue(end_height)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.animation.start()
    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class ProjectCreatorDialog(QDialog):
    def __init__(self, parent=None, root_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Criar Novo Projeto")
        self.setModal(True)
        if os.path.exists(ICON_FILE): self.setWindowIcon(QIcon(ICON_FILE))

        self.root_paths = root_paths or []
        self.layout = QVBoxLayout(self)
        self.formLayout = QFormLayout()

        self.rootPathComboBox = QComboBox(self)
        if self.root_paths:
            for path in self.root_paths:
                display_name = os.path.basename(path.rstrip(os.sep))
                self.rootPathComboBox.addItem(display_name, path)

        self.projectNameLineEdit = QLineEdit(self)
        self.projectTypeComboBox = QComboBox(self)

        self.formLayout.addRow("Raiz do Projeto:", self.rootPathComboBox)
        self.formLayout.addRow("Nome do Projeto:", self.projectNameLineEdit)
        self.formLayout.addRow("Tipo de Projeto:", self.projectTypeComboBox)

        self.layout.addLayout(self.formLayout)

        self.buttonsLayout = QHBoxLayout()
        self.createButton = QPushButton("Criar", self)
        self.createButton.setObjectName("dialogButton")
        self.cancelButton = QPushButton("Cancelar", self)
        self.cancelButton.setObjectName("dialogButton")
        self.buttonsLayout.addWidget(self.createButton)
        self.buttonsLayout.addWidget(self.cancelButton)
        self.layout.addLayout(self.buttonsLayout)

        self.createButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

        self.rootPathComboBox.activated.connect(self.update_project_types)
        self.update_project_types()

    def update_project_types(self):
        """Atualiza o dropdown de categorias com base na raiz selecionada."""
        selected_root = self.rootPathComboBox.currentData()
        self.projectTypeComboBox.clear()

        if selected_root and os.path.exists(selected_root):
            try:
                # Encontra todas as pastas (categorias) dentro da raiz selecionada
                categories = [d for d in os.listdir(selected_root) if os.path.isdir(os.path.join(selected_root, d))]
                self.projectTypeComboBox.addItems(sorted(categories))
            except OSError as e:
                print(f"Erro ao ler categorias de {selected_root}: {e}")

    def getProjectDetails(self):
        return (self.rootPathComboBox.currentData(),
                self.projectNameLineEdit.text(),
                self.projectTypeComboBox.currentText())

class SubProjectCreatorDialog(QDialog):
    def __init__(self, parent=None, year_folders=None):
        super().__init__(parent)
        self.setWindowTitle("Criar Novo Subprojeto")
        self.setModal(True)
        if os.path.exists(ICON_FILE): self.setWindowIcon(QIcon(ICON_FILE))

        self.layout = QVBoxLayout(self)
        self.formLayout = QFormLayout()

        self.subProjectNameLineEdit = QLineEdit(self)
        self.subProjectCategoryComboBox = QComboBox(self)
        self.subProjectCategoryComboBox.addItems([cat['display'] for cat in SUBPROJECT_CATEGORIES_CONFIG])

        self.yearComboBox = QComboBox(self)

        self.formLayout.addRow("Ano (Opcional):", self.yearComboBox)
        self.formLayout.addRow("Nome do Subprojeto:", self.subProjectNameLineEdit)
        self.formLayout.addRow("Categoria:", self.subProjectCategoryComboBox)

        self.yearComboBox.addItem("Nenhum", None)
        if year_folders:
            for year in sorted(year_folders):
                self.yearComboBox.addItem(year, year)
        self.yearComboBox.addItem("Criar Novo Ano...", "__CREATE_NEW__")

        self.layout.addLayout(self.formLayout)

        self.buttonsLayout = QHBoxLayout()
        self.createButton = QPushButton("Criar", self)
        self.createButton.setObjectName("dialogButton")
        self.cancelButton = QPushButton("Cancelar", self)
        self.cancelButton.setObjectName("dialogButton")
        self.buttonsLayout.addWidget(self.createButton)
        self.buttonsLayout.addWidget(self.cancelButton)
        self.layout.addLayout(self.buttonsLayout)

        self.createButton.clicked.connect(self.accept)
        self.cancelButton.clicked.connect(self.reject)

    def getSubProjectDetails(self):
        return (self.subProjectNameLineEdit.text(),
                self.subProjectCategoryComboBox.currentText(),
                self.yearComboBox.currentData())

class LineEditClickFilter(QObject):
    def __init__(self, line_edit):
        super().__init__(line_edit)
        self._line_edit = line_edit
    def eventFilter(self, watched, event):
        if watched == self._line_edit and event.type() == QEvent.Type.MouseButtonPress:
            self._line_edit.clear()
        return super().eventFilter(watched, event)

class ProjectControlApp(QWidget):

    def __init__(self):
        super().__init__()
        self.ROOT_PATHS = ROOT_PROJECT_PATHS
        self.active_project_path = None
        self.optimized_logo_path = None
        self.original_logo_path = None
        self.search_buttons = []
        self.thread = None
        self.worker = None
        self.all_projects_data = []

        # --- ALTERAÇÃO APLICADA AQUI ---
        # Inicializa o player de som
        self.search_sound_player = QSoundEffect()
        if os.path.exists(SOUND_FILE):
            self.search_sound_player.setSource(QUrl.fromLocalFile(SOUND_FILE))
        else:
            print(f"Aviso: Arquivo de som de busca '{SOUND_FILE}' não encontrado.")

        self.initUI()
        self.apply_dark_theme()

        self.populate_roots()
        self.populate_categories()
        self.populate_projects()

    # --- ALTERAÇÃO APLICADA AQUI ---
    # Cria um método central para tocar o som
    def play_search_sound(self):
        """Toca o som de busca se ele estiver carregado."""
        if self.search_sound_player.isLoaded():
            self.search_sound_player.play()

    def populate_categories(self):
        global PROJECT_CATEGORIES

        paths_to_scan = self.rootSelector.currentData()
        if not paths_to_scan:
            paths_to_scan = self.ROOT_PATHS

        found_categories = set()
        for root_path in paths_to_scan:
            if os.path.exists(root_path):
                try:
                    for category in os.listdir(root_path):
                        if os.path.isdir(os.path.join(root_path, category)):
                            found_categories.add(category)
                except OSError as e:
                    print(f"Erro ao acessar {root_path}: {e}")

        PROJECT_CATEGORIES = sorted(list(found_categories))

        current_selection = self.categorySelector.currentText()
        self.categorySelector.clear()
        self.categorySelector.addItems(["Todas as Categorias"] + PROJECT_CATEGORIES)

        index = self.categorySelector.findText(current_selection)
        if index != -1:
            self.categorySelector.setCurrentIndex(index)

    def open_search_dialog(self, predefined_terms=None, base_path=None):
        # --- ALTERAÇÃO APLICADA AQUI ---
        # Toca o som sempre que a caixa de diálogo de busca é aberta
        self.play_search_sound()

        search_base = base_path if base_path else self.active_project_path
        if not search_base:
            paths_to_scan = self.rootSelector.currentData()
            if paths_to_scan and len(paths_to_scan) == 1:
                search_base = paths_to_scan[0]
            else:
                QMessageBox.warning(self, "Aviso", "Selecione um projeto ou uma raiz específica para a busca.")
                return

        dialog = SearchDialog(self, predefined_terms)
        if dialog.exec():
            params = dialog.get_search_parameters()
            if not params["term"] and not params["filter_text"]:
                QMessageBox.warning(self, "Aviso", "Pelo menos um termo ou filtro deve ser preenchido.")
                return

            search_params = params.copy()
            search_params["date_range"] = (params["start_date"], params["end_date"]) if params["use_date"] else None

            self.execute_search(search_base, search_params)

    def on_app_button_clicked(self, subproject_path, config):
        """Lida com cliques nos botões de app (Ae, Ps, etc.)."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            # --- ALTERAÇÃO APLICADA AQUI ---
            # Toca o som na ação de busca com Shift+Clique
            #self.play_search_sound()

            search_params = {
                "term": "",
                "filter_text": ",".join(ext.replace('.', '') for ext in config["extensions"]),
                "date_range": None,
                "depth": 0
            }
            self.execute_search(subproject_path, search_params)
        else:
            found = False
            for alias in config["aliases"]:
                direct_path = os.path.join(subproject_path, "PROJETOS", alias)
                if os.path.exists(direct_path):
                    self.open_folder(direct_path)
                    found = True
                    break
            if not found:
                QMessageBox.information(self, "Pasta não encontrada", f"Nenhuma pasta correspondente a '{config['label']}' foi encontrada em 'PROJETOS'.\nVerificadas: {', '.join(config['aliases'])}")


    def initUI(self):
        self.setWindowTitle("Controle de Projetos")
        self.setGeometry(100, 100, 800, 720)
        self.setMinimumSize(700, 500)
        if os.path.exists(ICON_FILE):
            self.setWindowIcon(QIcon(ICON_FILE))
        else:
            print(f"Aviso: Arquivo de ícone '{ICON_FILE}' não encontrado.")

        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.mainLayout.setSpacing(10)

        topBarLayout = QHBoxLayout()

        self.rootSelector = QComboBox(self)
        self.rootSelector.activated.connect(self.on_root_selected)

        self.categorySelector = QComboBox(self)
        self.categorySelector.activated.connect(self.filter_projects_by_category)

        self.projectSelector = QComboBox(self)
        self.projectSelector.setFont(QFont("Arial", 10))
        self.projectSelector.setEditable(True)
        self.projectSelector.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.projectSelector.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.projectSelector.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.projectSelector.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.projectSelector.activated.connect(self.select_project)
        self.line_edit_filter = LineEditClickFilter(self.projectSelector.lineEdit())
        self.projectSelector.lineEdit().installEventFilter(self.line_edit_filter)

        topBarLayout.addWidget(self.rootSelector, 1)
        topBarLayout.addWidget(self.categorySelector, 1)
        topBarLayout.addWidget(self.projectSelector, 2)

        button_width = 35

        self.newProjectButton = QPushButton("+", self)
        self.newProjectButton.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.newProjectButton.setFixedWidth(button_width)
        self.newProjectButton.setToolTip("Criar novo projeto")
        self.newProjectButton.setObjectName("topBarBtn")
        self.newProjectButton.clicked.connect(self.create_new_project)
        topBarLayout.addWidget(self.newProjectButton)

        self.refreshButton = QPushButton("↻", self)
        self.refreshButton.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.refreshButton.setFixedWidth(button_width)
        self.refreshButton.setToolTip("Atualizar listas")
        self.refreshButton.setObjectName("topBarBtn")
        self.refreshButton.clicked.connect(self.refresh_all)
        topBarLayout.addWidget(self.refreshButton)

        self.openProjectRootButton = QPushButton("📁", self)
        self.openProjectRootButton.setFont(QFont("Arial", 14))
        self.openProjectRootButton.setFixedWidth(button_width)
        self.openProjectRootButton.setToolTip("Abrir pasta raiz do projeto")
        self.openProjectRootButton.setObjectName("topBarBtn")
        self.openProjectRootButton.setEnabled(False)
        self.openProjectRootButton.clicked.connect(self.open_project_root_folder)
        topBarLayout.addWidget(self.openProjectRootButton)

        self.searchRootButton = QPushButton("🔎", self)
        self.searchRootButton.setFont(QFont("Arial", 14))
        self.searchRootButton.setFixedWidth(button_width)
        self.searchRootButton.setToolTip("Busca avançada na raiz selecionada\nShift+clique para uma pasta específica")
        self.searchRootButton.setObjectName("topBarBtn")
        self.searchRootButton.setEnabled(True)
        self.searchRootButton.clicked.connect(self.on_search_root_clicked)
        topBarLayout.addWidget(self.searchRootButton)

        self.mainLayout.addLayout(topBarLayout)

        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.mainLayout.addWidget(separator)
        self.header_label = QLabel(self)
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_label.setFixedHeight(256)
        self.header_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.header_label.mousePressEvent = self.on_header_clicked
        self.header_label.hide()
        self.mainLayout.addWidget(self.header_label)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setStyleSheet("border: none; background-color: transparent;")
        self.scrollContentWidget = QWidget()
        self.subProjectArea = QVBoxLayout(self.scrollContentWidget)
        self.scrollArea.setWidget(self.scrollContentWidget)
        self.mainLayout.addWidget(self.scrollArea)

    def populate_roots(self):
        self.rootSelector.clear()
        self.rootSelector.addItem("Todas as Raízes", self.ROOT_PATHS)
        for root_path in self.ROOT_PATHS:
            display_name = os.path.basename(root_path.rstrip(os.sep))
            self.rootSelector.addItem(display_name, [root_path])

    def on_root_selected(self, index):
        self.populate_categories()
        self.filter_projects_by_category()

    def on_search_root_clicked(self):
        # O som já será tocado dentro de `open_search_dialog`,
        # então não precisamos adicioná-lo aqui novamente.
        modifiers = QApplication.keyboardModifiers()

        if modifiers == Qt.KeyboardModifier.ShiftModifier:
            custom_path = QFileDialog.getExistingDirectory(self, "Selecione uma Pasta para Pesquisar", os.path.expanduser('~'))
            if custom_path:
                self.open_search_dialog(base_path=custom_path)
        else:
            paths_to_scan = self.rootSelector.currentData()
            if paths_to_scan and len(paths_to_scan) == 1:
                root_to_search = paths_to_scan[0]
                self.open_search_dialog(base_path=root_to_search)

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        self.setPalette(dark_palette)
        self.setStyleSheet("""
            QWidget { font-family: Arial; font-size: 11pt; color: #f0f0f0; }
            QToolTip { border: 1px solid #555; background-color: #3c3c3c; color: #f0f0f0; padding: 5px; border-radius: 4px; }
            QComboBox { border: 1px solid #555; border-radius: 8px; padding: 5px 10px; background-color: #3c3c3c; }
            QComboBox:hover { border: 1px solid #777; }
            QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 25px; border-left-width: 1px; border-left-color: #555; border-left-style: solid; border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
            QComboBox QAbstractItemView { background-color: #3c3c3c; border: 1px solid #555; selection-background-color: #0078d7; outline: 0px; }
            QComboBox QLineEdit { background-color: #3c3c3c; }
            QPushButton#topBarBtn, QPushButton#dialogButton { border: 1px solid #555; background-color: #555; border-radius: 8px; padding: 5px; }
            QPushButton#topBarBtn:hover, QPushButton#dialogButton:hover { background-color: #666; }
            QPushButton#topBarBtn:pressed, QPushButton#dialogButton:pressed { background-color: #777; }
            QPushButton#linkBtn { border: none; background-color: transparent; color: #57a8e8; text-align: left; padding: 2px; }
            QPushButton#linkBtn:hover { color: #80c4ff; text-decoration: underline; }
            QPushButton#appBtn { border: 1px solid #4f4f4f; background-color: #4f4f4f; border-radius: 6px; padding: 2px; font-weight: bold; min-width: 25px; }
            QPushButton#appBtn:hover { background-color: #606060; }
            QPushButton#appBtn:pressed { background-color: #707070; }
            QLineEdit { border: 1px solid #555; padding: 5px; border-radius: 8px; background-color: #3c3c3c; }
            QScrollBar:vertical { border: none; background: #3c3c3c; width: 12px; margin: 0px; }
            QScrollBar::handle:vertical { background: #666; min-height: 20px; border-radius: 6px; }
            QScrollBar::handle:vertical:hover { background: #777; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

    def display_subprojects(self):
        self.clear_layout(self.subProjectArea)
        self.search_buttons.clear()
        if not self.active_project_path:
            self.subProjectArea.addStretch()
            return

        self.search_status_label = QLabel("")
        self.search_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.search_status_label.hide()
        self.subProjectArea.addWidget(self.search_status_label)

        new_subproject_button = QPushButton("Criar Novo Subprojeto")
        new_subproject_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        new_subproject_button.setStyleSheet(""" QPushButton { background-color: #555; color: white; font-weight: bold; border: 1px solid #555; border-radius: 8px; padding: 8px; margin-bottom: 5px; } QPushButton:hover { background-color: #666; } QPushButton:pressed { background-color: #777; } """)
        new_subproject_button.clicked.connect(self.create_new_subproject)
        self.subProjectArea.addWidget(new_subproject_button)

        search_widget = QWidget()
        search_layout = QHBoxLayout(search_widget)
        search_layout.setContentsMargins(0,0,0,0)
        search_layout.setSpacing(10)
        for config in FILENAME_SEARCH_CONFIG:
            search_btn = QPushButton(config["label"])
            search_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            search_btn.setToolTip(config.get("tooltip", ""))
            search_btn.setStyleSheet(""" QPushButton { border: 1px solid #404040; background-color: #404040; border-radius: 8px; padding: 6px; font-weight: bold; color: white; } QPushButton:hover { background-color: #505050; } QPushButton:pressed { background-color: #454545; } """)
            predefined_terms = config.get("search_terms")
            search_btn.clicked.connect(lambda checked, terms=predefined_terms: self.open_search_dialog(terms))
            search_layout.addWidget(search_btn)
            self.search_buttons.append(search_btn)
        self.subProjectArea.addWidget(search_widget)

        grouped_subprojects = {cat['display']: [] for cat in SUBPROJECT_CATEGORIES_CONFIG}
        try:
            top_level_items = [d for d in os.listdir(self.active_project_path) if os.path.isdir(os.path.join(self.active_project_path, d))]
        except OSError:
            top_level_items = []

        for item_name in top_level_items:
            item_path = os.path.join(self.active_project_path, item_name)
            if is_year_folder(item_name):
                year_path = item_path
                for category_name in os.listdir(year_path):
                    category_path = os.path.join(year_path, category_name)
                    if not os.path.isdir(category_path): continue

                    display_category = ALIAS_TO_CATEGORY_MAP.get(category_name.lower())
                    if display_category:
                        for subproject_name in os.listdir(category_path):
                            full_path = os.path.join(category_path, subproject_name)
                            if os.path.isdir(full_path):
                                grouped_subprojects[display_category].append({
                                    "display": f"{item_name} / {subproject_name}", "path": full_path
                                })

            display_category = ALIAS_TO_CATEGORY_MAP.get(item_name.lower())
            if display_category:
                category_path = item_path
                for sub_item_name in os.listdir(category_path):
                    sub_item_path = os.path.join(category_path, sub_item_name)
                    if os.path.isdir(sub_item_path) and not is_year_folder(sub_item_name):
                         grouped_subprojects[display_category].append({
                            "display": sub_item_name, "path": sub_item_path
                         })

        row_index = 0
        for category_config in SUBPROJECT_CATEGORIES_CONFIG:
            display_name = category_config['display']
            collapsible_section = CollapsibleSection(display_name.upper(), self)
            self.subProjectArea.addWidget(collapsible_section)

            subproject_list = grouped_subprojects.get(display_name, [])
            if not subproject_list:
                collapsible_section.add_widget(QLabel("Nenhum subprojeto encontrado."))
            else:
                for sp_data in sorted(subproject_list, key=lambda x: x['display'].lower()):
                    row_widget = QWidget()
                    if row_index % 2 == 0:
                        row_widget.setStyleSheet("QWidget { background-color: #2e2e2e; border-radius: 4px; } QWidget:hover { background-color: #505050; }")
                    else:
                        row_widget.setStyleSheet("QWidget { background-color: #404040; border-radius: 4px; } QWidget:hover { background-color: #505050; }")

                    row_layout = QHBoxLayout(row_widget)
                    row_layout.setContentsMargins(4,4,4,4)
                    row_layout.setSpacing(5)

                    sp_button = QPushButton(sp_data["display"])
                    sp_button.setObjectName("linkBtn")
                    sp_button.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                    sp_button.clicked.connect(lambda checked, path=sp_data["path"]: self.open_folder(path))

                    row_layout.addWidget(sp_button, 1)
                    for config_app in APP_BUTTONS_CONFIG:
                        app_btn = QPushButton(config_app["label"])
                        app_btn.setObjectName("appBtn")
                        app_btn.setToolTip(config_app["tooltip"])
                        app_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

                        if config_app.get("type") == "search":
                            app_btn.clicked.connect(lambda checked, path=sp_data["path"]: self.open_search_dialog(base_path=path))
                        else:
                            app_btn.clicked.connect(lambda checked, path=sp_data["path"], cfg=config_app: self.on_app_button_clicked(path, cfg))

                        row_layout.addWidget(app_btn)
                        self.search_buttons.append(app_btn)
                    collapsible_section.add_widget(row_widget)
                    row_index += 1

        self.subProjectArea.addStretch()

    def select_project(self, index=None):
        if index is not None and index <= 0:
            self.active_project_path = None
        else:
            self.active_project_path = self.projectSelector.currentData()

        self.openProjectRootButton.setEnabled(bool(self.active_project_path))
        self.update_header()
        self.display_subprojects()

        self.projectSelector.clearFocus()

    def update_header(self):
        self.header_label.hide()
        self.optimized_logo_path = None
        self.original_logo_path = None
        if self.active_project_path:
            logo_path = self.find_best_logo_path(self.active_project_path)
            if logo_path:
                self.original_logo_path = logo_path
                self.optimized_logo_path = self.create_optimized_logo_copy(logo_path)
                if self.optimized_logo_path:
                    self.set_header_pixmap()

    def set_header_pixmap(self):
        if not self.optimized_logo_path: return
        pixmap = QPixmap(self.optimized_logo_path)
        margins = self.mainLayout.contentsMargins()
        available_width = self.width() - (margins.left() + margins.right())
        scaled_pixmap = pixmap.scaled(available_width, self.header_label.height(),
                                      Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                      Qt.TransformationMode.SmoothTransformation)
        self.header_label.setPixmap(scaled_pixmap)
        self.header_label.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.optimized_logo_path and self.header_label.isVisible():
            self.set_header_pixmap()

    def on_header_clicked(self, event):
        if self.original_logo_path:
            folder_of_logo = os.path.dirname(self.original_logo_path)
            self.open_folder(folder_of_logo)

    def create_optimized_logo_copy(self, original_path):
        try:
            pixmap = QPixmap(original_path)
            if pixmap.isNull():
                print(f"Aviso: Falha ao carregar a imagem do logo em '{original_path}'.")
                return None
            max_width, max_height = 1920, 1080
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(max_width, max_height,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation)
            temp_path = os.path.join(tempfile.gettempdir(), "proj_manager_temp_logo.png")
            if pixmap.save(temp_path, "PNG"):
                return temp_path
            else:
                print(f"Aviso: Falha ao salvar a imagem temporária em '{temp_path}'.")
                return None
        except Exception as e:
            print(f"Erro ao processar imagem do logo: {e}")
            return None

    def _find_newest_logo_in_path(self, search_path):
        """
        Função auxiliar para encontrar o arquivo de logo mais recente em um caminho específico.
        Retorna o caminho do arquivo ou None se não encontrar.
        """
        if not os.path.exists(search_path):
            return None

        latest_logo = None
        latest_mtime = 0
        for root, _, files in os.walk(search_path):
            for file in files:
                if "logo" in file.lower() and file.lower().endswith(('.png', '.jpg')):
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_logo = full_path
                    except OSError:
                        continue
        return latest_logo

    def find_best_logo_path(self, project_path):
        # 1. Prioridade 1: Pasta SAIDA/LOGO
        logo = self._find_newest_logo_in_path(os.path.join(project_path, "SAIDA", "LOGO"))
        if logo:
            return logo

        # 2. Prioridade 2: Pasta SAIDA/LOGOS
        logo = self._find_newest_logo_in_path(os.path.join(project_path, "SAIDA", "LOGOS"))
        if logo:
            return logo

        # 3. Prioridade 3: Pasta SAIDA (geral)
        logo = self._find_newest_logo_in_path(os.path.join(project_path, "SAIDA"))
        if logo:
            return logo

        # 4. Fallback: Busca geral no projeto (excluindo a pasta SAIDA)
        best_match = None
        min_depth = float('inf')
        for root, dirs, files in os.walk(project_path):
            if "SAIDA" in dirs:
                dirs.remove("SAIDA")

            relative_path = os.path.relpath(root, project_path)
            current_depth = 0 if relative_path == '.' else len(relative_path.split(os.sep))

            if HEADER_LOGO_SEARCH_DEPTH > 0 and current_depth >= HEADER_LOGO_SEARCH_DEPTH:
                dirs[:] = []
                continue

            for file in files:
                if "logo" in file.lower() and file.lower().endswith(('.png', '.jpg')):
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if current_depth < min_depth:
                            min_depth = current_depth
                            best_match = (full_path, mtime)
                        elif current_depth == min_depth:
                            if best_match is None or mtime > best_match[1]:
                                best_match = (full_path, mtime)
                    except OSError:
                        continue

        return best_match[0] if best_match else None

    def execute_search(self, base_path, search_params):
        self.waiting_dialog = WaitingDialog(self)

        self.thread = QThread()
        self.worker = SearchWorker(base_path, search_params)
        self.worker.moveToThread(self.thread)

        self.waiting_dialog.rejected.connect(self.cancel_search)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_search_results)

        self.worker.finished.connect(self.thread.quit)

        self.thread.start()
        self.waiting_dialog.exec()

    def cancel_search(self):
        """Interrompe a thread de busca quando o usuário cancela."""
        print("Sinal de cancelamento recebido. Interrompendo a busca...")
        if self.thread and self.thread.isRunning():
            if self.worker:
                self.worker.cancel()
            self.thread.quit()
            self.thread.wait()

        if hasattr(self, 'waiting_dialog'):
            self.waiting_dialog.accept()

    def handle_search_results(self, found_items):
        """Lida com os resultados ao final da busca."""
        if hasattr(self, 'waiting_dialog'):
            self.waiting_dialog.accept()

        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None

        if found_items is None:
            return

        if not found_items:
            QMessageBox.information(self, "Busca Concluída", "Nenhum item foi encontrado.")
            return

        results_dialog = SearchResultsDialog(found_items, self)
        results_dialog.exec()

    def find_and_open_folders_simple(self, base_path, search_params):
        found_folders = _perform_search(base_path, search_params)
        self._handle_folder_opening(found_folders, search_params)

    def launch_search_process(self, base_path, search_params):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.warning(self, "Busca em Andamento", "Uma busca já está em andamento.")
            return

        for btn in self.search_buttons: btn.setEnabled(False)
        self.search_status_label.setText("Buscando...")
        self.search_status_label.show()

        self.thread = QThread()
        self.worker = SearchWorker(base_path, search_params)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_final_search_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @pyqtSlot(object)
    def handle_final_search_result(self, result_tuple):
        found_folders, search_params = result_tuple
        self.search_status_label.hide()
        for btn in self.search_buttons: btn.setEnabled(True)

        self._handle_folder_opening(found_folders, search_params)

        self.thread = None
        self.worker = None

    def _handle_folder_opening(self, folders_to_open, search_params={}):
        if not folders_to_open:
            term = search_params.get("term") or search_params.get("filter_text")
            QMessageBox.information(self, "Busca Concluída", f"Nenhum item correspondente a '{term}' foi encontrado.")
            return

        folder_count = len(folders_to_open)
        limit = 5
        if folder_count > limit:
            reply = QMessageBox.question(self, "Aviso: Múltiplas Janelas", f"A busca encontrou {folder_count} pastas.\n\nDeseja abrir todas as janelas?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No:
                return
        for folder in folders_to_open:
            self.open_folder(folder)

    def open_project_root_folder(self):
        if self.active_project_path: self.open_folder(self.active_project_path)

    def open_folder(self, path):
        if not os.path.exists(path):
            QMessageBox.critical(self, "Erro", f"O caminho não foi encontrado:\n{path}")
            return
        try:
            if sys.platform == 'win32': os.startfile(os.path.realpath(path))
            elif sys.platform == 'darwin': subprocess.Popen(['open', path])
            else: subprocess.Popen(['xdg-open', path])
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Não foi possível abrir a pasta:\n{e}")

    def populate_projects(self):
        self.all_projects_data.clear()

        paths_to_scan = self.rootSelector.currentData()
        if not paths_to_scan:
            paths_to_scan = self.ROOT_PATHS

        selected_category = self.categorySelector.currentText()
        categories_to_scan = PROJECT_CATEGORIES if selected_category == "Todas as Categorias" else [selected_category]

        for root_path in paths_to_scan:
            if not os.path.exists(root_path):
                continue

            for category_folder in categories_to_scan:
                category_path = os.path.join(root_path, category_folder)
                if os.path.isdir(category_path):
                    for project_folder in sorted(os.listdir(category_path)):
                        project_path = os.path.join(category_path, project_folder)
                        if os.path.isdir(project_path):
                            display_name = f"{category_folder} / {project_folder}"
                            self.all_projects_data.append({"display": display_name, "path": project_path})

        self.update_project_selector()

    def update_project_selector(self):
        self.projectSelector.clear()
        self.projectSelector.addItem("Selecione ou digite para filtrar...", None)
        for project in self.all_projects_data:
            self.projectSelector.addItem(project["display"], project["path"])

    def filter_projects_by_category(self):
        self.active_project_path = None
        self.projectSelector.setCurrentIndex(0)
        self.update_header()
        self.display_subprojects()
        self.populate_projects()

    def create_new_project(self):
        dialog = ProjectCreatorDialog(self, root_paths=self.ROOT_PATHS)
        if dialog.exec():
            root, name, p_type = dialog.getProjectDetails()

            if not name:
                QMessageBox.warning(self, "Aviso", "O nome do projeto não pode ser vazio.")
                return
            if not p_type:
                QMessageBox.warning(self, "Aviso", "A categoria selecionada não é válida ou a pasta raiz está vazia.")
                return

            project_path = os.path.join(root, p_type, name)

            if os.path.exists(project_path):
                QMessageBox.warning(self, "Aviso", f"O projeto '{name}' já existe em '{project_path}'.")
                return
            try:
                os.makedirs(project_path)
                QMessageBox.information(self, "Sucesso", f"Projeto '{name}' criado com sucesso em '{project_path}'!")

                root_index = self.rootSelector.findData([root])
                if root_index != -1:
                    self.rootSelector.setCurrentIndex(root_index)

                self.populate_categories()

                self.categorySelector.setCurrentText(p_type)

                self.populate_projects()

                index = self.projectSelector.findData(project_path)
                if index != -1:
                    self.projectSelector.setCurrentIndex(index)
                    self.select_project(index)

            except OSError as e:
                 QMessageBox.critical(self, "Erro", f"Não foi possível criar a pasta do projeto: {e}")

    def getProjectDetails(self):
        return (self.rootPathComboBox.currentData(),
                self.projectNameLineEdit.text(),
                self.projectTypeComboBox.currentText())



    def create_new_subproject(self):
        if not self.active_project_path:
            QMessageBox.warning(self, "Aviso", "Selecione um projeto primeiro.")
            return

        year_folders = []
        try:
            for item in os.listdir(self.active_project_path):
                if os.path.isdir(os.path.join(self.active_project_path, item)) and is_year_folder(item):
                    year_folders.append(item)
        except OSError as e:
            print(f"Erro ao ler pastas de ano: {e}")

        dialog = SubProjectCreatorDialog(self, year_folders=year_folders)
        if dialog.exec():
            name, category, selected_year = dialog.getSubProjectDetails()

            if not name:
                QMessageBox.warning(self, "Aviso", "O nome do subprojeto não pode ser vazio.")
                return

            if selected_year == "__CREATE_NEW__":
                current_year_str = str(datetime.now().year)
                text, ok = QInputDialog.getText(self, "Criar Novo Ano",
                                                "Digite o ano (4 dígitos):",
                                                QLineEdit.EchoMode.Normal,
                                                current_year_str)
                if ok and text:
                    if text.isdigit() and len(text) == 4:
                        selected_year = text
                    else:
                        QMessageBox.warning(self, "Entrada Inválida", "Por favor, digite um ano válido com 4 dígitos.")
                        return
                else:
                    return

            if selected_year:
                base_path = os.path.join(self.active_project_path, selected_year, category)
            else:
                base_path = os.path.join(self.active_project_path, category)

            subproject_path = os.path.join(base_path, name)

            if os.path.exists(subproject_path):
                QMessageBox.warning(self, "Aviso", f"O subprojeto '{name}' já existe neste local.")
                return
            try:
                os.makedirs(base_path, exist_ok=True)

                os.makedirs(subproject_path)
                for folder, subfolders in SUBPROJECT_DEFAULT_FOLDERS.items():
                    base_folder_path = os.path.join(subproject_path, folder)
                    os.makedirs(base_folder_path, exist_ok=True)
                    for subfolder in subfolders:
                        os.makedirs(os.path.join(base_folder_path, subfolder), exist_ok=True)
                QMessageBox.information(self, "Sucesso", f"Subprojeto '{name}' e sua estrutura de pastas foram criados com sucesso.")
                self.display_subprojects()
            except OSError as e:
                QMessageBox.critical(self, "Erro", f"Não foi possível criar a pasta do subprojeto: {e}")

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None: widget.deleteLater()
                else:
                    child_layout = item.layout()
                    if child_layout is not None: self.clear_layout(child_layout)

    def refresh_all(self):
        current_path = self.projectSelector.currentData()
        self.filter_projects_by_category()
        self.categorySelector.setCurrentIndex(0)

        index = self.projectSelector.findData(current_path)
        if index != -1:
            self.projectSelector.setCurrentIndex(index)
        self.select_project(index=self.projectSelector.currentIndex())
        print("Listas atualizadas.")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    if USE_MULTIPROCESSING:
        multiprocessing.freeze_support()
    ex = ProjectControlApp()
    ex.show()
    sys.exit(app.exec())
