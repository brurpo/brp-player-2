import sys
import os
import mpv
import tempfile
import psutil
import numpy as np
import uuid
import PyOpenColorIO as OCIO
try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE = True
except ImportError:
    OPENEXR_AVAILABLE = False
import shutil # Importa shutil para verificar a existência do ffmpeg
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread, QObject, pyqtSlot, QEvent
from PyQt6.QtGui import QAction, QPixmap, QColor, QBrush, QIcon
from PyQt6.QtWidgets import (QApplication, QDialog, QListWidgetItem, QDialogButtonBox, QVBoxLayout, QFormLayout, QLabel, QComboBox,
                             QFileDialog, QMessageBox, QProgressDialog, QSystemTrayIcon, QMenu, QStyle)
from ui_player import Ui_MainWindow
from ocio_logic import OcioManager
from filtering import apply_lut_filter
from ui_export_dialog import ExportDialog
from export_logic import ExportWorker
import subprocess
import json
import multiprocessing
import re
import ffmpeg
from ui_batch_export_dialog import BatchExportDialog
from time import sleep
from ui_effects_dialog import EffectsDialog
from effects_logic import EffectsManager
from PyQt6.QtNetwork import QTcpServer, QHostAddress
from ui_log_dialog import LogDialog
import traceback
import time
try:
    import av
except ImportError:
    av = None
APP_PORT = 61337
APP_SECRET_KEY = "BRP_PLAYER_IPC_KEY"
FILENAME_MAX_CHARS = 20
INFO_MAX_CHARS = 12
LOG_MAX_LINES = 5000

def check_vulkan_support():
    """
    Verifica se o MPV consegue inicializar a API Vulkan e encontrar dispositivos.
    Retorna True se o suporte Vulkan estiver funcional, False caso contrário.
    """
    try:
        # Usamos --no-config para uma saída limpa e --vulkan-device=help para listar.
        # Se este comando falhar, o Vulkan não está disponível para o MPV.
        cmd = ['mpv', '--no-config', '--vo=gpu-next', '--gpu-api=vulkan', '--vulkan-device=help']

        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        # Se o processo rodar sem erro (returncode 0), o Vulkan está OK.
        subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo, encoding='utf-8')
        print("Detecção de Vulkan: Suporte funcional encontrado.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # Ocorreu um erro, o que significa que o MPV não conseguiu usar a API Vulkan.
        print(f"Detecção de Vulkan: Suporte não encontrado ou com erro. {e}")
        return False

def get_gpu_devices():
    """
    Detecta dispositivos Vulkan disponíveis usando a API Python do Vulkan.
    Retorna uma lista de dicionários com os resultados.
    """
    devices = {'vulkan': [], 'd3d11': []} # Mantemos a estrutura, mas só popularemos o Vulkan.

    try:
        # Tenta importar a biblioteca vulkan. Se não estiver instalada, não há suporte.
        import vulkan as vk

        # --- Lógica Padrão de Inicialização do Vulkan ---
        app_info = vk.VkApplicationInfo(pApplicationName="MPV_Player_Check", apiVersion=vk.VK_API_VERSION_1_0)
        instance_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(instance_info, None)

        physical_devices = vk.vkEnumeratePhysicalDevices(instance)

        print(f"Vulkan detectou {len(physical_devices)} dispositivo(s).")

        for device in physical_devices:
            properties = vk.vkGetPhysicalDeviceProperties(device)
            device_name = properties.deviceName
            device_type_enum = properties.deviceType

            # Mapeia o enum do Vulkan para nossas strings amigáveis
            if device_type_enum == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                device_type = 'discrete'
            elif device_type_enum == vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                device_type = 'integrated'
            elif device_type_enum == vk.VK_PHYSICAL_DEVICE_TYPE_CPU:
                device_type = 'cpu'
            else:
                device_type = 'other'

            devices['vulkan'].append({'name': device_name, 'type': device_type})

    except ImportError:
        print("Biblioteca 'vulkan' não instalada. Detecção de GPU Vulkan desativada.")
    except Exception as e:
        print(f"Erro ao inicializar a API Vulkan para detectar dispositivos: {e}")
        # Se ocorrer qualquer erro aqui, significa que o Vulkan não está funcional.

    print(f"Dispositivos de GPU detectados: {devices}")
    return devices

def detect_spirv_compiler():
    """Verifica se o compilador SPIR-V (spirv-cross) está disponível no PATH do sistema."""
    # spirv-cross é o nome comum do executável do compilador/refletor SPIR-V
    # que vem com o SDK do Vulkan.
    if shutil.which('spirv-cross'):
        print("spirv-cross detectado. Padrão para Vulkan.")
        return True
    print("spirv-cross não detectado. Padrão para OpenGL.")
    return False

def detect_hardware_encoders():
    """
    Verifica quais encoders de hardware estão disponíveis tanto no FFmpeg quanto no sistema.
    Retorna um dicionário indicando suporte REAL para nvidia, intel, e amd.
    """
    # Passo 1: Verifica o que o build do FFmpeg suporta
    try:
        cmd = ['ffmpeg', '-v', 'quiet', '-encoders']
        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo, encoding='utf-8')
        output = result.stdout
        ffmpeg_support = {
            "nvidia": bool(re.search(r'h264_nvenc|hevc_nvenc|av1_nvenc', output)),
            "intel": bool(re.search(r'h264_qsv|hevc_qsv|av1_qsv', output)),
            "amd": bool(re.search(r'h264_amf|hevc_amf|av1_amf', output)),
        }
    except Exception as e:
        print(f"Não foi possível executar ffmpeg para detectar encoders, assumindo nenhum. Erro: {e}")
        return {"nvidia": False, "intel": False, "amd": False}

    # Passo 2: Verifica qual hardware está fisicamente presente usando nossa função existente
    gpu_list = get_gpu_devices().get('vulkan', [])

    # Passo 3: Cruza as informações para obter o resultado final e realista
    available_encoders = {
        "nvidia": ffmpeg_support["nvidia"] and any("nvidia" in gpu['name'].lower() for gpu in gpu_list),
        "intel": ffmpeg_support["intel"] and any("intel" in gpu['name'].lower() for gpu in gpu_list),
        "amd": ffmpeg_support["amd"] and any("amd" in gpu['name'].lower() or "radeon" in gpu['name'].lower() for gpu in gpu_list),
    }

    print(f"Suporte FFmpeg: {ffmpeg_support}")
    print(f"GPUs Físicas Encontradas: {[gpu['name'] for gpu in gpu_list]}")
    print(f"Encoders de hardware realmente disponíveis: {available_encoders}")

    return available_encoders
def truncate_text(text, max_length):
    """Corta o texto se ele exceder o comprimento máximo e adiciona '...'."""
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def prerender_frame_task(in_path, out_path, layer_name, progress_queue):
    """Executa o FFmpeg para um único quadro e reporta o progresso."""
    try:
        cmd = [
            'ffmpeg', '-y',
            '-layer', layer_name,
            '-i', in_path,
            '-frames:v', '1',
            '-c:v', 'exr', '-pix_fmt', 'rgbaf32le',
            out_path
        ]

        # --- CORREÇÃO AQUI: Usa STARTUPINFO para esconder a janela de forma segura ---
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE

        # Remove a flag 'creationflags' e adiciona 'startupinfo'
        result = subprocess.run(cmd, check=True, startupinfo=startupinfo,
                                capture_output=True, text=True)

        progress_queue.put(out_path)
    except subprocess.CalledProcessError as e:
        print(f"Falha ao renderizar o frame {in_path}: {e.stderr}")
        progress_queue.put(None)
    except Exception as e:
        print(f"Falha inesperada ao renderizar o frame {in_path}: {e}")
        progress_queue.put(None)

class PlayerResetWorker(QObject):
    finished = pyqtSignal(object) # Sinal que emitirá a nova instância do player

    def __init__(self, old_player, mpv_options):
        super().__init__()
        self.old_player = old_player
        self.mpv_options = mpv_options

    @pyqtSlot()
    def run(self):
        """Encerra o player antigo e cria um novo em segundo plano."""
        if self.old_player:
            try:
                self.old_player.terminate()
            except Exception as e:
                print(f"Erro ao terminar a instância antiga do MPV: {e}")

        try:
            new_player = mpv.MPV(**self.mpv_options)
            self.finished.emit(new_player)
        except Exception as e:
            print(f"Erro ao criar nova instância do MPV: {e}")
            self.finished.emit(None)

class PrerenderManagerWorker(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)

    def __init__(self, image_files, layer_name, temp_dir):
        super().__init__()
        self.image_files = image_files
        self.layer_name = layer_name
        self.temp_dir = temp_dir
        self.is_cancelled = False
        self.pool = None

    def cancel(self):
        self.is_cancelled = True
        if self.pool:
            self.pool.terminate()

    @pyqtSlot()
    def run(self):
        tasks = []
        for i, exr_path in enumerate(self.image_files):
            output_filename = f"frame_{i:06d}.exr"
            output_path = os.path.join(self.temp_dir, output_filename)
            tasks.append((exr_path, output_path, self.layer_name))

        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        tasks_with_queue = [t + (progress_queue,) for t in tasks]
        new_image_files = []

        try:
            self.pool = multiprocessing.Pool()

            # --- CORREÇÃO AQUI: Adiciona chunksize=1 ---
            # Isso força o pool a distribuir as tarefas uma a uma,
            # mantendo todos os núcleos ocupados.
            self.pool.starmap_async(prerender_frame_task, tasks_with_queue, chunksize=1)

            self.pool.close()

            completed_tasks = 0
            total_tasks = len(tasks)
            while completed_tasks < total_tasks:
                if self.is_cancelled:
                    break
                try:
                    result_path = progress_queue.get(timeout=0.1)
                    if result_path:
                        new_image_files.append(result_path)
                    completed_tasks += 1
                    self.progress.emit(int((completed_tasks / total_tasks) * 100))
                except multiprocessing.queues.Empty:
                    continue

            if self.is_cancelled:
                self.pool.terminate()

            self.pool.join()

        except Exception as e:
            print(f"Erro no pool de pré-render: {e}")
            if self.pool:
                self.pool.terminate()
                self.pool.join()

        self.finished.emit(sorted(new_image_files) if not self.is_cancelled else [])




class LayerSelectionDialog(QDialog):
    def __init__(self, parent, layers):
        super().__init__(parent)
        self.setWindowTitle("EXR layer selection")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        formLayout = QFormLayout()

        label = QLabel("Multiple layers detected.\nPlease, choose which one to load:")
        self.layer_combo = QComboBox()
        self.layer_combo.addItems(layers)
        formLayout.addRow(label)
        formLayout.addRow("Layer:", self.layer_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addLayout(formLayout)
        layout.addWidget(buttons)

    def get_selected_layer(self):
        return self.layer_combo.currentText()

class VideoPlayerApp(Ui_MainWindow):
    video_ended = pyqtSignal()
    fullscreen_state_changed = pyqtSignal(bool)
    was_window_fullscreen = False


    def __init__(self, base_path="."):
        super().__init__()
        self._is_quitting = False # Adicione esta flag
        self.base_path = base_path
        self.source_color_info = {}
        self.detected_encoders = detect_hardware_encoders()
        self.gpu_devices = get_gpu_devices()
        self.vulkan_supported = bool(self.gpu_devices.get('vulkan'))

        # Define a API padrão baseada na detecção real
        self.default_gpu_api = 'vulkan' if self.vulkan_supported else 'opengl'

        # Lógica para priorizar a GPU dedicada como padrão
        self.default_gpu_device_name = None
        if self.vulkan_supported:
            dedicated_gpu = next((d for d in self.gpu_devices['vulkan'] if d['type'] == 'discrete'), None)
            if dedicated_gpu:
                self.default_gpu_device_name = dedicated_gpu['name']
            else: # Se não houver GPU dedicada, pega a primeira da lista
                self.default_gpu_device_name = self.gpu_devices['vulkan'][0]['name']




        #self.setup_tray_icon()
        self.ipc_server = None # Adicione esta linha
        self.setup_tray_icon()

        self.player = None
        self.ocio_manager = OcioManager(self)
        self.is_interlaced = False
        self.temp_render_dir = None
        self.temp_dirs_to_cleanup = []
        self.temp_render_dir = None
        self.effects_manager = None
        self.is_repeat_on = True
        self.total_frames = 0
        self.framerate = 0.0
        self.original_framerate = 0.0
        self.video_info_loaded = False
        self.last_volume = 100
        self.current_media_path = None
        self.is_paused = True
        self.is_image_sequence = False
        self.image_files = []
        self.current_image_index = 0
        self.ffmpeg_concat_file = None # Para guardar o caminho do arquivo de lista do ffmpeg
        self.selected_exr_layer = None
        self.in_point_frame = 0
        self.out_point_frame = 0
        self.is_panning = False
        self.last_mouse_pos = None
        self.zoom_level = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_unscaled = False
        app_data_path = os.path.join(os.getenv('LOCALAPPDATA'), "BRP_Player")
        os.makedirs(app_data_path, exist_ok=True)

        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.active_shader_config = []
        self.shader_temp_dir = None
        self.current_upscale_factor = None
        self.single_click_timer = QTimer(self)
        # --- LINHA CORRIGIDA (VERSÃO FINAL) ---
        # Pega o intervalo de tempo de clique duplo diretamente da aplicação.
        QApplication.setDoubleClickInterval(200)
        self.single_click_timer.setInterval(QApplication.doubleClickInterval())
        self.single_click_timer.setSingleShot(True)
        self.single_click_timer.timeout.connect(self.play_pause)


        self.setup_logic_connections()
        self.setup_mpv_events()
        self.setup_mpv_property_observers()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_ui)
        self.image_seq_timer = QTimer(self)
        self.image_seq_timer.timeout.connect(self.next_image)
        self.video_frame.setFocus()
        self.audio_device_changed = False
        self.export_thread = None
        self.has_alpha = False
        self.source_resolution = (0, 0)
        self.setAcceptDrops(True)
        self.update_play_pause_icon()
        self.log_buffer = []
        self.center_on_screen()

        if len(sys.argv) > 1:
            filepath = sys.argv[1]
            if os.path.exists(filepath):
                # Usa um QTimer para garantir que a janela principal esteja totalmente
                # carregada antes de tentar carregar o vídeo.
                print(f"Arquivo recebido via argumento: {filepath}")
                QTimer.singleShot(100, lambda: self.handle_file_argument(filepath))

    def _detect_interlacing_fast(self, filepath):
        """
        Detecção rápida sem cache por filename:
        1) ffprobe stream.field_order (não decodifica frames)
        2) fallback: ffprobe -show_frames em ~6 frames (barato)
        3) se incerto, assume progressivo (MPV pode ajustar depois)
        """
        start_time = time.perf_counter()
        if not filepath or self.is_image_sequence:
            return False

        startupinfo = None
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        # 1) Stream-only (rapidíssimo, não decodifica)
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=field_order,codec_name',
                '-of', 'json', filepath
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               startupinfo=startupinfo, timeout=1.0)
            data = json.loads(r.stdout)
            st = (data.get('streams') or [{}])[0]
            fo = (st.get('field_order') or '').lower()
            print(f"field: {fo}")
            if fo in ('progressive',):
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.4f} seconds")
                return False
            if fo in ('tt', 'bb', 'tb', 'bt'):
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.4f} seconds")
                print("interlaced")
                return True
            # 'unknown' => cai no fallback
        except Exception:
            pass

        # 2) Fallback em poucos frames (ainda externo ao processo Python)
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-read_intervals', '%+#6',  # ~6 frames do começo
                '-select_streams', 'v:0',
                '-show_entries', 'frame=interlaced_frame,top_field_first',
                '-of', 'json', filepath
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, check=True,
                               startupinfo=startupinfo, timeout=1.5)
            frames = (json.loads(r.stdout).get('frames') or [])
            interl = sum(1 for f in frames if str(f.get('interlaced_frame', 0)) in ('1', 'true'))
            print(f"field 2: {(interl / len(frames)) > 0.5}")
            if frames:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Execution time: {elapsed_time:.4f} seconds")
                print("interlaced")
                return (interl / len(frames)) > 0.5
        except Exception:
            pass

        # 3) Incerto: assume progressivo (MPV com deinterlace=auto cobre casos raros)
        print(f"not detected")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time:.4f} seconds")
        return False

    def _detect_interlacing(self, filepath):
        """
        Usa a biblioteca PyAV para decodificar alguns frames e verificar a flag
        'interlaced_frame'. Este é o método mais robusto e direto.
        """
        print(f"Analisando entrelaçamento para: {filepath} (método PyAV)")

        if av is None:
            print("Biblioteca PyAV não encontrada. Instale com 'pip install av'. Assumindo progressivo.")
            return False

        try:
            interlaced_count = 0
            frame_limit = 20  # Analisa os primeiros 200 frames

            # Abre o container de mídia (ex: .mkv, .mov)
            with av.open(filepath) as container:
                # Seleciona o primeiro stream de vídeo
                stream = container.streams.video[0]

                # Itera sobre os frames decodificados
                for i, frame in enumerate(container.decode(stream)):
                    # A propriedade 'interlaced_frame' é um booleano direto.
                    if frame.interlaced_frame:
                        interlaced_count += 1

                    # Para de analisar após o limite para ser mais rápido
                    if i >= frame_limit:
                        break

            # Se mais de 50% dos frames analisados forem marcados como entrelaçados,
            # consideramos o vídeo como entrelaçado.
            # Este limiar evita falsos positivos.
            is_interlaced = (interlaced_count / frame_limit) > 0.5 if frame_limit > 0 else False

            if is_interlaced:
                print(f"Decisão Final (PyAV): Vídeo é Entrelaçado ({interlaced_count}/{frame_limit} frames entrelaçados)")
                return True
            else:
                print(f"Decisão Final (PyAV): Vídeo é Progressivo ({interlaced_count}/{frame_limit} frames entrelaçados)")
                return False

        except Exception as e:
            print(f"Erro ao executar a detecção com PyAV: {e}. Assumindo progressivo.")
            return False


    def _get_master_mpv_options(self):
        """Constrói e retorna o dicionário de opções mestre para todas as instâncias do MPV."""
        selected_api = self.gpu_api_combo.currentData() or self.default_gpu_api

        try:
            total_ram_bytes = psutil.virtual_memory().total
            cache_size_kb = int((total_ram_bytes * 0.7) / 1024)
            demuxer_cache_bytes = int(total_ram_bytes * 0.8)
            mpv_options = {
                'wid': str(int(self.video_frame.winId())), 'vo': 'gpu-next', 'gpu-api': selected_api,
                'target_prim': 'bt.709', 'target_trc': 'bt.1886', 'tone-mapping': 'auto',
                'dither-depth': 'auto', 'log_handler': self._mpv_log_handler, 'keep-open': 'always',
                'hr-seek': 'always', 'cache': 'yes', 'demuxer-max-bytes': str(demuxer_cache_bytes),
                'demuxer-max-back-bytes': str(demuxer_cache_bytes),
            }
        except Exception as e:
            print(f"Não foi possível detectar a RAM, usando configurações de cache padrão. Erro: {e}")
            mpv_options = {
                'wid': str(int(self.video_frame.winId())), 'vo': 'gpu-next', 'gpu-api': selected_api,
                'target_prim': 'bt.709', 'target_trc': 'bt.1886', 'tone-mapping': 'auto',
                'dither-depth': 'auto', 'log_handler': self._mpv_log_handler, 'keep-open': 'always',
                'hr-seek': 'always', 'cache': 'yes',
            }

        # Adiciona a seleção de dispositivo Vulkan, se aplicável
        if selected_api == 'vulkan' and self.gpu_device_combo.isVisible():
            device_name = self.gpu_device_combo.currentData()
            if device_name:
                mpv_options['vulkan-device'] = device_name

        return mpv_options

    def _update_gpu_device_selector(self):
        """Atualiza a visibilidade e o conteúdo do seletor de GPU com base na API escolhida."""
        selected_api = self.gpu_api_combo.currentData()
        self.gpu_device_combo.clear()

        # O seletor de dispositivo só é relevante para Vulkan com esta nova lógica
        if selected_api == 'vulkan' and self.vulkan_supported:
            self.gpu_device_label.setVisible(True)
            self.gpu_device_combo.setVisible(True)

            for device in self.gpu_devices['vulkan']:
                display_name = f"{device['name']} ({device['type']})"
                self.gpu_device_combo.addItem(display_name, device['name'])
                self.gpu_device_combo.setItemData(self.gpu_device_combo.count() - 1, display_name, Qt.ItemDataRole.ToolTipRole)

            if self.default_gpu_device_name:
                 default_index = self.gpu_device_combo.findData(self.default_gpu_device_name)
                 if default_index != -1: self.gpu_device_combo.setCurrentIndex(default_index)
        else:
            self.gpu_device_label.setVisible(False)
            self.gpu_device_combo.setVisible(False)



    def set_gpu_api(self, index):
        """
        Reinicia o player com a nova API de GPU selecionada.
        Requer uma reinicialização completa do MPV.
        """
        if not self.player: return

        new_api = self.gpu_api_combo.itemData(index)
        current_api = self.player.gpu_api

        if new_api and current_api != new_api:
            print(f"Trocando API da GPU de '{current_api}' para '{new_api}'. Reiniciando o player...")
            self.timer.stop()

            # Salva o estado atual para restaurá-lo após a recriação do player
            old_state = {
                "path": self.current_media_path,
                "time": self.player.time_pos,
                "paused": self.player.pause,
                "is_sequence": self.is_image_sequence,
                "image_files": self.image_files if self.is_image_sequence else None,
                "in_frame": self.in_point_frame,
                "out_frame": self.out_point_frame,
            }

            self.controls_container.setEnabled(False)
            self.thread = QThread()
            mpv_options = self._get_master_mpv_options() # Já pegará a nova API

            self.reset_worker = PlayerResetWorker(self.player, mpv_options)
            self.reset_worker.moveToThread(self.thread)

            # Passa None para o dispositivo de áudio, pois não estamos mudando isso
            self.reset_worker.finished.connect(
                lambda new_player: self._on_player_reset_finished(new_player, None, old_state)
            )

            self.thread.started.connect(self.reset_worker.run)
            self.thread.start()

    @pyqtSlot(str)
    def _handle_log_message(self, text):
        """Slot que recebe texto do LogStream e o guarda no buffer."""
        self.log_buffer.append(text)
        if len(self.log_buffer) > LOG_MAX_LINES:
            self.log_buffer.pop(0)

    def open_log_dialog(self):
        """Cria e exibe a janela de log com o conteúdo do buffer."""
        log_content = "".join(self.log_buffer)
        dialog = LogDialog(log_content, self)
        dialog.exec()

    def _mpv_log_handler(self, level, component, message):
        """
        Captura as mensagens de log diretamente da biblioteca MPV e as envia
        para o nosso gestor de logs principal.
        """
        # Formata a mensagem para ser semelhante a um print normal
        log_line = f"mpv [{level}] {component}: {message.strip()}\n"
        # Usa a mesma função que captura os prints do Python para guardar a mensagem
        self._handle_log_message(log_line)

    def center_on_screen(self):
        """Calcula a posição central do ecrã e move a janela para lá."""
        try:
            # Obtém a geometria do ecrã principal
            screen_geometry = QApplication.primaryScreen().geometry()
            # Obtém a geometria da janela da nossa aplicação
            window_geometry = self.frameGeometry()
            # Calcula o ponto central e move a janela
            center_point = screen_geometry.center()
            window_geometry.moveCenter(center_point)
            self.move(window_geometry.topLeft())
        except Exception as e:
            print(f"Não foi possível centrar a janela: {e}")

    def start_ipc_server(self):
        """Inicia o servidor TCP para escutar por novas instâncias."""
        self.ipc_server = QTcpServer(self)
        self.ipc_server.newConnection.connect(self.handle_new_ipc_connection)
        if not self.ipc_server.listen(QHostAddress.SpecialAddress.LocalHost, APP_PORT):
            print("ERRO: Não foi possível iniciar o servidor IPC. A instância única pode não funcionar.")
        else:
            print("Instância principal ouvindo por novos arquivos.")

    def handle_new_ipc_connection(self):
        """Chamado quando uma nova instância (mensageiro) se conecta."""
        socket = self.ipc_server.nextPendingConnection()
        if socket:
            # Espera o mensageiro enviar os dados e chama a função de leitura
            socket.readyRead.connect(lambda: self.read_ipc_data(socket))

    def read_ipc_data(self, socket):
        """Lê a mensagem enviada pelo mensageiro."""
        data = socket.readAll().data().decode('utf-8')
        secret_key, bar, filepath = data.partition('|')

        # Verifica a "senha" para ter certeza que é o nosso app
        if secret_key == APP_SECRET_KEY and filepath:
            print(f"Recebido novo arquivo via IPC: {filepath}")

            # Usa a configuração da bandeja para decidir o que fazer
            if self.single_instance_action.isChecked():
                # Usa QTimer para garantir que a carga do arquivo seja executada na thread principal da GUI
                QTimer.singleShot(0, lambda: self.load_file_from_ipc(filepath))
            else:
                # Futuramente, aqui poderia entrar a lógica para abrir uma nova janela
                # Por enquanto, vamos apenas carregar na janela atual
                QTimer.singleShot(0, lambda: self.load_file_from_ipc(filepath))

        socket.close()

    def load_file_from_ipc(self, filepath):
        """Carrega o arquivo recebido e traz a janela para a frente."""
        # Mostra a janela se estiver escondida na bandeja
        if not self.isVisible():
            self.show()

        # Traz a janela para o topo e a torna a janela ativa
        self.raise_()
        self.activateWindow()

        # Reutiliza sua função de carregamento existente
        self.handle_file_argument(filepath)

    def setup_tray_icon(self):
        """Cria o ícone e o menu de contexto da bandeja do sistema."""
        self.tray_icon = QSystemTrayIcon(self.windowIcon(), self)
        self.tray_icon.setToolTip("BRP Player")

        tray_menu = QMenu()

        self.show_hide_action = QAction("Mostrar Player", self)
        self.show_hide_action.triggered.connect(self.toggle_window_visibility)
        tray_menu.addAction(self.show_hide_action)

        tray_menu.addSeparator()

        self.single_instance_action = QAction("Substituir vídeo atual", self, checkable=True)
        self.single_instance_action.setChecked(True)
        tray_menu.addAction(self.single_instance_action)

        tray_menu.addSeparator()

        # Garante que a ação de sair está ligada à função correta
        quit_action = QAction("Sair do BRP Player", self)
        quit_action.triggered.connect(self.quit_application)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.show()
        self.tray_icon.activated.connect(self.on_tray_icon_activated)

    def toggle_window_visibility(self):
        """Mostra a janela se estiver escondida, ou esconde se estiver visível."""
        if self.isVisible():
            self.hide()
            self.show_hide_action.setText("Mostrar Player")
        else:
            self.show()
            self.activateWindow()
            self.show_hide_action.setText("Esconder Player")

    def on_tray_icon_activated(self, reason):
        """Ação para quando o ícone da bandeja é clicado."""
        # Clicar com botão esquerdo ou duplo-clique mostra/esconde
        if reason == QSystemTrayIcon.ActivationReason.Trigger or reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.toggle_window_visibility()

    def quit_application(self):
        """Fecha a aplicação de forma limpa."""
        print("Iniciando encerramento completo da aplicação...")

        if hasattr(self, 'effects_dialog_instance') and self.effects_dialog_instance and self.effects_dialog_instance.isVisible():
            print("Fechando a janela de efeitos aberta antes de sair...")

            self.effects_dialog_instance.close()


        self._is_quitting = True # ATIVA A FLAG! A aplicação está a sair.
        self.tray_icon.hide()
        self._cleanup_temp_dir()
        if self.player:
            try:
                self.player.terminate()
            except Exception as e:
                print(f"Erro ao terminar MPV na saída: {e}")
        QApplication.instance().quit()


    def _rehydrate_shaders_if_needed(self):
        """
        Garante que shaders GLSL na configuração tenham um 'runtime_path' válido
        para a sessão atual. Essencial para carregar configurações salvas e aplicar
        efeitos ao carregar novos arquivos.
        """
        if not self.active_shader_config:
            return

        temp_dir = self.get_temp_dir_for_shaders()
        for effect_data in self.active_shader_config:
            # Pula filtros nativos e shaders que já estão prontos para uso
            if effect_data.get("is_builtin") or (effect_data.get("runtime_path") and os.path.exists(effect_data.get("runtime_path"))):
                continue

            original_path = effect_data.get("path")
            if not original_path or not os.path.exists(original_path):
                print(f"Aviso: Shader em configuração não encontrado em '{original_path}', será ignorado.")
                continue

            # Cria a cópia temporária e atualiza o dicionário na configuração
            unique_suffix = uuid.uuid4().hex[:8]
            base, ext = os.path.splitext(os.path.basename(original_path))
            temp_filename = f"{base}_{unique_suffix}{ext}"
            runtime_path = os.path.join(temp_dir, temp_filename)
            shutil.copy2(original_path, runtime_path)

            effect_data["original_path"] = original_path
            effect_data["runtime_path"] = runtime_path

    def _get_video_dimensions(self, filepath):
        """Usa ffprobe para obter a largura e altura de um vídeo de forma síncrona."""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0', filepath
            ]
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
            width, height = map(int, result.stdout.strip().split('x'))
            return width, height
        except Exception as e:
            print(f"Não foi possível obter as dimensões do vídeo para {filepath}: {e}")
            return None, None


    def handle_file_argument(self, filepath):
        """Verifica a extensão e carrega vídeo ou sequência de imagem."""
        # --- ALTERAÇÃO AQUI ---
        video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.mxf', '.webm', '.gif')
        image_extensions = ('.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tga', '.bmp')

        if filepath.lower().endswith(video_extensions):
            self.load_video(filepath)
        elif filepath.lower().endswith(image_extensions):
            self.load_image_sequence(filepath)

    def dragEnterEvent(self, event):
        """Chamado quando um arquivo é arrastado sobre a janela."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        """Chamado quando o arquivo é solto na janela."""
        urls = event.mimeData().urls()
        if urls:
            filepath = urls[0].toLocalFile()

            # --- ALTERAÇÃO AQUI ---
            video_extensions = ('.mp4', '.mkv', '.mov', '.avi', '.mxf', '.webm', '.gif')
            image_extensions = ('.png', '.jpg', '.jpeg', '.exr', '.tiff', '.tga', '.bmp')

            if filepath.lower().endswith(video_extensions):
                self.load_video(filepath)
            elif filepath.lower().endswith(image_extensions):
                self.load_image_sequence(filepath)
            else:
                print(f"Formato de arquivo não suportado: {filepath}")

    def showEvent(self, event):
        """Este método é chamado quando a janela está prestes a ser exibida."""
        super().showEvent(event)
        if self.player is None:
            self._initialize_player_once()

    def _initialize_player_once(self):
        """Cria a instância do MPV uma única vez e configura os eventos."""
        try:
            # Pega as opções do nosso novo método centralizado
            mpv_options = self._get_master_mpv_options()

            self.player = mpv.MPV(**mpv_options)
            self.setup_mpv_events()
            self.setup_mpv_property_observers()
            self.populate_audio_devices()
        except Exception as e:
            raise RuntimeError(f"Não foi possível inicializar o MPV. Erro: {e}")

    def closeEvent(self, event):
        """
        Verifica se a aplicação está a ser encerrada. Se sim, permite o fecho.
        Se não, esconde para a bandeja.
        """
        # Se a flag _is_quitting for verdadeira, permite que a janela feche normalmente.
        if self._is_quitting:
            event.accept()
            return

        # Caso contrário, executa a lógica de "esconder para a bandeja".
        event.ignore()
        self.stop_media(clear_ui_only=True)
        self.hide()
        if hasattr(self, 'show_hide_action'):
            self.show_hide_action.setText("Mostrar Player")



    def _reinitialize_player(self):
        """Destrói o player antigo (se existir) e cria um novo. Usado para resolver deadlocks."""
        print("Reinicializando a instância do player MPV para evitar deadlock...")
        if self.player:
            try:
                self.player.terminate()
            except Exception as e:
                print(f"Erro ao terminar a instância do MPV: {e}")

        self._initialize_player_once()
        self.set_volume(self.volume_slider.value())
        self.player.loop = 'inf' if self.is_repeat_on else False

    def _cleanup_temp_dir(self):
        """Remove todas as pastas temporárias criadas."""
        dirs_to_clean = self.temp_dirs_to_cleanup + ([self.shader_temp_dir] if self.shader_temp_dir else [])
        for temp_dir in dirs_to_clean:
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Pasta temporária removida: {temp_dir}")
            except Exception as e:
                print(f"Erro ao remover pasta temporária {temp_dir}: {e}")
        self.temp_dirs_to_cleanup.clear()
        self.shader_temp_dir = None

    def get_temp_dir_for_shaders(self):
        """Cria (se necessário) e retorna o caminho para a pasta de shaders temporários."""
        if not self.shader_temp_dir or not os.path.exists(self.shader_temp_dir):
            self.shader_temp_dir = tempfile.mkdtemp(prefix="cvplayer_shaders_")
        return self.shader_temp_dir

    # Em player_logic.py, substitua a função apply_all_effects por esta:
    def _get_video_color_info(self, filepath):
        """Usa ffprobe para obter os metadados de cor (transfer, primaries, space)."""
        if not filepath or self.is_image_sequence:
            return {} # Retorna vazio para sequências de imagem (tratadas pela extensão)
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=color_space,color_primaries,color_transfer',
                '-of', 'json', filepath
            ]
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
            return json.loads(result.stdout)['streams'][0]
        except Exception as e:
            print(f"Não foi possível obter os metadados de cor de {filepath}: {e}")
            return {}

    def apply_all_effects(self):
        if not self.player: return

        self._rehydrate_shaders_if_needed()

        glsl_shader_paths = []
        glsl_shader_opts_list = []
        vf_filters = []
        libplacebo_opts = {}

        uses_glsl = any(not effect.get("is_builtin", False) and effect.get("enabled", True) for effect in self.active_shader_config)

        if uses_glsl:
            is_source_linear_exr = (self.is_image_sequence and
                                    self.current_media_path and
                                    self.current_media_path.lower().endswith('.exr'))

            # Obtém a "transfer function" que detetámos ao carregar o vídeo
            transfer_function = self.source_color_info.get('color_transfer')

            if is_source_linear_exr:
                # 1. PIPELINE LINEAR/FLOAT: A fonte é EXR.
                #print("Pipeline HDR/Float (Linear) ativada para o player.")
                vf_filters.append('zscale=t=linear,format=gbrapf32le')

            elif transfer_function in ['smpte2084', 'arib-std-b67']:
                # 2. PIPELINE HDR/VÍDEO: A fonte é um vídeo HDR (PQ ou HLG).
                #print(f"Pipeline HDR/Vídeo ({transfer_function}) ativada para o player.")
                # Converte de PQ/HLG para luz linear, e depois para 32-bit float.
                vf_filters.append(f'zscale=t={transfer_function},format=gbrapf32le')

            else:
                # 3. PIPELINE SDR: A fonte é padrão. Usa 16-bit para alta qualidade sem banding.
                #print("Pipeline SDR de Alta Precisão (16-bit) ativada para o player.")
                vf_filters.append('format=yuva444p16le')

        deinterlace_selection = self.deinterlace_combo.currentText() # Mantém o case original

        # Controlamos manualmente o deinterlace, então desligamos o do MPV por padrão.
        self.player.deinterlace = 'no'

        # Determina se um filtro de deinterlace deve ser adicionado para a reprodução
        should_add_filter = False
        method_to_use = ''

        if deinterlace_selection == 'Auto':
            if self.is_interlaced:
                should_add_filter = True
                method_to_use = 'bwdif' # Nosso método padrão para 'Auto'
        elif deinterlace_selection != 'Off':
            # Se o usuário escolheu uma opção específica (ex: "bwdif field")
            should_add_filter = True
            # Extrai apenas o nome do filtro ("bwdif") para a reprodução ao vivo
            method_to_use = deinterlace_selection.split(' ')[0]

        if should_add_filter:
            # Para a reprodução ao vivo, 'send_field' é um padrão aceitável.
            # O controle granular será aplicado na exportação.
            vf_filters.append(f"{method_to_use}=mode=send_field:deint=all")

        for effect_data in self.active_shader_config:
            if not effect_data.get("enabled", True): continue
            if effect_data.get("is_builtin"):
                filter_name = effect_data["path"].split(":")[1]
                params = effect_data.get("params", {})
                if effect_data.get("is_libplacebo_option"):
                    libplacebo_opts['deband'] = 'yes'
                    for p_name, p_data in params.items(): libplacebo_opts[p_name] = p_data.get("value", p_data["default"])
                else:
                    param_str_list = []
                    for k, v in params.items():
                        value = v['value']
                        if  'color' in k:
                            # Converte a cor de '#rrggbb' para '0xRRGGBB' que o FFmpeg entende
                            value_str = str(value).replace('#', '0x')
                            print(f"converted color to {value_str}")
                        elif isinstance(value, float):
                            value_str = f"{value:.6f}"
                        else:
                            value_str = str(value)
                        param_str_list.append(f"{k}={value_str}")

                    param_str = ":".join(param_str_list)
                    vf_filters.append(f"{filter_name}={param_str}")
            else:
                runtime_path = effect_data.get("runtime_path")
                if runtime_path and os.path.exists(runtime_path):
                    glsl_shader_paths.append(os.path.normpath(runtime_path).replace('\\', '/'))
                    shader_name = os.path.splitext(os.path.basename(runtime_path))[0]
                    for name, param in effect_data.get("params", {}).items():
                        # --- CORREÇÃO AQUI: Lógica de parâmetros para shaders agrupados ---
                        if param.get("type") == "color":
                            r_val = param["r_float_value"]
                            g_val = param["g_float_value"]
                            b_val = param["b_float_value"]
                            glsl_shader_opts_list.append(f"{shader_name}/{param['r_param_name']}={r_val:.6f}")
                            glsl_shader_opts_list.append(f"{shader_name}/{param['g_param_name']}={g_val:.6f}")
                            glsl_shader_opts_list.append(f"{shader_name}/{param['b_param_name']}={b_val:.6f}")
                        else:
                            current_value = param.get("value", param["default"])
                            opt_format = f"{float(current_value):.6f}" if param.get("type") in ["float", "DEFINE"] else str(int(current_value))
                            glsl_shader_opts_list.append(f"{shader_name}/{name}={opt_format}")

        try:

            #print("\n" + "="*50)
            #print("Parâmetros a serem enviados para o MPV:")
            #print(f"  -> self.player.glsl_shaders = {glsl_shader_paths}")
            #print(f"  -> self.player.glsl_shader_opts = '{','.join(glsl_shader_opts_list)}'")
            #print(f"  -> self.player.vf = '{','.join(vf_filters)}'")

            # Imprime as opções libplacebo separadamente
            #if libplacebo_opts:
                #print("  -> Opções libplacebo:")
                #for opt, val in libplacebo_opts.items():
                    #print(f"     - self.player.{opt} = {val}")

            #print("="*50 + "\n")
            # --- FIM DO PRINT DE DEBUG ---

            self.player.glsl_shaders = glsl_shader_paths
            self.player.glsl_shader_opts = ",".join(glsl_shader_opts_list)
            self.player.vf = ",".join(vf_filters)
            for opt, val in libplacebo_opts.items(): setattr(self.player, opt, val)
            if 'deband' not in libplacebo_opts: self.player.deband = 'no'
            # print(f"Efeitos aplicados. Shaders: {self.player.glsl_shaders}, Opts: {self.player.glsl_shader_opts}, VF: {self.player.vf}")

        except Exception as e:
            print(f"ERRO ao aplicar todos os efeitos: {e}")

    def setup_logic_connections(self):
        self.open_video_action.triggered.connect(self.open_video_file)
        self.open_folder_action.triggered.connect(self.open_image_folder)
        self.deinterlace_combo.currentIndexChanged.connect(self.apply_all_effects)
        self.original_size_button.clicked.connect(self.toggle_original_size)
        self.to_start_button.clicked.connect(self.go_to_in_point)
        self.to_end_button.clicked.connect(self.go_to_out_point)
        self.open_effects_action.triggered.connect(self.open_effects_dialog)

        # --- CORREÇÃO AQUI: Garante uma única conexão para o export ---
        try:
            # Tenta desconectar qualquer conexão anterior para evitar duplicação
            self.export_action.triggered.disconnect()
            self.batch_export_action.triggered.disconnect()
        except TypeError:
            # Isso acontece se não houver conexões anteriores, o que é normal na primeira vez
            pass
        self.export_action.triggered.connect(self.open_export_dialog)
        self.batch_export_action.triggered.connect(self.open_batch_export_dialog)
        # --- FIM DA CORREÇÃO ---
        self.ocio_button.clicked.connect(self.check_pause)
        self.exit_action.triggered.connect(self.quit_application)


        self.video_ended.connect(self.on_video_ended)
        self.clear_range_button.clicked.connect(self.clear_range)
        self.current_frame_input.returnPressed.connect(self.seek_to_frame)
        self.framerate_input.returnPressed.connect(self.update_framerate_and_rate)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.prev_frame_button.clicked.connect(self.frame_step_backward)
        self.play_pause_button.clicked.connect(self.play_pause)
        self.next_frame_button.clicked.connect(self.frame_step_forward)
        self.stop_button.clicked.connect(self.stop_media)
        self.repeat_button.clicked.connect(self.toggle_repeat)
        self.mute_button.clicked.connect(self.toggle_mute)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setValue(self.last_volume)

        self.audio_device_combo.activated.connect(self.set_audio_device)
        self.gpu_api_combo.addItem("Vulkan", "vulkan")
        self.gpu_api_combo.addItem("OpenGL", "opengl")
        self.gpu_api_combo.addItem("Direct3D 11", "d3d11")

        # Define o padrão detectado
        default_index = self.gpu_api_combo.findData(self.default_gpu_api)
        if default_index != -1:
            self.gpu_api_combo.setCurrentIndex(default_index)

        self.gpu_api_combo.activated.connect(self.set_gpu_api)

        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.video_fullscreen_button.clicked.connect(self.toggle_video_fullscreen)
        self.fullscreen_state_changed.connect(self._handle_fullscreen_ui_change)
        self.restore_zoom_button.clicked.connect(self.reset_zoom)
        self.video_frame.installEventFilter(self)
        self.set_in_button.clicked.connect(self.set_in_point)
        self.set_out_button.clicked.connect(self.set_out_point)
        self.in_frame_input.returnPressed.connect(self.update_range_from_inputs)
        self.out_frame_input.returnPressed.connect(self.update_range_from_inputs)
        self.open_log_action.triggered.connect(self.open_log_dialog)

        self.gpu_api_combo.clear()

        if self.vulkan_supported:
            self.gpu_api_combo.addItem("Vulkan", "vulkan")
            self.gpu_api_combo.setItemData(self.gpu_api_combo.count() - 1, "Vulkan: Modern API, preferred.", Qt.ItemDataRole.ToolTipRole)

        self.gpu_api_combo.addItem("OpenGL", "opengl")
        self.gpu_api_combo.setItemData(self.gpu_api_combo.count() - 1, "OpenGL: Legacy API, compatibility.", Qt.ItemDataRole.ToolTipRole)

        # A detecção de D3D11 foi removida por não ser possível via API Python pura
        if sys.platform == "win32":
            self.gpu_api_combo.addItem("Direct3D 11", "d3d11")
            self.gpu_api_combo.setItemData(self.gpu_api_combo.count() - 1, "Direct3D 11: Windows API, not recommended\nas it may conflict with the UI.", Qt.ItemDataRole.ToolTipRole)

        default_index = self.gpu_api_combo.findData(self.default_gpu_api)
        if default_index != -1:
            self.gpu_api_combo.setCurrentIndex(default_index)


        self.gpu_device_combo.activated.connect(self.set_gpu_api)
        self._update_gpu_device_selector()


        if self.player:
            self.populate_audio_devices()
        sleep(1.0)
        self.ocio_manager.connect_signals()

    def go_to_in_point(self):
        """Pula para o ponto de entrada (In) e pausa."""
        if not self.player or not self.player.path: return
        #self.is_paused = True
        #self.player.pause = True
        self.seek_to_frame(frame_number=self.in_point_frame)

        # --- CORREÇÃO AQUI ---
        # Força uma atualização da UI para garantir que o slider seja redesenhado
        QTimer.singleShot(10, self.update_ui)
        self.update_play_pause_icon()

    def open_effects_dialog(self):
        if not self.player: return
        if hasattr(self, 'effects_dialog_instance') and self.effects_dialog_instance.isVisible():
            self.effects_dialog_instance.activateWindow()
            return

        self.effects_dialog_instance = EffectsDialog(self)
        # Passa a lista de configuração da aplicação para o manager.
        # O manager irá operar diretamente NESTA lista.
        self.effects_manager = EffectsManager(self, self.effects_dialog_instance, self.active_shader_config)
        self.effects_dialog_instance.finished.connect(self.on_effects_dialog_closed)
        self.effects_dialog_instance.show()

    def on_effects_dialog_closed(self):
        # A configuração (self.active_shader_config) já foi atualizada em tempo real.
        # Apenas limpamos a referência ao manager.
        self.effects_manager = None
        # Garante que os filtros (como deinterlace) sejam aplicados corretamente mesmo após fechar.
        self.apply_all_effects()

    def _update_video_filters(self):
        """
        Constrói e aplica a cadeia de filtros de vídeo (-vf) a partir de múltiplas fontes.
        Esta é agora a única função que deve modificar 'self.player.vf'.
        """
        if not self.player: return

        final_vf_chain = []

        # 1. Filtro de Deinterlace
        deinterlace_mode = self.deinterlace_combo.currentText().lower()
        if self.is_interlaced and deinterlace_mode != 'off':
            method = 'bwdif' if deinterlace_mode == 'auto' else deinterlace_mode
            final_vf_chain.append(f"{method}=mode=send_field:deint=all")
            # Desativa o deinterlace nativo do mpv para usar o filtro
            self.player.deinterlace = 'no'
        elif not self.is_interlaced:
             self.player.deinterlace = 'no'
        else: # Vídeo entrelaçado mas filtro 'Off'
             self.player.deinterlace = 'yes' # Usa o deinterlace automático do mpv


        # 2. Filtros nativos da janela de Efeitos
        if self.active_shader_config:
            for effect_data in self.active_shader_config:
                if effect_data.get("enabled", True) and effect_data.get("is_builtin") and not effect_data.get("is_libplacebo_option"):
                    filter_name = effect_data["path"].split(":")[1]
                    params = effect_data.get("params", {})
                    param_str_list = []
                    for k, v in params.items():
                        # Garante que valores float sejam formatados corretamente
                        if isinstance(v['value'], float):
                            param_str_list.append(f"{k}={v['value']:.6f}")
                        else:
                            param_str_list.append(f"{k}={v['value']}")

                    param_str = ":".join(param_str_list)
                    final_vf_chain.append(f"{filter_name}={param_str}")

        # 3. Aplica a cadeia de filtros final ao MPV
        if final_vf_chain:
            vf_string = ",".join(final_vf_chain)
            print(f"Atualizando cadeia de filtros (-vf): {vf_string}")
            self.player.vf = vf_string
        else:
            # Limpa a cadeia se não houver filtros
            print("Limpando cadeia de filtros (-vf).")
            self.player.vf = ""

    def go_to_out_point(self):
        """Pula para o ponto de saída (Out) e pausa."""
        if not self.player or not self.player.path: return
        #self.is_paused = True
        #self.player.pause = True
        self.seek_to_frame(frame_number=self.out_point_frame)

        # --- CORREÇÃO AQUI ---
        # Força uma atualização da UI para garantir que o slider seja redesenhado
        QTimer.singleShot(10, self.update_ui)
        self.update_play_pause_icon()

    def toggle_original_size(self):
        """Alterna o modo de visualização 1:1 (video-unscaled)."""
        if not self.player or not self.player.path:
            self.original_size_button.setChecked(False)
            return

        self.is_unscaled = not self.is_unscaled
        self.player.video_unscaled = "yes" if self.is_unscaled else "no"
        self.original_size_button.setChecked(self.is_unscaled)
        self.original_size_button.style().polish(self.original_size_button)

        # Se estivermos ativando o modo 1:1, também resetamos o pan/zoom
        if self.is_unscaled:
            self.zoom_level = 0.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.player.video_zoom = self.zoom_level
            self.player.video_pan_x = self.pan_x
            self.player.video_pan_y = self.pan_y

        print(f"Modo 1:1 (video-unscaled) {'ativado' if self.is_unscaled else 'desativado'}.")


    def eventFilter(self, obj, event):
        """Filtra eventos para adicionar funcionalidade de clique e duplo clique."""
        # Verifica se o evento é para o video_frame
        if obj is self.video_frame:
            # --- LÓGICA ATUALIZADA PARA CLIQUE ÚNICO E DUPLO ---
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                # Se o botão esquerdo for pressionado, inicia o timer.
                self.single_click_timer.start()
                return True # Evento tratado

            if event.type() == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                # Se for um clique duplo, para o timer (para evitar o play/pause) e ativa a tela cheia.
                self.single_click_timer.stop()
                self.toggle_video_fullscreen()
                return True # Evento tratado
            # --- FIM DA LÓGICA ATUALIZADA ---

            # Lógica existente para Zoom com Scroll (sem alterações)
            if event.type() == QEvent.Type.Wheel:
                self.is_unscaled = False
                self.original_size_button.setChecked(False)
                self.original_size_button.style().polish(self.original_size_button)
                delta = event.angleDelta().y()
                mouse_pos = event.position()
                self.handle_zoom(delta, mouse_pos.x(), mouse_pos.y())
                return True

            # Lógica existente para Pan (sem alterações)
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.MiddleButton:
                self.is_panning = True
                self.last_mouse_pos = event.position()
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                return True

            if event.type() == QEvent.Type.MouseMove and self.is_panning:
                if self.last_mouse_pos:
                    delta = event.position() - self.last_mouse_pos
                    self.last_mouse_pos = event.position()
                    scale = 2 ** self.zoom_level
                    pan_delta_x = delta.x() / (self.video_frame.width() * -scale)
                    pan_delta_y = delta.y() / (self.video_frame.height() * scale)
                    self.pan_x -= pan_delta_x
                    self.pan_y += pan_delta_y
                    self.player.video_pan_x = self.pan_x
                    self.player.video_pan_y = self.pan_y
                return True

            if event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.MiddleButton:
                self.is_panning = False
                self.last_mouse_pos = None
                self.unsetCursor()
                return True

        # Passa todos os outros eventos para o manipulador padrão
        return super().eventFilter(obj, event)

    def check_pause(self):
        if not self.is_paused:
            self.player.pause = True
            self.play_pause_button.setIcon(self.play_icon)
            self.is_paused = True
            self.timer.stop()

    def open_batch_export_dialog(self):
        if not self.is_paused:
            self.player.pause = True
            self.play_pause_button.setIcon(self.play_icon)
            self.is_paused = True
            self.timer.stop()
            self.update_play_pause_icon()
        if shutil.which('ffmpeg') is None:
            QMessageBox.critical(self, "Erro - FFmpeg não encontrado", "...")
            return

        selected_gpu_name = None
        if self.gpu_device_combo.isVisible():
            selected_gpu_name = self.gpu_device_combo.currentData()

        # 2. Passa o nome da GPU para o construtor do BatchExportDialog.
        dialog = BatchExportDialog(self, detected_encoders=self.detected_encoders, selected_gpu=selected_gpu_name)


        dialog.add_video_button.clicked.connect(lambda: self.add_files_to_batch(dialog, "video"))
        dialog.add_sequence_button.clicked.connect(lambda: self.add_files_to_batch(dialog, "sequence"))
        dialog.remove_button.clicked.connect(lambda: self.remove_from_batch(dialog.file_list))
        dialog.export_requested.connect(lambda: self.on_batch_export_request(dialog))

        try:
            dialog.exec()
        finally:
            # --- ADICIONADO AQUI: Garante a limpeza ao fechar o diálogo ---
            self._cleanup_temp_dir()

    def on_batch_export_request(self, dialog):
        """Função chamada pelo sinal para iniciar a exportação em lote."""
        file_list = dialog.get_batch_list()
        settings = dialog.get_settings()

        if not file_list:
            QMessageBox.warning(self, "Aviso", "Nenhum arquivo foi adicionado à lista de exportação.")
            return
        settings['deinterlace_mode'] = self.deinterlace_combo.currentText()
        output_folder = QFileDialog.getExistingDirectory(self, "Selecione a Pasta de Saída para o Lote")
        if output_folder:
            settings['active_shaders'] = self.active_shader_config
            self.start_batch_export(file_list, output_folder, settings, dialog)

    def add_files_to_batch(self, dialog, file_type):
        """Adiciona arquivos à lista de lote, pré-renderizando EXRs imediatamente."""
        list_widget = dialog.file_list # Obtém a lista a partir do diálogo

        if file_type == "video":
            files, _ = QFileDialog.getOpenFileNames(self, "Adicionar Vídeos", os.path.expanduser('~'), "Video Files (*.mp4 *.mkv *.mov *.avi)")
            if files:
                for f in files:
                    item = QListWidgetItem(f)
                    item.setData(Qt.ItemDataRole.UserRole, {"path": f, "is_sequence": False})
                    list_widget.addItem(item)

        elif file_type == "sequence":
            first_file, _ = QFileDialog.getOpenFileName(self, "Adicionar Sequência de Imagens", os.path.expanduser('~'), "Image Files (*.png *.jpg *.jpeg *.exr *.tiff *.tga *.bmp)")
            if first_file:
                if first_file.lower().endswith('.exr'):
                    directory = os.path.dirname(first_file)
                    base_name = os.path.basename(first_file)
                    match = re.match(r'^(.*?)(\d+)(\..*)$', base_name)
                    if not match:
                        QMessageBox.warning(self, "Erro", "Padrão de sequência EXR não reconhecido.")
                        return
                    prefix, _, extension = match.groups()
                    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().startswith(prefix.lower()) and f.lower().endswith(extension.lower())])

                    selected_layer, status = self._handle_exr_layers(image_files[0])
                    if status == "cancelled":
                        return

                    if selected_layer:
                        # --- CORREÇÃO AQUI: Passa o 'dialog' para a função de pré-render ---
                        self._prerender_exr_layer_for_batch(dialog, image_files, selected_layer, original_filename=first_file)
                        return

                item = QListWidgetItem(first_file)
                item.setData(Qt.ItemDataRole.UserRole, {"path": first_file, "is_sequence": True})
                list_widget.addItem(item)

    def _prerender_exr_layer_for_batch(self, dialog, image_files, layer_name, original_filename):
        """Versão do pré-render específica para o lote, que recebe o diálogo."""
        self.temp_render_dir = tempfile.mkdtemp(prefix="cvplayer_render_")
        self.temp_dirs_to_cleanup.append(self.temp_render_dir) # --- ADICIONADO AQUI ---

        progress_dialog = dialog.progress_bar
        progress_dialog.setRange(0, 100)
        dialog.status_label.setText(f"Pré-renderizando camada '{layer_name}'...")

        self.thread = QThread()
        self.prerender_worker = PrerenderManagerWorker(image_files, layer_name, self.temp_render_dir)

        self.prerender_worker.progress.connect(progress_dialog.setValue)
        # --- CORREÇÃO AQUI: Passa o 'dialog' para o callback de finalização ---
        self.prerender_worker.finished.connect(lambda new_files: self._on_prerender_finished_for_batch(dialog, new_files, original_filename))

        self.prerender_worker.moveToThread(self.thread)
        self.thread.started.connect(self.prerender_worker.run)

        self.thread.start()

    def _on_prerender_finished_for_batch(self, dialog, new_files, original_filename):
        """Callback chamado quando o pré-render do lote termina."""
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # --- CORREÇÃO AQUI: Usa o 'dialog' recebido como parâmetro ---
        dialog.progress_bar.setValue(0)
        dialog.status_label.setText("Pré-render concluído. Pronto para o lote.")

        if new_files:
            item = QListWidgetItem(f"[PRÉ-RENDERIZADO] {os.path.basename(original_filename)}")
            item.setData(Qt.ItemDataRole.UserRole, {
                "path": new_files[0],
                "is_sequence": True,
                "files": new_files
            })
            dialog.file_list.addItem(item)
        else:
            QMessageBox.warning(self, "Aviso", "O pré-render falhou ou foi cancelado.")

    def remove_from_batch(self, list_widget):
        """Remove todos os itens selecionados da lista de lote."""
        # Pega a lista de todos os itens selecionados de uma vez
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return

        # Itera sobre a lista estática de itens para removê-los do widget
        for item in selected_items:
            list_widget.takeItem(list_widget.row(item))

    def start_batch_export(self, file_list, output_folder, settings, dialog):
        """Prepara a fila e inicia o primeiro item da exportação em lote."""
        self.export_queue = file_list
        self.output_folder = output_folder
        self.batch_settings = settings
        self.batch_dialog = dialog
        self.total_batch_files = len(file_list)
        self.current_batch_index = 0

        dialog.export_button.setEnabled(False)

        # --- CORREÇÃO AQUI: Pinta todos os itens de vermelho ---
        red_brush = QBrush(QColor("#5a2d2d")) # Vermelho escuro
        for i in range(dialog.file_list.count()):
            item = dialog.file_list.item(i)
            item.setBackground(red_brush)

        self.process_next_in_batch()


    def process_next_in_batch(self):
        """Processa o próximo item da fila de exportação."""
        if self.current_batch_index >= self.total_batch_files:
            self.batch_dialog.status_label.setText("Lote de exportação concluído!")
            self.batch_dialog.export_button.setEnabled(True)
            QMessageBox.information(self, "Sucesso", "Todos os arquivos foram exportados.")
            return

        # --- CORREÇÃO AQUI: Acessa os dados do item diretamente ---
        # A fila 'export_queue' já contém os dicionários com os dados.
        item_data = self.export_queue[self.current_batch_index]
        in_file_path = item_data["path"]
        is_sequence = item_data["is_sequence"]
        prerendered_files = item_data.get("files")

        self.current_batch_index += 1
        self.batch_dialog.status_label.setText(f"Analisando arquivo {self.current_batch_index} de {self.total_batch_files}...")

        try:
            input_for_ffmpeg = in_file_path

            base_name_for_output = os.path.splitext(os.path.basename(in_file_path))[0]
            if is_sequence:
                 match = re.match(r'^(.*?)(\d+)(\..*)$', os.path.basename(in_file_path))
                 if match: base_name_for_output = match.groups()[0].strip('._- ')

            if is_sequence:
                if prerendered_files:
                    image_files = prerendered_files
                else:
                    directory = os.path.dirname(in_file_path)
                    base_name = os.path.basename(in_file_path)
                    match = re.match(r'^(.*?)(\d+)(\..*)$', base_name)
                    if not match: raise ValueError("Padrão de sequência não reconhecido.")
                    prefix, _, extension = match.groups()
                    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().startswith(prefix.lower()) and f.lower().endswith(extension.lower())])

                input_for_ffmpeg = self.create_ffmpeg_concat_file(image_files)
                if not input_for_ffmpeg: raise RuntimeError("Falha ao criar arquivo de lista.")
                self.ffmpeg_concat_file = input_for_ffmpeg
                total_frames = len(image_files)
                framerate = 30.0
            else: # Para vídeos
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=avg_frame_rate,duration', '-of', 'json', in_file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                stream_info = json.loads(result.stdout)['streams'][0]
                fr_str = stream_info.get('avg_frame_rate', '30/1')
                num, den = map(int, fr_str.split('/'))
                framerate = num / den if den > 0 else 30.0
                duration = float(stream_info.get('duration', '0'))
                total_frames = int(duration * framerate)

            export_range = {"in_frame": 0, "out_frame": total_frames - 1, "framerate": framerate, "total_frames": total_frames}

            container = self.batch_settings['container']

            is_image_sequence_export = container in [".jpg", ".png", ".exr"]
            if is_image_sequence_export:
                out_file = os.path.join(self.output_folder, f"{base_name_for_output}_converted")
                os.makedirs(out_file, exist_ok=True)
            else:
                out_file = os.path.join(self.output_folder, f"{base_name_for_output}_converted{container}")

            is_linear = is_sequence and in_file_path.lower().endswith('.exr')

            active_shaders_for_batch = self.batch_settings.get('active_shaders', [])
            self.start_export(input_for_ffmpeg, out_file, self.batch_settings, export_range, self.batch_dialog,
                              is_batch_item=True, is_sequence_input=is_sequence,
                              active_shaders=active_shaders_for_batch,
                              is_source_linear=is_linear)

        except Exception as e:
            print(f"Erro ao analisar o arquivo {in_file_path}: {e}")
            QMessageBox.warning(self, "Erro de Análise", f"Não foi possível ler os metadados do arquivo:\n{os.path.basename(in_file_path)}\n\nPulando para o próximo.")
            QTimer.singleShot(10, self.process_next_in_batch)

    def setup_mpv_property_observers(self):
        if self.player:
            self.player.observe_property('fullscreen', self.on_fullscreen_change)

    def on_fullscreen_change(self, name, is_fullscreen):
        self.fullscreen_state_changed.emit(is_fullscreen)

    def _handle_fullscreen_ui_change(self, is_fullscreen):
        """
        Este slot é executado na thread principal e pode modificar a UI com segurança.
        """
        if is_fullscreen:
            # --- CORREÇÃO AQUI: Remove as margens do layout principal ---
            self.main_layout.setContentsMargins(1, 1, 1, 1)
            self.video_frame.setStyleSheet("border: none;")
        else:
            # Restaura as margens para o layout normal
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            self.video_frame.setStyleSheet("") # Reverte para o estilo padrão

        # Esconde/mostra os outros containers
        self.controls_container.setVisible(not is_fullscreen)
        self.ocio_container.setVisible(not is_fullscreen)
        self.metadata_container.setVisible(not is_fullscreen)
        self.menuBar().setVisible(not is_fullscreen)

        if not is_fullscreen and not self.was_window_fullscreen and self.isFullScreen():
            self.showNormal()

    def toggle_video_fullscreen(self):
        """Alterna (liga/desliga) o modo de tela cheia do vídeo."""
        if not self.player or not self.player.path: return

        # --- CORREÇÃO AQUI ---
        # Verifica se o player já está em tela cheia
        if self.player.fullscreen:
            # Se estiver, apenas desativa. O callback on_fullscreen_change cuidará do resto.
            self.player.fullscreen = False
        else:
            # Se não estiver, executa a lógica para entrar em tela cheia
            self.was_window_fullscreen = self.isFullScreen()
            if not self.isFullScreen():
                self.showFullScreen()
            self.player.fullscreen = True

    def open_export_dialog(self):
        if not self.is_paused:
            self.player.pause = True
            self.play_pause_button.setIcon(self.play_icon)
            self.is_paused = True
            self.timer.stop()
            self.update_play_pause_icon()
        if shutil.which('ffmpeg') is None:
            QMessageBox.critical(self, "Erro - FFmpeg não encontrado",
                                 "O executável 'ffmpeg.exe' não foi encontrado no PATH do seu sistema.\n\n"
                                 "Por favor, instale o FFmpeg e adicione sua pasta 'bin' ao PATH do Windows para usar a funcionalidade de exportação.")
            return

        if not self.player or not self.player.path:
            QMessageBox.warning(self, "Aviso", "Por favor, abra um vídeo antes de exportar.")
            return
        selected_gpu_name = None
        if self.gpu_device_combo.isVisible():
            selected_gpu_name = self.gpu_device_combo.currentData()
        is_hdr_source = (
            (self.is_image_sequence and self.current_media_path and self.current_media_path.lower().endswith('.exr')) or
            (self.source_color_info.get('color_transfer') in ['smpte2084', 'arib-std-b67'])
        )
        dialog = ExportDialog(self, source_has_alpha=self.has_alpha, source_resolution=self.source_resolution,
                              source_is_hdr=is_hdr_source, detected_encoders=self.detected_encoders,
                              selected_gpu=selected_gpu_name)

        dialog.export_requested.connect(self.on_export_request)

        # --- CORREÇÃO AQUI ---
        # Apenas executa o diálogo. A lógica de finalização cuidará de fechá-lo
        # ou mantê-lo aberto conforme necessário.
        dialog.exec()

    def detect_alpha_channel(self, filepath):
        """Usa o ffprobe para verificar se o vídeo tem um canal alfa."""
        if not filepath or self.is_image_sequence:
            ext = os.path.splitext(filepath)[1].lower()
            return ext in ['.png', '.exr', '.tiff', '.tga']

        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_streams',
                '-select_streams', 'v:0', '-print_format', 'json', filepath
            ]
            # --- CORREÇÃO ADICIONADA AQUI ---
            # Define startupinfo para esconder a janela do console no Windows
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
            # --- FIM DA CORREÇÃO ---

            stream_info = json.loads(result.stdout)['streams'][0]
            pix_fmt = stream_info.get('pix_fmt', '')
            return 'a' in pix_fmt.lower()
        except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Não foi possível detectar o canal alfa: {e}")
            return False

    def on_export_request(self, settings):
        """Lida com a exportação individual."""
        dialog = self.sender()
        container = settings["container"]
        is_image_sequence_export = container in [".jpg", ".png", ".exr"]

        default_name = os.path.splitext(os.path.basename(self.current_media_path))[0]
        out_path = None
        input_for_export = self.current_media_path
        is_linear = self.is_image_sequence and self.current_media_path.lower().endswith('.exr')

        if is_image_sequence_export:
            folder = QFileDialog.getExistingDirectory(self, "Selecione a Pasta de Saída para a Sequência")
            if folder:
                out_path = os.path.join(folder, default_name)
                # Garante que a pasta de saída seja criada
                os.makedirs(out_path, exist_ok=True)
        else:
            filter_str = f"Video (*{container})"
            file_path, _ = QFileDialog.getSaveFileName(self, "Salvar Vídeo Como...", f"{default_name}_export{container}", filter_str)
            if file_path:
                if not file_path.lower().endswith(container):
                    file_path += container
                out_path = file_path

        if out_path:
            export_range = {
                "in_frame": self.in_point_frame,
                "out_frame": self.out_point_frame,
                "framerate": self.framerate,
                "total_frames": self.total_frames
            }

            if self.is_image_sequence:
                try:
                    files_in_range = self.image_files[export_range['in_frame'] : export_range['out_frame'] + 1]
                    input_for_export = self.create_ffmpeg_concat_file(files_in_range)
                    if not input_for_export:
                        raise RuntimeError("Falha ao criar arquivo de lista para a sequência.")
                    self.ffmpeg_concat_file = input_for_export
                except Exception as e:
                    self.finish_export(dialog, f"Erro ao preparar arquivos: {e}", is_error=True)
                    return
            settings['deinterlace_mode'] = self.deinterlace_combo.currentText()
            settings['source_is_interlaced'] = self.is_interlaced

            # Passa a nova flag is_source_linear para a função start_export
            self.start_export(input_for_export, out_path, settings, export_range, dialog,
                              is_batch_item=False, is_sequence_input=self.is_image_sequence,
                              active_shaders=self.active_shader_config,
                              is_source_linear=is_linear)

    def set_deinterlace(self):
        """
        Aplica a seleção de deinterlace ao player MPV.
        Esta função agora simplesmente chama o gerenciador central de filtros.
        """
        self._update_video_filters()

    def start_export(self, in_file, out_file, settings, export_range, dialog, is_batch_item=False, is_sequence_input=False, active_shaders=None, is_source_linear=False,
                     source_color_info=None, ocio_settings=None):
        """Inicia a thread de exportação."""
        dialog.export_button.setEnabled(False)
        dialog.status_label.setText("Preparando exportação...")

        exr_layer_to_export = self.selected_exr_layer if is_sequence_input else None

        if hasattr(self, 'export_thread') and self.export_thread and self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait(5000)

        self.export_thread = QThread()
        ocio_settings = {"lut_path": self.ocio_manager.lut_temp_file_path if hasattr(self, 'ocio_manager') else None}

        # --- CORREÇÃO AQUI: Passa a configuração de efeitos correta (self.active_shader_config) ---
        self.export_worker = ExportWorker(in_file, out_file, settings, ocio_settings, export_range,
                                          is_sequence_input, exr_layer_to_export,
                                          self.active_shader_config, is_source_linear)
        self.export_worker.moveToThread(self.export_thread)

        self.export_worker.progress.connect(lambda p, eta: self.update_export_progress(dialog, p, eta))
        self.export_worker.finished.connect(lambda msg: self.finish_export(dialog, msg, is_batch_item=is_batch_item))
        self.export_worker.error.connect(lambda err: self.finish_export(dialog, err, is_error=True, is_batch_item=is_batch_item))
        self.export_worker.log_message.connect(self._handle_log_message)
        self.export_worker.finished.connect(self.export_thread.quit)
        self.export_worker.error.connect(self.export_thread.quit)

        self.export_thread.started.connect(self.export_worker.run)
        self.export_thread.start()


    def create_ffmpeg_concat_file(self, image_files):
        """Cria um arquivo de texto persistente para o demuxer concat do FFmpeg."""
        if not image_files:
            return None
        try:
            # --- CORREÇÃO APLICADA AQUI ---
            # Abordagem manual para criar um arquivo temporário robusto
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"cvplayer_concat_{os.urandom(6).hex()}.txt")

            with open(temp_path, 'w', encoding='utf-8') as tf:
                for img_path in image_files:
                    tf.write(f"file '{img_path.replace(os.sep, '/')}'\n")

            return temp_path
        except Exception as e:
            print(f"Erro ao criar arquivo de concatenação para FFmpeg: {e}")
            return None

    def update_export_progress(self, dialog, percent, eta):
        dialog.progress_bar.setValue(percent)
        dialog.status_label.setText(f"Exporting... {percent}% (ETA: {eta})")

    def finish_export(self, dialog, message, is_error=False, is_batch_item=False):
        """Finaliza uma exportação e lida com o estado do diálogo."""
        if hasattr(self, 'export_thread') and self.export_thread.isRunning():
            self.export_thread.quit()
            self.export_thread.wait()

        if is_batch_item:
            # --- ALTERAÇÃO AQUI: Pinta o item concluído ---
            if self.current_batch_index > 0:
                # O item que acabou de ser processado é o do índice anterior
                item_index = self.current_batch_index - 1
                item = dialog.file_list.item(item_index)
                if item:
                    if is_error:
                        item.setBackground(QBrush(QColor("#7a3d2d"))) # Laranja/vermelho para erro
                    else:
                        item.setBackground(QBrush(QColor("#2d5a2d"))) # Verde escuro para sucesso

            # Continua para o próximo item
            self.process_next_in_batch()
        else: # Comportamento para export individual
            dialog.status_label.setText(message)
            dialog.export_button.setEnabled(True)
            if is_error:
                QMessageBox.critical(self, "Export error", message)
            else:
                QMessageBox.information(self, "Finished", message)
                dialog.accept()

    def setup_mpv_events(self):
        """Os eventos agora são tratados de forma diferente, esta função pode ser vazia
           ou usada para outros eventos no futuro."""
        pass # Não vamos mais usar o @self.player.event_callback('end-file')

    def on_video_ended(self):
        """Função chamada por update_ui quando o vídeo termina sem repetição."""
        self.timer.stop()
        self.is_paused = True
        self.player.pause = True

        # Agenda a atualização final da UI para o próximo ciclo de eventos
        QTimer.singleShot(50, self.update_play_pause_icon)
        QTimer.singleShot(50, self.update_ui)


    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.play_pause_button.click()
            event.accept()
            return
        elif key == Qt.Key.Key_Enter:
            self.activateWindow()
            event.accept()
            return
        elif key == Qt.Key.Key_G:
            self.toggle_original_size()
            event.accept()
            return
        elif key == Qt.Key.Key_R:
            self.reset_zoom()
            event.accept()
            return
        elif key == Qt.Key.Key_Right:
            self.next_frame_button.click()
            event.accept()
            return
        elif key == Qt.Key.Key_Left:
            self.prev_frame_button.click()
            event.accept()
            return
        elif key == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        elif key == Qt.Key.Key_I:
            self.set_in_point()
            event.accept()
            return
        elif key == Qt.Key.Key_O:
            self.set_out_point()
            event.accept()
            return
        elif key == Qt.Key.Key_F:
            self.toggle_video_fullscreen() # Fullscreen do vídeo
            event.accept()
            return
        elif key == Qt.Key.Key_Escape:
             # Prioriza sair do fullscreen do VÍDEO
            if self.player and self.player.fullscreen:
                self.player.fullscreen = False
            # Se não estiver em fullscreen de vídeo, sai do fullscreen da JANELA
            elif self.isFullScreen():
                self.showNormal()
            event.accept()
            return
        super().keyPressEvent(event)

    def handle_zoom(self, delta, mx, my):
        """Calcula e aplica o zoom e o pan centrados no cursor, corrigindo o aspect ratio."""
        if not self.player or not self.player.path or not self.video_info_loaded: return

        zoom_speed = 0.05
        old_zoom_level = self.zoom_level

        if delta > 0:
            self.zoom_level += zoom_speed
        else:
            self.zoom_level -= zoom_speed

        self.zoom_level = max(-1.0, min(self.zoom_level, 5.0))

        frame_w = self.video_frame.width()
        frame_h = self.video_frame.height()
        if frame_w == 0 or frame_h == 0: return

        video_params = self.player.video_params
        if not video_params: return

        vid_w = video_params.get('w', frame_w)
        vid_h = video_params.get('h', frame_h)
        if vid_w == 0 or vid_h == 0: return

        dar = video_params.get('dar', vid_w / vid_h)

        view_w, view_h = frame_w, frame_h
        x_offset, y_offset = 0, 0

        if frame_w / frame_h > dar:
            view_w = view_h * dar
            x_offset = (frame_w - view_w) / 2
        else:
            view_h = view_w / dar
            y_offset = (frame_h - view_h) / 2

        adj_mx = mx - x_offset
        adj_my = my - y_offset

        # Garante que o zoom só funcione dentro da área do vídeo
        if not (0 <= adj_mx <= view_w and 0 <= adj_my <= view_h):
            return

        norm_mx_adj = (adj_mx / view_w - 0.5) * -2
        norm_my_adj = (adj_my / view_h - 0.5) * -2 # Y é invertido no MPV

        old_scale = 2 ** old_zoom_level
        new_scale = 2 ** self.zoom_level

        scale_factor = (1.0 / old_scale) - (1.0 / new_scale)

        # --- CORREÇÃO AQUI: Inverte o sinal da correção de pan ---
        self.pan_x += norm_mx_adj * scale_factor
        self.pan_y += norm_my_adj * scale_factor

        self.player.video_zoom = self.zoom_level
        self.player.video_pan_x = self.pan_x
        self.player.video_pan_y = self.pan_y

    def reset_zoom(self):
        """Reseta o zoom, o pan e o modo 1:1."""
        if not self.player: return
        self.zoom_level = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # --- ALTERAÇÃO AQUI: Garante que o modo 1:1 seja desativado ---
        self.is_unscaled = False
        self.player.video_unscaled = "no"
        self.original_size_button.setChecked(False)
        self.original_size_button.style().polish(self.original_size_button)

        self.player.video_zoom = self.zoom_level
        self.player.video_pan_x = self.pan_x
        self.player.video_pan_y = self.pan_y
        print("Zoom, Pan e modo 1:1 resetados.")

    def play_pause(self):
        """Alterna o estado de pausa e atualiza a UI."""
        if not self.player or not self.player.path: return

        # --- CORREÇÃO AQUI ---
        # Verifica se o player está pausado, o repeat está desligado e se a agulha está no ponto de saída (ou além).
        current_frame = int((self.player.time_pos or 0) * self.original_framerate)
        is_at_the_end = current_frame >= self.out_point_frame

        if self.is_paused and not self.is_repeat_on and is_at_the_end:
            # Se todas as condições forem verdadeiras, o play reinicia do ponto de entrada (In).
            self.seek_to_frame(frame_number=self.in_point_frame)
            self.is_paused = False
            self.player.pause = False
            self.update_play_pause_icon()
            return

        # Comportamento normal de play/pause para todos os outros casos.
        self.is_paused = not self.is_paused
        self.player.pause = self.is_paused
        self.update_play_pause_icon()

        if self.is_paused:
            QTimer.singleShot(50, self.update_ui)

    def update_play_pause_icon(self):
        """Atualiza o ícone e o estilo do botão de play/pause com base em self.is_paused."""
        if not self.player:
            # Garante o estado inicial correto antes de um vídeo carregar
            self.play_pause_button.setIcon(self.play_icon)
            self.play_pause_button.setProperty("state", "paused")
            self.play_pause_button.style().polish(self.play_pause_button)
            return

        if self.is_paused:
            self.play_pause_button.setIcon(self.play_icon)
            self.timer.stop()
            self.play_pause_button.setProperty("state", "paused")
        else:
            self.play_pause_button.setIcon(self.pause_icon)
            self.timer.start()
            self.play_pause_button.setProperty("state", "playing")

        # Força o botão a reavaliar sua folha de estilo
        self.play_pause_button.style().polish(self.play_pause_button)

    def clear_range(self):
        if not self.video_info_loaded: return

        self.in_point_frame = 0
        final_frame = self.total_frames - 1 if self.total_frames > 0 else 0
        self.out_point_frame = final_frame

        self.in_frame_input.setText("0")
        self.out_frame_input.setText(str(final_frame))
        self.update_ui() # Força a atualização do slider

    def stop_media(self, clear_ui_only=False):
        """
        Para o player. Se for um 'stop' do usuário, pausa e volta ao In.
        Se for para carregar um novo arquivo (clear_ui_only=True), limpa tudo.
        """
        if not self.player: return

        # Se o usuário clicou no botão Stop (comportamento padrão)
        if not clear_ui_only and self.player.path:
            # --- CORREÇÃO AQUI ---
            # Define nosso estado interno como pausado ANTES de atualizar a UI
            self.is_paused = True
            self.player.pause = True
            self.update_play_pause_icon()
            self.seek_to_frame(frame_number=self.in_point_frame)
            return
        if self.player:
            self.player.target_trc = 'bt.1886'

        # Lógica de limpeza total (chamada ao carregar um novo arquivo)
        self._cleanup_temp_dir()
        if self.player.fullscreen:
            self.player.fullscreen = False
        apply_lut_filter(self.player, None)
        self.player.command('stop')

        if hasattr(self, 'ffmpeg_concat_file') and self.ffmpeg_concat_file:
            try:
                if os.path.exists(self.ffmpeg_concat_file):
                    os.unlink(self.ffmpeg_concat_file)
                self.ffmpeg_concat_file = None
            except Exception as e:
                print(f"Não foi possível remover o arquivo temporário do FFmpeg: {e}")

        self.current_media_path = None
        self.filename_label.setText("N/A")
        self.field_label.setText("N/A")
        self.codec_label.setText("N/A")
        self.original_fps_label.setText("N/A")
        self.color_label.setText("N/A")
        self.alpha_label.setText("N/A")
        self.is_image_sequence = False
        self.play_pause_button.setIcon(self.play_icon)
        self.video_info_loaded = False
        self.timecode_label.setText("00:00:00.000")
        self.current_frame_input.setText("0")
        self.total_frames_label.setText("0")
        self.framerate_input.setText("0.0")
        self.time_label.setText("00:00 / 00:00")
        self.framerate = 0.0
        self.original_framerate = 0.0
        self.position_slider.setValue(0)
        self.in_point_frame = 0
        self.out_point_frame = 0
        self.timer.stop()

    def toggle_repeat(self):
        self.is_repeat_on = not self.is_repeat_on
        self.repeat_button.setChecked(self.is_repeat_on)

        # --- ALTERAÇÃO AQUI ---
        # Força a reavaliação da folha de estilo para o botão
        self.repeat_button.style().polish(self.repeat_button)

        if self.player:
            # O loop do vídeo inteiro é controlado pelo MPV
            is_range_set = self.in_point_frame > 0 or (self.out_point_frame != -1 and self.out_point_frame < self.total_frames - 1)
            # Desativa o loop nativo do MPV se estivermos usando o loop de range
            if self.is_repeat_on and is_range_set:
                self.player.loop = False
            else:
                self.player.loop = 'inf' if self.is_repeat_on else False

    def seek_to_frame(self, frame_number=None):
        if not self.player or not self.video_info_loaded or self.framerate == 0: return
        try:
            if frame_number is None:
                frame_number = int(self.current_frame_input.text())
        except ValueError: return

        target_time_sec = frame_number / self.framerate
        self.player.time_pos = target_time_sec
        self.current_frame_input.clearFocus()
        self.update_ui()

    def update_framerate_and_rate(self):
        if not self.video_info_loaded or not self.original_framerate > 0:
            return

        try:
            new_fps_text = self.framerate_input.text().replace(',', '.')
            new_fps = float(new_fps_text)
        except ValueError:
            self.framerate_input.setText(str(self.framerate))
            return

        if new_fps > 0:
            # --- LÓGICA DE VELOCIDADE CORRIGIDA ---
            # Atualiza o FPS de reprodução
            self.framerate = new_fps
            # A velocidade do player é a razão entre o FPS desejado e o original
            self.player.speed = self.framerate / self.original_framerate

        self.framerate_input.clearFocus()
        self.update_ui() # Força a atualização da UI para refletir a nova duração total
    def _find_upscale_shader(self):
        """Verifica a configuração de shaders ativa por um shader de upscale e retorna o fator."""
        if not self.active_shader_config:
            return None
        for shader_data in self.active_shader_config:
            if shader_data.get("enabled", True):
                path = shader_data.get("path", "")
                match = re.search(r'upscale(\d+)x', os.path.basename(path), re.IGNORECASE)
                if match:
                    factor = int(match.group(1))
                    return factor
        return None


    def load_video(self, filename, restore_state=None):
        self.reset_zoom()
        self.is_unscaled = False
        self.original_size_button.setChecked(False)

        # A LÓGICA DE UPSCALE FOI REMOVIDA DAQUI

        if not restore_state:
            self.stop_media(clear_ui_only=True)
            self.current_media_path = filename
            self.source_color_info = self._get_video_color_info(filename)
            self.is_interlaced = self._detect_interlacing_fast(filename)

        self.audio_device_combo.setEnabled(True)
        self.volume_slider.setEnabled(True)
        self.mute_button.setEnabled(True)
        self.is_image_sequence = False

        def do_load():
            if self.player:
                if filename.lower().endswith('.webm'):
                    codec_name = ''
                    try:
                        # Usa ffprobe para detectar o codec de vídeo antes da reprodução
                        probe_cmd = [
                            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name',
                            '-of', 'default=noprint_wrappers=1:nokey=1', filename
                        ]

                        startupinfo = None
                        if sys.platform == "win32":
                            startupinfo = subprocess.STARTUPINFO()
                            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo, encoding='utf-8')
                        codec_name = result.stdout.strip()
                    except Exception as e:
                        print(f"AVISO: Falha ao detectar o codec do WebM para reprodução. Erro: {e}")

                    if codec_name == 'vp9':
                        print("WebM (VP9) file detected, forcing libvpx-vp9 decoder for alpha channel playback.")
                        self.player.loadfile(filename, 'replace', vd='libvpx-vp9')
                    else:
                        print(f"WebM ({codec_name or 'unknown'}) file detected. Using default decoder for playback.")
                        self.player.loadfile(filename, 'replace')
                else:
                    self.player.loadfile(filename, 'replace')
                QTimer.singleShot(250, lambda: self.finish_loading_video(restore_state))

        QTimer.singleShot(50, do_load)

    def finish_loading_video(self, restore_state=None):
        """Preenche a UI com os metadados do vídeo, truncando o texto quando necessário."""
        if not self.player or self.player.duration is None or not hasattr(self.player, 'video_params') or self.player.video_params is None:
            QTimer.singleShot(50, lambda: self.finish_loading_video(restore_state))
            return
        is_source_linear_exr = (self.is_image_sequence and
                                self.current_media_path and
                                self.current_media_path.lower().endswith('.exr'))

        if is_source_linear_exr:
            # Se for um EXR, diz ao player para renderizar em luz linear (sem gama).
            # A imagem irá parecer escura, como esperado.
            print("Fonte EXR detetada. A definir a saída do player para linear.")
            self.player.target_trc = 'linear'
        else:
            # Para qualquer outra fonte (SDR ou HDR de vídeo), usa um padrão de visualização
            # moderno (gama ~2.4), que é o correto para monitores de vídeo.
            print("Fonte de vídeo padrão detetada. A definir a saída do player para bt.1886.")
            self.player.target_trc = 'bt.1886'

        # --- LÓGICA DE TRUNCATE E TOOLTIP APLICADA ---

        # Nome do Ficheiro
        full_filename = os.path.basename(self.current_media_path or "N/A")
        self.filename_label.setText(truncate_text(full_filename, FILENAME_MAX_CHARS))
        self.filename_label.setToolTip(full_filename)

        # Alpha
        self.has_alpha = self.detect_alpha_channel(self.current_media_path)
        alpha_text = "Sim" if self.has_alpha else "Não"
        self.alpha_label.setText(alpha_text)
        self.alpha_label.setToolTip(alpha_text)

        # HDR
        is_hdr = (
            (self.is_image_sequence and self.current_media_path and self.current_media_path.lower().endswith('.exr')) or
            (self.source_color_info.get('color_transfer') in ['smpte2084', 'arib-std-b67'])
        )
        hdr_text = "Sim" if is_hdr else "Não"
        self.hdr_label.setText(hdr_text)
        self.hdr_label.setToolTip(hdr_text)

        # Áudio
        if self.player.audio_codec:
            audio_codec = self.player.audio_codec
            audio_params = self.player.audio_params
            if audio_params:
                channels = audio_params.get('channel-count', 0)
                samplerate = audio_params.get('samplerate', 0)
                full_audio_text = f"{audio_codec.upper()}, {channels} canais, {samplerate / 1000:.1f}kHz"
            else:
                full_audio_text = audio_codec.upper()
            self.audio_info_label.setText(truncate_text(full_audio_text, INFO_MAX_CHARS))
            self.audio_info_label.setToolTip(full_audio_text)
        else:
            self.audio_info_label.setText("Sem Áudio")
            self.audio_info_label.setToolTip("Sem Áudio")

        # Codec de Vídeo
        full_codec_text = self.player.video_codec or "n/a"
        self.codec_label.setText(truncate_text(full_codec_text, INFO_MAX_CHARS))
        self.codec_label.setToolTip(full_codec_text)

        # Framerate
        self.original_framerate = self.player.container_fps or 30.0
        fps_text = f"{self.original_framerate:.3f}"
        self.original_fps_label.setText(fps_text)
        self.original_fps_label.setToolTip(fps_text)

        # Field Order e Color Space
        field_order_text = "Interlaced" if self.is_interlaced else "Progressive"
        self.field_label.setText(field_order_text)
        self.field_label.setToolTip(field_order_text)

        if self.is_interlaced:
            self.player.deinterlace = 'yes'
            self.deinterlace_combo.setCurrentText("Auto")
        else:
            self.player.deinterlace = 'no'
            self.deinterlace_combo.setCurrentText("Off")
        video_params = self.player.video_params
        if video_params:

            self.is_interlaced = field_order_text != 'Progressive'
            self.field_label.setText(field_order_text)
            self.field_label.setToolTip(field_order_text)

            color_space_text = video_params.get('space', 'n/a')
            self.color_label.setText(truncate_text(color_space_text, INFO_MAX_CHARS))
            self.color_label.setToolTip(color_space_text)

            if self.is_interlaced:
                self.player.deinterlace = 'yes'
                self.deinterlace_combo.setCurrentText("Auto")
            else:
                self.player.deinterlace = 'no'
                self.deinterlace_combo.setCurrentText("Off")
        if self.is_image_sequence:
            self.total_frames = len(self.image_files)
            self.original_framerate = 30.0
        else:
            self.original_framerate = self.player.container_fps or 30.0
            self.total_frames = self.player.estimated_frame_count or int(self.player.duration * self.original_framerate)

        self.framerate = self.original_framerate
        final_frame = self.total_frames - 1 if self.total_frames > 0 else 0
        self.original_fps_label.setText(f"{self.original_framerate:.3f}")

        video_params = self.player.video_params



        if restore_state:
            self.in_point_frame = restore_state.get("in_frame", 0)
            self.out_point_frame = restore_state.get("out_frame", final_frame)
        else:
            self.in_point_frame = 0
            self.out_point_frame = final_frame

        self.in_frame_input.setText(str(self.in_point_frame))
        self.out_frame_input.setText(str(self.out_point_frame))
        self.framerate_input.setText(f"{self.framerate:.3f}".replace('.',','))
        display_total = final_frame if self.is_image_sequence else self.total_frames
        self.total_frames_label.setText(str(display_total))
        self.player.loop = 'inf' if self.is_repeat_on else False

        self.video_info_loaded = True
        self.set_volume(self.volume_slider.value())

        self.is_paused = False
        self.player.pause = False
        self.update_play_pause_icon()

        if restore_state:
            self.player.time_pos = restore_state["time"]
            self.player.pause = restore_state["paused"]
        else:
            self.player.pause = False

        self.update_play_pause_icon()
        self.apply_all_effects()
        self.controls_container.setEnabled(True)

        if hasattr(self, 'ocio_manager') and self.ocio_manager.ocio_config:
            self.ocio_manager.handle_ocio_selection()

    def open_video_file(self):
        if not self.is_paused:
            self.player.pause = True
            self.play_pause_button.setIcon(self.play_icon)
            self.is_paused = True
            self.timer.stop()
            self.update_play_pause_icon()
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Vídeo", os.path.expanduser('~'),
                                                  "Video Files (*.mp4 *.mkv *.avi *.mov *.mxf *.webm *.gif)")
        if filename:
            self.load_video(filename)

    def open_image_folder(self):
        """Abre um diálogo para selecionar UM arquivo da sequência."""
        if not self.is_paused:
            self.player.pause = True
            self.play_pause_button.setIcon(self.play_icon)
            self.is_paused = True
            self.timer.stop()
            self.update_play_pause_icon()

        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Arquivo da Sequência de Imagens",
                                                  os.path.expanduser('~'),
                                                  "Image Files (*.png *.jpg *.jpeg *.exr *.tiff *.tga *.bmp)")
        if filename:
            self.load_image_sequence(filename)

    def update_ui(self):
        if not self.player or not self.video_info_loaded or self.original_framerate <= 0:
            return

        current_time_sec = self.player.time_pos or 0
        current_frame = int(current_time_sec * self.original_framerate)
        is_range_set = self.in_point_frame > 0 or (self.out_point_frame != -1 and self.out_point_frame < self.total_frames - 1)

        if not self.is_paused:
            if not self.is_repeat_on:
                if is_range_set and current_frame >= self.out_point_frame:
                    self.on_video_ended()
                    return
                elif not is_range_set:
                    if (not self.is_image_sequence and self.player.percent_pos is not None and self.player.percent_pos >= 99.9) or \
                       (self.is_image_sequence and current_frame >= self.total_frames - 1):
                        self.on_video_ended()
                        return
            elif self.is_repeat_on and is_range_set:
                if current_frame >= self.out_point_frame:
                    in_point_time = self.in_point_frame / self.original_framerate
                    self.player.time_pos = in_point_time
                    return

        total_duration_sec = self.total_frames / self.framerate if self.framerate > 0 else 0
        current_playback_time_sec = current_frame / self.framerate if self.framerate > 0 else 0

        self.time_label.setText(f"{self.format_time(int(current_playback_time_sec*1000), show_ms=False)} / {self.format_time(int(total_duration_sec*1000), show_ms=False)}")
        self.timecode_label.setText(self.format_time(int(current_time_sec*1000), show_ms=True))



        if self.total_frames > 1:
            # Converte os pontos de In/Out de frames para a escala do slider (0-1000)
            in_point_slider = int((self.in_point_frame / (self.total_frames - 1)) * 1000)
            out_point_slider = int((self.out_point_frame / (self.total_frames - 1)) * 1000)

            # Atualiza os pontos de range no widget do slider
            self.position_slider.set_range_points(in_point_slider, out_point_slider)

        if not self.position_slider.isSliderDown():
            percent_pos = (current_time_sec / self.player.duration * 100) if self.player.duration else 0
            self.position_slider.setValue(int(percent_pos * 10))


        if not self.current_frame_input.hasFocus():
            self.current_frame_input.setText(str(current_frame))

    def set_in_point(self):
        if self.player and self.player.path:
            current_frame = int((self.player.time_pos or 0) * self.original_framerate)
            self.in_point_frame = current_frame
            self.in_frame_input.setText(str(current_frame))
            self.update_ui() # Força a atualização do slider

    def set_out_point(self):
        if self.player and self.player.path:
            current_frame = int((self.player.time_pos or 0) * self.original_framerate)
            self.out_point_frame = current_frame
            self.out_frame_input.setText(str(current_frame))
            self.update_ui() # Força a atualização do slider

    def update_range_from_inputs(self):
        try:
            in_f = int(self.in_frame_input.text())
            out_f = int(self.out_frame_input.text())

            self.in_point_frame = max(0, in_f)
            self.out_point_frame = min(self.total_frames - 1, out_f)

            if self.in_point_frame >= self.out_point_frame:
                self.in_point_frame = self.out_point_frame - 1

            self.in_frame_input.setText(str(self.in_point_frame))
            self.out_frame_input.setText(str(self.out_point_frame))

            self.seek_to_frame(frame_number=self.in_point_frame)
        except ValueError:
            print("Valor de frame inválido.")


    @staticmethod
    def format_time(ms, show_ms=True):
        seconds, milliseconds = divmod(ms, 1000)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if show_ms:
            return f"{hours:02d}:{mins:02d}:{secs:02d},{milliseconds:03d}"
        else:
            return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

    def frame_step_forward(self):
        if not self.player or not self.player.path: return

        # --- LÓGICA CORRIGIDA ---
        self.is_paused = True # Define nosso estado como pausado
        self.player.pause = self.is_paused # Sincroniza o player
        self.player.command('frame-step')

        self.update_play_pause_icon() # Atualiza a UI
        QTimer.singleShot(50, self.update_ui)

    def frame_step_backward(self):
        if not self.player or not self.player.path: return

        # --- LÓGICA CORRIGIDA ---
        self.is_paused = True # Define nosso estado como pausado
        self.player.pause = self.is_paused # Sincroniza o player
        self.player.command('frame-back-step')

        self.update_play_pause_icon() # Atualiza a UI
        QTimer.singleShot(50, self.update_ui)



    def _prerender_exr_layer(self, image_files, layer_name):
        """Inicia a thread para o pré-render paralelo da camada EXR."""
        self.temp_render_dir = tempfile.mkdtemp(prefix="cvplayer_render_")
        print(f"Criando pasta temporária para render: {self.temp_render_dir}")
        self.temp_dirs_to_cleanup.append(self.temp_render_dir)
        self.progress_dialog = QProgressDialog(f"Preparing '{layer_name}'...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setWindowTitle("Processing EXR")

        self.thread = QThread()
        self.prerender_worker = PrerenderManagerWorker(image_files, layer_name, self.temp_render_dir)

        self.prerender_worker.progress.connect(self.progress_dialog.setValue)
        self.progress_dialog.canceled.connect(self.prerender_worker.cancel)
        self.prerender_worker.finished.connect(self._on_prerender_finished)

        self.prerender_worker.moveToThread(self.thread)
        self.thread.started.connect(self.prerender_worker.run)

        self.thread.finished.connect(self.thread.deleteLater)
        self.prerender_worker.finished.connect(self.prerender_worker.deleteLater)

        self.thread.start()
        self.progress_dialog.exec()

    def _on_prerender_finished(self, new_files):
        """É chamado quando o pré-render termina."""
        self.progress_dialog.close() # Fecha a janela de progresso

        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if new_files:
            self.image_files = new_files

            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
                for img_path in self.image_files:
                    tf.write(img_path.replace('\\', '/') + '\n')
                list_filepath = tf.name

            media_path = f"mf://@{list_filepath}"

            self.player.loadfile(media_path, 'replace', mf_fps='30')
            self.player.pause = False
            QTimer.singleShot(250, self.finish_loading_video)
        else:
            self.stop_media() # Limpa tudo se a operação foi cancelada ou falhou

    def _handle_exr_layers(self, first_exr_path):
        # Este método permanece exatamente o mesmo, pois a extração do nome está correta
        if not OPENEXR_AVAILABLE:
            print("Aviso: Biblioteca OpenEXR não encontrada. Não é possível selecionar camadas EXR.")
            return None, None

        try:
            exr_file = OpenEXR.InputFile(first_exr_path)
            header = exr_file.header()
            channels = header['channels'].keys()

            layer_names = set()
            default_channels_present = False
            for channel in channels:
                parts = channel.rsplit('.', 1)
                if len(parts) == 2:
                    layer_names.add(parts[0])
                else:
                    default_channels_present = True

            layers = sorted(list(layer_names))
            if default_channels_present and 'rgba' not in layers:
                layers.insert(0, 'rgba')

            if len(layers) > 1:
                dialog = LayerSelectionDialog(self, layers)
                if dialog.exec():
                    selected = dialog.get_selected_layer()
                    return (selected if selected != 'rgba' else None), "ok"
                else:
                    return None, "cancelled"

            return None, None
        except Exception as e:
            print(f"Erro ao ler camadas do arquivo EXR: {e}")
            QMessageBox.warning(self, "Erro de EXR", f"Não foi possível ler as camadas do arquivo:\n{os.path.basename(first_exr_path)}\n\nErro: {e}")
            return None, None

    def load_image_sequence(self, first_file_path):
        """Encontra todos os arquivos da sequência a partir de um arquivo e a carrega."""
        if self.player is None:
            self._initialize_player_once()

        self.stop_media(clear_ui_only=True)

        self.current_media_path = first_file_path # Usa o primeiro arquivo como referência
        self.is_image_sequence = True
        self.selected_exr_layer = None

        self.audio_device_combo.setEnabled(False)
        self.volume_slider.setEnabled(False)
        self.mute_button.setEnabled(False)

        try:
            # --- NOVA LÓGICA DE DETECÇÃO DE SEQUÊNCIA ---
            directory = os.path.dirname(first_file_path)
            base_name = os.path.basename(first_file_path)

            match = re.match(r'^(.*?)(\d+)(\..*)$', base_name)
            if not match:
                QMessageBox.warning(self, "Erro de Sequência", "Não foi possível detectar um padrão de numeração no nome do arquivo (ex: nome_1234.png).")
                self.stop_media()
                return

            prefix, number_str, extension = match.groups()

            # Encontra todos os arquivos na pasta que correspondem ao padrão
            original_image_files = sorted([
                os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().startswith(prefix.lower()) and f.lower().endswith(extension.lower())
            ])
            # --- FIM DA NOVA LÓGICA ---

        except Exception as e:
            print(f"Erro ao ler a pasta de imagens: {e}")
            original_image_files = []

        if not original_image_files:
            self.stop_media()
            return

        self.image_files = original_image_files

        # O resto da sua lógica original de tratamento de EXR é mantida
        if self.image_files[0].lower().endswith('.exr'):
            self.selected_exr_layer, status = self._handle_exr_layers(self.image_files[0])
            if status == "cancelled":
                self.stop_media()
                return
            if self.selected_exr_layer:
                self._prerender_exr_layer(self.image_files, self.selected_exr_layer)
            else:
                self.continue_loading_sequence()
        else:
            self.continue_loading_sequence()

    def _execute_load_image_sequence(self, folder_path):
        self.reset_zoom()
        self.is_unscaled = False
        self.original_size_button.setChecked(False)
        """Função auxiliar que contém a lógica de carregamento real da sequência."""
        self.stop_media()

        self.current_media_path = folder_path
        self.is_image_sequence = True
        self.selected_exr_layer = None

        self.audio_device_combo.setEnabled(False)
        self.volume_slider.setEnabled(False)
        self.mute_button.setEnabled(False)

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.exr', '.tga')
        try:
            original_image_files = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(valid_extensions)
            ])
        except Exception as e:
            print(f"Erro ao ler a pasta de imagens: {e}")
            original_image_files = []

        if not original_image_files:
            self.stop_media()
            return

        self.image_files = original_image_files

        if self.image_files[0].lower().endswith('.exr'):
            self.selected_exr_layer, status = self._handle_exr_layers(self.image_files[0])
            if status == "cancelled":
                self.stop_media()
                return
            if self.selected_exr_layer:
                self._prerender_exr_layer(self.image_files, self.selected_exr_layer)
            else:
                self.continue_loading_sequence()
        else:
            self.continue_loading_sequence()

    def continue_loading_sequence(self, restore_state=None):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tf:
            for img_path in self.image_files:
                tf.write(img_path.replace('\\', '/') + '\n')
            list_filepath = tf.name
        media_path = f"mf://@{list_filepath}"

        self.player.loadfile(media_path, 'replace', mf_fps='30')
        # A lógica de restaurar estado é movida para finish_loading_video
        QTimer.singleShot(250, lambda: self.finish_loading_video(restore_state))



    def set_position(self, position):
        if self.player:
            self.player.percent_pos = position / 10
        if not self.player or self.player.pause:
            self.update_ui()

    def next_image(self):
        """Exibe o próximo frame da sequência de imagens."""
        if not self.image_files or not self.is_image_sequence:
            self.image_seq_timer.stop()
            return

        # Carrega e exibe a imagem atual
        pixmap = QPixmap(self.image_files[self.current_image_index])
        # Escala a imagem para caber na label, mantendo o aspect ratio
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(),
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

        # Atualiza os contadores na UI
        self.current_frame_input.setText(str(self.current_image_index + 1))

        # Avança para o próximo índice
        self.current_image_index += 1

        # Faz o loop se a repetição estiver ligada
        if self.current_image_index >= len(self.image_files):
            if self.is_repeat_on:
                self.current_image_index = 0
            else:
                self.play_pause() # Pausa no final se a repetição estiver desligada

    def populate_audio_devices(self):
        if not self.player: return
        self.audio_device_combo.clear()
        try:
            # Estas linhas são do seu código original e funcionam.
            devices = self.player.audio_device_list
            current_device_name = self.player.audio_device

            current_index = 0
            for i, dev in enumerate(devices):
                # O texto completo do dispositivo
                full_description = dev['description']

                # Adiciona o item ao dropdown
                self.audio_device_combo.addItem(full_description, dev['name'])

                # --- LINHA ADICIONADA AQUI ---
                # Define o texto completo como o tooltip para o item que acabamos de adicionar.
                self.audio_device_combo.setItemData(i, full_description, Qt.ItemDataRole.ToolTipRole)
                # --- FIM DA LINHA ADICIONADA ---

                if dev['name'] == current_device_name:
                    current_index = i

            if current_index > 0:
                self.audio_device_combo.setCurrentIndex(current_index)

        except Exception as e:
            print(f"Não foi possível listar dispositivos de áudio: {e}")

    def set_audio_device(self, index):
        if not self.player: return
        device_name = self.audio_device_combo.itemData(index)

        if device_name and self.player.audio_device != device_name:
            print(f"Iniciando troca assíncrona para o dispositivo de áudio: {device_name}")

            self.timer.stop()

            # Salva o estado atual para restaurá-lo após a recriação do player
            old_state = {
                "path": self.current_media_path,
                "time": self.player.time_pos,
                "paused": self.player.pause,
                "is_sequence": self.is_image_sequence,
                "image_files": self.image_files if self.is_image_sequence else None,
                "in_frame": self.in_point_frame,
                "out_frame": self.out_point_frame,
            }
            new_device_name = device_name # Salva o nome do novo dispositivo

            self.controls_container.setEnabled(False)

            self.thread = QThread()

            # --- CORREÇÃO PRINCIPAL AQUI ---
            # Pega as opções do nosso novo método centralizado em vez de usar um dicionário hardcoded
            mpv_options = self._get_master_mpv_options()

            self.reset_worker = PlayerResetWorker(self.player, mpv_options)
            self.reset_worker.moveToThread(self.thread)

            self.reset_worker.finished.connect(lambda new_player: self._on_player_reset_finished(new_player, new_device_name, old_state))

            self.thread.started.connect(self.reset_worker.run)
            self.thread.start()

    def _on_player_reset_finished(self, new_player, new_device_name, old_state):
        """Slot chamado quando o worker termina de recriar o player."""
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if new_player:
            self.player = new_player
            if new_device_name:
                self.player.audio_device = new_device_name

            # Reconfigura tudo para o novo player
            self.setup_mpv_events()
            self.setup_mpv_property_observers()
            self.populate_audio_devices() # Agora irá selecionar o dispositivo correto
            self.set_volume(self.volume_slider.value())
            self.player.loop = 'inf' if self.is_repeat_on else False

            if old_state.get("path"):
                self.current_media_path = old_state["path"]
                self.is_image_sequence = old_state["is_sequence"]
                self.image_files = old_state["image_files"]

                if self.is_image_sequence:
                    self.continue_loading_sequence(restore_state=old_state)
                else:
                    self.load_video(old_state["path"], restore_state=old_state)
            else:
                self.controls_container.setEnabled(True)
        else:
            QMessageBox.critical(self, "Erro Fatal", "Não foi possível reinicializar o player de vídeo.")
            self.controls_container.setEnabled(True)

    def set_volume(self, volume):
        if not self.player: return
        self.player.volume = volume
        if volume > 0:
            self.player.mute = False
            self.last_volume = volume
            self.mute_button.setIcon(self.volume_on_icon)
        else:
            self.player.mute = True
            self.mute_button.setIcon(self.volume_muted_icon)

    def toggle_mute(self):
        if not self.player: return
        self.player.mute = not self.player.mute
        if self.player.mute:
            self.volume_slider.setValue(0)
        else:
            if self.last_volume == 0: self.last_volume = 100
            self.player.volume = self.last_volume
            self.volume_slider.setValue(self.last_volume)

    def restore_playback_state(self, was_playing, time_ms):
        if self.player and self.player.path:
            self.player.time_pos = time_ms / 1000.0
            if not was_playing:
                QTimer.singleShot(50, lambda: setattr(self.player, 'pause', True))
