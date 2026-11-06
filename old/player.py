# player.py
import sys
import os
import multiprocessing
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
import psutil

try:
    from log_redirector import LogStream
    LOG_REDIRECTOR_AVAILABLE = True
except ImportError:
    LOG_REDIRECTOR_AVAILABLE = False

basepath = os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Extrai o PID do lançador dos argumentos da linha de comando
    launcher_pid = None
    for arg in sys.argv:
        if arg.startswith('--launcher-pid='):
            try:
                launcher_pid = int(arg.split('=')[1])
                break
            except (ValueError, IndexError):
                pass
    log_stream = None
    if LOG_REDIRECTOR_AVAILABLE:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_stream = LogStream(original_stdout)
        sys.stdout = log_stream
        sys.stderr = log_stream
        print("--- LOG INITIATED ---")
    # --- FIM DA ATIVAÇÃO ---
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(os.getcwd())

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    from player_logic import VideoPlayerApp
    from ui_player import DarkModernStyle

    app.setStyleSheet(DarkModernStyle)

    app_icon_path = resource_path(os.path.join(basepath, "app_icon.png"))
    if os.path.exists(app_icon_path):
        app.setWindowIcon(QIcon(app_icon_path))

    try:
        # A janela principal é criada
        player_window = VideoPlayerApp(base_path=basepath)
        if log_stream:
            log_stream.message.connect(player_window._handle_log_message)

        # --- LINHA CORRIGIDA E RESTAURADA AQUI ---
        # Inicia o servidor TCP para que a aplicação possa escutar por outras instâncias.
        player_window.start_ipc_server()
        # --- FIM DA CORREÇÃO ---

        player_window.show()

        # Imediatamente após agendar a exibição da janela, mata o processo do lançador/splash.
        if launcher_pid:
            print(f"Janela principal pronta. A terminar o processo do lançador (PID: {launcher_pid}).")
            try:
                p = psutil.Process(launcher_pid)
                p.terminate()
            except psutil.NoSuchProcess:
                print("O processo do lançador já não existia.")
            except Exception as e:
                print(f"Erro ao terminar o processo do lançador: {e}")

        sys.exit(app.exec())
    except RuntimeError as e:
        print(f"ERRO CRÍTICO: {e}")
        sys.exit(1)
