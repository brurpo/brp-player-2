# messenger.py
import sys
import socket
import subprocess
import os
import tkinter as tk
from PIL import Image, ImageTk

basepath = os.path.dirname(os.path.abspath(__file__))

# --- Constantes ---
APP_PORT = 61337
APP_SECRET_KEY = "BRP_PLAYER_IPC_KEY"
PLAYER_EXE_NAME = "core.brp"
PLAYER_SCRIPT_NAME = "player.py"

def resource_path(relative_path):
    """ Obtém o caminho absoluto para o recurso. """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def show_splash_and_wait():
    """Cria e exibe uma janela de splash screen e entra no seu loop principal."""
    try:
        splash_path = resource_path(basepath + "\\"+"splash.png")
        if not os.path.exists(splash_path): return

        splash_root = tk.Tk()
        splash_root.withdraw()
        pil_image = Image.open(splash_path)
        splash_image = ImageTk.PhotoImage(pil_image)
        splash_window = tk.Toplevel(splash_root)
        splash_window.overrideredirect(True)
        splash_window.attributes("-topmost", True)
        transparent_color = '#ff00ff'
        splash_window.config(bg=transparent_color)
        splash_window.wm_attributes('-transparentcolor', transparent_color)
        splash_label = tk.Label(splash_window, image=splash_image, bd=0, bg=transparent_color)
        splash_label.image = splash_image
        splash_label.pack()
        splash_root.update_idletasks()

        splash_width = splash_window.winfo_width()
        splash_height = splash_window.winfo_height()
        screen_width = splash_root.winfo_screenwidth()
        screen_height = splash_root.winfo_screenheight()
        x = (screen_width // 2) - (splash_width // 2)
        y = (screen_height // 2) - (splash_height // 2)
        splash_window.geometry(f'{splash_width}x{splash_height}+{x}+{y}')

        splash_root.update()
        splash_root.mainloop()
    except Exception as e:
        # Escreve o erro num ficheiro para depuração
        with open("splash_error.log", "w") as f:
            f.write(str(e))

def main():
    filepath_to_open = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            s.connect(("127.0.0.1", APP_PORT))
            if filepath_to_open:
                message = f"{APP_SECRET_KEY}|{filepath_to_open}"
                s.sendall(message.encode('utf-8'))
            sys.exit(0)
    except (socket.timeout, ConnectionRefusedError):
        # Nenhuma instância encontrada. Lança o splash e o player.

        # Obtém o ID do processo atual (o lançador)
        launcher_pid = os.getpid()

        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        player_path_brp = os.path.join(base_dir, PLAYER_EXE_NAME)
        player_path_py = os.path.join(base_dir, PLAYER_SCRIPT_NAME)

        args_to_pass = []
        if filepath_to_open:
            args_to_pass.append(filepath_to_open)
        # Adiciona o PID do lançador como um argumento para o player principal
        args_to_pass.append(f'--launcher-pid={launcher_pid}')

        command_to_run = None
        if os.path.exists(player_path_brp):
            command_to_run = [player_path_brp] + args_to_pass
        elif os.path.exists(player_path_py):
            command_to_run = [sys.executable, player_path_py] + args_to_pass

        if command_to_run:
            subprocess.Popen(command_to_run)
            # Depois de lançar o player, este processo fica vivo, a mostrar o splash
            show_splash_and_wait()
        else:
            print(f"ERRO: Não foi possível encontrar '{PLAYER_EXE_NAME}' ou '{PLAYER_SCRIPT_NAME}'")

if __name__ == '__main__':
    main()
