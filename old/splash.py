# splash.py
import tkinter as tk
from PIL import Image, ImageTk
import os
import sys

def resource_path(relative_path):
    """ Obtém o caminho absoluto para o recurso. """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def show_splash():
    """Cria e exibe a janela de splash screen e entra no seu loop principal."""
    try:
        splash_path = resource_path("splash.png")
        if not os.path.exists(splash_path):
            return

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

        splash_window.update()

        # Mantém o splash visível até o processo ser terminado
        splash_root.mainloop()

    except Exception as e:
        # Escreve o erro num ficheiro para depuração, já que não haverá consola
        with open("splash_error.log", "w") as f:
            f.write(str(e))

if __name__ == '__main__':
    show_splash()
