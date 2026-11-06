# log_redirector.py
from PyQt6.QtCore import QObject, pyqtSignal
import sys

class LogStream(QObject):
    """
    Um objeto que se comporta como um ficheiro (com um método 'write')
    e emite um sinal do Qt com o texto que recebe.
    Agora é robusto o suficiente para lidar com streams de saída inexistentes.
    """
    message = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        # --- CORREÇÃO AQUI ---
        # Só tenta escrever para o stream original se ele existir.
        if self.original_stream:
            self.original_stream.write(text)
        # --- FIM DA CORREÇÃO ---

        # Emite sempre o sinal para a nossa janela de log capturar.
        self.message.emit(text)

    def flush(self):
        # --- CORREÇÃO AQUI ---
        # Só tenta fazer flush se o stream original existir.
        if self.original_stream:
            self.original_stream.flush()
        # --- FIM DA CORREÇÃO ---
