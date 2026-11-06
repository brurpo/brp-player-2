# video_widget.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QImage
from PyQt6.QtCore import QSize, Qt # <-- A CORREÇÃO ESTÁ AQUI

class VideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = QImage()
        self.setMinimumSize(640, 360)

    def set_frame(self, frame_image):
        if isinstance(frame_image, QImage):
            self.frame = frame_image
            # Agenda um evento de pintura para redesenhar o widget
            self.update()

    def paintEvent(self, event):
        if not self.frame.isNull():
            painter = QPainter(self)

            # Escala a imagem para preencher o widget mantendo a proporção
            scaled_frame = self.frame.scaled(self.size(),
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)

            # Centraliza a imagem no widget
            x = (self.width() - scaled_frame.width()) // 2
            y = (self.height() - scaled_frame.height()) // 2
            painter.drawImage(x, y, scaled_frame)

    def sizeHint(self):
        return QSize(1280, 720)
