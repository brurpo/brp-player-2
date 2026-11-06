import sys
import cv2
import numpy as np
import math

# Removido: multiprocessing, partial, scipy (não são mais usados diretamente aqui)
# NOVO: Importa nosso módulo Cython compilado.
# Este import só funcionará depois que você compilar o .pyx!
import keyer_optimizer

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QFileDialog, QColorDialog, QHBoxLayout, QProgressDialog, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QPoint, QPointF


class ColorUnmixer:
    """
    Esta classe continua responsável por construir o modelo de cor a partir dos
    dados da UI. A lógica de otimização foi movida para o Cython.
    """
    def build_global_model(self, image, scribbles_by_color):
        color_model = []
        for color_name, scribbles in scribbles_by_color.items():
            pixel_points = []
            for scribble in scribbles:
                for point in scribble:
                    if 0 <= point.y() < image.shape[0] and 0 <= point.x() < image.shape[1]:
                        bgr_pixel = image[point.y(), point.x()]
                        rgb_pixel = bgr_pixel[::-1]
                        pixel_points.append(rgb_pixel)
            if len(pixel_points) < 2: continue
            pixels = np.array(pixel_points, dtype=np.float64) / 255.0
            mean_color, covariance = np.mean(pixels, axis=0), np.cov(pixels, rowvar=False)
            covariance += np.eye(3) * 1e-6
            color_model.append({'name': color_name, 'mean': mean_color, 'cov_inv': np.linalg.inv(covariance)})
        print(f"Modelo global construído com {len(color_model)} distribuições (em RGB).")
        return color_model

class UnmixingWorker(QObject):
    """
    O Worker agora é muito mais simples. Ele apenas prepara os dados
    e chama a função Cython otimizada.
    """
    finished = pyqtSignal(np.ndarray)

    def __init__(self, image, scribbles):
        super().__init__()
        self.image_data = image
        self.scribbles_data = scribbles
        self.unmixer = ColorUnmixer()

    def run(self):
        # Prepara os dados da mesma forma que antes
        image_rgb_float = (self.image_data.astype(np.float64)[:, :, ::-1]) / 255.0

        global_model = self.unmixer.build_global_model(self.image_data, self.scribbles_data)
        if len(global_model) < 2:
            self.finished.emit(np.zeros((1,1,4), dtype=np.uint8)); return

        print(f"Iniciando processamento com Cython (paralelizado)...")

        # Chamada única para a função Cython compilada.
        # Todo o trabalho pesado acontece aqui, em código C otimizado.
        final_image_bgra = keyer_optimizer.process_image_cython(image_rgb_float, global_model)

        print("Processamento com Cython concluído.")
        self.finished.emit(final_image_bgra)

class ScribbleArea(QLabel):
    def __init__(self):
        super().__init__(); self.setMinimumSize(640, 480); self.setAlignment(Qt.AlignmentFlag.AlignCenter); self.setMouseTracking(True)
        self.original_pixmap = None; self.scribbles_by_color = {}; self.current_pen_color = QColor(Qt.GlobalColor.transparent)
        self.is_drawing = False; self.current_scribble = []
    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path); self.clear_scribbles(); self.resizeEvent(None)
    def set_pen_color(self, color): self.current_pen_color = color
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap() and not self.pixmap().isNull():
            self.is_drawing = True; self.current_scribble = [self._map_widget_to_pixmap(event.pos())]; self.update()
    def mouseMoveEvent(self, event):
        if self.is_drawing and self.pixmap() and not self.pixmap().isNull():
            self.current_scribble.append(self._map_widget_to_pixmap(event.pos())); self.update()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_drawing:
            self.is_drawing = False
            if self.current_pen_color.isValid() and self.current_pen_color.alpha() != 0:
                color_name = self.current_pen_color.name()
                if color_name not in self.scribbles_by_color: self.scribbles_by_color[color_name] = []
                scaled_scribble = self._scale_pixmap_to_image(self.current_scribble)
                self.scribbles_by_color[color_name].append(scaled_scribble)
            self.current_scribble = []; self.update()
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap() or self.pixmap().isNull(): return
        painter = QPainter(self)
        for color_name, scribbles in self.scribbles_by_color.items():
            pen = QPen(QColor(color_name), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            for scribble_img_coords in scribbles:
                widget_points = self._scale_image_to_widget(scribble_img_coords)
                if widget_points: painter.drawPath(self.points_to_path(widget_points))
        if self.current_scribble:
            pen = QPen(self.current_pen_color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen); widget_points = self._map_pixmap_to_widget(self.current_scribble)
            painter.drawPath(self.points_to_path(widget_points))
    def _get_pixmap_geometry(self):
        if not self.pixmap() or self.pixmap().isNull(): return QPoint(0, 0), None
        pixmap_size, widget_size = self.pixmap().size(), self.size()
        scaled_size = pixmap_size.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio)
        x_offset, y_offset = (widget_size.width() - scaled_size.width()) / 2, (widget_size.height() - scaled_size.height()) / 2
        return QPoint(int(x_offset), int(y_offset)), scaled_size
    def _map_widget_to_pixmap(self, widget_point):
        offset, _ = self._get_pixmap_geometry(); return widget_point - offset
    def _map_pixmap_to_widget(self, pixmap_points):
        offset, _ = self._get_pixmap_geometry(); return [p + offset for p in pixmap_points]
    def _scale_pixmap_to_image(self, pixmap_points):
        if not self.original_pixmap: return []
        _, scaled_size = self._get_pixmap_geometry(); original_size = self.original_pixmap.size()
        if scaled_size.width() == 0 or scaled_size.height() == 0: return []
        scale_x, scale_y = original_size.width() / scaled_size.width(), original_size.height() / scaled_size.height()
        return [QPoint(int(p.x() * scale_x), int(p.y() * scale_y)) for p in pixmap_points]
    def _scale_image_to_widget(self, image_points):
        if not self.original_pixmap: return []
        offset, scaled_size = self._get_pixmap_geometry(); original_size = self.original_pixmap.size()
        if original_size.width() == 0 or original_size.height() == 0: return []
        scale_x, scale_y = scaled_size.width() / original_size.width(), scaled_size.height() / original_size.height()
        return [QPoint(int(p.x() * scale_x + offset.x()), int(p.y() * scale_y + offset.y())) for p in image_points]
    def points_to_path(self, points):
        path = QPainterPath();
        if not points: return path
        path.moveTo(QPointF(points[0]));
        for p in points[1:]: path.lineTo(QPointF(p))
        return path
    def clear_scribbles(self): self.scribbles_by_color = {}; self.update()
    def resizeEvent(self, event):
        if self.original_pixmap:
            self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Interactive Keying Tool (Cython Otimizado)"); self.setGeometry(100, 100, 1400, 800)
        self.image_path = None; self.current_color = QColor(64, 255, 64); self.setup_ui(); self.set_dark_theme()
    def setup_ui(self):
        main_layout, left_layout = QHBoxLayout(), QVBoxLayout()
        self.btn_load = QPushButton("Carregar Imagem"); self.btn_load.clicked.connect(self.load_image)
        self.btn_add_color = QPushButton("Selecionar Cor do Scribble"); self.btn_add_color.clicked.connect(self.select_color)
        self.color_preview = QLabel(); self.color_preview.setFixedSize(50, 20); self.update_color_preview()
        self.btn_clear = QPushButton("Limpar Scribbles"); self.btn_clear.clicked.connect(self.clear_all_scribbles)
        self.btn_compute = QPushButton("COMPUTAR KEY"); self.btn_compute.clicked.connect(self.start_computation)
        left_layout.addWidget(self.btn_load); color_layout = QHBoxLayout(); color_layout.addWidget(self.btn_add_color); color_layout.addWidget(self.color_preview)
        left_layout.addLayout(color_layout); left_layout.addWidget(self.btn_clear); left_layout.addStretch(); left_layout.addWidget(self.btn_compute)
        self.scribble_area = ScribbleArea(); self.scribble_area.setText("Carregue uma imagem para começar"); self.scribble_area.set_pen_color(self.current_color)
        self.result_label = QLabel("O resultado aparecerá aqui"); self.result_label.setMinimumSize(640, 480); self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(left_layout, stretch=1); main_layout.addWidget(self.scribble_area, stretch=4); main_layout.addWidget(self.result_label, stretch=4)
        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)
    def set_dark_theme(self):
        self.setStyleSheet("""QMainWindow, QWidget { background-color: #2b2b2b; color: #f0f0f0; } QLabel { background-color: #3c3f41; border: 1px solid #555; } QPushButton { background-color: #555; border: 1px solid #777; padding: 5px; min-height: 25px; } QPushButton:hover { background-color: #666; } QPushButton:pressed { background-color: #444; } QProgressDialog { background-color: #3c3f41; }""")
        self.btn_compute.setStyleSheet("background-color: #0d6eab; color: white; font-weight: bold; padding: 10px;")
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagem", "", "Imagens (*.png *.jpg *.bmp)");
        if file_path: self.image_path = file_path; self.scribble_area.set_image(file_path); self.result_label.setText("O resultado aparecerá aqui"); self.result_label.setPixmap(QPixmap())
    def select_color(self):
        color = QColorDialog.getColor(self.current_color, self);
        if color.isValid(): self.current_color = color; self.scribble_area.set_pen_color(color); self.update_color_preview()
    def update_color_preview(self):
        pixmap = QPixmap(self.color_preview.size()); pixmap.fill(self.current_color); self.color_preview.setPixmap(pixmap)
    def clear_all_scribbles(self): self.scribble_area.clear_scribbles()
    def start_computation(self):
        if not self.image_path or not self.scribble_area.scribbles_by_color:
            QMessageBox.warning(self, "Aviso", "Por favor, carregue uma imagem e desenhe pelo menos um scribble."); return
        self.progress_dialog = QProgressDialog("Processando em paralelo (Cython), aguarde...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Computando"); self.progress_dialog.setCancelButton(None); self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        image = cv2.imread(self.image_path); scribbles = self.scribble_area.scribbles_by_color
        self.thread = QThread(); self.worker = UnmixingWorker(image, scribbles); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_computation_finished)
        self.btn_compute.setEnabled(False); self.thread.start(); self.progress_dialog.show()
    def on_computation_finished(self, result_image_bgra):
        self.progress_dialog.close(); self.thread.quit(); self.thread.wait(); self.btn_compute.setEnabled(True)
        if result_image_bgra.shape[:2] == (1,1): QMessageBox.warning(self, "Aviso", "Cálculo falhou: modelo de cor inválido (precisa de pelo menos 2 cores)."); return
        height, width, _ = result_image_bgra.shape; q_image = QImage(result_image_bgra.data, width, height, 4 * width, QImage.Format.Format_ARGB32)
        bg_pixmap = QPixmap(self.result_label.size()); bg_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(bg_pixmap); tile_size = 10
        for y in range(0, self.result_label.height(), tile_size):
            for x in range(0, self.result_label.width(), tile_size):
                color = QColor(128, 128, 128) if (x//tile_size + y//tile_size) % 2 == 0 else QColor(192, 192, 192)
                painter.fillRect(x, y, tile_size, tile_size, color)
        scaled_result = QPixmap.fromImage(q_image).scaled(self.result_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x_pos, y_pos = (self.result_label.width() - scaled_result.width()) // 2, (self.result_label.height() - scaled_result.height()) // 2
        painter.drawPixmap(x_pos, y_pos, scaled_result); painter.end(); self.result_label.setPixmap(bg_pixmap)
        print("Processamento concluído.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
