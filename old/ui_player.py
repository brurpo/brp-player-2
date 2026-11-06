from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QFrame, QLabel, QLineEdit, QComboBox, QSizePolicy, QStyleOptionSlider, QStyle)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QIntValidator, QDoubleValidator, QAction,QPainter, QColor
import sys
import os

basepath= os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class RangePaintSlider(QSlider):
    def __init__(self, orientation):
        super().__init__(orientation)
        self.in_point = 0
        self.out_point = self.maximum()

    def set_range_points(self, in_point, out_point):
        """Receives In and Out points in the same scale as the slider (0-1000)."""
        self.in_point = in_point
        self.out_point = out_point
        self.update() # Forces the widget to redraw

    def paintEvent(self, event):
        """Draws the bar with custom colors and the correct handle."""
        painter = QPainter(self)
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        groove_rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self)
        handle_rect = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self)

        # 1. Draw the "out of range" bar (red) as the base
        painter.setBrush(QColor("#a93226"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(groove_rect)

        # 2. Draw the "in range" part (gray) on top
        in_pos = self.value_to_pixel(self.in_point, groove_rect)
        out_pos = self.value_to_pixel(self.out_point, groove_rect)
        if out_pos > in_pos:
            valid_range_rect = groove_rect.adjusted(in_pos, 0, -(groove_rect.width() - out_pos), 0)
            painter.setBrush(QColor("#3a3a3a"))
            painter.drawRect(valid_range_rect)

        # 3. Draw the already played part (blue) on top of everything
        handle_pixel_pos = self.value_to_pixel(self.value(), groove_rect)
        start_pos = in_pos
        end_pos = min(handle_pixel_pos, out_pos)
        if end_pos > start_pos:
            played_range_rect = groove_rect.adjusted(start_pos, 0, -(groove_rect.width() - end_pos), 0)
            painter.setBrush(QColor("#007bff"))
            painter.drawRect(played_range_rect)

        # 4. Draw the handle as a white circle
        painter.setBrush(QColor("#e0e0e0")) # Handle color from your QSS
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(handle_rect)

    def value_to_pixel(self, value, groove_rect):
        """Converts a slider value to a pixel position."""
        if self.maximum() == self.minimum():
            return 0
        return int(((value - self.minimum()) / (self.maximum() - self.minimum())) * groove_rect.width())

    def mousePressEvent(self, event):
        """Allows the user to click anywhere on the bar to seek."""
        if event.button() == Qt.MouseButton.LeftButton:
            val = self.style().sliderValueFromPosition(self.minimum(), self.maximum(), event.pos().x(), self.width())
            self.setValue(val)
            self.sliderMoved.emit(val)
        super().mousePressEvent(event)
class RangeSlider(QWidget):
    def __init__(self, orientation):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 1. A barra de fundo que terá as cores
        self.range_bar = QLabel()
        self.range_bar.setFixedHeight(8) # Mesma altura do groove do slider

        # 2. O slider interativo que ficará por cima
        self.slider = SeekSlider(orientation)

        layout.addWidget(self.range_bar)
        # Move o slider para cima para sobrepor a barra (margem negativa)
        layout.addWidget(self.slider)
        self.slider.setStyleSheet("QSlider::groove:horizontal { background: transparent; }")

    def update_style(self, in_percent, out_percent, handle_percent):
        """Atualiza o gradiente de cores da barra de fundo."""
        in_stop = in_percent / 100.0
        out_stop = out_percent / 100.0
        handle_stop = handle_percent / 100.0

        # Garante que os valores não se sobreponham de forma inválida
        in_stop = max(0.0, in_stop)
        out_stop = min(1.0, out_stop)
        handle_stop = max(in_stop, min(handle_stop, out_stop))

        style = f"""
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop: 0 #a93226,
                stop: {in_stop - 0.001} #a93226,
                stop: {in_stop} #3a3a3a,
                stop: {handle_stop - 0.001} #3a3a3a,
                stop: {handle_stop} #007bff,
                stop: {out_stop - 0.001} #007bff,
                stop: {out_stop} #a93226,
                stop: 1 #a93226
            );
            border-radius: 4px;
        """
        self.range_bar.setStyleSheet(style)



DarkModernStyle = """
    QWidget {
        background-color: #2b2b2b;
        color: #ffffff;
        font-family: Segoe UI, Arial, sans-serif;
    }
    QMainWindow {
        background-color: #1e1e1e;
    }
    QFrame#video_frame {
        border: none;
    }
    QPushButton {
        background-color: #3c3f41;
        color: #ffffff;
        border: 1px solid #555;
        padding: 2px 8px;
        min-height: 22px;
        max-height: 22px;
        border-radius: 8px;
    }
    #main_controls QPushButton {
        min-width: 40px;
        min-height: 30px;
        max-height: 30px;
    }
    QPushButton:hover {
        background-color: #4f5355;
    }
    QPushButton:pressed {
        background-color: #2a2c2e;
    }
    QPushButton#repeat_button[checked="true"],
    QPushButton#original_size_button[checked="true"] {
        background-color: #007bff;
        border-color: #007bff;
    }
    #play_pause_button[state="paused"] { background-color: #218838; border-color: #218838; }
    #play_pause_button[state="paused"]:hover { background-color: #28a745; }
    #play_pause_button[state="playing"] { background-color: #c05f01; border-color: #c05f01; }
    #play_pause_button[state="playing"]:hover { background-color: #e97300; }
    QPushButton#stop_button { background-color: #8b0000; }
    QPushButton#stop_button:hover { background-color: #a93226; }
    QPushButton#stop_button:pressed { background-color: #670000; }

    QLineEdit, QComboBox {
        background-color: #3c3f41;
        border: 1px solid #555;
        color: #ffffff;
        padding: 1px 4px;
        border-radius: 8px;
        min-height: 22px;
        max-height: 22px;
    }
    QSlider::groove:horizontal {
        height: 8px;
        background: #3a3a3a;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: #e0e0e0;
        width: 16px;
        margin: -4px 0;
        border-radius: 8px;
    }

    /* --- REGRA RESTAURADA AQUI --- */
    /* Esta regra se aplica ao QSlider padrão (slider de volume) */
    QSlider::sub-page:horizontal {
        background: #007bff;
        border-radius: 4px;
    }

    /* Regra geral para a maioria dos ComboBoxes */
    QComboBox {
        min-width: 150px;
    }

    /* Regras específicas para os ComboBoxes da barra de controle inferior */
    QComboBox#audio_device_combo {
        min-width: 150px;
        max-width: 250px;
    }
    QComboBox#gpu_api_combo {
        min-width: 110px;
        max-width: 110px;
    }
    QComboBox#gpu_device_combo {
        min-width: 220px;
        max-width: 220px;
    }

    QComboBox::drop-down { border: none; }
    QComboBox QAbstractItemView {
        background-color: #3c3f41;
        border: 1px solid #555;
        selection-background-color: #007bff;
    }
    QLabel { font-size: 12px; }
    #metadata_label {
        font-weight: bold;
        color: #bbbbbb;
    }
    QMenuBar { background-color: #2b2b2b; color: #ffffff; }
    QMenu { background-color: #2b2b2b; border: 1px solid #444; }
    QMenu::item:selected { background-color: #007bff; }
"""

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRP Player v1.2.101")
        self.setGeometry(100, 100, 1024, 768)

        self.play_icon = QIcon(basepath+"/"+"play_icon.png")
        self.pause_icon = QIcon(basepath+"/"+"pause_icon.png")
        self.stop_icon = QIcon(basepath+"/"+"stop_icon.png")
        self.repeat_icon = QIcon(basepath+"/"+"repeat_icon.png")
        self.replay_icon = QIcon(basepath+"/"+"replay_icon.png")
        self.prev_frame_icon = QIcon(basepath+"/"+"prev_frame_icon.png")
        self.next_frame_icon = QIcon(basepath+"/"+"next_frame_icon.png")
        self.fullscreen_icon = QIcon(basepath+"/"+"fullscreen_icon.png")
        self.video_fullscreen_icon = QIcon(basepath+"/"+"video_fullscreen_icon.png")
        self.volume_on_icon = QIcon(basepath+"/"+"volume_on_icon.png")
        self.volume_muted_icon = QIcon(basepath+"/"+"volume_muted_icon.png")
        self.clear_icon = QIcon(resource_path(basepath+"/"+"clear.png"))
        self.restore_zoom_icon = QIcon(resource_path(basepath+"/"+"restoreZoom.png"))
        self.original_size_icon = QIcon(resource_path(basepath+"/"+"originalSize.png"))
        self.to_start_icon = QIcon(resource_path(basepath+"/"+"toStart.png"))
        self.to_end_icon = QIcon(resource_path(basepath+"/"+"toEnd.png"))
        self.scopes_icon = QIcon(resource_path(basepath+"/"+"scopes_icon.png"))

        self.setup_menu()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        self.ocio_container = QWidget()
        ocio_layout = QHBoxLayout(self.ocio_container)
        ocio_layout.setContentsMargins(10, 5, 10, 0)
        self.ocio_button = QPushButton("OCIO")
        self.ocio_button.setObjectName("ocio_button")
        self.ocio_button.setToolTip("Load OpenColorIO configuration (.ocio)")
        ocio_layout.addWidget(self.ocio_button)
        ocio_layout.addWidget(QLabel("Input Transform:"))
        self.input_transform_combo = QComboBox()
        self.input_transform_combo.setObjectName("input_transform_combo")
        self.input_transform_combo.setEnabled(False)
        ocio_layout.addWidget(self.input_transform_combo)
        ocio_layout.addWidget(QLabel("Output Transform:"))
        self.output_transform_combo = QComboBox()
        self.output_transform_combo.setObjectName("output_transform_combo")
        self.output_transform_combo.setEnabled(False)
        ocio_layout.addWidget(self.output_transform_combo)
        ocio_layout.addStretch(1)

        ocio_layout.addWidget(QLabel("Deinterlace:"))
        self.deinterlace_combo = QComboBox()
        self.deinterlace_combo.setObjectName("deinterlace_combo")
        self.deinterlace_combo.addItems(["Auto", "Off", "bwdif field", "yadif field", "bwdif frame", "yadif frame"])
        self.deinterlace_combo.setToolTip("Select the deinterlacing method")
        ocio_layout.addWidget(self.deinterlace_combo)
        self.main_layout.addWidget(self.ocio_container)

        self.metadata_container = QWidget()
        metadata_layout = QHBoxLayout(self.metadata_container)
        metadata_layout.setContentsMargins(10, 2, 10, 2)

        metadata_layout.addWidget(QLabel("File:", objectName="metadata_label"))
        self.filename_label = QLabel("N/A")
        metadata_layout.addWidget(self.filename_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Field:", objectName="metadata_label"))
        self.field_label = QLabel("N/A")
        metadata_layout.addWidget(self.field_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Codec:", objectName="metadata_label"))
        self.codec_label = QLabel("N/A")
        metadata_layout.addWidget(self.codec_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Alpha:", objectName="metadata_label"))
        self.alpha_label = QLabel("N/A")
        metadata_layout.addWidget(self.alpha_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("HDR:", objectName="metadata_label"))
        self.hdr_label = QLabel("N/A")
        metadata_layout.addWidget(self.hdr_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Audio:", objectName="metadata_label"))
        self.audio_info_label = QLabel("N/A")
        metadata_layout.addWidget(self.audio_info_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Framerate:", objectName="metadata_label"))
        self.original_fps_label = QLabel("N/A")
        metadata_layout.addWidget(self.original_fps_label)
        metadata_layout.addStretch(1)

        metadata_layout.addWidget(QLabel("Space:", objectName="metadata_label"))
        self.color_label = QLabel("N/A")
        metadata_layout.addWidget(self.color_label)
        self.main_layout.addWidget(self.metadata_container)

        self.video_frame = QFrame()
        self.video_frame.setObjectName("video_frame")
        self.video_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(True)
        video_frame_layout = QVBoxLayout(self.video_frame)
        video_frame_layout.setContentsMargins(0, 0, 0, 0)
        video_frame_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.video_frame)

        self.controls_container = QWidget()
        controls_group_layout = QVBoxLayout(self.controls_container)
        controls_group_layout.setContentsMargins(10, 5, 10, 5)

        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Timecode:"))
        self.timecode_label = QLabel("00:00:00.000")
        self.timecode_label.setMinimumWidth(100)
        info_layout.addWidget(self.timecode_label)
        info_layout.addStretch(1)
        info_layout.addWidget(QLabel("Frame:"))
        self.current_frame_input = QLineEdit("0")
        self.current_frame_input.setValidator(QIntValidator(0, 9999999))
        self.current_frame_input.setMaximumWidth(80)
        self.current_frame_input.setMinimumWidth(80)
        info_layout.addWidget(self.current_frame_input)
        info_layout.addWidget(QLabel("/"))
        self.total_frames_label = QLabel("0")
        self.total_frames_label.setMaximumWidth(80)
        self.total_frames_label.setMinimumWidth(80)
        info_layout.addWidget(self.total_frames_label)
        info_layout.addStretch(1)
        info_layout.addWidget(QLabel("FPS:"))
        self.framerate_input = QLineEdit("0.0")
        self.framerate_input.setValidator(QDoubleValidator(0.0, 1000.0, 3))
        self.framerate_input.setMaximumWidth(60)
        info_layout.addWidget(self.framerate_input)
        controls_group_layout.addLayout(info_layout)

        slider_layout = QHBoxLayout()
        self.set_in_button = QPushButton("In")
        self.set_in_button.setMinimumWidth(50)
        self.set_in_button.setToolTip("Set In Point (I)")
        slider_layout.addWidget(self.set_in_button)
        self.in_frame_input = QLineEdit("0")
        self.in_frame_input.setValidator(QIntValidator(0, 9999999))
        self.in_frame_input.setMaximumWidth(80)
        slider_layout.addWidget(self.in_frame_input)

        self.position_slider = RangePaintSlider(Qt.Orientation.Horizontal)
        self.position_slider.setMaximum(1000)
        self.position_slider.setStyle(self.style())
        slider_layout.addWidget(self.position_slider)

        self.time_label = QLabel("00:00 / 00:00")
        slider_layout.addWidget(self.time_label)
        self.out_frame_input = QLineEdit("0")
        self.out_frame_input.setMaximumWidth(80)
        self.set_out_button = QPushButton("Out")
        slider_layout.addWidget(self.out_frame_input)
        slider_layout.addWidget(self.set_out_button)
        controls_group_layout.addLayout(slider_layout)
        self.set_out_button.setMaximumWidth(50)
        self.set_out_button.setToolTip("Set Out Point (O)")
        slider_layout.addWidget(self.set_out_button)

        media_buttons_container = QWidget()
        media_buttons_container.setObjectName("main_controls")

        media_buttons_layout = QHBoxLayout(media_buttons_container)
        media_buttons_layout.setContentsMargins(0,0,0,0)
        media_buttons_layout.addStretch(1)
        self.to_start_button = QPushButton(self.to_start_icon, "")
        self.to_start_button.setToolTip("Go to In Point")
        media_buttons_layout.addWidget(self.to_start_button)

        self.prev_frame_button = QPushButton(self.prev_frame_icon, "")
        self.prev_frame_button.setToolTip("Previous Frame (Left Arrow)")
        media_buttons_layout.addWidget(self.prev_frame_button)
        self.play_pause_button = QPushButton(self.play_icon, "")
        self.play_pause_button.setObjectName("play_pause_button")
        self.play_pause_button.setToolTip("Play/Pause (Spacebar)")
        media_buttons_layout.addWidget(self.play_pause_button)

        self.next_frame_button = QPushButton(self.next_frame_icon, "")
        media_buttons_layout.addWidget(self.next_frame_button)
        self.next_frame_button.setToolTip("Next Frame (Right Arrow)")

        self.to_end_button = QPushButton(self.to_end_icon, "")
        self.to_end_button.setToolTip("Go to Out Point")
        media_buttons_layout.addWidget(self.to_end_button)

        self.stop_button = QPushButton(self.stop_icon, "")
        self.stop_button.setToolTip("Pause and go to In point")
        self.stop_button.setObjectName("stop_button")
        media_buttons_layout.addWidget(self.stop_button)
        self.clear_range_button = QPushButton(self.clear_icon, "")
        self.clear_range_button.setToolTip("Clear In/Out range")
        media_buttons_layout.addWidget(self.clear_range_button)
        self.repeat_button = QPushButton(self.repeat_icon, "")
        self.repeat_button.setObjectName("repeat_button")
        self.repeat_button.setToolTip("Repeat on/off")
        self.repeat_button.setCheckable(True)
        self.repeat_button.setChecked(True)
        self.repeat_button.setProperty("active", "true")
        media_buttons_layout.addWidget(self.repeat_button)
        media_buttons_layout.addStretch(1)

        controls_group_layout.addWidget(media_buttons_container)

        extra_controls_layout = QHBoxLayout()
        self.mute_button = QPushButton(self.volume_on_icon, "")
        extra_controls_layout.addWidget(self.mute_button)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimumWidth(120) # Manter o mínimo para o slider está OK.
        self.volume_slider.setRange(0, 100)
        extra_controls_layout.addWidget(self.volume_slider)

        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setObjectName("audio_device_combo") # <--- NOME ADICIONADO
        extra_controls_layout.addWidget(self.audio_device_combo)

        extra_controls_layout.addWidget(QLabel("GPU API:"))
        self.gpu_api_combo = QComboBox()
        self.gpu_api_combo.setObjectName("gpu_api_combo") # <--- NOME ADICIONADO
        self.gpu_api_combo.setToolTip("Select the GPU rendering API for MPV")
        extra_controls_layout.addWidget(self.gpu_api_combo)

        self.gpu_device_label = QLabel("GPU Device:")
        self.gpu_device_combo = QComboBox()
        self.gpu_device_combo.setObjectName("gpu_device_combo") # <--- NOME ADICIONADO
        self.gpu_device_combo.setToolTip("Select the specific GPU to use for the selected API")
        extra_controls_layout.addWidget(self.gpu_device_label)
        extra_controls_layout.addWidget(self.gpu_device_combo)

        extra_controls_layout.addStretch(1)

        self.original_size_button = QPushButton(self.original_size_icon, "")
        self.original_size_button.setToolTip("View at original size (G)")
        self.original_size_button.setCheckable(True)
        self.original_size_button.setObjectName("original_size_button")

        extra_controls_layout.addWidget(self.original_size_button)
        self.restore_zoom_button = QPushButton(self.restore_zoom_icon, "")
        self.restore_zoom_button.setToolTip("Reset Zoom and Pan (R)")
        extra_controls_layout.addWidget(self.restore_zoom_button)

        self.fullscreen_button = QPushButton(self.fullscreen_icon, "")
        self.fullscreen_button.setToolTip("Window Fullscreen (F11)")
        self.video_fullscreen_button = QPushButton(self.video_fullscreen_icon, "")
        self.video_fullscreen_button.setToolTip("Video Fullscreen (F)")
        extra_controls_layout.addWidget(self.fullscreen_button)
        extra_controls_layout.addWidget(self.video_fullscreen_button)
        controls_group_layout.addLayout(extra_controls_layout)

        self.main_layout.addWidget(self.controls_container)
        self.main_layout.setStretch(2, 1)

        all_interactive_widgets = self.findChildren(QPushButton) + \
                                  self.findChildren(QSlider) + \
                                  self.findChildren(QComboBox)
        for widget in all_interactive_widgets:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def setup_menu(self):
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu("&File")
        self.open_video_action = QAction("Open Video...", self)
        self.open_video_action.setShortcut("Ctrl+O")
        self.file_menu.addAction(self.open_video_action)
        self.open_folder_action = QAction("Open Image Sequence...", self)
        self.open_folder_action.setShortcut("Ctrl+I")
        self.file_menu.addAction(self.open_folder_action)
        self.file_menu.addSeparator()
        self.effects_menu = menu_bar.addMenu("&Effects")
        self.open_effects_action = QAction("Filters...", self)
        self.effects_menu.addAction(self.open_effects_action)
        self.file_menu.addSeparator()
        self.export_menu = menu_bar.addMenu("&Export")
        self.export_action = QAction("Export Video...", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_menu.addAction(self.export_action)
        self.batch_export_action = QAction("Batch Export...", self)
        self.batch_export_action.setShortcut("Ctrl+Shift+E")
        self.export_menu.addAction(self.batch_export_action)
        self.file_menu.addSeparator()
        self.utils_menu = menu_bar.addMenu("&Utilities")
        self.open_log_action = QAction("Log...", self)
        self.utils_menu.addAction(self.open_log_action)
        self.file_menu.addSeparator()
        self.exit_action = QAction("Exit", self)
        self.file_menu.addAction(self.exit_action)
