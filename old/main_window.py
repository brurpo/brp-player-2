# main_window.py (CORREÇÃO FINAL DO ATTRIBUTEERROR)
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
from video_widget import VideoWidget
from player_thread import PlayerThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ProPlayer")
        self.setGeometry(100, 100, 1280, 720)

        self.player_thread = None
        self.is_playing = False
        self.total_frames = 0
        self.frame_rate = 30.0

        # UI Widgets
        self.video_widget = VideoWidget(self)
        self.time_label = QLabel("TC: 00:00:00:00")
        self.fps_label = QLabel("FPS: --")
        self.frame_line_edit = QLineEdit("0")
        self.frame_line_edit.setValidator(QIntValidator(0, 999999))
        self.frame_line_edit.setFixedWidth(80)

        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.setEnabled(False)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setEnabled(False)
        self.prev_frame_button = QPushButton("<<")
        self.next_frame_button = QPushButton(">>")
        self.ocio_checkbox = QCheckBox("Enable OCIO")
        self.ocio_checkbox.setChecked(True)

        # Layouts
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        controls_layout = QHBoxLayout()
        main_layout.addWidget(self.video_widget, stretch=1)
        main_layout.addWidget(self.timeline_slider)
        main_layout.addLayout(controls_layout)

        controls_layout.addWidget(self.prev_frame_button)
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.next_frame_button)
        controls_layout.addStretch()
        controls_layout.addWidget(self.ocio_checkbox)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(QLabel("Frame:"))
        controls_layout.addWidget(self.frame_line_edit)

        self.setCentralWidget(central_widget)
        self._create_menu()
        self._connect_signals()

    def keyPressEvent(self, event):
        if self.frame_line_edit.hasFocus():
            super().keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key.Key_Right: self.next_frame()
        elif key == Qt.Key.Key_Left: self.prev_frame()
        elif key == Qt.Key.Key_Space: self.toggle_playback()
        else: super().keyPressEvent(event)

    def _create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&Arquivo")
        open_action = file_menu.addAction("Abrir Mídia...")
        open_action.triggered.connect(self.open_file)

    def _connect_signals(self):
        self.play_pause_button.clicked.connect(self.toggle_playback)
        self.timeline_slider.sliderMoved.connect(self.seek_to_frame)
        self.timeline_slider.sliderPressed.connect(self.pause_on_drag)
        self.timeline_slider.sliderReleased.connect(self.unpause_after_drag)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.prev_frame_button.clicked.connect(self.prev_frame)
        self.ocio_checkbox.stateChanged.connect(self.toggle_ocio)
        self.frame_line_edit.returnPressed.connect(self.seek_from_line_edit)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Abrir Mídia", "", "Vídeos (*.mp4 *.mov *.mkv);;Imagens (*.exr *.dpx *.jpg *.png)")
        if not file_path: return

        if self.player_thread and self.player_thread.isRunning():
            self.player_thread.stop()

        self.play_pause_button.setEnabled(False)
        self.timeline_slider.setEnabled(False)
        # --- CORREÇÃO APLICADA AQUI ---
        self.frame_line_edit.setText("...")

        self.player_thread = PlayerThread(file_path)
        self.player_thread.frame_ready.connect(self.update_ui_frame)
        self.player_thread.media_loaded.connect(self.on_media_loaded)
        self.player_thread.start()

    def on_media_loaded(self, total_frames, frame_rate):
        self.total_frames = total_frames
        self.frame_rate = frame_rate
        self.timeline_slider.setRange(0, self.total_frames - 1 if self.total_frames > 0 else 0)
        self.timeline_slider.setEnabled(True)
        self.play_pause_button.setEnabled(True)
        self.fps_label.setText(f"FPS: {self.frame_rate:.2f}")
        self.update_to_paused_state()
        self.player_thread.seek_and_display_frame(0)

    def update_ui_frame(self, frame_image, frame_num, timecode):
        if not frame_image.isNull(): self.video_widget.set_frame(frame_image)
        if not self.timeline_slider.isSliderDown():
            self.timeline_slider.setValue(frame_num)

        self.time_label.setText(timecode)

        if not self.frame_line_edit.hasFocus():
            self.frame_line_edit.setText(str(frame_num))

    def update_to_paused_state(self):
        self.is_playing = False
        self.play_pause_button.setText("Play")
        if self.player_thread: self.player_thread.toggle_pause(True)

    def update_to_playing_state(self):
        self.is_playing = True
        self.play_pause_button.setText("Pause")
        if self.player_thread: self.player_thread.toggle_pause(False)

    def toggle_playback(self):
        if not self.player_thread or not self.player_thread.is_loaded: return
        if self.is_playing: self.update_to_paused_state()
        else: self.update_to_playing_state()

    def pause_on_drag(self):
        if self.player_thread and self.is_playing:
            self.update_to_paused_state()

    def unpause_after_drag(self):
        self.update_to_paused_state()

    def seek_to_frame(self, frame_num):
        if self.player_thread:
            if self.is_playing:
                self.update_to_paused_state()
            self.player_thread.seek_and_display_frame(frame_num)

    def next_frame(self):
        if not self.player_thread or not self.timeline_slider.isEnabled(): return
        current_frame = self.timeline_slider.value()
        new_frame = current_frame + 1
        if new_frame < self.total_frames:
            self.seek_to_frame(new_frame)

    def prev_frame(self):
        if not self.player_thread or not self.timeline_slider.isEnabled(): return
        current_frame = self.timeline_slider.value()
        new_frame = current_frame - 1
        if new_frame >= 0:
            self.seek_to_frame(new_frame)

    def toggle_ocio(self, state):
        if self.player_thread:
            is_enabled = (state == Qt.CheckState.Checked.value)
            self.player_thread.set_ocio_enabled(is_enabled)

    def seek_from_line_edit(self):
        if not self.player_thread or not self.timeline_slider.isEnabled(): return
        try:
            frame_num = int(self.frame_line_edit.text())
            if 0 <= frame_num < self.total_frames:
                self.seek_to_frame(frame_num)
            else:
                self.frame_line_edit.setText(str(self.timeline_slider.value()))
        except ValueError:
            self.frame_line_edit.setText(str(self.timeline_slider.value()))
        self.frame_line_edit.clearFocus()
