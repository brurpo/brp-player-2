import sys
import os
import mpv
import tempfile
import numpy as np
import PyOpenColorIO as OCIO
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog
from ui_player import Ui_MainWindow
from ocio_logic import OcioManager
from playback_logic import PlaybackManager # Nova importação

class VideoPlayerApp(Ui_MainWindow):
    video_ended = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.ocio_manager = OcioManager(self)

        try:
            self.player = mpv.MPV(wid=str(int(self.video_frame.winId())),
                                  vo='gpu',
                                  log_handler=print)
        except Exception as e:
            raise RuntimeError(f"Não foi possível inicializar o MPV. Erro: {e}")

        self.playback_manager = PlaybackManager(self)

        self.is_repeat_on = True
        self.total_frames = 0
        self.framerate = 0.0
        self.original_framerate = 0.0
        self.video_info_loaded = False
        self.last_volume = 100
        self.current_media_path = None

        self.is_paused = False
        self.is_image_sequence = False
        self.image_files = []
        self.current_image_index = 0

        self.setup_logic_connections()
        self.setup_mpv_events()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_ui)
        self.image_seq_timer = QTimer(self)
        self.image_seq_timer.timeout.connect(self.next_image)

    def setup_logic_connections(self):
        self.open_video_action.triggered.connect(self.open_video_file)
        self.open_folder_action.triggered.connect(self.open_image_folder)
        self.exit_action.triggered.connect(self.close)

        self.ocio_manager.connect_signals()

        self.video_ended.connect(self.on_video_ended)

        # Conecta os widgets às funções do PlaybackManager
        self.current_frame_input.returnPressed.connect(self.playback_manager.seek_to_frame)
        self.position_slider.sliderMoved.connect(self.playback_manager.set_position)
        self.prev_frame_button.clicked.connect(self.playback_manager.frame_step_backward)
        self.play_pause_button.clicked.connect(self.playback_manager.play_pause)
        self.next_frame_button.clicked.connect(self.playback_manager.frame_step_forward)
        self.stop_button.clicked.connect(self.playback_manager.stop)

        self.framerate_input.returnPressed.connect(self.update_framerate_and_rate)
        self.repeat_button.clicked.connect(self.toggle_repeat)
        self.mute_button.clicked.connect(self.toggle_mute)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setValue(self.last_volume)
        self.audio_device_combo.activated.connect(self.set_audio_device)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.populate_audio_devices()

    def setup_mpv_events(self):
        if not self.player: return
        @self.player.event_callback('end-file')
        def handle_end_file(event):
            if event['reason'] == 0:
                self.video_ended.emit()

    def on_video_ended(self):
        if not self.is_repeat_on:
            self.play_pause_button.setIcon(self.replay_icon)
            self.is_paused = True
            self.timer.stop()
            if self.player and self.player.percent_pos is not None:
                self.position_slider.setValue(1000)
        else:
            self.timer.start()

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
        elif key == Qt.Key.Key_Period:
            self.next_frame_button.click()
            event.accept()
            return
        elif key == Qt.Key.Key_Comma:
            self.prev_frame_button.click()
            event.accept()
            return
        elif key == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def load_video(self, filename):
        self.playback_manager.stop()
        self.current_media_path = filename
        self.is_image_sequence = False
        self.image_label.hide()

        self.player.loadfile(filename, 'replace')
        self.player.pause = False

        QTimer.singleShot(250, self.finish_loading_video)

    def finish_loading_video(self):
        if not self.player: return
        self.original_framerate = self.player.container_fps or 30.0
        self.framerate = self.original_framerate
        self.player.loop = 'inf' if self.is_repeat_on else False

        self.update_play_pause_icon()
        self.video_info_loaded = False
        self.set_volume(self.volume_slider.value())
        self.timer.start()

        if self.ocio_manager.ocio_config:
            self.ocio_manager.handle_ocio_selection()

    def open_video_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir Vídeo", os.path.expanduser('~'), "Video Files (*.mp4 *.mkv *.avi *.mov)")
        if filename:
            self.load_video(filename)

    def open_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Abrir Pasta de Imagens", os.path.expanduser('~'))
        if folder_path:
            self.load_image_sequence(folder_path)

    def update_ui(self):
        if self.is_image_sequence or not self.player or not self.player.duration:
            return
        if not self.video_info_loaded:
            duration_sec = self.player.duration
            self.total_frames = int(duration_sec * self.framerate)
            self.framerate_input.setText(f"{self.framerate:.3f}".replace('.',','))
            self.total_frames_label.setText(str(self.total_frames))
            self.video_info_loaded = True

        current_time_sec = self.player.time_pos or 0
        total_time_sec = self.player.duration or 0

        self.time_label.setText(f"{self.format_time(int(current_time_sec*1000), show_ms=False)} / {self.format_time(int(total_time_sec*1000), show_ms=False)}")
        self.timecode_label.setText(self.format_time(int(current_time_sec*1000), show_ms=True))

        if not self.position_slider.isSliderDown():
            self.position_slider.setValue(int((self.player.percent_pos or 0) * 10))

        if self.video_info_loaded and not self.current_frame_input.hasFocus():
            current_frame = int(current_time_sec * self.framerate)
            self.current_frame_input.setText(str(current_frame))

    @staticmethod
    def format_time(ms, show_ms=True):
        seconds, milliseconds = divmod(ms, 1000)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if show_ms:
            return f"{hours:02d}:{mins:02d}:{secs:02d},{milliseconds:03d}"
        else:
            return f"{hours:02d}:{mins:02d}:{secs:02d}" if hours > 0 else f"{mins:02d}:{secs:02d}"

    def load_image_sequence(self, folder_path):
        self.playback_manager.stop()
        self.is_image_sequence = True
        self.image_label.show()
        # Lógica para carregar e exibir imagens

    def next_image(self):
        pass

    def populate_audio_devices(self):
        if not self.player: return
        self.audio_device_combo.clear()
        try:
            devices = self.player.audio_device_list
            for dev in devices:
                self.audio_device_combo.addItem(dev['description'], dev['name'])
        except Exception as e:
            print(f"Não foi possível listar dispositivos de áudio: {e}")

    def set_audio_device(self, index):
        if not self.player: return
        device_name = self.audio_device_combo.itemData(index)
        if device_name:
            self.player.audio_device = device_name

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

    def toggle_repeat(self):
        self.is_repeat_on = not self.is_repeat_on
        self.repeat_button.setChecked(self.is_repeat_on)
        if self.player:
            self.player.loop = 'inf' if self.is_repeat_on else False

    def update_framerate_and_rate(self):
        if not self.player or not self.original_framerate: return
        try:
            new_fps = float(self.framerate_input.text().replace(',', '.'))
        except ValueError:
            self.framerate_input.setText(str(self.framerate))
            return
        if new_fps > 0:
            self.framerate = new_fps
            if self.player.duration:
                self.total_frames = int(self.player.duration * self.framerate)
                self.total_frames_label.setText(str(self.total_frames))
            self.player.speed = self.framerate / self.original_framerate
        self.framerate_input.clearFocus()

    # As funções de playback agora são delegadas ao PlaybackManager
    def stop_media(self): self.playback_manager.stop()
    def set_position(self, p): self.playback_manager.set_position(p)
    def frame_step_forward(self): self.playback_manager.frame_step_forward()
    def frame_step_backward(self): self.playback_manager.frame_step_backward()
