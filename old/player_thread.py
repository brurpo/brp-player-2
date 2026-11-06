# player_thread.py (FINAL COM OTIMIZAÇÃO DE OCIO)
import av
import numpy as np
import PyOpenColorIO as OCIO
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt6.QtGui import QImage
from ocio_manager import OCIOManager

class PlayerThread(QThread):
    frame_ready = pyqtSignal(QImage, int, str)
    media_loaded = pyqtSignal(int, float)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.mutex = QMutex()
        self.is_running = True

        self.container = None
        self.video_stream = None
        self.frame_generator = None
        self.is_paused = True
        self.is_loaded = False
        self.total_frames = 0
        self.frame_rate = 30.0
        self.ocio_enabled = True # OCIO ligado por padrão

        self.ocio_manager = OCIOManager()
        self.input_colorspace = "ACES - ACEScg"
        self.display_colorspace = "Output - sRGB"

    def run(self):
        try:
            with QMutexLocker(self.mutex):
                # ... (código de abertura e carregamento de metadados igual ao anterior) ...
                self.container = av.open(self.file_path)
                self.video_stream = self.container.streams.video[0]
                self.frame_generator = self.container.decode(video=0)
                if self.video_stream.frames > 0: self.total_frames = self.video_stream.frames
                else: self.total_frames = int(self.video_stream.duration * self.video_stream.average_rate)
                self.frame_rate = float(self.video_stream.average_rate)
                self.is_loaded = True

            self.media_loaded.emit(self.total_frames, self.frame_rate)

            while self.is_running:
                if self.is_paused:
                    self.msleep(30)
                    continue

                with QMutexLocker(self.mutex):
                    if self.is_paused: continue
                    try:
                        frame = next(self.frame_generator)
                        q_img, frame_num, tc = self._process_frame(frame)
                        self.frame_ready.emit(q_img, frame_num, tc)
                    except StopIteration:
                        self.is_paused = True

                if self.frame_rate > 0:
                    self.msleep(int(1000 / self.frame_rate))
        except Exception as e:
            print(f"Erro fatal na thread: {e}")
        finally:
            if self.container: self.container.close()
            print("Thread finalizada.")

    def _process_frame(self, frame):
        total_seconds = frame.pts * self.video_stream.time_base
        frame_number_display = int(total_seconds * self.frame_rate)

        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        frame_of_second = int((total_seconds - int(total_seconds)) * self.frame_rate)
        timecode = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{frame_of_second:02d}"

        img = frame.to_ndarray(format='rgb24')

        # --- OTIMIZAÇÃO DE PERFORMANCE AQUI ---
        if self.ocio_enabled:
            img_float = img.astype(np.float32) / 255.0
            processed_img_float = self.ocio_manager.process(img_float, self.input_colorspace, self.display_colorspace)
            processed_img_8bit = (np.clip(processed_img_float, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            # Pula todo o processamento pesado se o OCIO estiver desligado
            processed_img_8bit = img

        h, w, ch = processed_img_8bit.shape
        q_img = QImage(processed_img_8bit.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        return q_img, frame_number_display, timecode

    def seek_and_display_frame(self, frame_num):
        if not self.is_loaded: return
        with QMutexLocker(self.mutex):
            try:
                target_time = frame_num / self.frame_rate
                timestamp = int(target_time / self.video_stream.time_base)

                self.container.seek(timestamp, stream=self.video_stream, backward=True, any_frame=True)
                self.video_stream.codec_context.flush_buffers()
                self.frame_generator = self.container.decode(video=0)

                frame = next(self.frame_generator)
                q_img, current_frame_num, tc = self._process_frame(frame)
                self.frame_ready.emit(q_img, current_frame_num, tc)
            except StopIteration: pass
            except Exception as e: print(f"ERRO NO SEEK: {e}")

    def toggle_pause(self, paused):
        with QMutexLocker(self.mutex): self.is_paused = paused

    def set_ocio_enabled(self, enabled):
        """Controla se o processamento OCIO deve ser executado."""
        with QMutexLocker(self.mutex):
            self.ocio_enabled = enabled

    def stop(self): self.is_running = False
