# ui_export_dialog.py
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QComboBox,
                             QSlider, QLineEdit, QDialogButtonBox, QProgressBar, QLabel, QCheckBox, QWidget, QHBoxLayout, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIntValidator
from preset_manager import PresetWidget

CODEC_OPTIONS = {
    ".mp4": {
        "H.264": {
            "software": {'c:v': 'libx264', 'pix_fmt': 'yuv420p', 'quality_param': 'crf', 'supports_alpha': False, 'supports_hdr': False},
            "nvidia":   {'c:v': 'h264_nvenc', 'pix_fmt': 'nv12', 'quality_param': 'cq', 'preset': 'p5', 'supports_alpha': False, 'supports_hdr': False},
            "intel":    {'c:v': 'h264_qsv', 'pix_fmt': 'nv12', 'quality_param': 'global_quality', 'supports_alpha': False, 'supports_hdr': False},
            "amd":      {'c:v': 'h264_amf', 'pix_fmt': 'nv12', 'quality_param': 'quality', 'supports_alpha': False, 'supports_hdr': False}
        },
        "H.265/HEVC": {
            "software": {'c:v': 'libx265', 'pix_fmt': 'yuv420p10le', 'quality_param': 'crf', 'supports_alpha': False, 'supports_hdr': True},
            "nvidia":   {'c:v': 'hevc_nvenc', 'pix_fmt': 'p010le', 'quality_param': 'cq', 'preset': 'p5', 'supports_alpha': False, 'supports_hdr': True},
            "intel":    {'c:v': 'hevc_qsv', 'pix_fmt': 'p010le', 'quality_param': 'global_quality', 'supports_alpha': False, 'supports_hdr': True},
            "amd":      {'c:v': 'hevc_amf', 'pix_fmt': 'yuv420p10le', 'quality_param': 'quality', 'supports_alpha': False, 'supports_hdr': True}
        }
    },
    ".mov": {
        "ProRes 4444 XQ": {"software": {'c:v': 'prores_ks', 'profile:v': '5', 'pix_fmt': 'yuv444p10le', 'supports_alpha': True, 'supports_hdr': True}},
        "ProRes 4444":    {"software": {'c:v': 'prores_ks', 'profile:v': '4', 'pix_fmt': 'yuv444p10le', 'supports_alpha': True, 'supports_hdr': True}},
        "ProRes 422 HQ":  {"software": {'c:v': 'prores_ks', 'profile:v': '3', 'pix_fmt': 'yuv422p10le', 'supports_alpha': False, 'supports_hdr': True}},
        "ProRes 422":     {"software": {'c:v': 'prores_ks', 'profile:v': '2', 'pix_fmt': 'yuv422p10le', 'supports_alpha': False, 'supports_hdr': True}},
        "DNxHR 444":      {"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_444', 'pix_fmt': 'yuv444p10le', 'supports_alpha': True, 'supports_hdr': True}},
        "DNxHR HQX 10-bit":{"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_hqx', 'pix_fmt': 'yuv422p10le', 'supports_alpha': False, 'supports_hdr': True}},
        "DNxHR HQ 8-bit": {"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_hq', 'pix_fmt': 'yuv422p', 'supports_alpha': False, 'supports_hdr': False}},
        "DNxHR SQ 8-bit": {"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_sq', 'pix_fmt': 'yuv422p', 'supports_alpha': False, 'supports_hdr': False}},
        "DNxHR LB 8-bit": {"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_lb', 'pix_fmt': 'yuv422p', 'supports_alpha': False, 'supports_hdr': False}},
        "XDCAM HD422":    {"software": {'c:v': 'mpeg2video', 'pix_fmt': 'yuv422p', 'b:v': '50M', 'supports_alpha': False, 'supports_hdr': False}},
    },
    ".mxf": {
        "DNxHR HQX":      {"software": {'c:v': 'dnxhd', 'profile:v': 'dnxhr_hqx', 'pix_fmt': 'yuv422p10le', 'supports_alpha': False, 'supports_hdr': True}},
        "XDCAM HD422":    {"software": {'c:v': 'mpeg2video', 'pix_fmt': 'yuv422p', 'b:v': '50M', 'supports_alpha': False, 'supports_hdr': False}},
    },
    ".webm": {
        "AV1": {
            "software": {'c:v': 'libaom-av1', 'pix_fmt': 'yuv420p10le', 'quality_param': 'crf', 'supports_hdr': True, 'supports_alpha': False},
            "nvidia":   {'c:v': 'av1_nvenc', 'pix_fmt': 'p010le', 'quality_param': 'cq', 'preset': 'p5', 'supports_hdr': True, 'supports_alpha': False},
            "intel":    {'c:v': 'av1_qsv', 'pix_fmt': 'p010le', 'quality_param': 'global_quality', 'supports_hdr': True, 'supports_alpha': False},
            "amd":      {'c:v': 'av1_amf', 'pix_fmt': 'yuv420p10le', 'quality_param': 'quality', 'supports_hdr': True, 'supports_alpha': False}
        },
        "VP9": { "software": {'c:v': 'libvpx-vp9', 'pix_fmt': 'yuv420p', 'supports_hdr': True, 'supports_alpha': True} }
    },
    ".gif": {}, ".jpg": {}, ".png": {}, ".exr": {}
}

class ExportOptionsWidget(QWidget):
    # Esta classe foi reescrita para ser robusta contra crashes e bugs de visibilidade.
    def __init__(self, source_has_alpha=False, source_resolution=(1920, 1080), source_is_hdr=False, detected_encoders=None, selected_gpu=None):
        super().__init__()
        self.source_has_alpha = source_has_alpha
        self.source_is_hdr = source_is_hdr
        self.source_width, self.source_height = source_resolution
        self.detected_encoders = detected_encoders if detected_encoders else {"nvidia": False, "intel": False, "amd": False}

        # Armazena a GPU selecionada
        self.selected_gpu = selected_gpu

        self.form_layout = QFormLayout(self)
        self.form_layout.setContentsMargins(0, 0, 0, 0)

        # Criação de widgets
        self.hw_accel_checkbox = QCheckBox("Enable Hardware Acceleration")
        self.hw_accel_checkbox.setEnabled(any(self.detected_encoders.values()))
        if not self.hw_accel_checkbox.isEnabled():
            self.hw_accel_checkbox.setToolTip("No compatible hardware encoders found in FFmpeg.")

        self.container_combo = QComboBox()
        self.container_combo.addItem("MP4 (H.264/H.265)", ".mp4")
        self.container_combo.addItem("MOV (ProRes/DNxHR/XDCAM)", ".mov")
        self.container_combo.addItem("MXF (DNxHR/XDCAM)", ".mxf")
        self.container_combo.addItem("WebM (VP9/AV1)", ".webm")
        self.container_combo.addItem("Animated GIF", ".gif")
        self.container_combo.addItem("JPG Sequence", ".jpg")
        self.container_combo.addItem("PNG Sequence", ".png")
        self.container_combo.addItem("EXR Sequence", ".exr")

        self.codec_combo = QComboBox()

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem(f"Match Source ({self.source_width}x{self.source_height})", "match_source")
        self.resolution_combo.addItem("3840x2160 (4K UHD)", "3840:2160")
        self.resolution_combo.addItem("1920x1080 (Full HD)", "1920:1080")
        self.resolution_combo.addItem("1280x720 (HD)", "1280:720")
        self.resolution_combo.addItem("Custom", "custom")

        self.custom_res_widget = QWidget()
        custom_res_layout = QHBoxLayout(self.custom_res_widget)
        custom_res_layout.setContentsMargins(0,0,0,0)
        self.width_input = QLineEdit(str(self.source_width)); self.width_input.setValidator(QIntValidator(1, 8000))
        self.height_input = QLineEdit(str(self.source_height)); self.height_input.setValidator(QIntValidator(1, 8000))
        custom_res_layout.addWidget(self.width_input); custom_res_layout.addWidget(QLabel("x")); custom_res_layout.addWidget(self.height_input)

        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["Fit (Black bars)", "Fill (Crop)", "Stretch (Distort)"])

        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(0, 51)
        self.quality_slider.setValue(23)
        self.quality_slider.setInvertedAppearance(True)

        self.bitrate_input = QLineEdit("20M")
        self.alpha_checkbox = QCheckBox("Alpha Export")
        self.hdr_export_checkbox = QCheckBox("HDR Export (PQ, Rec.2020)")

        # Adição de widgets ao layout
        self.form_layout.addRow(self.hw_accel_checkbox)
        self.form_layout.addRow("Container:", self.container_combo)
        self.form_layout.addRow("Codec:", self.codec_combo)
        self.form_layout.addRow("Resolution:", self.resolution_combo)
        self.form_layout.addRow("Custom Resolution:", self.custom_res_widget)
        self.form_layout.addRow("Framing:", self.scaling_combo)
        self.form_layout.addRow("Quality:", self.quality_slider)
        self.form_layout.addRow("Bitrate:", self.bitrate_input)
        self.form_layout.addRow("", self.alpha_checkbox)
        self.form_layout.addRow("", self.hdr_export_checkbox)

        # Conexão de sinais
        self.container_combo.currentIndexChanged.connect(self.update_codec_options)
        self.codec_combo.currentIndexChanged.connect(self.update_widget_visibility)
        self.resolution_combo.currentIndexChanged.connect(self.update_widget_visibility)
        self.container_combo.currentIndexChanged.connect(self._update_alpha_default_state)
        self.codec_combo.currentIndexChanged.connect(self._update_alpha_default_state)

        # Chamadas iniciais
        self.update_codec_options()
        self._update_alpha_default_state()

    def apply_settings(self, settings):
        self.container_combo.blockSignals(True)
        self.codec_combo.blockSignals(True)
        self.resolution_combo.blockSignals(True)

        container_index = self.container_combo.findData(settings.get("container"))
        if container_index > -1:
            self.container_combo.setCurrentIndex(container_index)
            self.update_codec_options()

        codec_c_v_to_find = settings.get("c:v")
        profile_to_find = settings.get("profile:v")
        found_index = -1
        for i in range(self.codec_combo.count()):
            item_data = self.codec_combo.itemData(i)
            if not item_data: continue
            for implementation in item_data.values():
                if implementation.get('c:v') == codec_c_v_to_find:
                    if profile_to_find:
                        if implementation.get('profile:v') == profile_to_find:
                            found_index = i
                            break
                    else:
                        found_index = i
                        break
            if found_index > -1: break

        if found_index > -1: self.codec_combo.setCurrentIndex(found_index)

        self.quality_slider.setValue(settings.get("quality", 23))
        self.bitrate_input.setText(settings.get("bitrate", "20M"))
        res_index = self.resolution_combo.findData(settings.get("resolution", "match_source"))
        if res_index > -1: self.resolution_combo.setCurrentIndex(res_index)
        self.width_input.setText(str(settings.get("custom_width", "1920")))
        self.height_input.setText(str(settings.get("custom_height", "1080")))
        scale_index = self.scaling_combo.findText(settings.get("scaling_mode", "Fit (Black bars)"))
        if scale_index > -1: self.scaling_combo.setCurrentIndex(scale_index)

        self.container_combo.blockSignals(False)
        self.codec_combo.blockSignals(False)
        self.resolution_combo.blockSignals(False)

        self.update_widget_visibility()

        if self.alpha_checkbox.isVisible(): self.alpha_checkbox.setChecked(settings.get("export_alpha", False))
        if self.hdr_export_checkbox.isVisible(): self.hdr_export_checkbox.setChecked(settings.get("export_as_hdr", False))

    def update_codec_options(self):
        container = self.container_combo.currentData()
        codecs_for_container = CODEC_OPTIONS.get(container, {})
        self.codec_combo.clear()
        for name, data in codecs_for_container.items():
            self.codec_combo.addItem(name, data)
        self.update_widget_visibility()

    def _update_alpha_default_state(self):
        container = self.container_combo.currentData()
        codec_data = self.codec_combo.currentData()
        output_supports_alpha = False
        if container in [".png", ".exr"]:
            output_supports_alpha = True
        elif codec_data:
            sw_data = codec_data.get('software', {})
            output_supports_alpha = sw_data.get('supports_alpha', False)
        should_be_checked_by_default = self.source_has_alpha and output_supports_alpha
        self.alpha_checkbox.setChecked(should_be_checked_by_default)

    def update_widget_visibility(self):
        container = self.container_combo.currentData()
        is_video_container = container in [".mp4", ".mov", ".mxf", ".webm"]
        is_image_sequence = container in [".jpg", ".png", ".exr"]
        codec_data = self.codec_combo.currentData()

        # Visibilidade do Codec
        label = self.form_layout.labelForField(self.codec_combo)
        if label:
            label.setVisible(is_video_container)
            self.codec_combo.setVisible(is_video_container)

        # Visibilidade da Resolução Customizada e Framing
        resolution_mode = self.resolution_combo.currentData()
        is_custom_res = resolution_mode == "custom"
        self.custom_res_widget.setVisible(is_custom_res)
        label = self.form_layout.labelForField(self.custom_res_widget)
        if label: label.setVisible(is_custom_res)
        show_framing = resolution_mode != "match_source"
        self.scaling_combo.setVisible(show_framing)
        label = self.form_layout.labelForField(self.scaling_combo)
        if label: label.setVisible(show_framing)

        # Visibilidade dos Checkboxes Alpha e HDR
        output_supports_alpha = False
        output_supports_hdr = False
        if is_image_sequence:
            output_supports_alpha = container in [".png", ".exr"]
            output_supports_hdr = container == ".exr"
        elif codec_data:
            sw_data = codec_data.get('software', {})
            output_supports_alpha = sw_data.get('supports_alpha', False)
            output_supports_hdr = sw_data.get('supports_hdr', False)
        self.alpha_checkbox.setVisible(output_supports_alpha)
        self.hdr_export_checkbox.setVisible(self.source_is_hdr and output_supports_hdr)

        # Visibilidade da Qualidade e Bitrate
        show_quality = False
        show_bitrate = False
        if is_video_container and codec_data:
            sw_data = codec_data.get('software', {})
            has_quality_param = bool(sw_data.get('quality_param'))
            has_fixed_bitrate = bool(sw_data.get('b:v'))
            show_quality = has_quality_param
            if not has_fixed_bitrate and sw_data.get('c:v') and ('libx26' in sw_data['c:v'] or 'libvpx' in sw_data['c:v']):
                show_bitrate = True

        self.quality_slider.setVisible(show_quality)
        label = self.form_layout.labelForField(self.quality_slider)
        if label: label.setVisible(show_quality)
        self.bitrate_input.setVisible(show_bitrate)
        label = self.form_layout.labelForField(self.bitrate_input)
        if label: label.setVisible(show_bitrate)

    def get_settings(self):
        base_settings = {
            "container": self.container_combo.currentData(),
            "quality": self.quality_slider.value(),
            "bitrate": self.bitrate_input.text(),
            "export_alpha": self.alpha_checkbox.isChecked() and self.alpha_checkbox.isVisible(),
            "resolution": self.resolution_combo.currentData(),
            "custom_width": int(self.width_input.text()) if self.width_input.text() else 0,
            "custom_height": int(self.height_input.text()) if self.height_input.text() else 0,
            "scaling_mode": self.scaling_combo.currentText(),
            "export_as_hdr": self.hdr_export_checkbox.isChecked() and self.hdr_export_checkbox.isVisible()
        }

        chosen_encoder_params = {}
        selected_codec_data = self.codec_combo.currentData()

        if selected_codec_data:
            use_hw = self.hw_accel_checkbox.isChecked() and self.hw_accel_checkbox.isEnabled()
            if use_hw:
                # --- LÓGICA DE SELEÇÃO MODIFICADA ---
                encoder_selected = False
                # 1. Tenta usar a GPU selecionada na UI principal
                if self.selected_gpu:
                    gpu_name_lower = self.selected_gpu.lower()
                    if "nvidia" in gpu_name_lower and self.detected_encoders["nvidia"] and "nvidia" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["nvidia"]
                        encoder_selected = True
                        print(f"Exportação usará o encoder NVIDIA, conforme seleção da UI: {self.selected_gpu}")
                    elif "intel" in gpu_name_lower and self.detected_encoders["intel"] and "intel" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["intel"]
                        encoder_selected = True
                        print(f"Exportação usará o encoder Intel, conforme seleção da UI: {self.selected_gpu}")
                    elif ("amd" in gpu_name_lower or "radeon" in gpu_name_lower) and self.detected_encoders["amd"] and "amd" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["amd"]
                        encoder_selected = True
                        print(f"Exportação usará o encoder AMD, conforme seleção da UI: {self.selected_gpu}")

                # 2. Se não foi possível usar a GPU da UI (ou se nenhuma foi passada), usa o fallback de prioridade
                if not encoder_selected:
                    print("Usando fallback de prioridade para selecionar encoder de hardware.")
                    if self.detected_encoders["nvidia"] and "nvidia" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["nvidia"]
                    elif self.detected_encoders["intel"] and "intel" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["intel"]
                    elif self.detected_encoders["amd"] and "amd" in selected_codec_data:
                        chosen_encoder_params = selected_codec_data["amd"]

            # Se nenhuma opção de hardware funcionar, usa software
            if not chosen_encoder_params:
                chosen_encoder_params = selected_codec_data.get("software", {})

        final_settings = {**base_settings, **chosen_encoder_params}
        final_settings["hw_accel_enabled"] = bool(chosen_encoder_params.get('c:v', '').endswith(('_nvenc', '_qsv', '_amf')))
        return final_settings

class ExportDialog(QDialog):
    export_requested = pyqtSignal(dict)
    cancellation_requested = pyqtSignal()

    def __init__(self, parent=None, source_has_alpha=False, source_resolution=(1920, 1080), source_is_hdr=False, detected_encoders=None, selected_gpu=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(450)
        main_layout = QVBoxLayout(self)
        self.is_exporting = False

        options_groupbox = QGroupBox("Export Options")
        group_layout = QVBoxLayout(options_groupbox)

        self.options_widget = ExportOptionsWidget(source_has_alpha, source_resolution, source_is_hdr,
                                                  detected_encoders=detected_encoders, selected_gpu=selected_gpu)

        apply_func = lambda settings: self.options_widget.apply_settings(settings)
        self.preset_widget = PresetWidget(get_settings_func=self.options_widget.get_settings)
        self.preset_widget.preset_loaded.connect(apply_func)

        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready to export.")

        group_layout.addWidget(self.preset_widget)
        group_layout.addWidget(self.options_widget)
        group_layout.addWidget(self.progress_bar)
        group_layout.addWidget(self.status_label)

        main_layout.addWidget(options_groupbox)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close)
        self.export_button = button_box.button(QDialogButtonBox.StandardButton.Save)
        self.export_button.setText("Export")
        self.export_button.clicked.connect(self.on_export_clicked)
        button_box.rejected.connect(self.reject) # Conecta o sinal reject do QDialogButtonBox

        main_layout.addWidget(button_box)

    def reject(self):
        """Overrides the default close behavior."""
        if self.is_exporting:
            reply = QMessageBox.question(self, "Cancel Export",
                                         "An export is in progress. Are you sure you want to cancel?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.cancellation_requested.emit()
        else:
            super().reject()

    def on_export_clicked(self):
        self.export_requested.emit(self.get_settings())

    def get_settings(self):
        return self.options_widget.get_settings()
