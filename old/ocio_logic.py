# ocio_logic.py
import os
import tempfile
import numpy as np
import PyOpenColorIO as OCIO
from PyQt6.QtWidgets import QFileDialog
from filtering import apply_lut_filter

class OcioManager:
    def __init__(self, main_app):
        """
        Inicializa o gerenciador de OCIO.
        :param main_app: Uma referência à instância principal do VideoPlayerApp.
        """
        self.main_app = main_app
        self.ocio_config = None
        self.lut_temp_file_path = None


    def connect_signals(self):
        """Conecta os sinais dos widgets da UI aos métodos desta classe."""

        self.main_app.ocio_button.clicked.connect(self.load_ocio_config)
        self.main_app.input_transform_combo.currentIndexChanged.connect(self.handle_ocio_selection)
        self.main_app.output_transform_combo.currentIndexChanged.connect(self.handle_ocio_selection)

    def load_ocio_config(self):

        """Abre um diálogo para carregar um arquivo de configuração .ocio."""
        filepath, _ = QFileDialog.getOpenFileName(self.main_app, "Carregar Config OCIO", "", "OCIO Config Files (*.ocio)")
        if not filepath:
            return

        try:
            self.ocio_config = OCIO.Config.CreateFromFile(filepath)
            self.populate_ocio_dropdowns()
            if self.main_app.player and self.main_app.player.path:
                self.handle_ocio_selection()
        except Exception as e:
            print(f"Erro ao carregar configuração OCIO: {e}")
            self.ocio_config = None
            self.main_app.input_transform_combo.clear()
            self.main_app.input_transform_combo.setEnabled(False)
            self.main_app.output_transform_combo.clear()
            self.main_app.output_transform_combo.setEnabled(False)

    def populate_ocio_dropdowns(self):
        """Preenche os dropdowns com os color spaces da config carregada."""
        if not self.ocio_config: return
        self.main_app.input_transform_combo.clear()
        self.main_app.output_transform_combo.clear()
        self.main_app.input_transform_combo.addItem("None")
        self.main_app.output_transform_combo.addItem("None")
        try:
            color_spaces = self.ocio_config.getColorSpaceNames()
            for cs in color_spaces:
                self.main_app.input_transform_combo.addItem(cs)
                self.main_app.output_transform_combo.addItem(cs)
            self.main_app.input_transform_combo.setEnabled(True)
            self.main_app.output_transform_combo.setEnabled(True)
        except Exception as e:
            print(f"Erro ao popular dropdowns OCIO: {e}")

    def handle_ocio_selection(self):
        """Gera a LUT usando o OCIO.Baker para ser compatível com HDR e chama a função de filtro."""
        if not self.ocio_config:
            return

        input_space = self.main_app.input_transform_combo.currentText()
        output_space = self.main_app.output_transform_combo.currentText()

        if input_space in ["None", ""] or output_space in ["None", ""]:
            self.lut_temp_file_path = None
            apply_lut_filter(self.main_app.player, None)
            return

        try:
            print(f"Gerando LUT HDR-aware para: {input_space} -> {output_space}")

            baker = OCIO.Baker()
            baker.setConfig(self.ocio_config)

            baker.setInputSpace(input_space)
            baker.setTargetSpace(output_space)

            shaper_space = ""
            available_spaces = self.ocio_config.getColorSpaceNames()
            if "ACEScct" in available_spaces:
                shaper_space = "ACEScct"
            elif any("log" in s.lower() for s in available_spaces):
                shaper_space = next((s for s in available_spaces if "log" in s.lower()), "")

            if shaper_space:
                print(f"Usando shaper space para HDR: {shaper_space}")
                baker.setShaperSpace(shaper_space)
            else:
                print("Aviso: Nenhum shaper space logarítmico encontrado. O LUT pode não cobrir todo o range HDR.")

            lut_size = 64
            baker.setFormat("resolve_cube")

            # --- LINHA REMOVIDA DAQUI ---
            # baker.setType("3d") # Este método não existe mais.

            # A linha abaixo já define implicitamente o tipo como 3D.
            baker.setCubeSize(lut_size)

            lut_data = baker.bake()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.cube', delete=False, encoding='utf-8') as tf:
                if self.lut_temp_file_path and os.path.exists(self.lut_temp_file_path):
                    try: os.remove(self.lut_temp_file_path)
                    except OSError: pass
                tf.write(lut_data)
                self.lut_temp_file_path = tf.name

            apply_lut_filter(self.main_app.player, self.lut_temp_file_path)

        except Exception as e:
            print(f"ERRO ao gerar LUT OCIO: {e}")
            import traceback
            traceback.print_exc()
            self.lut_temp_file_path = None
            apply_lut_filter(self.main_app.player, None)
