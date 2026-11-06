# ocio_manager.py
import PyOpenColorIO as OCIO  # CORREÇÃO APLICADA AQUI
import numpy as np

class OCIOManager:
    def __init__(self, config_path=None):
        try:
            if config_path:
                self.config = OCIO.Config.CreateFromFile(config_path)
            else:
                # Tenta carregar a partir da variável de ambiente $OCIO
                self.config = OCIO.Config.CreateFromEnv()
            print(f"OCIO config loaded: {self.config.getDescription()}")
        except Exception as e:
            print(f"Error loading OCIO config: {e}")
            self.config = None

    def get_colorspaces(self):
        if not self.config:
            return []
        return [cs.getName() for cs in self.config.getColorSpaces()]

    def process(self, numpy_array, input_space, output_space):
        if not self.config or input_space == output_space:
            return numpy_array

        # Em versões mais recentes do PyOpenColorIO, a API pode ser ligeiramente diferente
        # Esta é a abordagem comum:
        processor = self.config.getProcessor(input_space, output_space)

        # O processamento do OCIO é feito "in-place" em arrays CPU
        processed_array = np.copy(numpy_array) # Evita modificar o original

        # Para a API v2, o método pode ser cpuProcessor.applyRGB()
        cpu_processor = processor.getCpuProcessor()
        cpu_processor.applyRGB(processed_array)

        return processed_array
