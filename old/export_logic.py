# export_logic.py
import subprocess
import re
from PyQt6.QtCore import QObject, pyqtSignal
import tempfile
import os
import sys
import shutil
import time
import collections
import uuid
import traceback

basepath = os.path.dirname(os.path.abspath(__file__))

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class ExportWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self, in_file, out_file, settings, ocio_settings, export_range,
                 is_image_sequence=False, exr_layer=None, active_effects=None,
                 is_source_linear=False, source_color_info=None):
        super().__init__()
        self.in_file = in_file
        self.out_file = out_file
        self.settings = settings
        self.ocio_settings = ocio_settings
        self.export_range = export_range
        self.is_image_sequence_input = is_image_sequence
        self.exr_layer = exr_layer
        self.active_effects = active_effects or []
        self.temp_shader_dir_for_export = None
        self.is_source_linear = is_source_linear
        self.source_color_info = source_color_info or {}
        self.process = None
        self.is_cancelled = False
    def _get_rational_framerate(self, fps):
        """Converte framerates de ponto flutuante conhecidos para sua representação racional padrão."""
        # Compara com uma pequena tolerância para evitar problemas com ponto flutuante
        if abs(fps - 23.976) < 0.01:
            return "24000/1001"
        if abs(fps - 29.97) < 0.01:
            return "30000/1001"
        if abs(fps - 59.94) < 0.01:
            return "60000/1001"

        # Se for um número inteiro (24, 25, 30, 50, 60), retorna como string
        if fps == int(fps):
            return str(int(fps))

        # Como fallback para outros framerates, retorna como string formatada
        return f"{fps:.3f}"

    def cancel(self):
        print("Cancellation requested.")
        self.is_cancelled = True
        if self.process:
            try:
                self.process.terminate()
            except Exception as e:
                print(f"Could not terminate FFmpeg process: {e}")

    def _escape_path_for_ffmpeg(self, path):
        # A lógica original do player que funcionava: apenas substitui as barras invertidas.
        # Adicione a barra invertida antes do ':' para o Windows, pois isso é necessário em alguns casos.
        # A sintaxe 'C\:/' é uma forma de escapar a pontuação em strings de shell do ffmpeg,
        # e a string raw 'r' pode ter sido o problema. Vamos usar uma abordagem mais robusta.
        escaped_path = os.path.normpath(path).replace('\\', '/')
        if sys.platform == "win32" and escaped_path[1] == ':':
            return escaped_path[0] + r'\:' + escaped_path[2:]
        return escaped_path

    def _rewrite_temp_shader_for_export(self, shader_data, temp_dir):
        try:
            original_path = shader_data.get("path") or shader_data.get("original_path")
            with open(original_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()

            rewritten_lines = []
            param_values = {}

            # Mapeia os nomes dos parâmetros para os valores ATUAIS
            if "params" in shader_data:
                for param_name, param_data in shader_data.get("params", {}).items():
                    if param_data.get("type") == "color":
                        param_values[param_data["r_param_name"]] = f"{param_data['r_float_value']:.6f}"
                        param_values[param_data["g_param_name"]] = f"{param_data['g_float_value']:.6f}"
                        param_values[param_data["b_param_name"]] = f"{param_data['b_float_value']:.6f}"
                    else:
                        value = param_data.get("value", param_data["default"])
                        if param_data.get("type") in ["float", "DEFINE"]:
                            param_values[param_name] = f"{float(value):.6f}"
                        else:
                            param_values[param_name] = str(int(value))

            # Novo dicionário para armazenar a ordem e os valores
            param_order = {}
            current_param = None
            for line in original_lines:
                stripped_line = line.strip()
                if stripped_line.startswith('//!PARAM'):
                    param_match = re.match(r'//!PARAM\s+(\w+)', stripped_line)
                    if param_match:
                        current_param = param_match.group(1)
                    # Adiciona a linha de declaração do parâmetro
                    rewritten_lines.append(line)
                elif current_param and stripped_line.startswith('//!'):
                    # Adiciona outras diretivas do parâmetro
                    rewritten_lines.append(line)
                elif current_param:
                    # Linha do valor padrão
                    if current_param in param_values:
                        new_value = param_values[current_param]
                        rewritten_lines.append(f"{new_value}\n")
                    else:
                        rewritten_lines.append(line)
                    current_param = None
                else:
                    rewritten_lines.append(line)

            temp_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(original_path)}"
            temp_path = os.path.join(temp_dir, temp_filename)
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.writelines(rewritten_lines)

            return temp_path
        except Exception as e:
            print(f"Erro ao reescrever shader '{original_path}' para export: {e}")
            traceback.print_exc()
            return None

    def run(self):
        try:
            # --- 1. SETUP INICIAL ---
            container = self.settings.get("container", ".mp4")
            is_image_sequence_export = container in [".jpg", ".png", ".exr"]
            is_gif_export = container == ".gif"
            is_float_export = container == ".exr"
            codec = self.settings.get("c:v")
            is_hdr_export = self.settings.get("export_as_hdr", False)

            ffmpeg_path = shutil.which('ffmpeg') or 'ffmpeg'
            cmd = [ffmpeg_path, '-y']
            if self.settings.get("hw_accel_enabled"):
                codec_hw = self.settings.get("c:v", "")
                hwaccel_params = []
                if "nvenc" in codec_hw: hwaccel_params = ['-hwaccel', 'cuda']
                elif "qsv" in codec_hw: hwaccel_params = ['-hwaccel', 'qsv']
                elif "amf" in codec_hw: hwaccel_params = ['-hwaccel', 'd3d11va']
                if hwaccel_params: cmd.extend(hwaccel_params)

            uses_glsl = any(not effect.get("is_builtin", False) and effect.get("enabled", True) for effect in self.active_effects)
            if uses_glsl:
                cmd.extend(['-init_hw_device', 'vulkan=vk'])
                cmd.extend(['-filter_hw_device', 'vk'])

            if self.in_file.lower().endswith('.webm'):
                try:
                    # Usa ffprobe para detectar o codec de vídeo do arquivo WebM
                    probe_cmd = [
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=codec_name',
                        '-of', 'default=noprint_wrappers=1:nokey=1', self.in_file
                    ]

                    startupinfo = None
                    if sys.platform == "win32":
                        startupinfo = subprocess.STARTUPINFO()
                        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

                    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo, encoding='utf-8')
                    codec_name = result.stdout.strip()

                    # Força o decoder libvpx-vp9 APENAS se o codec for de fato vp9
                    if codec_name == 'vp9':
                        self.log_message.emit("Arquivo WebM (VP9) detectado. Forçando decoder libvpx-vp9 para garantir canal alfa.")
                        cmd.extend(['-c:v', 'libvpx-vp9'])
                    else:
                        self.log_message.emit(f"Arquivo WebM ({codec_name}) detectado. Usando decoder padrão.")

                except Exception as e:
                    self.log_message.emit(f"AVISO: Falha ao detectar o codec do WebM, usando decoder padrão. Erro: {e}")

            framerate = self.export_range.get('framerate', 30.0)
            in_frame = self.export_range['in_frame']
            out_frame = self.export_range['out_frame']
            total_frames_in_range = out_frame - in_frame + 1
            duration_sec = total_frames_in_range / framerate
            audio_filter_chain = []
            final_vf_chain = [] # Renomeado de final_vf_chain para clareza

            output_framerate = framerate
            final_frame_count = total_frames_in_range+1
            deinterlace_selection = self.settings.get("deinterlace_mode", "Auto")

            if " field" in deinterlace_selection:
                final_frame_count *= 2
                output_framerate *= 1
                self.log_message.emit(f"Modo 'send_field' detectado. Contagem de frames de saída ajustada para {final_frame_count} e framerate para {output_framerate:.3f} fps.")

            # Define o framerate de escrita para o vídeo de saída
            cmd.extend(['-r', self._get_rational_framerate(output_framerate)])

            if self.is_image_sequence_input:
                if self.is_source_linear: cmd.extend(['-color_trc', 'linear'])
                # Usa o formato racional para definir o framerate de leitura da sequência
                cmd.extend(['-f', 'concat', '-safe', '0', '-i', self.in_file])
            else:
                # Carrega o arquivo de entrada sem fazer seek (-ss). Isso é crucial para a precisão.
                cmd.extend(['-i', self.in_file])

                # Calcula o ponto de início e a duração para os filtros de corte
                start_time_sec = in_frame / framerate
                duration_sec = total_frames_in_range / framerate

                # Adiciona filtros de corte precisos no início da cadeia
                # O setpts reseta o timestamp para o clipe cortado começar em 0, evitando problemas de sincronia.
                final_vf_chain.append(f"trim=start_frame={in_frame}:end_frame={out_frame + 2},setpts=PTS-STARTPTS")
                audio_filter_chain.append(f"atrim=start={start_time_sec:.6f}:duration={duration_sec:.6f},asetpts=PTS-STARTPTS")




            # --- 2. CONSTRUÇÃO DA CADEIA DE FILTROS (-VF) ---
            was_float_pipeline = False

            if uses_glsl:
                transfer_function = self.source_color_info.get('color_transfer')
                is_source_hdr_video = transfer_function in ['smpte2084', 'arib-std-b67']
                was_float_pipeline = self.is_source_linear or is_source_hdr_video

                # Inicia a cadeia de filtros com a conversão de formato e espaço de cor
                if self.is_source_linear:
                    final_vf_chain.append('zscale=t=linear,format=gbrapf32le')
                elif is_source_hdr_video:
                    final_vf_chain.append(f'zscale=t={transfer_function},format=gbrapf32le')
                else:
                    final_vf_chain.append('format=yuva444p16le')

            deinterlace_selection = self.settings.get("deinterlace_mode", "Auto")
            source_is_interlaced = self.settings.get("source_is_interlaced", False)

            method_to_use = ''
            mode = ''

            if deinterlace_selection == "Auto":
                # Se for 'Auto', só aplica o filtro se o vídeo for entrelaçado.
                # Usa 'bwdif' com 'send_frame' como padrão para Auto, conforme solicitado.
                if source_is_interlaced:
                     method_to_use = 'bwdif'
                     mode = 'send_frame'
            elif " field" in deinterlace_selection:
                method_to_use = deinterlace_selection.split(' ')[0]
                mode = 'send_field'
            elif " frame" in deinterlace_selection:
                method_to_use = deinterlace_selection.split(' ')[0]
                mode = 'send_frame'

            # Adiciona o filtro à cadeia do FFmpeg se um método e modo válidos foram definidos
            if method_to_use and mode:
                 self.log_message.emit(f"Aplicando filtro de deinterlace '{method_to_use}' com modo '{mode}' para exportação.")
                 final_vf_chain.append(f"{method_to_use}=mode={mode}:deint=all")

            # Aplica os shaders GLSL e filtros nativos
            for effect_data in self.active_effects:
                if not effect_data.get("enabled", True): continue
                if effect_data.get("is_builtin") and effect_data.get("is_libplacebo_option"):
                    params = effect_data.get("params", {})
                    param_str = ":".join([f"{k}={v['value']}" for k, v in params.items()])
                    final_vf_chain.append(f"libplacebo=deband=1:{param_str}")
                elif effect_data.get("is_builtin"):
                    filter_name = effect_data["path"].split(":")[1]
                    params = effect_data.get("params", {})

                    param_str_list = []
                    for k, v in params.items():
                        value = v['value']
                        if  'color' in k:
                            value_str = str(value).replace('#', '0x')
                        elif isinstance(value, float):
                            value_str = f"{value:.6f}"
                        else:
                            value_str = str(value)
                        param_str_list.append(f"{k}={value_str}")
                    param_str = ":".join(param_str_list)
                    final_vf_chain.append(f"{filter_name}={param_str}")

                elif not effect_data.get("is_builtin"):
                    if not self.temp_shader_dir_for_export:
                        self.temp_shader_dir_for_export = tempfile.mkdtemp(prefix="cvplayer_export_shaders_")
                    temp_shader_path = self._rewrite_temp_shader_for_export(effect_data, self.temp_shader_dir_for_export)
                    if temp_shader_path:
                        final_vf_chain.append(f"libplacebo=custom_shader_path='{self._escape_path_for_ffmpeg(temp_shader_path)}'")

            # *** ATENÇÃO AQUI: A lógica de compositing com fundo preto foi removida da última sugestão,
            # porque você está usando ProRes4444. Mas pode haver um problema de mapeamento de alpha.
            # O problema provável é que o ProRes4444 está lendo a máscara de forma diferente.
            # Se você estiver usando um filtro de keying como o que eu sugeri,
            # a máscara é um canal alfa. O ProRes4444 espera que o canal alfa seja
            # um valor de 0 a 1.
            # Um ajuste que pode ajudar é garantir que os filtros de cor não afetem
            # o canal alfa de forma inesperada.
            # Por isso, a ordem é crucial.

            # A sua alteração para `yuva444p16le` já é um bom passo. O problema pode
            # estar no `libplacebo` ou em alguma outra parte da cadeia.
            # No entanto, vamos adicionar uma verificação de segurança no final.

            # Removendo a lógica de conversão de cor que pode interferir com o alpha
            # no final da cadeia. O shader já gera o alpha. Vamos apenas garantir
            # que a cadeia final preserve o formato correto.

            # A ordem deve ser: [format de entrada] -> [libplacebo] -> [formato final de saída]

            if uses_glsl and not is_float_export and self.settings.get("export_alpha"):
                if codec == "prores_ks":
                    final_vf_chain.append('format=yuva444p16le')
                elif codec == "dnxhd":
                    final_vf_chain.append('format=yuva444p10le')

                # Adicione outros codecs com suporte a alpha aqui.

            # A sua lógica de remap de pretos e brancos no shader é a mais provável
            # causa da transparência excessiva. Verifique os valores de BlackRemap e WhiteRemap.
            # Tente exportar com BlackRemap=0.0 e WhiteRemap=1.0 para ver se o problema
            # de transparência melhora. Se sim, o problema está nesses parâmetros.
            #print("lut applied")
            ocio_lut_was_applied = False
            if self.ocio_settings.get("lut_path"):
                ocio_lut_was_applied = True
                print("lut applied")
                escaped_lut_path = self._escape_path_for_ffmpeg(self.ocio_settings['lut_path'])
                # A lógica abaixo de format é importante para que o filtro de LUT não quebre o alpha.
                if uses_glsl and was_float_pipeline:
                    final_vf_chain.append("format=gbrp16le")
                final_vf_chain.append(f"lut3d=file='{escaped_lut_path}'")

            #print(f"lutpath:{escaped_lut_path}")

            res_mode = self.settings.get("resolution")
            if res_mode != "match_source":
                w, h = (int(self.settings["custom_width"]), int(self.settings["custom_height"])) if res_mode == "custom" else map(int, res_mode.split(':'))
                mode = self.settings.get("scaling_mode")
                if mode == "Stretch (distorcer)": final_vf_chain.append(f"scale={w}:{h}")
                elif mode == "Fill (cortando excedente)": final_vf_chain.append(f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")
                else: final_vf_chain.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")

            # --- 3. PARÂMETROS FINAIS E LÓGICA DE GAMA ---
            pixel_format = self.settings.get("pix_fmt")
            codec_profile = self.settings.get('profile:v')

            if is_image_sequence_export:
                if container == ".jpg": pixel_format = "yuvj420p"
                elif container == ".png": pixel_format = "rgba" if self.settings.get("export_alpha") else "rgb24"
                elif container == ".exr": pixel_format = "rgbaf32le" if self.settings.get("export_alpha") else "rgbaf32le"

            if is_hdr_export:
                if codec and 'libx265' in codec:
                    if self.settings.get("export_alpha"):
                        pixel_format = 'yuva420p10le'
                    else:
                        pixel_format = 'yuv420p10le'

                elif codec and ('prores' in codec or 'dnxhd' in codec):
                    if 'dnxhr_444' in codec_profile or codec_profile in ['4', '5']: pixel_format = 'yuv444p10le'
                    else: pixel_format = 'yuv422p10le'
                elif codec and ('libvpx-vp9' in codec or 'libaom-av1' in codec): pixel_format = 'yuv420p10le'
            elif self.settings.get("export_alpha"):
                if codec == "prores_ks": pixel_format, codec_profile = 'yuva444p16le', '4444'
                elif codec == "libvpx-vp9": pixel_format = 'yuva420p'
                elif container == ".png": pixel_format = 'rgba'
                elif container == ".exr": pixel_format = 'rgbaf32le'
                elif codec == "libx265": pixel_format = 'yuva420p'

            zscale_filter_str = ""
            print("TRIGGER!!")
            here = "no"
            if ocio_lut_was_applied and (is_hdr_export or (was_float_pipeline or not is_float_export)):
                here = "yes!"
                if is_hdr_export:
                    print("OCIO LUT to HDR applied. Re-tagging stream as HDR (PQ).")
                    zscale_filter_str = f"zscale=p=bt2020:t=smpte2084:m=bt2020nc:pin=bt2020:tin=smpte2084:min=bt2020nc"
                else:
                    print("OCIO LUT to SDR applied. Re-tagging stream as Rec.709.")
                    zscale_filter_str = f"zscale=p=709:t=709:m=709:pin=709:tin=709:min=709"

            elif was_float_pipeline and not is_float_export:
                if is_hdr_export:
                    print("No OCIO LUT. Converting from Linear to HDR (PQ).")
                    zscale_filter_str = f"zscale=p=bt2020:t=smpte2084:m=bt2020nc:tin=linear"
                else:
                    print("No OCIO LUT. Converting from Linear to SDR (Rec.709).")
                    zscale_filter_str = f"zscale=p=709:t=709:m=709:tin=linear"

            print(f"exr to sdr? {here}")

            if zscale_filter_str: final_vf_chain.append(zscale_filter_str)


            if final_vf_chain:
                if is_gif_export:
                    base_filters = ",".join(final_vf_chain)
                    palette_filter = f"{base_filters},fps=15,scale=540:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
                    cmd.extend(['-vf', palette_filter])
                else:
                    cmd.extend(['-vf', ",".join(final_vf_chain)])

            if audio_filter_chain:
                cmd.extend(['-af', ",".join(audio_filter_chain)])


                #cmd.append(f"-r 30000/1001")

            # --- 4. PARÂMETROS DO CODEC ---
            if not is_image_sequence_export and not is_gif_export:
                cmd.extend(['-c:v', codec])
                if pixel_format and container != ".exr":
                    cmd.extend(['-pix_fmt', pixel_format])
                if codec_profile: cmd.extend(['-profile:v', codec_profile])
                if self.settings.get('preset'): cmd.extend(['-preset', self.settings.get('preset')])
                quality_flag = self.settings.get('quality_param')
                if quality_flag: cmd.extend([f'-{quality_flag}', str(self.settings.get("quality"))])
                fixed_bitrate = self.settings.get('b:v')
                if fixed_bitrate: cmd.extend(['-b:v', fixed_bitrate])

            if is_hdr_export:
                if codec and 'libx265' in codec:
                    hdr_params = ":".join(["colorprim=bt2020", "transfer=smpte2084", "colormatrix=bt2020nc"])
                    cmd.extend(['-x265-params', hdr_params])
                else:
                    cmd.extend(['-color_primaries', 'bt2020', '-color_trc', 'smpte2084', '-colorspace', 'bt2020nc'])

            if is_image_sequence_export:
                output_ext = self.settings["container"]
                base_name = os.path.basename(os.path.normpath(self.out_file))
                output_pattern = os.path.join(self.out_file, f"{base_name}_%06d{output_ext}")
                # Usa a contagem de frames final, que pode ter sido dobrada.
                cmd.extend(['-frames:v', str(final_frame_count)])
                if output_ext == ".jpg": cmd.extend(['-q:v', '2'])
                cmd.append(output_pattern)
            elif is_gif_export:
                cmd.append(self.out_file)
            else:
                if not self.is_image_sequence_input:
                    if container in ['.mp4', '.mov']: cmd.extend(['-c:a', 'aac', '-b:a', '320k'])
                    elif container == '.mxf': cmd.extend(['-c:a', 'pcm_s16le'])
                cmd.extend(['-frames:v', str(final_frame_count)])
                cmd.append(self.out_file)



            # --- 5. EXECUÇÃO E LOG ---
            print("Final ffmpeg cmd:", " ".join(f'"{c}"' if " " in c or ":" in c else c for c in cmd))


            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                                            text=True, errors='replace', encoding='utf-8')
            output_log = ""
            rates = collections.deque(maxlen=5) # Armazena as últimas 5 taxas de progresso
            last_percent = 0
            last_progress_time = time.time()
            eta_string = "Calculating..."

            for line in iter(self.process.stdout.readline, ''):
                output_log += line
                self.log_message.emit(line)

                percent = -1

                time_match = re.search(r"time=(\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
                frame_match = re.search(r'frame=\s*(\d+)', line)

                if time_match:
                    h, m, s, ms = map(int, time_match.groups())
                    current_time_sec = h * 3600 + m * 60 + s + ms / 100.0
                    percent = int((current_time_sec / duration_sec) * 100) if duration_sec > 0 else 0
                elif frame_match:
                    current_frame = int(frame_match.group(1))
                    percent = int((current_frame / total_frames_in_range) * 100) if total_frames_in_range > 0 else 0

                if percent > last_percent:
                    now = time.time()
                    time_diff = now - last_progress_time

                    if time_diff > 0:
                        percent_diff = percent - last_percent
                        current_rate = percent_diff / time_diff # % por segundo
                        rates.append(current_rate)

                    if rates:
                        avg_rate = sum(rates) / len(rates)
                        if avg_rate > 0:
                            remaining_percent = 100 - percent
                            eta_seconds = int(remaining_percent / avg_rate)

                            mins, secs = divmod(eta_seconds, 60)
                            hours, mins = divmod(mins, 60)
                            if hours > 0:
                                eta_string = f"{hours:d}:{mins:02d}:{secs:02d}"
                            else:
                                eta_string = f"{mins:02d}:{secs:02d}"

                    last_progress_time = now
                    last_percent = percent

                if percent != -1:
                    if percent > 100: percent = 100
                    self.progress.emit(percent, eta_string)

            self.process.wait()

            if self.is_cancelled:
                self.finished.emit("Export cancelled by user.")
                return

            if self.process.returncode == 0:
                self.finished.emit("Exporting finished!")
            else:
                error_message = (f"Error (code: {self.process.returncode}).\n\n"
                                 f"FFMPEG log:\n{output_log[-1500:]}")
                print(error_message)
                self.error.emit(error_message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Unexpected error: {str(e)}")
        finally:
            if self.temp_shader_dir_for_export and os.path.exists(self.temp_shader_dir_for_export):
                try:
                    #shutil.rmtree(self.temp_shader_dir_for_export)
                    pass
                except Exception as e:
                    print(f"Erro ao remover pasta temporária de shaders: {e}")
            if self.is_image_sequence_input and self.in_file and self.in_file.endswith('.txt') and os.path.exists(self.in_file):
                try:
                    os.unlink(self.in_file)
                except Exception as e:
                    print(f"Erro ao remover arquivo de lista temporário: {e}")
