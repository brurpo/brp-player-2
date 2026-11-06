# converter.py
import ffmpeg

def convert_media(input_path, output_path, output_format, ocio_config, input_cs, output_cs):
    """
    Converte uma mídia aplicando uma transformação OCIO via filtros do FFmpeg.
    """
    try:
        # A sintaxe exata do filtro OCIO pode variar com a versão do FFmpeg
        # ffmpeg -h filter=ocio
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.filter_(stream, 'ocio',
                                config=ocio_config,
                                'dir=from', # Direção da transformação
                                'in_color_space', input_cs,
                                'out_color_space', output_cs)

        stream = ffmpeg.output(stream, output_path, format=output_format, vcodec='libx264')

        print("Comando FFmpeg a ser executado:")
        print(' '.join(ffmpeg.compile(stream)))

        ffmpeg.run(stream, overwrite_output=True)
        print("Conversão concluída com sucesso!")
        return True

    except ffmpeg.Error as e:
        print("Erro na conversão FFmpeg:")
        print(e.stderr.decode())
        return False

# Exemplo de como usar:
# if __name__ == '__main__':
#    convert_media(
#        'input.mov',
#        'output.mp4',
#        'mp4',
#        '/path/to/your/config.ocio',
#        'ACES - ACEScg',
#        'Output - sRGB'
#    )
