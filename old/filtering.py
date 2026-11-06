# filtering.py

def apply_lut_filter(player, lut_path):
    """
    Aplica ou remove uma LUT 3D usando a propriedade 'target-lut' do MPV.

    :param player: A instância do player MPV.
    :param lut_path: O caminho para o arquivo .cube da LUT, ou None para remover.
    """
    if not player:
        return

    # Normaliza o caminho para evitar problemas com barras invertidas no Windows
    clean_path = ''
    if lut_path:
        clean_path = lut_path.replace('\\', '/')

    try:
        player.target_lut = clean_path
        if clean_path:
            print(f"Filtro LUT aplicado: {clean_path}")
        else:
            print("Filtro LUT removido.")
    except Exception as e:
        print(f"Erro ao aplicar LUT via 'target-lut': {e}")
