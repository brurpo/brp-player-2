//! Interface strings in English and Portuguese (Brazil).

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Lang {
    #[default]
    En,
    Pt,
}

impl Lang {
    pub fn label(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Pt => "Português",
        }
    }

    pub fn t(self, key: &str) -> &'static str {
        let (en, pt) = match key {
            "open" => ("Open…", "Abrir…"),
            "open_title" => ("Open media", "Abrir mídia"),
            "drop_hint" => ("Drop a video here, or click Open", "Solte um vídeo aqui ou clique em Abrir"),
            "play" => ("Play (Space)", "Reproduzir (Espaço)"),
            "pause" => ("Pause (Space)", "Pausar (Espaço)"),
            "prev_frame" => ("Previous frame (←)", "Quadro anterior (←)"),
            "next_frame" => ("Next frame (→)", "Próximo quadro (→)"),
            "to_in" => ("Go to In point (Home)", "Ir para o ponto de entrada (Home)"),
            "to_out" => ("Go to Out point (End)", "Ir para o ponto de saída (End)"),
            "set_in" => ("Set In point (I)", "Marcar entrada (I)"),
            "set_out" => ("Set Out point (O)", "Marcar saída (O)"),
            "clear_range" => ("Clear In/Out", "Limpar entrada/saída"),
            "loop" => ("Loop", "Repetir"),
            "fps" => ("Playback frame rate", "Taxa de quadros da reprodução"),
            "fit" => ("Fit to window (R)", "Ajustar à janela (R)"),
            "one_to_one" => ("Original size 1:1 (G)", "Tamanho original 1:1 (G)"),
            "fullscreen" => ("Fullscreen (F)", "Tela cheia (F)"),
            "lock" => ("Lock: new files open in another window", "Travar: novos arquivos abrem em outra janela"),
            "unlock" => ("Unlocked: new files replace this video", "Destravado: novos arquivos substituem este vídeo"),
            "info" => ("Info", "Info"),
            "language" => ("Language", "Idioma"),
            "settings" => ("Settings", "Configurações"),
            "effects" => ("Effects", "Efeitos"),
            "fx_add" => ("Add effect…", "Adicionar efeito…"),
            "fx_browse" => ("Browse for a .glsl file…", "Procurar arquivo .glsl…"),
            "fx_presets" => ("Presets", "Predefinições"),
            "fx_save" => ("Save preset…", "Salvar predefinição…"),
            "fx_load" => ("Load preset…", "Carregar predefinição…"),
            "fx_clear" => ("Clear", "Limpar"),
            "fx_empty" => ("No effects. Add one above, or drop a .glsl file on the window.", "Nenhum efeito. Adicione acima ou solte um arquivo .glsl na janela."),
            "fx_compiling" => ("Compiling…", "Compilando…"),
            "fx_remove" => ("Remove", "Remover"),
            "fx_reset" => ("Reset parameters", "Restaurar parâmetros"),
            "fx_preset_missing" => ("Shaders not found", "Shaders não encontrados"),
            "audio_device" => ("Audio output", "Saída de áudio"),
            "default_device" => ("System default", "Padrão do sistema"),
            "volume" => ("Volume", "Volume"),
            "mute" => ("Mute (M)", "Silenciar (M)"),
            "unmute" => ("Unmute (M)", "Ativar som (M)"),
            "gpu_hint" => ("Set BRP_GPU=<name> to choose another GPU", "Use BRP_GPU=<nome> para escolher outra GPU"),
            "deinterlace" => ("Deinterlace", "Desentrelaçar"),
            "deint_auto" => ("Auto (interlaced files: best)", "Automático (arquivos entrelaçados: melhor)"),
            "deint_off" => ("Off", "Desligado"),
            "deint_best" => ("Best: NNEDI3 + motion-adaptive", "Melhor: NNEDI3 + adaptativo ao movimento"),
            "deint_output" => ("Output", "Saída"),
            "field_rate" => ("Field rate (smooth)", "Taxa de campos (suave)"),
            "frame_rate_out" => ("Frame rate", "Taxa de quadros"),
            "field_short" => ("field", "campo"),
            "deint_hint" => ("Deinterlacing — click to change", "Desentrelaçando — clique para mudar"),
            "file" => ("File", "Arquivo"),
            "codec" => ("Codec", "Codec"),
            "resolution" => ("Resolution", "Resolução"),
            "frame_rate" => ("Frame rate", "Taxa de quadros"),
            "frames" => ("Frames", "Quadros"),
            "pixel_format" => ("Pixel format", "Formato de pixel"),
            "color" => ("Color", "Cor"),
            "field" => ("Field order", "Campos"),
            "progressive" => ("Progressive", "Progressivo"),
            "interlaced" => ("Interlaced", "Entrelaçado"),
            "unknown" => ("Unknown", "Desconhecido"),
            "alpha" => ("Alpha", "Alfa"),
            "yes" => ("Yes", "Sim"),
            "no" => ("No", "Não"),
            "hdr" => ("HDR", "HDR"),
            "audio" => ("Audio", "Áudio"),
            "no_audio" => ("None", "Nenhum"),
            "timecode" => ("Timecode", "Timecode"),
            "container" => ("Container", "Contêiner"),
            "gpu" => ("GPU", "GPU"),
            "indexing" => ("Indexing…", "Indexando…"),
            "error_open" => ("Could not open file", "Não foi possível abrir o arquivo"),
            _ => ("?", "?"),
        };
        match self {
            Lang::En => en,
            Lang::Pt => pt,
        }
    }
}
