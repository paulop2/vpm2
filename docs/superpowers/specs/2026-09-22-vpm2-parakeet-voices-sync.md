# VPM2 — Parakeet ASR, biblioteca de vozes e correção do timeline

**Data:** 2026-09-22
**Status:** Pronto para implementação
**Baseado em:** `2026-06-24-vpm2-design.md` (spec original) e conversa de 2026-09-22

## Problem Statement

Hoje o VPM2 transcreve o áudio-fonte com **faster-whisper large-v3**. O modelo é
bom, mas para vídeo de narração com música de fundo e pausas longas (ex: Bob Ross)
ele **alucina/entra em loop no silêncio** e é ~20x mais lento que as alternativas
atuais. Além disso, o estágio de transcrição é uma casca acoplada ao faster-whisper,
sem interface plugável — trocar de ASR exige editar o pipeline.

Do lado da voz, cada vídeo **re-extrai a referência do locutor e re-sintetiza tudo
do zero**. Quando o mesmo locutor aparece em vários vídeos, isso gera:

- **voz inconsistente** entre vídeos;
- desperdício: o TTS (estágio mais caro) refaz clipes idênticos que se repetem
  entre vídeos (intro/outro/frases recorrentes);
- nenhuma forma de **melhorar o clone com o tempo** — a referência nunca é curada
  nem reaproveitada.

Por fim, o algoritmo de sincronismo (`plan_timeline`) espera segmentos com a
**duração real do clipe PT sintetizado**, mas nenhum estágio produz esse artefato
enriquecido. Ao ligar o TTS, o plano de timeline quebraria.

## Solution

Três frentes, todas atrás de seams já existentes ou espelhando padrões do repo:

1. **ASR plugável com Parakeet como padrão.** Introduzir uma interface `ASRBackend`
   espelhando o `TTSBackend` já existente, com `ParakeetBackend` (padrão) e
   `WhisperBackend` (fallback para CPU / idiomas fora dos 25 suportados / comparação).
   O estágio de transcrição vira casca fina. A lógica testável é a **normalização**
   da saída crua do backend para o shape canônico de segmentos.

2. **Biblioteca de vozes persistida + cache de TTS.** Perfis de voz salvos em disco
   (referência + metadados) que são reaproveitados entre vídeos, podendo ser
   recurados/substituídos ao longo do tempo. Clipes TTS são cacheados por
   `hash(perfil, texto, params)`, eliminando ressíntese de falas repetidas.

3. **Correção do contrato do timeline.** Deixar explícito que `plan_timeline`
   consome os segmentos enriquecidos com a duração do clipe PT (artefato de nível
   05, produzido pelo estágio de síntese), e não o transcript cru. O algoritmo em
   si permanece; o que falta é o contrato e o estágio que o produz.

## User Stories

1. Como autor do VPM2, quero que a transcrição use o Parakeet por padrão, para ter
   melhor acurácia em inglês e muito mais velocidade.
2. Como autor do VPM2, quero que o Parakeet não alucine em silêncio/música de fundo,
   para não gerar linhas fantasma em vídeos com pausas.
3. Como autor do VPM2, quero poder escolher o backend de ASR por config, para trocar
   entre Parakeet e Whisper sem editar o pipeline.
4. Como autor do VPM2, quero que o Whisper continue disponível como fallback, para
   rodar em CPU ou transcrever idiomas fora dos 25 do Parakeet.
5. Como autor do VPM2, quero que a saída de qualquer backend de ASR seja
   normalizada para o mesmo shape de segmentos, para que os estágios seguintes
   não saibam qual modelo rodou.
6. Como autor do VPM2, quero timestamps por segmento confiáveis, para alimentar o
   sincronismo da dublagem.
7. Como autor do VPM2, quero importar o modelo pesado do NeMo de forma preguiçosa
   (sob demanda), para não pagar o custo de VRAM ao importar o pacote.
8. Como autor do VPM2, quero que um erro do NeMo/GPU vire mensagem acionável, para
   não receber um stack trace cru de CUDA.
9. Como autor do VPM2, quero salvar um perfil de voz do locutor em disco, para
   reaproveitá-lo em vídeos futuros.
10. Como autor do VPM2, quero selecionar um perfil de voz por nome na config, para
    não re-extrair a referência a cada vídeo.
11. Como autor do VPM2, quero que o perfil guarde metadados (vídeo de origem, span,
    transcript, hash, data), para poder auditar de onde veio a voz.
12. Como autor do VPM2, quero poder substituir a referência de um perfil, para
    melhorar o clone ao longo do tempo sem retreinar nada.
13. Como autor do VPM2, quero manter múltiplos perfis de voz, para cobrir mais de
    um locutor no futuro.
14. Como autor do VPM2, quero que falhas de download no meio da extração não
    corrompam um perfil existente, para não perder uma referência boa.
15. Como autor do VPM2, quero cachear clipes de TTS por `hash(perfil, texto, params)`,
    para não pagar de novo pela síntese de falas repetidas entre vídeos.
16. Como autor do VPM2, quero que o cache seja invalidado automaticamente quando a
    referência ou os parâmetros mudam, para nunca servir áudio desatualizado.
17. Como autor do VPM2, quero que o cache sobreviva entre execuções, para economizar
    ao traduzir muitos vídeos do mesmo locutor.
18. Como autor do VPM2, quero que a remoção do cache seja explícita e simples, para
    conseguir forçar regeneração quando necessário.
19. Como autor do VPM2, quero que o estágio de síntese grave a duração real de cada
    clipe, para o sincronismo ter o dado de que precisa.
20. Como autor do VPM2, quero que o timeline receba um contrato claro de entrada,
    para não quebrar ao ligar o TTS.
21. Como autor do VPM2, quero que um artefato de clipes inválido (clipe ausente ou
    duração faltando) seja descartado e refeito, para a retomada ser confiável.
22. Como autor do VPM2, quero manter a numeração de artefatos (01–06), para o
    pipeline continuar inspecionável em disco.
23. Como autor do VPM2, quero que a normalização de segmentos descarte texto vazio
    e ordene por tempo, para nunca alimentar o sync com lixo.
24. Como autor do VPM2, quero que trocar ASR não invalide artefatos bons a jusante,
    para aproveitar a retomada.
25. Como mantenedor, quero testar o normalizador e a política de perfis/cache sem
    GPU, para ter CI rápido e determinístico.
26. Como mantenedor, quero que os backends de ASR e TTS sejam cascas finas não
    testadas unitariamente, para focar os testes na nossa lógica.
27. Como autor do VPM2, quero poder comparar Parakeet vs Whisper no meu próprio
    áudio, para decidir com dado real e não com benchmark de terceiros.

## Implementation Decisions

### Frente 1 — ASR plugável

- Nova interface `ASRBackend` (ABC) espelhando `TTSBackend`, com um método que
  recebe o caminho do WAV e devolve uma lista de segmentos crus.
- Factory `get_asr_backend(config)` seleciona por `config.asr_backend`. Backends
  são importados de forma preguiçosa (padrão já usado pelo Chatterbox).
- Backends concretos:
  - `ParakeetBackend` — NeMo (`ASRModel.from_pretrained`), padrão.
    Modelo padrão `nvidia/parakeet-tdt-0.6b-v2` (English-only); config permite
    trocar para `...-v3` (EN + 25 europeus, inclui PT) para QA do áudio gerado.
    Transcrição com timestamps por segmento; suporte a áudio longo (até ~24 min
    num passe) — sem necessidade de chunking manual na v1.
  - `WhisperBackend` — código atual do faster-whisper movido do estágio para o
    backend, preservando `device`/`compute_type`/`vad_filter` atuais.
- **Normalizador puro** (a lógica testável): os backends devolvem um tipo
  intermediário canônico `RawSegment(start, end, text)`; uma função pura
  `normalize_segments(raw)` atribui `id` sequencial, **descarta texto vazio**,
  ordena por `start` e devolve o shape canônico
  `{id, start, end, text}`. Nenhum backend monta JSON sozinho.
- `Config` ganha separação de backend e modelo:
  - `asr_backend: str = "parakeet"`
  - `parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"`
  - `whisper_model: str = "large-v3"`
  - `asr_model` deixa de existir (ou vira alias depreciado).
- `TranscribeStage` passa a: resolver backend → chamar → normalizar → gravar
  `03_transcript.json`. `is_done` continua validando via `valid_transcript`.
- Erros de NeMo/CUDA/VRAM devem ser capturados e reemitidos com mensagem
  acionável (ex: sugestão de usar `asr_backend=whisper`), não stack trace cru.

### Frente 2 — Biblioteca de vozes + cache de TTS

- **Perfil de voz persistido** em `voices/<speaker_id>/` contendo:
  - `ref.wav` — clipe de referência (idealmente 6–10s, só voz);
  - `profile.json` — `{id, source_video, span:[start,end], transcript, sha256,
    created_at, backend, notes}`.
- Módulo puro de perfis (prior art: `voice_sample.py`):
  - `profile_id(...)` — id determinístico/legível;
  - `make_profile(...)` e `resolve_reference(profile)` → caminho do `ref.wav`;
  - `cache_key(profile_sha, text, params)` → sha256 estável.
- `Config.voice_mode` passa a aceitar `"cloning" | "preset" | "profile"`:
  - `cloning` mantém o comportamento atual (extrai referência do vídeo);
  - `profile` usa um perfil existente (campo `voice_profile: str | None`);
  - `preset` segue como gancho para uma referência PT curada fixa.
- Quando `voice_mode == "cloning"` e o autor pedir para salvar, a referência
  extraída (via `pick_reference_window` + VAD) é **promovida a um perfil**.
- **Curadoria/“melhorar o clone”**: substituir `ref.wav` de um perfil é a forma
  de melhorar (não há treino na v1). Recomenda-se manter candidatos e um campo
  de qualidade; fine-tune real fica reservado ao XTTS (fora de escopo).
- **Cache de TTS**: `cache/tts/<sha256>.wav`, chave = `hash(perfil_sha, texto,
  backend, sample_rate, params)`. O estágio de síntese consulta antes de gerar;
  hit → reutiliza o wav. Invalidação é natural (mudou referência/params → chave
  nova). Limpeza do cache é comando explícito.
- **Estágio de síntese (05)** novo: para cada segmento traduzido, resolve
  backend + referência (perfil/cache), gera/recupera o clipe em `05_clips/`,
  mede a duração real e grava `05_clips.json` com
  `{id, start, end, text, text_pt, clip, duration}`. `duration` é a duração do
  **clipe PT**, não do segmento original.

### Frente 3 — Contrato do timeline

- **Esclarecimento (corrige a análise anterior):** o campo `duration` que
  `plan_timeline` consome é a duração do **clipe PT sintetizado**, portanto pertence
  ao artefato de nível 05 — **não** deve ser derivado de `end - start` no transcript
  (isso seria a duração da fala original, dado errado para o sync). O algoritmo
  `plan_timeline` **não muda**.
- A correção é: (a) formalizar o shape de entrada do timeline como o artefato
  enriquecido de 05; (b) o estágio de síntese passa a produzi-lo.
- Estender `valid_clips` para exigir `duration` (além de `clip` existente), para a
  retomada descartar artefatos parciais.
- Adicionar teste de regressão que garante que o artefato de 05 satisfaz o contrato
  que `plan_timeline` espera.

### Pipeline

- `STAGES` passa a incluir `SynthesizeStage` (05) após `TranslateStage`.
- `AssembleStage` (06, mux final com ffmpeg) fica **fora do escopo** desta spec,
  mas o contrato de entrada (`05_clips.json`) é definido aqui para destravá-lo.

## Testing Decisions

**O que faz um bom teste aqui:** testar comportamento externo e lógica pura — nunca
detalhes internos de NeMo/Chatterbox nem chamadas de GPU. Modelos pesados são cascas
finas; a nossa lógica (normalização, perfis, cache, sync) é o alvo.

- **Seam principal (novo):** `ASRBackend` + `normalize_segments`. Testa-se o
  normalizador com saída crua fictícia de backend (fora de ordem, texto vazio,
  duplicados, pontuação), sem tocar no NeMo.
- **Seam de perfis/cache:** módulo de `VoiceProfile` testado puramente — id,
  resolução de referência, estabilidade da `cache_key`, e invalidação quando
  referência/params mudam.
- **Cache, comportamento externo:** o estágio de síntese com um **TTS backend
  fake** (mesmo padrão do `FakeStage` em `test_pipeline.py`) deve (a) gerar uma vez
  e (b) na segunda vez reaproveitar do cache sem chamar o backend de novo. Usa
  `tmp_path`.
- **Timeline:** manter os testes puros existentes (`test_timeline.py`) e adicionar
  um teste de contrato sobre o shape de `05_clips.json`.
- **Prior art:** `tests/test_timeline.py` (função pura), `tests/test_voice_sample.py`
  (função pura), `tests/test_pipeline.py` (stage fake + `tmp_path`),
  `tests/test_artifacts.py` (validação de JSON).
- **Não testado unitariamente:** `ParakeetBackend`, `WhisperBackend`,
  `ChatterboxBackend` — cascas sobre ferramentas externas, conforme já definido na
  spec original.
- **Smoke ponta-a-ponta** (manual, GPU + rede) continua documentado no README.

## Out of Scope

- `AssembleStage` (06) e o mux final com ffmpeg.
- Fine-tune de TTS no locutor (caminho XTTS-v2).
- XTTS-v2 como backend implementado (segue como gancho).
- Diarização multi-locutor / mais de um locutor por vídeo.
- Ensemble de ASR + LLM juiz.
- Sincronismo agressivo (esticar vídeo).
- UI web.
- Migração das deps para NeMo **não** inclui suporte CPU do Parakeet na v1.

## Further Notes

- **Escolha do modelo Parakeet:** como a fonte é inglês, o padrão é o
  `parakeet-tdt-0.6b-v2` (English-only, WER menor). O `v3` cobre 25 idiomas
  europeus (inclui PT) e serve se quisermos transcrever o PT gerado para QA.
  Licença **CC-BY-4.0** (uso comercial OK).
- **Benchmarks:** a referência é o **Hugging Face Open ASR Leaderboard**
  (`hf-audio/open-asr-leaderboard`), com trilhas English / long-form / multilingual,
  reportando **WER + RTFx**. Aviso: os números mudam com rebaseline (houve um em
  2026) e nenhum Parakeet está no topo de acurácia hoje — o topo são híbridos
  Conformer + decoder LLM (Granite Speech 4.1 2B, Cohere Transcribe, Canary-Qwen),
  porém muito mais lentos. Também: qualquer modelo published piora ~5x em
  dialeto/acento (CORAAL). Por isso a user story 27 — medir **no nosso áudio**.
- **Risco de setup:** Parakeet roda via `nemo_toolkit`, que é bem mais chato de
  instalar no WSL + cu128/Blackwell (sm_120) que um `pip install` de faster-whisper.
  Precisa de uma tarefa de verificação de instalação/VRAM antes de virar padrão.
  O Whisper como fallback mitiga o risco.
- **Ganho-chave do Parakeet para este caso:** o decoder transducer emite "blank",
  então o modelo fica quieto no silêncio onde o Whisper alucina — exatamente o
  cenário de narração com música/pausas.
- **Deps:** adicionar `nemo_toolkit[asr]`; manter `faster-whisper`. Não remover
  `chatterbox-tts`.
