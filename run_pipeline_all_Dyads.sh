#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

# Configuración
STEP=5
SMOOTH=5
SR=16000
BACKEND="opencv"   # o retinaface si lo prefieres
DEVICE="cpu"
COMPUTE="int8"
MODEL="small"
VAD="silero"

# Asegura HF_TOKEN definido si tu whisperx lo necesita
: "${HF_TOKEN:?ERROR: export HF_TOKEN='TU_TOKEN' antes de ejecutar}"

for DYAD_DIR in "${ROOT}"/Dyad*; do
  [[ -d "${DYAD_DIR}" ]] || continue
  DYAD="$(basename "${DYAD_DIR}")"

  A_MP4="${DYAD_DIR}/${DYAD}-A_cut/${DYAD}-A_sync.mp4"
  B_MP4="${DYAD_DIR}/${DYAD}-B_cut/${DYAD}-B_sync.mp4"

  if [[ ! -f "${A_MP4}" || ! -f "${B_MP4}" ]]; then
    echo "[SKIP] ${DYAD}: no encuentro A o B mp4"
    continue
  fi

  echo "=============================="
  echo "[DYAD] ${DYAD}"
  echo "A: ${A_MP4}"
  echo "B: ${B_MP4}"

  PROBE_DIR="${DYAD_DIR}/_probe_audio"
  DIAR_DIR="${DYAD_DIR}/diarization"
  EMO_DIR="${DYAD_DIR}/emotions"

  mkdir -p "${PROBE_DIR}" "${DIAR_DIR}" "${EMO_DIR}"

  A_WAV="${PROBE_DIR}/${DYAD}-A_16k_mono.wav"
  B_WAV="${PROBE_DIR}/${DYAD}-B_16k_mono.wav"
  STEREO_WAV="${PROBE_DIR}/${DYAD}_stereo.wav"

  # 1) Extraer mono wavs (16kHz)
  ffmpeg -y -i "${A_MP4}" -vn -ac 1 -ar "${SR}" -c:a pcm_s16le "${A_WAV}" -loglevel error
  ffmpeg -y -i "${B_MP4}" -vn -ac 1 -ar "${SR}" -c:a pcm_s16le "${B_WAV}" -loglevel error

  # 2) Construir estéreo (A=Left, B=Right)
  ffmpeg -y -i "${A_WAV}" -i "${B_WAV}" \
    -filter_complex "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]" \
    -map "[a]" -ar "${SR}" -c:a pcm_s16le "${STEREO_WAV}" -loglevel error

  # 3) WhisperX -> words (ajusta input según tu script; aquí uso el estéreo)
  WORDS_CSV="${DIAR_DIR}/${DYAD}_words.csv"
  python3 transcribe_diarize_whisperx.py "${STEREO_WAV}" \
    --output "${WORDS_CSV}" \
    --device "${DEVICE}" --compute-type "${COMPUTE}" \
    --vad-method "${VAD}" --model "${MODEL}"

  # 4) Diarización v11 (salida final)
  OUT_V11="${DIAR_DIR}/${DYAD}_diarized_v11.csv"
  python3 diarizing_by_RMS_v11_roles_viterbi.py \
    --audio "${STEREO_WAV}" \
    --words "${WORDS_CSV}" \
    --out "${OUT_V11}"

  # 5) Emociones A y B (requiere video_emotion_analyzer.py con --outdir)
  python3 video_emotion_analyzer.py --videos "${A_MP4}" --step "${STEP}" --smooth "${SMOOTH}" --backend "${BACKEND}" --outdir "${EMO_DIR}"
  python3 video_emotion_analyzer.py --videos "${B_MP4}" --step "${STEP}" --smooth "${SMOOTH}" --backend "${BACKEND}" --outdir "${EMO_DIR}"

  echo "[OK] ${DYAD} -> diarization: ${OUT_V11} | emotions: ${EMO_DIR}"
done

echo "DONE."
