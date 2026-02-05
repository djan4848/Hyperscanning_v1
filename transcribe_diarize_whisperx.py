#!/usr/bin/env python3
"""
transcribe_diarize_whisperx.py

Run WhisperX transcription + (optional) word-level alignment + (optional) speaker diarization (pyannote)
and export a word-level CSV with speaker labels.

Example:
  python transcribe_diarize_whisperx.py dyad_stereo.wav \
    --output Dyad01/diarization/Dyad01_words.csv \
    --device cpu --compute-type int8 --vad-method silero --model small

Notes:
- If diarization is enabled, you must provide a Hugging Face token via HF_TOKEN env var or --hf-token.
- Default diarization model: pyannote/speaker-diarization-3.1 (gated, accept terms on HF).
"""

"""
transcribe_diarize_whisperx.py

Run WhisperX transcription + (optional) word-level alignment + (optional) speaker diarization (pyannote)
and export a word-level CSV with speaker labels.
"""

# ======================================================================
# Silence non-actionable warnings for long batch runs (CPU pipeline)
# ======================================================================
import os
import warnings

# TorchAudio deprecations (TorchCodec transition)
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio\\._backend\\..*has been deprecated.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=".*In 2\\.9, this function's implementation will be changed to use torchaudio\\.load_with_torchcodec.*",
    category=UserWarning
)

# NVML warnings (CPU-only systems)
warnings.filterwarnings(
    "ignore",
    message=".*Can't initialize NVML.*",
    category=UserWarning
)

# TensorFlow INFO logs (oneDNN / CPU features)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# ======================================================================
# Normal imports start here
# ======================================================================
import argparse
import sys
from pathlib import Path
import csv

import os
import sys
from pathlib import Path
import csv

import torch
import whisperx
from pyannote.audio import Pipeline

from contextlib import contextmanager


@contextmanager
def torch_load_weights_only_false_temporarily():
    """
    PyTorch >= 2.6 changed torch.load default weights_only=True which can break loading
    of Lightning/pyannote checkpoints that include non-weight objects (e.g., Specifications).
    We force weights_only=False *during* diarization model loading (trusted HF checkpoint).

    This avoids errors like:
      _pickle.UnpicklingError: Weights only load failed ... Unsupported global ...
    """
    _orig_load = torch.load

    def _load(*args, **kwargs):
        # IMPORTANT: override unconditionally (setdefault is not enough)
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = _orig_load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Path to WAV/FLAC/MP3 audio file (ideally 16kHz+).")
    p.add_argument("--output", help="Output CSV path. Default: <audio>_word_level.csv")
    p.add_argument("--model", default="small", help="Whisper model name (tiny/base/small/medium/large-v3...).")
    p.add_argument("--language", default="en", help="Language code for Whisper/align (e.g., en, es, fr).")
    p.add_argument("--device", default=None, help="cpu or cuda. Default: auto.")
    p.add_argument("--compute-type", default=None, help="CTranslate2 compute type (int8, int8_float16, float16...).")
    p.add_argument("--vad-method", default="silero", choices=["silero", "pyannote"], help="VAD backend for WhisperX.")
    p.add_argument("--no-diarization", action="store_true", help="Disable speaker diarization.")
    p.add_argument("--diarization-model", default="pyannote/speaker-diarization-3.1",
                   help="HF diarization pipeline id (gated).")
    p.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var).")
    p.add_argument("--num-speakers", type=int, default=None, help="Optional fixed number of speakers.")
    return p.parse_args()


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main():
    args = parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        sys.exit(2)

    out_csv = Path(args.output).expanduser().resolve() if args.output else audio_path.with_name(
        audio_path.stem + "_word_level.csv"
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # CPU-safe default for ctranslate2: avoid float16 on CPU
    if device == "cpu" and not args.compute_type:
        args.compute_type = "int8"

    asr_kwargs = {}
    if args.compute_type:
        asr_kwargs["compute_type"] = args.compute_type

    # 1) ASR
    model = whisperx.load_model(
        args.model,
        device,
        language=args.language,
        vad_method=args.vad_method,
        **asr_kwargs
    )
    result = model.transcribe(str(audio_path))

    # 2) Word-level alignment
    align_model, metadata = whisperx.load_align_model(language_code=args.language, device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, str(audio_path), device)

    # 3) Diarization via pyannote (optional)
    diar = None
    if not args.no_diarization:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: diarization requires HF_TOKEN env var or --hf-token", file=sys.stderr)
            sys.exit(2)

        # Force torch.load(weights_only=False) only while loading pyannote checkpoint
        with torch_load_weights_only_false_temporarily():
            diar_pipe = Pipeline.from_pretrained(args.diarization_model, use_auth_token=hf_token)

        if diar_pipe is None:
            raise RuntimeError(
                f"Failed to load diarization pipeline '{args.diarization_model}'. "
                "Likely gated/private. Accept model terms on Hugging Face and ensure your token has access."
            )
        diar = diar_pipe(str(audio_path), num_speakers=args.num_speakers)

    # Helper: map pyannote speaker labels to A/B in order of appearance
    speaker_map = {}
    next_letter = 0

    def map_speaker(spk):
        nonlocal next_letter
        if spk is None:
            return "unknown"
        if spk not in speaker_map:
            letter = chr(ord("A") + next_letter) if next_letter < 26 else f"S{next_letter}"
            speaker_map[spk] = letter
            next_letter += 1
        return speaker_map[spk]

    # Function: assign a speaker to an interval by maximum overlap with diarization turns
    def speaker_for_interval(t0, t1):
        if diar is None:
            return None
        best_spk = None
        best_ov = 0.0
        for turn, _, spk in diar.itertracks(yield_label=True):
            ov = overlap(t0, t1, float(turn.start), float(turn.end))
            if ov > best_ov:
                best_ov = ov
                best_spk = spk
        return best_spk

    # 4) Build word-level CSV rows
    rows = []
    for seg in aligned["segments"]:
        words = seg.get("words", [])
        for w in words:
            w0 = w.get("start", None)
            w1 = w.get("end", None)
            word = w.get("word", "").strip()
            if w0 is None or w1 is None or not word:
                continue
            spk = speaker_for_interval(float(w0), float(w1))
            rows.append({
                "speaker_raw": spk if spk is not None else "",
                "speaker": map_speaker(spk),
                "start_time": float(w0),
                "end_time": float(w1),
                "word": word,
                "confidence": w.get("score", "")
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[ "speaker_raw", "speaker", "start_time","end_time","word", "confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"OK: wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

