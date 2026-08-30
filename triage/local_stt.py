"""Local speech-to-text fallback -- Vosk, run entirely on this machine.

Mirrors triage/ollama.py's own posture: the browser's Web Speech API
(web/voice.js) is the first tier -- free, live, zero setup, but it ships your
patient's audio to Google's (or Apple's) servers to transcribe. This is the
fallback tier for when that isn't available or isn't acceptable: no network
call at all, at the cost of running the model yourself.

If the `vosk` package isn't installed, or the model directory isn't present,
every function here degrades to "not available" rather than raising --
exactly like `_ollama.available()` -- so a browser that never needed the
fallback (Web Speech API worked) never has to care whether this is set up.
"""

from __future__ import annotations

import io
import json
import os
import threading
import wave

MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH",
    str((__import__("pathlib").Path(__file__).parent.parent
         / "models" / "vosk-model-small-en-us-0.15")),
)

_model = None
_load_error: str | None = None
_lock = threading.Lock()


def _get_model():
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return _model
    with _lock:
        if _model is not None or _load_error is not None:
            return _model
        try:
            import vosk
        except ImportError:
            _load_error = "the vosk package isn't installed (pip install vosk)"
            return None
        if not os.path.isdir(MODEL_PATH):
            _load_error = f"no model at {MODEL_PATH} -- download one from alphacephei.com/vosk/models"
            return None
        try:
            vosk.SetLogLevel(-1)
            _model = vosk.Model(MODEL_PATH)
        except Exception as exc:
            _load_error = f"failed to load the model: {exc}"
    return _model


def available() -> tuple[bool, str]:
    """(True, 'ready') once the model is loaded; (False, why) otherwise --
    checked at /api/config time so the UI can grey out the fallback rather
    than let a patient click a mic button that quietly does nothing."""
    model = _get_model()
    return (True, "ready") if model is not None else (False, _load_error or "not loaded")


def transcribe_wav(wav_bytes: bytes) -> str | None:
    """wav_bytes must be mono, 16-bit PCM WAV -- exactly what web/voice.js's
    fallback recorder encodes client-side (no ffmpeg, no server-side decode).
    Returns None if the model isn't available or the audio isn't in that
    format; never raises for the caller to catch."""
    import vosk

    model = _get_model()
    if model is None:
        return None
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                return None
            rec = vosk.KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(False)
            parts = []
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                if rec.AcceptWaveform(data):
                    parts.append(json.loads(rec.Result()).get("text", ""))
            parts.append(json.loads(rec.FinalResult()).get("text", ""))
    except (wave.Error, EOFError):
        return None
    return " ".join(p for p in parts if p).strip()
