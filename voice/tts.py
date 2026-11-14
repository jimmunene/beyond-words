"""Fail-closed voice output for approved messages.

No cloud TTS is used unless BOTH message approval and explicit cloud consent are
present. Offline voice works without any API key. The older speak(text, ...)
entrypoint is retained for compatibility, but now requires approved=True; old
UI calls without approval intentionally produce no speech until Part B integrates
its confirmation screen.

Speech is asynchronous, so the main pygame UI isn't blocked.
"""

import hashlib
import os
import pathlib
import subprocess
import sys
import threading

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
_lock = threading.Lock()


def cache_path(text, voice_id):
    model = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
    key = hashlib.sha256(f"{voice_id}|{model}|{text}".encode()).hexdigest()[:24]
    return CACHE / f"{key}.mp3"


def speak(text, voice_id, enabled=True, *, approved=False, allow_cloud=False):
    """Start speech ONLY after approval. Returns True if worker was started."""
    if not enabled or not approved or not isinstance(text, str) or not text.strip():
        return False
    worker = threading.Thread(
        target=_speak, args=(text, voice_id, allow_cloud), daemon=True,
        name="beyond-words-speech",
    )
    worker.start()
    return True


def speak_confirmed(message, voice_id, enabled=True):
    """Preferred Part B entrypoint; works with experience.contracts.ConfirmedMessage."""
    from experience.contracts import ConfirmedMessage
    if not isinstance(message, ConfirmedMessage):
        raise TypeError("Expected a ConfirmedMessage from the Part B confirmation handler")
    message.validate_for_output()
    return speak(
        message.text, voice_id, enabled=enabled,
        approved=True, allow_cloud=message.cloud_voice_consent,
    )


def _speak(text, voice_id, allow_cloud):
    with _lock:
        if allow_cloud and voice_id and voice_id != "YOUR_VOICE_ID":
            path = cache_path(text, voice_id)
            try:
                if not path.exists():
                    fetch_elevenlabs(text, voice_id, path)
                _play_mp3(path)
                return
            except Exception as exc:
                print(f"[tts] Cloud/cached playback unavailable: {type(exc).__name__}; falling back offline")
        try:
            _offline_speak(text)
        except Exception as exc:
            print(f"[tts] Offline sound unavailable ({type(exc).__name__}). Confirmed message remains on screen.")


def fetch_elevenlabs(text, voice_id, path):
    """Fetch intentionally cloud-consented text. Cache is LOCAL and gitignored."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    if not voice_id or voice_id == "YOUR_VOICE_ID":
        raise ValueError("Configure an actual ElevenLabs voice ID")
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=key, timeout=10)
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=os.environ.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
        output_format="mp3_44100_128",
    )
    data = audio if isinstance(audio, (bytes, bytearray)) else b"".join(audio)
    if not data:
        raise RuntimeError("ElevenLabs returned empty audio")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _offline_speak(text):
    # macOS has a built-in speech synthesizer: no heavyweight pyobjc dependency.
    if sys.platform == "darwin":
        subprocess.run(["say", text], check=True, timeout=40)
        return
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def _play_mp3(path):
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=True, timeout=40)
        return
    import time
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
