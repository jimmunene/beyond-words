"""Speak text aloud, with fallbacks so the demo NEVER goes silent or crashes.

Order of attempts (each in a background thread, so the flicker never freezes):
  1. Cached audio      voice/cache/<hash>.mp3, made by ElevenLabs earlier (instant, offline)
  2. ElevenLabs API    natural voice; the result is saved to the cache for next time
  3. Offline voice     pyttsx3 (the computer's built-in voice; robotic but works without internet)
  4. Text only         prints a note; the message is still shown on screen and on the postcard

    pip install elevenlabs pyttsx3
    export ELEVENLABS_API_KEY=...          (Windows: set ELEVENLABS_API_KEY=...)
    python voice/precache.py               # pre-generate patient phrases before the demo
"""
import hashlib
import os
import pathlib
import threading

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
_lock = threading.Lock()          # one sentence at a time, no overlapping audio


def cache_path(text, voice_id):
    key = hashlib.sha1(f"{voice_id}|{text}".encode()).hexdigest()[:16]
    return CACHE / f"{key}.mp3"


def speak(text, voice_id, enabled=True):
    if not enabled or not text:
        return
    threading.Thread(target=_speak, args=(text, voice_id), daemon=True).start()


def _speak(text, voice_id):
    with _lock:
        path = cache_path(text, voice_id)
        try:
            if not path.exists():
                fetch_elevenlabs(text, voice_id, path)
            _play_mp3(path)
            return
        except Exception as e:
            print(f"[tts] ElevenLabs/cache unavailable ({e}); trying offline voice")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[tts] (not spoken, shown on screen only) {text!r}: {e}")


def fetch_elevenlabs(text, voice_id, path):
    if not os.environ.get("ELEVENLABS_API_KEY"):
        raise RuntimeError("ELEVENLABS_API_KEY not set")
    from elevenlabs.client import ElevenLabs      # check current SDK docs at the event
    audio = ElevenLabs(timeout=8).text_to_speech.convert(
        voice_id=voice_id, model_id="eleven_flash_v2_5", text=text, output_format="mp3_44100_128")
    data = audio if isinstance(audio, (bytes, bytearray)) else b"".join(audio)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)             # only a complete file ever lands in the cache


def _play_mp3(path):
    """Play with pygame (already installed), so no ffmpeg/mpv is needed."""
    import time
    import pygame
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(str(path))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
