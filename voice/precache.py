"""Pre-generate ElevenLabs audio for every patient-mode phrase (and any extra lines),
so they play instantly and still work if the Wi-Fi or API fails during the demo.

    python voice/precache.py
    python voice/precache.py "Hello from BrainType" "Hi Mom"     # extra lines
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402
from speller.state_machine import SCREENS  # noqa: E402
from voice.tts import cache_path, fetch_elevenlabs  # noqa: E402

voice_id = load_config()["elevenlabs_voice_id"]
phrases = [a[1] for s in SCREENS.values() for item in s if item for _, a in [item] if a[0] == "phrase"]
phrases += sys.argv[1:]

for text in phrases:
    path = cache_path(text, voice_id)
    if path.exists():
        print("cached  ", text)
        continue
    try:
        fetch_elevenlabs(text, voice_id, path)
        print("saved   ", text)
    except Exception as e:
        print("FAILED  ", text, "->", e)
