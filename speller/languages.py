"""Speak in the listener's language.

The person picks boxes labelled in English; what is said aloud can be in another language, so a
patient can talk to a nurse or a grandparent who speaks Spanish, Hindi, French, ...
ElevenLabs' multilingual voice model speaks whatever language the text is written in, so all this
module has to do is hand it the sentence in the right language:

  built-in phrases   come from speller/translations.json (made once with Gemini, works offline)
  anything else      (spelled sentences, suggested answers, personal phrases) is translated live
                     by Gemini; without Gemini it is spoken in English

    python speller/languages.py --show es            # print the built-in phrases in Spanish
    python speller/languages.py --build              # (re)make translations.json with Gemini
    python speller/languages.py --precache es hi     # pre-generate ElevenLabs audio for these languages

The translations are machine-made. Have a native speaker check them before real use with a patient.
"""
import json
import os
import pathlib
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.state_machine import SCREENS  # noqa: E402

# Languages the ElevenLabs voice can speak (Flash v2.5, or Eleven v3 for V3_LANGS). First = no translation.
LANGUAGES = [("en", "English"), ("es", "Spanish"), ("hi", "Hindi"), ("fr", "French"), ("de", "German"),
             ("pt", "Portuguese"), ("ar", "Arabic"), ("zh", "Chinese"), ("ja", "Japanese"), ("ta", "Tamil"),
             ("sw", "Swahili")]
NAMES = dict(LANGUAGES)
# Languages the fast Flash model cannot speak. For these the speller fetches the audio itself with
# the Eleven v3 model and puts it in the voice cache, where voice/tts.py finds and plays it.
V3_LANGS = {"sw"}
V3_MODEL = "eleven_v3"
TABLE = pathlib.Path(__file__).resolve().parent / "translations.json"


def prefetch(text, lang, voice_id):
    """For a V3_LANGS language: make sure the audio for `text` is in the voice cache.
    Returns True if it is there afterwards. Blocks for a few seconds, so call it off the UI thread."""
    if lang not in V3_LANGS:
        return True
    import urllib.request
    from voice.tts import cache_path
    path = pathlib.Path(cache_path(text, voice_id))
    if path.exists():
        return True
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key or not voice_id or voice_id == "YOUR_VOICE_ID":
        return False
    try:
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128",
            data=json.dumps({"text": text, "model_id": V3_MODEL, "language_code": lang}).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        if not data:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)                                   # only a complete file lands in the cache
        return True
    except Exception as e:
        print(f"[languages] no {NAMES[lang]} audio:", str(e)[:120])
        return False


def builtin_phrases():
    """Every sentence the speller can say by itself, in a stable order."""
    seen = []
    for screen in SCREENS.values():
        for item in screen:
            if item and item[1][0] == "phrase" and item[1][1] not in seen:
                seen.append(item[1][1])
    return seen


class Translator:
    def __init__(self, assistant=None):
        self.assistant = assistant              # speller.assistant.Assistant, for sentences not in the table
        self.memory = {}                        # (lang, english) -> translation found during this run
        try:
            self.table = json.loads(TABLE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.table = {}

    def ready(self, lang):
        """How this language is covered: 'table', 'gemini' (live only) or 'none'."""
        if lang == "en" or self.table.get(lang):
            return "table"
        return "gemini" if getattr(self.assistant, "enabled", False) else "none"

    def lookup(self, text, lang):
        if lang == "en":
            return text
        return self.table.get(lang, {}).get(text) or self.memory.get((lang, text))

    def translate(self, text, lang, done):
        """Call done(sentence in `lang`). Falls back to the English text rather than staying silent.
        Known sentences answer at once; new ones ask Gemini in a background thread."""
        known = self.lookup(text, lang)
        if known or not getattr(self.assistant, "enabled", False):
            done(known or text)
            return
        threading.Thread(target=self._live, args=(text, lang, done), daemon=True).start()

    def _live(self, text, lang, done):
        try:
            out = " ".join(self.assistant.generate(LIVE_PROMPT.format(lang=NAMES[lang], text=text)).split())
            out = out.strip('"')
            if out and len(out) <= 3 * len(text) + 40:
                self.memory[(lang, text)] = out
                done(out)
                return
        except Exception as e:
            print("[languages] could not translate, speaking English:", str(e)[:120])
        done(text)


LIVE_PROMPT = """Translate this sentence into {lang}. It will be spoken aloud by a text-to-speech voice on
behalf of a person who cannot speak, to their caregiver or family. Use natural, polite, everyday
{lang} in its own script. Reply with the translation only.

{text}"""

BUILD_PROMPT = """Translate each of these English sentences into {lang}. They are spoken aloud by a
text-to-speech voice on behalf of a person who cannot speak, to their caregiver or family.
Use natural, polite, everyday {lang} in its own script. Keep each one short.
Reply with ONLY a JSON array of {n} strings, in the same order, nothing else.

{items}"""


def build(langs):
    """Ask Gemini once per language for the whole list and save speller/translations.json."""
    from speller.assistant import Assistant
    a = Assistant()
    if not a.enabled:
        sys.exit("GEMINI_API_KEY is not set in this window")
    phrases = builtin_phrases()
    try:
        table = json.loads(TABLE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        table = {}
    for lang in langs:
        raw = a.generate(BUILD_PROMPT.format(lang=NAMES[lang], n=len(phrases), items=json.dumps(phrases, indent=0)))
        raw = raw[raw.find("["):raw.rfind("]") + 1]
        try:
            out = json.loads(raw)
        except ValueError:
            out = None
        if not isinstance(out, list) or len(out) != len(phrases) or not all(isinstance(x, str) and x.strip() for x in out):
            print(f"  {NAMES[lang]}: unusable reply, skipped")
            continue
        table[lang] = dict(zip(phrases, (x.strip() for x in out)))
        print(f"  {NAMES[lang]}: {len(out)} phrases")
        TABLE.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
        time.sleep(5)                                       # stay under the free requests-per-minute limit


def precache(langs):
    """Pre-generate ElevenLabs audio for the built-in phrases, so they play at once and offline."""
    from common import load_config
    from voice.tts import cache_path, fetch_elevenlabs
    voice_id = load_config()["elevenlabs_voice_id"]
    tr = Translator()
    for lang in langs:
        for english in builtin_phrases():
            text = tr.lookup(english, lang)
            if not text:
                print("no translation", lang, english)
                continue
            path = cache_path(text, voice_id)
            if path.exists():
                continue
            try:
                if lang in V3_LANGS:
                    if not prefetch(text, lang, voice_id):
                        raise RuntimeError("Eleven v3 request failed")
                else:
                    fetch_elevenlabs(text, voice_id, path)
                print("saved  ", lang, english)
            except Exception as e:
                print("FAILED ", lang, english, "->", str(e)[:100])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    codes = [c for c in args if c in NAMES and c != "en"] or [c for c, _ in LANGUAGES[1:]]
    if "--build" in sys.argv:
        build(codes)
    elif "--precache" in sys.argv:
        precache(codes)
    else:
        tr = Translator()
        for code in (codes if args else ["es"]):
            print(f"--- {NAMES[code]} ({tr.ready(code)})")
            for english in builtin_phrases():
                line = f"  {english}  ->  {tr.lookup(english, code) or '(not translated)'}"
                print(line.encode(sys.stdout.encoding or "utf-8", "replace").decode(sys.stdout.encoding or "utf-8"))
