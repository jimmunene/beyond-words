"""Gemini helpers that save a person who cannot speak many looks at once.

  answers(question)  A helper asked something out loud and typed it in. Gemini proposes the three
                     most useful short answers, which become one-look boxes.
  expand(words)      The person spelled a few key words ("COLD FEET BLANKET"). Gemini offers one
                     natural sentence with the same meaning; the person chooses whether to say it.

Both run in a background thread (the flicker never waits) and both are optional: without a key,
without internet or over quota, the speller simply carries on with its built-in pages.
Gemini only ever proposes. Nothing it writes is spoken until the person picks it.

    python speller/assistant.py "What would you like for dinner?"     # needs GEMINI_API_KEY
    python speller/assistant.py --expand "COLD FEET BLANKET"
"""
import os
import pathlib
import re
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import BLOCKED, DEFAULT_MODEL  # noqa: E402

MAX_SPOKEN = 120            # longest sentence we will offer to speak


class Assistant:
    def __init__(self, enabled=True):
        self.has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.enabled = bool(enabled) and self.has_key
        self.results = {}           # (kind, text) -> answer, or False when Gemini could not help
        self.pending = set()
        self.last_error = ""
        self._client = None

    # ------------------------------------------------------------ public
    def answers(self, question):
        self._start("answers", question.strip())

    def expand(self, words):
        self._start("expand", words.strip())

    def get(self, kind, text):
        """The result: a list / sentence, False if Gemini could not help, None while still waiting."""
        return self.results.get((kind, text.strip()))

    # ------------------------------------------------------------ internals
    def _start(self, kind, text):
        key = (kind, text)
        if not text or key in self.results or key in self.pending:
            return
        if not self.enabled:
            self.results[key] = False
            return
        self.pending.add(key)
        threading.Thread(target=self._run, args=key, daemon=True).start()

    def _run(self, kind, text):
        try:
            raw = self.generate(ANSWER_PROMPT.format(q=text) if kind == "answers" else EXPAND_PROMPT.format(w=text))
            self.results[(kind, text)] = (parse_answers(raw) if kind == "answers" else parse_sentence(raw, text)) or False
        except Exception as e:      # no key, no Wi-Fi, quota: the speller falls back to its own pages
            self.last_error = str(e)[:160]
            print("[assistant] Gemini unavailable:", self.last_error)
            self.results[(kind, text)] = False
        finally:
            self.pending.discard((kind, text))

    def generate(self, prompt):
        from google import genai   # pip install google-genai
        from google.genai import types
        if self._client is None:
            self._client = genai.Client(http_options=types.HttpOptions(timeout=10000))   # ms; 10 s is the minimum
        resp = self._client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL), contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3))
        return resp.text or ""


ANSWER_PROMPT = """A person who cannot speak or move was asked this by their caregiver:
"{q}"
They answer by looking at one of three boxes on a screen. Write the 3 most useful, clearly different
answers they may want to give. Each answer is 1 to 3 plain words. If it is a yes/no question, use
YES, NO and one more useful answer (for example LATER, A LITTLE, NOT SURE).
Reply with exactly 3 lines, one answer per line, no numbering, no punctuation."""

EXPAND_PROMPT = """A person who cannot speak spelled these words on a slow eye-controlled speller, to be
spoken aloud to their caregiver:
"{w}"
Rewrite them as ONE short, natural, polite spoken sentence (two short sentences at most).
Keep exactly their meaning. Do not add any request, fact or feeling they did not type.
Reply with the sentence only."""


def parse_answers(raw):
    """Up to 3 clean, short, different labels (CAPITALS), or [] if the reply is unusable."""
    out = []
    for line in raw.splitlines():
        words = re.findall(r"[A-Z]+", line.upper().replace("'", ""))
        label = " ".join(words[:3])
        if (label and len(label) <= 18 and label not in out and not any(w in BLOCKED for w in words)
                and "EMERGENCY" not in words):               # that word belongs to the real alarm box
            out.append(label)
    return out[:3] if len(out) >= 2 else []


def parse_sentence(raw, typed):
    """One speakable sentence, or '' if the reply is unusable or adds nothing."""
    s = " ".join(raw.replace('"', "").split())
    plain = re.sub(r"[^A-Z ]", "", s.upper())
    if not s or len(s) > MAX_SPOKEN or any(w in BLOCKED for w in plain.split()):
        return ""
    if plain.strip() == " ".join(typed.upper().split()):
        return ""                                            # same words: nothing to offer
    return s


if __name__ == "__main__":
    import time
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = "expand" if "--expand" in sys.argv else "answers"
    text = " ".join(args) or ("COLD FEET BLANKET" if kind == "expand" else "What would you like for dinner?")
    a = Assistant()
    if not a.enabled:
        sys.exit("GEMINI_API_KEY is not set in this window")
    t0 = time.time()
    (a.expand if kind == "expand" else a.answers)(text)
    while a.get(kind, text) is None and time.time() - t0 < 15:
        time.sleep(0.05)
    print(f"{text!r} -> {a.get(kind, text)}   ({time.time() - t0:.1f} s)")
