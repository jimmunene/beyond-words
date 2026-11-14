"""Word prediction for Box 4: instant local corpus model + optional Gemini, merged.

Local model always answers (offline, milliseconds). Gemini runs in a background thread;
its answer is used only if it arrives in time AND matches the letters already typed.
"""
import json
import os
import pathlib
import threading
from collections import defaultdict

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
FALLBACK = ["HELLO", "HI", "I", "THE", "YOU", "LOVE", "THANK", "WANT", "WATER", "HELP", "MOM", "PLEASE"]

AAC_EXAMPLES = """I want some water please.
Can you help me sit up?
I am in pain.
Thank you so much.
I love you.
Please call my daughter."""


class Predictor:
    def __init__(self, use_llm=False):
        self.uni, self.next_words = {}, defaultdict(list)
        path = DATA / "corpus.json"
        if path.exists():
            data = json.load(open(path))
            self.uni = data["unigram"]
            for pair, p in data["bigram"].items():
                a, b = pair.split(" ")
                self.next_words[a].append((p, b))
            for a in self.next_words:
                self.next_words[a].sort(reverse=True)
        else:
            self.uni = {w: 1.0 / (i + 1) for i, w in enumerate(FALLBACK)}
            print("[predictor] data/corpus.json not found, using tiny fallback list")
        self.ranked = sorted(self.uni, key=self.uni.get, reverse=True)
        self.llm_cache, self.llm_pending = {}, set()
        self.use_llm = use_llm and bool(os.environ.get("GEMINI_API_KEY"))

    @staticmethod
    def split(typed):
        """'I WANT WA' -> (prev='WANT', prefix='WA')."""
        parts = typed.split(" ")
        prefix = parts[-1]
        done = [p for p in parts[:-1] if p]
        return (done[-1] if done else "<S>"), prefix

    def local(self, typed):
        prev, prefix = self.split(typed)
        for _, w in self.next_words.get(prev, []):
            if w.startswith(prefix) and w != prefix:
                return w
        for w in self.ranked:
            if w.startswith(prefix) and w != prefix and len(w) > 1:
                return w
        return None

    def suggest(self, typed):
        typed = typed.upper()
        _, prefix = self.split(typed)
        llm = self.llm_cache.get(typed)
        if llm and llm.startswith(prefix) and llm != prefix:
            return llm
        if self.use_llm and typed not in self.llm_cache and typed not in self.llm_pending:
            self.llm_pending.add(typed)
            threading.Thread(target=self._ask_llm, args=(typed,), daemon=True).start()
        return self.local(typed)

    def _ask_llm(self, typed):
        try:
            from google import genai   # pip install google-genai ; check current SDK docs
            client = genai.Client()
            _, prefix = self.split(typed)
            prompt = (
                "You predict the next word for a person typing with a brain-computer interface "
                "(an AAC user). They write short, practical, everyday messages like:\n"
                f"{AAC_EXAMPLES}\n\nTyped so far: '{typed}'.\n"
                f"Reply with ONE word that starts with '{prefix}' (completing the current word), "
                "uppercase letters only, nothing else."
            )
            resp = client.models.generate_content(model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                                                  contents=prompt)
            word = "".join(c for c in resp.text.strip().upper() if c.isalpha())
            if word:
                self.llm_cache[typed] = word
        except Exception as e:  # never let the LLM break the speller
            print("[predictor] LLM error:", e)
        finally:
            self.llm_pending.discard(typed)


if __name__ == "__main__":
    p = Predictor()
    for t in ["", "H", "I ", "I W", "I WANT ", "I WANT WA", "THANK "]:
        print(repr(t), "->", p.suggest(t))
