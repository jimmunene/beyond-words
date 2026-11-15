"""Word prediction for Box 4: instant local corpus model + optional Gemini, merged.

Local model always answers (offline, milliseconds). Gemini runs in a background thread;
its answer is used only if it arrives quickly AND matches the letters already typed, so the
word in the box never changes while somebody is already looking at it.

    Predictor.suggest(typed) -> WORD or None

    python speller/predictor.py           # sample suggestions from the local model
    python speller/predictor.py --llm     # also ask Gemini (needs GEMINI_API_KEY) and show timing
"""
import json
import os
import pathlib
import re
import threading
import time
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SEED = HERE / "seed_sentences.txt"
FALLBACK = ["HELLO", "HI", "I", "THE", "YOU", "LOVE", "THANK", "WANT", "WATER", "HELP", "MOM", "PLEASE"]
DEFAULT_MODEL = "gemini-3.5-flash-lite"      # override with the GEMINI_MODEL environment variable

BIGRAM_WEIGHT = 0.75        # how much "what usually follows the previous word" counts vs. plain frequency
TRIGRAM_WEIGHT = 0.5        # extra weight for "what follows the previous TWO words" (from AAC-style sentences)
EVENT_BOOST = 8.0           # postcard mode: words from data/custom_words.txt (HELLO, EMU, ...) count this much more
EXACT_RATIO = 0.3           # offer "WORD + space" when the typed word is at least this likely vs. the best longer word

# Web word lists contain junk; the speller is on a public screen, so keep suggestions clean.
BLOCKED = set("""FUCK FUCKING FUCKED FUCKER SHIT SHITTY BITCH BITCHES ASS ASSHOLE DICK COCK PUSSY CUNT SLUT WHORE
PORN PORNO XXX SEX SEXY SEXUAL NUDE BOOBS TITS RAPE RAPED NAZI DAMN CRAP PISS BASTARD
FAG FAGGOT NIGGER NIGGA RETARD RETARDED VIAGRA CIALIS HENTAI ESCORT ESCORTS EROTIC FETISH
WWW HTTP HTTPS HTML COM ORG NET EDU GOV PDF FAQ URL RSS PHP USB CSS XML ISBN EBAY""".split())
# Words a patient may genuinely need for their care, but that must never be offered on a public
# postcard: only suggested when Predictor.clinical is on (the Speller turns it on in patient mode).
CLINICAL = set("PENIS VAGINA ANAL RECTAL NAKED GENITALS TESTICLES KILL SUICIDE SUICIDAL DIE DYING DEATH".split())
TWO_LETTER = set("AM AN AS AT BE BY DO GO HE HI IF IN IS IT ME MY NO OF OH OK ON OR SO TO UP US WE IM TV IV".split())

AAC_EXAMPLES = """I want some water please.
Can you help me sit up?
I am in pain.
Thank you so much.
I love you.
Please call my daughter.
Hi mom, I love you.
Hello from Hack The Nest."""


def valid_word(w):
    """Real, clean, letters-only words (the speller has no digits or punctuation)."""
    if not w.isalpha() or not w.isascii() or w in BLOCKED:
        return False
    if len(w) == 1:
        return w in ("A", "I")
    return len(w) > 2 or w in TWO_LETTER


def tokenize(line):
    return re.findall(r"[A-Z]+", line.upper().replace("'", ""))


def count_sentences(lines):
    """Word, word-pair and word-triple counts from sentences (one per line). '<S>' = sentence start."""
    uni, bi, tri = Counter(), Counter(), Counter()
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        w = tokenize(line)
        if not w:
            continue
        uni.update(w)
        bi[f"<S> {w[0]}"] += 1
        bi.update(f"{a} {b}" for a, b in zip(w, w[1:]))
        w = ["<S>"] + w
        tri.update(f"{a} {b} {c}" for a, b, c in zip(w, w[1:], w[2:]))
    return uni, bi, tri


def read_lines(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read().splitlines()


class Predictor:
    def __init__(self, use_llm=False, llm_deadline_s=0.8):
        uni, bi, tri, self.source = {}, {}, {}, "corpus"
        path = DATA / "corpus.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                uni, bi, tri = data["unigram"], data["bigram"], data.get("trigram", {})
            except Exception as e:
                print("[predictor] could not read data/corpus.json:", e)
        if not uni:
            # no corpus built yet: a small built-in vocabulary so suggestions still make sense
            self.source = "built-in"
            u, b, tri = count_sentences(read_lines(SEED))
            for i, w in enumerate(FALLBACK):
                u[w] += max(1, 6 - i // 2)
            total_u, total_b = sum(u.values()), sum(b.values()) or 1
            uni = {w: c / total_u for w, c in u.items()}
            bi = {k: c / total_b for k, c in b.items()}
            print("[predictor] data/corpus.json not found, using the small built-in vocabulary "
                  "(run: python speller/build_corpus.py)")

        self.uni = {w: p for w, p in uni.items() if valid_word(w)}
        follow = defaultdict(list)
        for pair, p in bi.items():
            a, _, b = pair.partition(" ")
            if b in self.uni or valid_word(b):
                follow[a].append((p, b))
        follow3 = defaultdict(list)
        for triple, p in tri.items():
            ab, _, c = triple.rpartition(" ")
            if c in self.uni or valid_word(c):
                follow3[ab].append((p, c))

        def conditional(groups):                    # context -> [(P(word | context), word)], best first
            out = {}
            for ctx, lst in groups.items():
                total = sum(p for p, _ in lst)
                out[ctx] = sorted(((p / total, w) for p, w in lst), reverse=True)
            return out

        self.next_words, self.next_words3 = conditional(follow), conditional(follow3)
        self.ranked = sorted(self.uni, key=self.uni.get, reverse=True)
        self.by_prefix = defaultdict(list)          # first 1-3 letters -> words, most common first
        for w in self.ranked:
            for k in range(1, min(3, len(w)) + 1):
                self.by_prefix[w[:k]].append(w)
        self._memo = {}

        self.llm_deadline_s = llm_deadline_s
        self.llm_cache, self.llm_pending = {}, set()
        self.has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        self.use_llm = bool(use_llm) and self.has_key
        if use_llm and not self.has_key:
            print("[predictor] use_llm is on but GEMINI_API_KEY is not set: local suggestions only")
        self.llm_errors, self._llm_off_until, self._client = 0, 0.0, None
        self.clinical = False                       # patient mode: also offer CLINICAL words, no event boost
        self.event_words = {w.strip().upper() for w in read_lines(DATA / "custom_words.txt")} & set(self.uni)
        self._visit = ("\0", 0.0, None)             # (typed text, first seen, Gemini word in use)
        self.last_source = "local"                  # where the latest suggestion came from: local / llm

    # ------------------------------------------------------------ helpers
    @staticmethod
    def split(typed):
        """'I WANT WA' -> (prev='WANT', prefix='WA')."""
        parts = typed.split(" ")
        prefix = parts[-1]
        done = [p for p in parts[:-1] if p]
        return (done[-1] if done else "<S>"), prefix

    @staticmethod
    def context(typed):
        """'I WANT WA' -> (prev2='I', prev='WANT', prefix='WA'). '<S>' stands for the start."""
        parts = typed.split(" ")
        done = ["<S>", "<S>"] + [p for p in parts[:-1] if p]
        return done[-2], done[-1], parts[-1]

    def describe(self):
        """Short status line for the UI."""
        base = f"{len(self.uni):,} words ({self.source})"
        return base + " + Gemini" if self.use_llm else base

    # ------------------------------------------------------------ local model
    def local(self, typed, max_len=None):
        key = self.context(typed.upper()) + (max_len,)
        memo = (key, self.clinical)
        if memo not in self._memo:
            self._memo[memo] = self._local(*key)
        return self._memo[memo]

    def _scores(self, prev2, prev, prefix, max_len):
        def fits(w):
            return (w.startswith(prefix) and w != prev and (max_len is None or len(w) <= max_len)
                    and (self.clinical or w not in CLINICAL))

        scores, found = {}, 0
        for p, w in self.next_words.get(prev, ()):          # what usually follows the previous word
            if fits(w):
                scores[w] = BIGRAM_WEIGHT * p + (1 - BIGRAM_WEIGHT) * self.uni.get(w, 0.0)
                found += 1
                if found >= 40:
                    break
        for p, w in self.next_words3.get(f"{prev2} {prev}", ()):   # ...and the previous two words
            if fits(w):
                scores[w] = scores.get(w, (1 - BIGRAM_WEIGHT) * self.uni.get(w, 0.0)) + TRIGRAM_WEIGHT * p
        found = 0
        for w in (self.by_prefix.get(prefix[:3], ()) if prefix else self.ranked):   # most common words overall
            if fits(w):
                scores.setdefault(w, (1 - BIGRAM_WEIGHT) * self.uni[w])
                found += 1
                if found >= 40:
                    break
        if prefix and not self.clinical:                         # at the booth, greetings and event words first
            for w in scores:
                if w in self.event_words:
                    scores[w] *= EVENT_BOOST
        return scores

    def _local(self, prev2, prev, prefix, max_len):
        scores = self._scores(prev2, prev, prefix, max_len)
        if not scores:
            return None
        exact = scores.pop(prefix, None) if prefix else None
        best = max(scores, key=scores.get) if scores else None
        if exact is not None and (best is None or exact >= EXACT_RATIO * scores[best]):
            return prefix                                   # the word is complete: box 4 adds the space
        return best

    def candidates(self, typed, n=3, max_len=None):
        """The n most likely words for the current position, best first (local model only)."""
        scores = self._scores(*self.context(typed.upper()), max_len)
        first = self.local(typed, max_len)
        rest = sorted((w for w in scores if w != first), key=scores.get, reverse=True)
        return ([first] if first else []) + rest[:n - 1 if first else n]

    # ------------------------------------------------------------ public API
    def suggest(self, typed, max_len=None):
        typed = typed.upper()
        now = time.monotonic()
        text, t0, llm = self._visit
        if typed != text:                                   # the message changed: decide afresh
            t0, llm = now, self._llm_word(typed)
        elif llm is None and now - t0 <= self.llm_deadline_s:
            llm = self._llm_word(typed)                     # Gemini answered in time
        self._visit = (typed, t0, llm)
        if (self.use_llm and typed not in self.llm_cache and typed not in self.llm_pending
                and now >= self._llm_off_until):
            self.llm_pending.add(typed)
            threading.Thread(target=self._ask_llm, args=(typed,), daemon=True).start()
        if llm and (max_len is None or len(llm) <= max_len):
            self.last_source = "llm"
            return llm
        self.last_source = "local"
        return self.local(typed, max_len)

    # ------------------------------------------------------------ Gemini
    def _llm_word(self, typed):
        word = self.llm_cache.get(typed)
        _, prefix = self.split(typed)
        ok = word and word.startswith(prefix) and (self.clinical or word not in CLINICAL)
        return word if ok else None

    def _prompt(self, typed):
        _, prefix = self.split(typed)
        if prefix:
            task = (f"The last word is unfinished: it starts with '{prefix}'. Reply with the ONE complete word "
                    f"they most likely mean. It must start with '{prefix}'.")
        elif typed.strip():
            task = "The last word is finished. Reply with the ONE word most likely to come next."
        else:
            task = "Nothing is typed yet. Reply with the ONE word most likely to start the message."
        return (
            "You are the word predictor of a brain-computer-interface speller. Every letter costs the user "
            "about ten seconds, so a correct guess saves real effort. Users write short, warm, practical "
            "messages (postcard greetings, or everyday needs of a person who cannot speak), for example:\n"
            f"{AAC_EXAMPLES}\n\n"
            f"Typed so far: \"{typed}\"\n{task}\n"
            "Answer with that single word in UPPERCASE letters A-Z only. No punctuation, no explanation."
        )

    def _ask_llm(self, typed):
        try:
            from google import genai   # pip install google-genai
            from google.genai import types
            if self._client is None:
                try:
                    self._client = genai.Client(http_options=types.HttpOptions(timeout=4000))   # milliseconds
                except Exception:
                    self._client = genai.Client()
            model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=self._prompt(typed),
                    config=types.GenerateContentConfig(temperature=0.2))
            except TypeError:
                resp = self._client.models.generate_content(model=model, contents=self._prompt(typed))
            first = ((resp.text or "").strip().upper().split() or [""])[0]
            word = "".join(c for c in first if "A" <= c <= "Z")
            _, prefix = self.split(typed)
            if word and len(word) <= 15 and word.startswith(prefix) and (valid_word(word) or word in self.uni):
                self.llm_cache[typed] = word
            else:
                self.llm_cache[typed] = ""          # unusable answer: don't ask again for this text
            self.llm_errors = 0
        except Exception as e:  # never let the LLM break the speller
            self.llm_errors += 1
            if self.llm_errors >= 3:                # Wi-Fi or quota trouble: stay local for a while
                self._llm_off_until = time.monotonic() + 30
                self.llm_errors = 0
            print("[predictor] LLM error:", str(e)[:200])
        finally:
            self.llm_pending.discard(typed)


if __name__ == "__main__":
    import sys
    llm = "--llm" in sys.argv
    p = Predictor(use_llm=llm)
    print("Model:", p.describe())
    for t in ["", "H", "HI", "HI ", "HI M", "I", "I ", "I L", "I LOVE ", "I W", "I WANT ", "I WANT WA",
              "THANK ", "PLEASE ", "HACK THE ", "CAN YOU "]:
        if llm and p.use_llm:
            t0 = time.time()
            p.suggest(t)
            while t in p.llm_pending and time.time() - t0 < 6:
                time.sleep(0.02)
            print(f"{t!r:14} local={str(p.local(t)):10} gemini={p.llm_cache.get(t) or '-':10} "
                  f"({(time.time() - t0) * 1000:.0f} ms)")
        else:
            print(f"{t!r:14} -> {p.suggest(t)}")
