"""Build the word-prediction model (data/corpus.json) from free corpora.

Put these files in data/ (any that are missing are skipped):
  count_1w.txt        Norvig unigram counts   (norvig.com/ngrams)       word<TAB>count
  count_2w.txt        Norvig bigram counts    (norvig.com/ngrams)       "w1 w2"<TAB>count
  aac_raw/            crowd-sourced AAC messages from aactext.org (Keith Vertanen), fetched by
                      --download: what people say when they talk through a communication device
  aac_sentences.txt   any extra sentences of your own, ONE PER LINE (optional)
  custom_words.txt    names / event words, one per line (always boosted)
Always included from speller/: seed_sentences.txt (our own postcard + care sentences) and
needs_vocabulary.txt (body, symptoms, equipment, care words: guaranteed to be predictable).

    python speller/build_corpus.py --download     # first time: fetch the AAC files (4 MB), then build
    python speller/build_corpus.py                # rebuild after editing words or sentences
    python speller/eval_predictor.py              # how many looks does prediction save?

Credits: AAC text collections (c) Keith Vertanen and collaborators, https://www.aactext.org
(Creative Commons, see the readme files in data/aac_raw/). Word counts: norvig.com/ngrams.
"""
import json
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import DATA, SEED, count_sentences, read_lines, valid_word  # noqa: E402

TOP_UNIGRAMS, TOP_BIGRAMS = 60000, 300000
AAC_WEIGHT = 0.6            # how much AAC-style sentences count vs. general web text
SEED_SHARE = 0.25           # our own postcard/care sentences count as this fraction of the AAC text
NEEDS = SEED.parent / "needs_vocabulary.txt"

RAW = DATA / "aac_raw"
SITE = "https://www.aactext.org/"
DOWNLOADS = {               # file in data/aac_raw/ -> path on aactext.org
    "sent_train_aac.txt": "imagine/aac_comm/sent_train_aac.txt",
    "sent_dev_aac.txt": "imagine/aac_comm/sent_dev_aac.txt",
    "sent_test_aac.txt": "imagine/aac_comm/sent_test_aac.txt",          # only for eval_predictor.py
    "american_words.txt": "imagine/aac_comm/american_words.txt",
    "imagine_readme.txt": "imagine/aac_comm/readme.txt",
    "comm2.txt": "comm2/comm2/comm2.txt",
    "comm2_readme.txt": "comm2/comm2/readme.txt",
    "turk-dialogues-filtered.txt": "turk/turk/turk-dialogues-filtered.txt",
    "turk_readme.txt": "turk/turk/readme.txt",
}
TRAIN_FILES = ["sent_train_aac.txt", "sent_dev_aac.txt", "comm2.txt", "turk-dialogues-filtered.txt"]


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in DOWNLOADS.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            req = urllib.request.Request(SITE + path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            print("  downloaded", name)
        except Exception as e:
            print("  could not download", name, "->", e)


def read_aac_raw():
    """Sentences from the aactext.org files. Lines may start with an id and hold several
    tab-separated turns of a dialogue; every turn is one sentence."""
    out = []
    for name in TRAIN_FILES:
        n0 = len(out)
        for line in read_lines(RAW / name):
            cells = [c.strip() for c in line.split("\t")]
            if len(cells) > 1:                               # first cell is an id (or the header row)
                cells = cells[1:]
            out += [c for c in cells if c and not c.startswith("turn")]
        print(f"  {name}: {len(out) - n0:,} sentences" if len(out) > n0 else f"  (missing) aac_raw/{name}")
    return out


def read_words(path):
    words = []
    for line in read_lines(path):
        if not line.lstrip().startswith("#"):
            words += [w.upper() for w in line.split() if w.isalpha()]
    return words

DEFAULT_CUSTOM = """HELLO HI LOVE MOM DAD YOU THANK THANKS WATER PAIN HELP YES NO PLEASE
EMU HACK NEST BRAIN MIND WOW COOL FRIEND FAMILY HAPPY""".split()


def read_counts(path, top):
    """Read 'key<TAB>count' lines and keep the `top` most frequent (files are not always sorted by count)."""
    out = {}
    if not path.exists():
        print("  (missing)", path.name)
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1].isdigit():
                key = parts[0].upper()
                out[key] = out.get(key, 0) + int(parts[1])
    keep = sorted(out, key=out.get, reverse=True)[:top]
    print(f"  {path.name}: {len(out):,} entries, keeping {len(keep):,}")
    return {k: out[k] for k in keep}


def normalize(counts):
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def good_pair(pair):
    a, _, b = pair.partition(" ")
    return (a == "<S>" or valid_word(a)) and valid_word(b)


def main():
    print("Reading corpora from", DATA)
    if "--download" in sys.argv:
        download()
    web_uni = read_counts(DATA / "count_1w.txt", 10 ** 9)
    web_bi = read_counts(DATA / "count_2w.txt", 10 ** 9)
    # the web list is full of typos, names and fragments: keep only real dictionary words
    real = {w.strip().upper() for w in read_lines(RAW / "american_words.txt")}
    if real:
        print(f"  american_words.txt: {len(real):,} dictionary words used as a filter")
    web_uni = {k: v for k, v in web_uni.items() if valid_word(k) and (not real or k in real)}
    web_uni = normalize({k: web_uni[k] for k in sorted(web_uni, key=web_uni.get, reverse=True)[:TOP_UNIGRAMS]})
    web_bi = {k: v for k, v in web_bi.items() if good_pair(k)}
    web_bi = normalize({k: web_bi[k] for k in sorted(web_bi, key=web_bi.get, reverse=True)[:TOP_BIGRAMS]})

    aac_lines = read_aac_raw() + read_lines(DATA / "aac_sentences.txt")
    if not aac_lines:
        print("  no AAC text found (run with --download): using only our own seed sentences")
    seed_lines = [s for s in read_lines(SEED) if not s.lstrip().startswith("#")]
    # our own postcard/care sentences are few: repeat them so they still matter next to the AAC files
    repeat = max(1, round(SEED_SHARE * len(aac_lines) / max(len(seed_lines), 1)))
    aac_uni, aac_bi, aac_tri = count_sentences(aac_lines + seed_lines * repeat)
    aac_tri = {k: v for k, v in aac_tri.items() if good_pair(k.partition(" ")[2]) and
               (k.startswith("<S> ") or valid_word(k.partition(" ")[0]))}
    aac_tri = {k: aac_tri[k] for k in sorted(aac_tri, key=aac_tri.get, reverse=True)[:TOP_BIGRAMS]}
    aac_uni = normalize({k: v for k, v in aac_uni.items() if valid_word(k)})
    aac_bi = normalize({k: v for k, v in aac_bi.items() if good_pair(k)})
    print(f"  AAC-style text: {len(aac_lines):,} + {len(seed_lines):,} seed lines")

    def mix(a, b):
        if not a or not b:
            return dict(a or b)
        keys = set(a) | set(b)
        return {k: AAC_WEIGHT * a.get(k, 0) + (1 - AAC_WEIGHT) * b.get(k, 0) for k in keys}

    uni, bi = mix(aac_uni, web_uni), mix(aac_bi, web_bi)

    # care vocabulary: always present, with a small floor so it can be completed from its first letters
    needs = [w for w in read_words(NEEDS) if valid_word(w)]
    floor = sorted(uni.values())[len(uni) // 2] if uni else 0.0      # as likely as a middle-of-the-list word
    added = [w for w in needs if w not in uni]
    for w in needs:
        uni[w] = max(uni.get(w, 0.0), floor)
    print(f"  needs_vocabulary.txt: {len(set(needs)):,} care words ({len(added)} were not in the corpus before)")

    custom = DEFAULT_CUSTOM[:]
    custom += [w.strip().upper() for w in read_lines(DATA / "custom_words.txt") if w.strip()]
    top = max(uni.values()) if uni else 1.0
    for w in custom:
        if w.isalpha():
            uni[w] = max(uni.get(w, 0), top * 0.05)

    out = DATA / "corpus.json"
    DATA.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"unigram": uni, "bigram": bi, "trigram": aac_tri}, f)
    print(f"Saved {out}: {len(uni):,} words, {len(bi):,} bigrams, {len(aac_tri):,} trigrams")


if __name__ == "__main__":
    main()
