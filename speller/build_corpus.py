"""Build the word-prediction model (data/corpus.json) from free corpora.

Put these files in data/ (any that are missing are skipped):
  count_1w.txt        Norvig unigram counts   (norvig.com/ngrams)       word<TAB>count
  count_2w.txt        Norvig bigram counts    (norvig.com/ngrams)       "w1 w2"<TAB>count
  aac_sentences.txt   AAC-style sentences, ONE PER LINE (e.g. Vertanen's AAC collection,
                      keithv.com; convert to one sentence per line if needed)
  custom_words.txt    names / event words, one per line (always boosted)

    python speller/build_corpus.py
"""
import collections
import json
import pathlib
import re

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
TOP_UNIGRAMS, TOP_BIGRAMS = 30000, 200000
AAC_WEIGHT = 0.6            # how much AAC sentences count vs. general web text

DEFAULT_CUSTOM = """HELLO HI LOVE MOM DAD YOU THANK THANKS WATER PAIN HELP YES NO PLEASE
EMU HACK NEST BRAIN MIND WOW COOL FRIEND FAMILY HAPPY""".split()


def read_counts(path, top):
    out = {}
    if not path.exists():
        print("  (missing)", path.name)
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= top:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[1].isdigit():
                out[parts[0].upper()] = int(parts[1])
    return out


def normalize(counts):
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def words(line):
    return re.findall(r"[A-Z']+", line.upper())


def main():
    print("Reading corpora from", DATA)
    web_uni = normalize(read_counts(DATA / "count_1w.txt", TOP_UNIGRAMS))
    web_bi = normalize(read_counts(DATA / "count_2w.txt", TOP_BIGRAMS))

    aac_uni, aac_bi = collections.Counter(), collections.Counter()
    aac_path = DATA / "aac_sentences.txt"
    if aac_path.exists():
        for line in open(aac_path, encoding="utf-8", errors="ignore"):
            w = words(line)
            aac_uni.update(w)
            aac_bi.update(f"{a} {b}" for a, b in zip(w, w[1:]))
            if w:
                aac_bi[f"<S> {w[0]}"] += 1          # sentence-start words
    else:
        print("  (missing) aac_sentences.txt")
    aac_uni, aac_bi = normalize(aac_uni), normalize(aac_bi)

    def mix(a, b):
        keys = set(a) | set(b)
        return {k: AAC_WEIGHT * a.get(k, 0) + (1 - AAC_WEIGHT) * b.get(k, 0) for k in keys}

    uni, bi = mix(aac_uni, web_uni), mix(aac_bi, web_bi)

    custom = DEFAULT_CUSTOM[:]
    if (DATA / "custom_words.txt").exists():
        custom += [w.strip().upper() for w in open(DATA / "custom_words.txt") if w.strip()]
    top = max(uni.values()) if uni else 1.0
    for w in custom:
        uni[w] = max(uni.get(w, 0), top * 0.05)

    uni = {k: v for k, v in uni.items() if re.fullmatch(r"[A-Z]+", k)}   # letters only (speller has no ')
    bi = {k: v for k, v in bi.items() if re.fullmatch(r"(<S>|[A-Z]+) [A-Z]+", k)}
    out = DATA / "corpus.json"
    json.dump({"unigram": uni, "bigram": bi}, open(out, "w"))
    print(f"Saved {out}: {len(uni)} words, {len(bi)} bigrams")


if __name__ == "__main__":
    main()
