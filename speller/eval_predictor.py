"""How many looks does word prediction save? Types every test sentence the way a user would
(3 looks per letter, 1 look when box 4 shows the right word, 2 looks when the letter-group
page shows it) and compares with no prediction. It drives the real Speller, so it measures the product.

    python speller/eval_predictor.py                       # held-out AAC test sentences
    python speller/eval_predictor.py my_sentences.txt      # any file, one sentence per line
    python speller/eval_predictor.py --misses              # also list the words it never predicted
    python speller/eval_predictor.py --postcard            # score postcard mode instead of patient mode

The default test file (data/aac_raw/sent_test_aac.txt) is never used to build the corpus,
so the numbers are honest. Get it with: python speller/build_corpus.py --download
"""
import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import DATA, Predictor, read_lines, tokenize  # noqa: E402
from speller.state_machine import SCREENS, Speller  # noqa: E402

LOOKS_PER_LETTER = 3
TEST = DATA / "aac_raw" / "sent_test_aac.txt"


def letter_paths():
    """Box sequence for every letter, read from the real screens: group, subgroup, letter."""
    paths = {}
    for gi, group in enumerate(["AI", "JR", "SZ"]):
        for si, (_, action) in enumerate(SCREENS[group][:3]):
            for li, item in enumerate(SCREENS[action[1]]):
                if item[1][0] == "char":
                    paths[item[1][1]] = (gi, si, li)
    return paths


PATHS = letter_paths()
GROUPS = ["AI", "JR", "SZ"]


def type_sentence(pred, target, missed=None, mode="patient"):
    """Type `target` on the real Speller, always taking a word box when it shows the right word.
    Returns (looks used, letters in target, words completed by a suggestion, words)."""
    sp = Speller(pred, mode, max_chars=10 ** 6, patient_max_chars=10 ** 6)
    sp.screen, sp.stack = "root", []
    words, looks, hits = target.split(" "), 0, 0
    while sp.text.rstrip() != target or (sp.text != target and sp.text != target + " "):
        done = sp.text.count(" ")
        if done >= len(words):
            break
        word, typed_word = words[done], sp.text.split(" ")[-1]
        if sp.suggestion == word and not (mode == "postcard" and not sp.text):
            sp.select(3)                                       # box 4: the word in one look
            looks, hits = looks + 1, hits + 1
            continue
        nxt = word[len(typed_word)] if len(typed_word) < len(word) else " "
        g = PATHS[nxt][0]
        if nxt != " " and sp.group_word(GROUPS[g]) == word:
            sp.select(g)
            sp.select(3)                                       # the second word, on the letter-group page
            looks, hits = looks + 2, hits + 1
            continue
        if nxt == " " and done == len(words) - 1:
            break
        for box in PATHS[nxt]:
            sp.select(box)
        looks += LOOKS_PER_LETTER
        if missed is not None and nxt != " " and len(typed_word) + 1 == len(word):
            missed[word] += 1                                  # spelled to the last letter without help
    assert sp.text.strip() == target, (sp.text, target)
    return looks, len(target), hits, len(words)


def evaluate(pred, lines, missed=None, mode="patient"):
    looks = letters = hits = words = n = 0
    for line in lines:
        target = " ".join(tokenize(line))
        if not target:
            continue
        a, b, c, d = type_sentence(pred, target, missed, mode)
        looks, letters, hits, words, n = looks + a, letters + b, hits + c, words + d, n + 1
    return {"sentences": n, "letters": letters, "looks": looks, "baseline": letters * LOOKS_PER_LETTER,
            "saved": 1 - looks / max(1, letters * LOOKS_PER_LETTER), "word_hits": hits / max(1, words),
            "looks_per_letter": looks / max(1, letters)}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = pathlib.Path(args[0]) if args else TEST
    lines = read_lines(path)
    if not lines:
        sys.exit(f"No sentences in {path}. Run: python speller/build_corpus.py --download")
    pred = Predictor()
    mode = "postcard" if "--postcard" in sys.argv else "patient"
    missed = collections.Counter()
    r = evaluate(pred, lines, missed, mode)
    print(f"Word prediction: {pred.describe()}")
    print(f"{r['sentences']:,} sentences, {r['letters']:,} letters from {path.name}")
    print(f"  without prediction : {r['baseline']:,} looks  ({LOOKS_PER_LETTER}.00 per letter)")
    print(f"  with prediction    : {r['looks']:,} looks  ({r['looks_per_letter']:.2f} per letter)")
    print(f"  looks saved        : {100 * r['saved']:.1f}%")
    print(f"  words finished by a word box : {100 * r['word_hits']:.1f}%")
    if "--misses" in sys.argv:
        print("\nMost common words that had to be spelled out in full:")
        print("  " + ", ".join(f"{w} ({n})" for w, n in missed.most_common(40)))
