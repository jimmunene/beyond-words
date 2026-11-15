"""Offline accuracy of the decoder on the recorded datasets, for every setting worth trying.

Grid: window 1 / 1.5 / 2 / 3 s  x  CCA / FBCCA  x  electrode sets. For each: the share of windows
where the strongest score is the frequency the person was really looking at, and the ITR
(Wolpaw, 4 targets, one pick per window; the live system adds about 0.75 s of confirmation).
Electrode sets come from the eight positions in the datasets (PO7 PO3 POz PO4 PO8 O1 Oz O2);
"PO7 Oz PO8" are the ones a g.tec Unicorn also has (its Pz is not in the recordings).

    python eval/offline_accuracy.py            # table on screen + docs/offline_results.md
    python eval/offline_accuracy.py --quick    # 8 benchmark people instead of 35

Also reports how often a window would be thrown away by the classifier's artifact check.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from acquisition.dataset_stream import load_benchmark, load_gtec  # noqa: E402
from common import ROOT, load_config  # noqa: E402
from eval.metrics import itr_bits_per_min  # noqa: E402
from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess  # noqa: E402

WINDOWS = [1.0, 1.5, 2.0, 3.0]
SETS = {"Oz": [6], "PO7 Oz PO8 (Unicorn has these)": [0, 6, 4], "all 8": list(range(8))}


def evaluate(people, cfg, freqs, window_s, Clf, chans):
    """-> (accuracy per person, share of windows over the artifact limit)."""
    limit = float(cfg.get("artifact_uv", 100.0))
    accs, over, total = [], 0, 0
    for fs, trials, _ in people:
        n = int(window_s * fs)
        filt = make_filters(fs, cfg["bandpass"], cfg["notch_hz"])
        clf = Clf(fs, freqs, n, cfg["harmonics"])
        right = count = 0
        for f in freqs:
            for seg in trials[f]:
                for a in range(0, len(seg) - n + 1, int(fs) // 2):
                    w = preprocess(np.asarray(seg[a:a + n], dtype=float)[:, chans], filt)
                    right += freqs[int(np.argmax(clf.scores(w)))] == f
                    over += bool(limit) and float(np.abs(w).max()) > limit
                    count += 1
        accs.append(right / max(count, 1))
        total += count
    return np.array(accs), over / max(total, 1)


def main():
    cfg = load_config()
    freqs = [float(f) for f in cfg["freqs"]]
    files = sorted((ROOT / "data" / "benchmark").glob("S??.npz"))
    if "--quick" in sys.argv:
        files = files[:8]
    groups = {f"Tsinghua benchmark, {len(files)} people": [load_benchmark(ROOT / "data" / "benchmark", int(p.stem[1:])) for p in files],
              "g.tec dataset, 2 people": [load_gtec(ROOT / "data" / "ssvep", s) for s in ("1", "2")]}
    lines = ["# Offline decoder accuracy on recorded EEG", "",
             f"Frequencies {freqs} Hz, chance = {100 / len(freqs):.0f}%. Windows start 0.3 s after the flicker begins, "
             "every 0.5 s. No per-person training. Made by `python eval/offline_accuracy.py`.", "",
             "These are recordings from public datasets, not people wearing our headset.", ""]
    best = {}
    for title, people in groups.items():
        if not people:
            continue
        lines += [f"## {title}", "", "| Window | Method | Electrodes | Mean accuracy | Worst person | People >= 90% | ITR (bits/min) |",
                  "|---|---|---|---|---|---|---|"]
        for window_s in WINDOWS:
            for kind, Clf in (("CCA", CCAClassifier), ("FBCCA", FBCCAClassifier)):
                for name, chans in SETS.items():
                    accs, over = evaluate(people, cfg, freqs, window_s, Clf, chans)
                    itr = itr_bits_per_min(len(freqs), float(accs.mean()), window_s)
                    row = (f"| {window_s:g} s | {kind} | {name} | {100 * accs.mean():.1f}% | {100 * accs.min():.1f}% | "
                           f"{int((accs >= 0.9).sum())} of {len(accs)} | {itr:.1f} |")
                    print(title[:9], row, f" artifact-rejected {100 * over:.2f}%", flush=True)
                    lines.append(row)
                    if kind == ("FBCCA" if cfg["use_fbcca"] else "CCA") and window_s == cfg["window_s"] and name == "all 8":
                        best[title] = (accs.mean(), over)
        lines.append("")
    kind = "FBCCA" if cfg["use_fbcca"] else "CCA"
    lines += ["## What we run", "",
              f"`config.yaml`: {kind}, {cfg['window_s']:g} s window. Shorter windows have a higher ITR on paper but more wrong "
              "picks; for a person who cannot easily undo a mistake we chose accuracy.", ""]
    for title, (acc, over) in best.items():
        lines.append(f"- {title}: {100 * acc:.1f}% with all 8 electrodes; {100 * over:.2f}% of windows exceed the "
                     f"artifact limit ({cfg.get('artifact_uv', 100.0):g} uV) and would be skipped.")
    out = ROOT / "docs" / "offline_results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSaved", out)


if __name__ == "__main__":
    main()
