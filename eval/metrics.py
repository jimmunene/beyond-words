"""Compute the numbers for the demo slide from logs/selections.csv.

    python eval/metrics.py logs/selections.csv
"""
import csv
import math
import sys


def itr_bits_per_min(n_targets, accuracy, seconds_per_selection):
    """Wolpaw information transfer rate."""
    N, P = n_targets, min(max(accuracy, 1e-9), 1 - 1e-9)
    bits = math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))
    return bits * 60.0 / seconds_per_selection


def main(path):
    rows = list(csv.DictReader(open(path)))
    cued = [r for r in rows if r["cued_box"] not in ("", "-1")]
    if cued:
        acc = sum(r["cued_box"] == r["box"] for r in cued) / len(cued)
        times = [float(r["seconds_since_last"]) for r in cued if float(r["seconds_since_last"]) > 0]
        spt = sum(times) / len(times) if times else 3.0
        print(f"Cued test selections : {len(cued)}")
        print(f"Accuracy             : {acc * 100:.1f}%  (chance = 25%)")
        print(f"Seconds / selection  : {spt:.2f}")
        print(f"ITR                  : {itr_bits_per_min(4, acc, spt):.1f} bits/min")
    sessions = [r for r in rows if r["event"] == "postcard_done"]
    if sessions:
        chars = sum(len(r["text"]) for r in sessions)
        secs = sum(float(r["session_seconds"]) for r in sessions)
        print(f"Postcards            : {len(sessions)}")
        print(f"Characters / minute  : {chars / (secs / 60):.1f}")
        print(f"Avg time / postcard  : {secs / len(sessions):.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/selections.csv")
