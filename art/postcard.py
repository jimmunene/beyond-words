"""Turn a visitor's EEG into a unique postcard with their brain-typed message.

    make_postcard(eeg, fs, message, nickname) -> path to PNG
    python art/postcard.py            # demo with random EEG
"""
import hashlib
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.signal import welch  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "postcards"

PALETTES = {  # chosen by the visitor's alpha/beta balance
    "calm":   ["#0b1d51", "#1f4e9c", "#3fa7d6", "#a8e6f0"],
    "focus":  ["#2b0b3f", "#7b2cbf", "#e05780", "#ffb4a2"],
    "bright": ["#3d1308", "#c44900", "#f6aa1c", "#fff3b0"],
}


def band_power(eeg, fs, lo, hi):
    f, p = welch(eeg, fs=fs, nperseg=min(len(eeg), fs * 2), axis=0)
    return p[(f >= lo) & (f < hi)].mean(axis=0)


def features(eeg, fs):
    eeg = eeg - eeg.mean(axis=0)
    theta, alpha, beta = (band_power(eeg, fs, *b) for b in [(4, 8), (8, 13), (13, 30)])
    ratio = float(alpha.mean() / (beta.mean() + 1e-9))
    mood = "calm" if ratio > 1.3 else "focus" if ratio > 0.8 else "bright"
    return {
        "mood": mood,
        "ch_power": (alpha + beta + theta) / ((alpha + beta + theta).max() + 1e-9),
        "variability": float(np.std(eeg) / (np.abs(eeg).mean() + 1e-9)),
        "seed": int(hashlib.sha256(eeg.tobytes()).hexdigest()[:8], 16),
    }


def make_postcard(eeg, fs, message, nickname="", out_dir=OUT):
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2 or len(eeg) < fs:      # too short: use noise so we still make a card
        eeg = np.random.default_rng().normal(size=(fs * 4, 8))
    ft = features(eeg, fs)
    rng = np.random.default_rng(ft["seed"])
    cmap = LinearSegmentedColormap.from_list("p", PALETTES[ft["mood"]])

    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    # background gradient
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(np.vstack([grad] * 8), extent=[0, 12, 0, 8], cmap=cmap, alpha=0.95, aspect="auto")

    # glowing circles, one cluster per channel, sized by its power
    for ch, pw in enumerate(ft["ch_power"]):
        for _ in range(int(3 + 10 * ft["variability"])):
            x, y = rng.uniform(0, 12), rng.uniform(0, 8)
            r = 0.2 + 1.6 * pw * rng.uniform(0.3, 1)
            ax.add_patch(plt.Circle((x, y), r, color=cmap(rng.uniform(0.4, 1)), alpha=0.12, lw=0))

    # brainwave lines: each channel drawn as a flowing ribbon across the card
    n = min(len(eeg), fs * 4)
    x = np.linspace(0, 12, n)
    for ch in range(eeg.shape[1]):
        sig = eeg[:n, ch] - eeg[:n, ch].mean()
        sig = np.convolve(sig, np.ones(15) / 15, mode="same")
        sig = sig / (np.abs(sig).max() + 1e-9)
        y0 = 1.2 + ch * (5.6 / max(eeg.shape[1] - 1, 1))
        ax.plot(x, y0 + 0.45 * sig, color="white", lw=1.4, alpha=0.35 + 0.4 * ft["ch_power"][ch])

    # message
    ax.text(6, 4.3, message.upper(), ha="center", va="center", fontsize=64 if len(message) < 10 else 46,
            fontweight="bold", color="white",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=6, foreground="black", alpha=0.45)])
    if nickname:
        ax.text(6, 3.0, f"— {nickname}", ha="center", fontsize=26, fontweight="bold", color="white",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=4, foreground="black", alpha=0.45)])
    ax.text(6, 0.35, "Typed with brain signals only  ·  Hack The Nest 2026  ·  BrainType Postcards",
            ha="center", fontsize=13, color="white", alpha=0.85)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in (nickname or "guest") if c.isalnum())[:20] or "guest"
    path = out_dir / f"{int(time.time())}_{safe}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


import matplotlib.patheffects  # noqa: E402,F401  (used above)

if __name__ == "__main__":
    fake = np.random.default_rng(1).normal(size=(250 * 10, 8))
    print(make_postcard(fake, 250, "HI MOM", "Ava"))
