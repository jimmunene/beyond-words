"""Turn an ACTUAL EEG recording into artwork; never silently invent EEG data.

Use make_demo_postcard() for an explicitly labeled synthetic-data example.
Source refers to the *signal*; input_mode describes how text was selected.
Neither a recorded replay nor keyboard input is presented as live brain typing.
"""

import hashlib
import pathlib
import textwrap
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.signal import welch  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "postcards"

PALETTES = {
    "calm": ["#0b1d51", "#1f4e9c", "#3fa7d6", "#a8e6f0"],
    "focus": ["#2b0b3f", "#7b2cbf", "#e05780", "#ffb4a2"],
    "bright": ["#3d1308", "#c44900", "#f6aa1c", "#fff3b0"],
}


def validate_eeg(eeg, fs):
    if not isinstance(fs, (int, float)) or not 64 <= fs <= 2048:
        raise ValueError("Expected sample rate between 64 and 2048 Hz")
    try:
        arr = np.asarray(eeg, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("EEG recording must be numerical") from exc
    if arr.ndim != 2 or not 1 <= arr.shape[1] <= 64:
        raise ValueError("EEG must be an (samples, channels) array")
    if len(arr) < fs:
        raise ValueError("EEG recording must contain at least one second of actual data")
    if not np.isfinite(arr).all():
        raise ValueError("EEG recording contains NaN or Infinity")
    return arr


def band_power(eeg, fs, lo, hi):
    f, p = welch(eeg, fs=fs, nperseg=min(len(eeg), int(fs * 2)), axis=0)
    idx = (f >= lo) & (f < hi)
    return p[idx].mean(axis=0) if idx.any() else np.zeros(eeg.shape[1])


def features(eeg, fs):
    eeg = validate_eeg(eeg, fs)
    centered = eeg - eeg.mean(axis=0)
    theta, alpha, beta = (band_power(centered, fs, *band) for band in [(4, 8), (8, 13), (13, 30)])
    ratio = float(alpha.mean() / (beta.mean() + 1e-9))
    mood = "calm" if ratio > 1.3 else "focus" if ratio > 0.8 else "bright"
    power = alpha + beta + theta
    return {
        "mood": mood,
        "ch_power": power / (power.max() + 1e-9),
        "variability": float(np.clip(np.std(centered) / (np.abs(centered).mean() + 1e-9), 0.0, 4.0)),
        "seed": int(hashlib.sha256(eeg.tobytes()).hexdigest()[:8], 16),
    }


def _source_caption(signal_source, input_mode):
    signal_desc = {
        "live": "LIVE EEG RECORDING",
        "replay": "RECORDED EEG REPLAY",
        "synthetic": "SYNTHETIC DEMO — NOT LIVE EEG",
    }
    input_desc = {
        "eeg": "EEG-SELECTED TEXT",
        "keyboard": "KEYBOARD-SELECTED TEXT",
        "synthetic": "SIMULATED TEXT SELECTION",
    }
    if signal_source not in signal_desc or input_mode not in input_desc:
        raise ValueError("Use an explicit live/replay/synthetic source and eeg/keyboard/synthetic input mode")
    return signal_desc[signal_source] + "  ·  " + input_desc[input_mode]


def make_postcard(eeg, fs, message, nickname="", out_dir=OUT, *,
                  signal_source="live", input_mode="eeg", gallery_url=None):
    """Create PNG from actual provided samples. Never silently replace missing data."""
    arr = validate_eeg(eeg, fs)
    caption = _source_caption(signal_source, input_mode)
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Cannot make a postcard without a message")
    if len(message) > 120:
        raise ValueError("Message must be 120 characters or fewer")
    ft = features(arr, fs)
    rng = np.random.default_rng(ft["seed"])
    cmap = LinearSegmentedColormap.from_list("postcard", PALETTES[ft["mood"]])
    fig = plt.figure(figsize=(12, 8), dpi=100)
    try:
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis("off")
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(np.repeat(gradient, 8, axis=0), extent=[0, 12, 0, 8], cmap=cmap, alpha=0.96, aspect="auto")
        dots_per_channel = min(16, 3 + int(3 * ft["variability"]))
        for pw in ft["ch_power"]:
            for _ in range(dots_per_channel):
                x, y = rng.uniform(0, 12), rng.uniform(0, 8)
                radius = 0.2 + 1.6 * pw * rng.uniform(0.3, 1)
                ax.add_patch(plt.Circle((x, y), radius, color=cmap(rng.uniform(0.4, 1)), alpha=0.12, lw=0))
        n = min(len(arr), int(fs * 4))
        x = np.linspace(0, 12, n)
        for ch in range(arr.shape[1]):
            signal = arr[:n, ch] - arr[:n, ch].mean()
            signal = np.convolve(signal, np.ones(15) / 15, mode="same")
            signal = signal / (np.abs(signal).max() + 1e-9)
            base = 1.4 + ch * (5.2 / max(arr.shape[1] - 1, 1))
            ax.plot(x, base + 0.32 * signal, color="white", lw=1.1,
                    alpha=0.3 + 0.5 * float(ft["ch_power"][ch]))
        # Wrap longer text; don't truncate or change a confirmed message.
        lines = textwrap.wrap(message.strip().upper(), width=18, break_long_words=True)[:7]
        main_text = "\n".join(lines)
        size = 56 if len(main_text) < 20 else 36 if len(main_text) < 70 else 25
        ax.text(6, 4.25, main_text, ha="center", va="center", fontsize=size,
                linespacing=1.22, fontweight="bold", color="white",
                path_effects=[path_effects.withStroke(linewidth=5, foreground="black", alpha=0.5)])
        if nickname:
            ax.text(6, 2.15, "— " + nickname[:32], ha="center", fontsize=19,
                    color="white", fontweight="bold")
        ax.text(6, 0.48, caption, ha="center", fontsize=12, color="white", fontweight="bold")
        ax.text(6, 0.18, "BEYOND WORDS  ·  Hack The Nest 2026", ha="center",
                fontsize=10, color="white", alpha=0.9)
        if gallery_url:
            # QR is only for a real configured HTTPS gallery URL and only on the
            # consented publish path; it never automatically uploads anything.
            if not gallery_url.startswith("https://") or "YOUR-" in gallery_url.upper():
                raise ValueError("Gallery URL must be a configured HTTPS URL")
            import qrcode
            qr = qrcode.make(gallery_url).convert("RGB").resize((190, 190))
            fig.figimage(np.asarray(qr), xo=975, yo=50, zorder=12)
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / ("bw_" + uuid.uuid4().hex[:16] + ".png")
        fig.savefig(output)
        return str(output)
    finally:
        plt.close(fig)


def make_demo_postcard(message="HI MOM", nickname="DEMO", out_dir=OUT):
    """Explicit synthetic-only demonstration; never claim a real brain typed it."""
    fake = np.random.default_rng(42).normal(size=(250 * 4, 8))
    return make_postcard(fake, 250, message, nickname, out_dir,
                         signal_source="synthetic", input_mode="synthetic")


if __name__ == "__main__":
    print("SYNTHETIC EXAMPLE:", make_demo_postcard())
