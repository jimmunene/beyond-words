"""Download the Tsinghua SSVEP benchmark (35 people) and keep the small part we need.

Wang, Chen, Gao, Gao: "A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces",
IEEE TNSRE 2017. 35 people each looked at 40 flickering targets (8-15.8 Hz), 6 times, 5 s each,
with a 64-electrode cap, stored at 250 Hz. Files: https://bci.med.tsinghua.edu.cn/download.html

Each subject is a 105 MB archive. This script downloads it, keeps the eight electrodes over the
visual cortex (PO7 PO3 POz PO4 PO8 O1 Oz O2: the same eight, in the same order, as the g.tec
dataset) and the eight whole-number frequencies (8-15 Hz), saves that as data/benchmark/Sxx.npz
(about 2 MB) and deletes the big files again.

    python acquisition/prepare_benchmark.py              # all 35 subjects (about 4 GB of download)
    python acquisition/prepare_benchmark.py 1 2 3        # only these subjects
    python acquisition/prepare_benchmark.py --report     # accuracy table from what is already on disk

Needs 7-Zip (7z.exe) or `pip install py7zr` to unpack the archives.
"""
import pathlib
import shutil
import subprocess
import sys
import urllib.request

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, load_config  # noqa: E402

URL = "https://bci.med.tsinghua.edu.cn/upload/yijun/S{n}.mat.7z"
OUT = ROOT / "data" / "benchmark"
FS = 250
CHANNELS = [53, 55, 56, 57, 59, 61, 62, 63]          # 1-based numbers in 64-channels.loc
CHANNEL_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
FREQS = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]   # target indices 0-7 in the benchmark
STIM = slice(int(0.5 * FS), int(5.5 * FS))            # each 6 s epoch: 0.5 s cue, 5 s flicker, 0.5 s blank


def unpack(archive):
    seven = shutil.which("7z") or shutil.which("7za") or next(
        (p for p in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe")
         if pathlib.Path(p).exists()), None)
    if seven:
        subprocess.run([seven, "x", str(archive), f"-o{archive.parent}", "-y"], check=True,
                       stdout=subprocess.DEVNULL)
    else:
        import py7zr   # pip install py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(archive.parent)


def fetch(n):
    """Download subject n and save the reduced file. Returns its path."""
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"S{n:02d}.npz"
    if out.exists():
        return out
    archive, mat = OUT / f"S{n}.mat.7z", OUT / f"S{n}.mat"
    try:
        req = urllib.request.Request(URL.format(n=n), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        unpack(archive)
        data = sio.loadmat(str(mat))["data"]                    # (64 electrodes, 1500 samples, 40 targets, 6 blocks)
        keep = data[[c - 1 for c in CHANNELS]][:, :, :len(FREQS), :]
        eeg = np.transpose(keep, (2, 3, 1, 0)).astype(np.float32)   # -> (frequency, block, sample, channel)
        np.savez_compressed(out, eeg=eeg, freqs=np.array(FREQS), fs=FS, channels=np.array(CHANNEL_NAMES))
    finally:
        archive.unlink(missing_ok=True)
        mat.unlink(missing_ok=True)
    return out


def accuracy(path, cfg, freqs):
    """Share of windows our classifier gets right for these frequencies (chance = 1 / len(freqs))."""
    from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess
    z = np.load(path)
    n = int(cfg["window_s"] * FS)
    filt = make_filters(FS, cfg["bandpass"], cfg["notch_hz"])
    clf = (FBCCAClassifier if cfg["use_fbcca"] else CCAClassifier)(FS, freqs, n, cfg["harmonics"])
    right = total = 0
    for f in freqs:
        for block in z["eeg"][FREQS.index(f)]:
            stim = block[STIM][int(0.3 * FS):]                  # skip the first 0.3 s after the flicker starts
            for a in range(0, len(stim) - n + 1, FS // 2):
                scores = clf.scores(preprocess(stim[a:a + n][:, cfg["eeg_channels"]].astype(float), filt))
                right += freqs[int(np.argmax(scores))] == f
                total += 1
    return right / max(total, 1)


def report(cfg):
    freqs = [float(f) for f in cfg["freqs"]]
    if any(f not in FREQS for f in freqs):
        sys.exit(f"config.yaml freqs {freqs} are not all in the benchmark subset {FREQS}")
    files = sorted(OUT.glob("S??.npz"))
    if not files:
        sys.exit("No subjects yet. Run: python acquisition/prepare_benchmark.py")
    kind = "FBCCA" if cfg["use_fbcca"] else "CCA"
    print(f"\nOur classifier ({kind}, {cfg['window_s']:g} s windows, channels {cfg['eeg_channels']}) "
          f"on {freqs} Hz, chance = {100 / len(freqs):.0f}%")
    accs = []
    for path in files:
        acc = accuracy(path, cfg, freqs)
        accs.append(acc)
        print(f"  {path.stem}: {100 * acc:5.1f}%  " + "#" * int(round(40 * acc)))
    accs = np.array(accs)
    print(f"\n{len(accs)} people: mean {100 * accs.mean():.1f}%, median {100 * np.median(accs):.1f}%, "
          f"worst {100 * accs.min():.1f}%, best {100 * accs.max():.1f}%; "
          f"{int((accs >= 0.9).sum())} of {len(accs)} at 90% or better")


if __name__ == "__main__":
    cfg = load_config()
    if "--report" not in sys.argv:
        wanted = [int(a) for a in sys.argv[1:] if a.isdigit()] or list(range(1, 36))
        for n in wanted:
            try:
                path = fetch(n)
                print(f"S{n:02d} ready ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
            except Exception as e:
                print(f"S{n:02d} FAILED: {e}", flush=True)
    report(cfg)
