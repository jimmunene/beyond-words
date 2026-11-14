"""Replay a recorded EEG file as a live LSL stream (same name/format as the headset).

Input: .npy or .csv shaped (n_samples, n_channels), in microvolts, at cfg['fs'].
Convert the g.tec dataset into this shape first (Person A, step A1).

    python acquisition/replay_stream.py data/ssvep_subject1.npy
"""
import pathlib
import sys
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402

cfg = load_config()
path = pathlib.Path(sys.argv[1])
data = np.load(path) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
fs, n_ch = cfg["fs"], data.shape[1]
outlet = StreamOutlet(StreamInfo(cfg["stream_name"], "EEG", n_ch, fs, "float32", "replay01"))
print(f"Replaying {path.name}: {data.shape[0] / fs:.1f} s, {n_ch} ch @ {fs} Hz (loops forever)")

chunk, i, t0, sent = 10, 0, time.perf_counter(), 0
while True:
    block = data[i:i + chunk]
    if len(block) < chunk:
        i = 0
        continue
    outlet.push_chunk(block.astype(np.float32).tolist())
    i += chunk
    sent += chunk
    sleep = t0 + sent / fs - time.perf_counter()
    if sleep > 0:
        time.sleep(sleep)
