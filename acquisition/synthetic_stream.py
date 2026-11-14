"""Synthetic SSVEP EEG -> LSL stream. Lets the WHOLE pipeline run without a headset.

Run the UI with `--input sim`: pressing keys 1-4 tells this script which box you are
"looking at", and it injects that frequency into the occipital channels. 0 = rest.

    python acquisition/synthetic_stream.py            # default signal strength
    python acquisition/synthetic_stream.py 0.5        # weaker signal = harder, more realistic
"""
import pathlib
import sys
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import UDPReceiver, load_config  # noqa: E402

cfg = load_config()
fs, n_ch, freqs = cfg["fs"], cfg["n_channels"], cfg["freqs"]
strength = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

outlet = StreamOutlet(StreamInfo(cfg["stream_name"], "EEG", n_ch, fs, "float32", "synthetic01"))
ctrl = UDPReceiver(cfg["ports"]["sim_target"])
rng = np.random.default_rng()

target, sample, chunk = -1, 0, 10
print(f"Synthetic EEG streaming as '{cfg['stream_name']}' ({n_ch} ch @ {fs} Hz). Strength={strength}")
t0 = time.perf_counter()
while True:
    for m in ctrl.poll():
        target = int(m.get("target", -1))
        print("gaze target ->", target)
    t = (sample + np.arange(chunk)) / fs
    data = rng.normal(0, 5.0, (chunk, n_ch))                       # background noise (uV)
    data += 4.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6))[:, None] * 0.3  # some alpha
    if target >= 0:
        f = freqs[target]
        ssvep = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t + 0.4)
        data[:, cfg["eeg_channels"]] += strength * 3.0 * ssvep[:, None]
    outlet.push_chunk(data.astype(np.float32).tolist())
    sample += chunk
    # keep real-time pace without drift
    sleep = t0 + sample / fs - time.perf_counter()
    if sleep > 0:
        time.sleep(sleep)
