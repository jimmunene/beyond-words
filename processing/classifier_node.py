"""Reads EEG from LSL, classifies SSVEP every step, sends scores + selections to the UI.
Also records raw EEG for each visitor session (used to draw their postcard art).

    python processing/classifier_node.py
"""
import pathlib
import sys
import time
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, UDPReceiver, UDPSender, load_config  # noqa: E402
from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess  # noqa: E402
from processing.selector import Selector  # noqa: E402

cfg = load_config()
fs = cfg["fs"]
n_win = int(cfg["window_s"] * fs)
step = cfg["step_s"]
chans = cfg["eeg_channels"]

print(f"Looking for LSL stream '{cfg['stream_name']}' ...")
streams = resolve_byprop("name", cfg["stream_name"], timeout=30)
if not streams:
    sys.exit("No EEG stream found. Start the headset LSL app, replay_stream.py or synthetic_stream.py first.")
inlet = StreamInlet(streams[0], max_chunklen=32)
print("Connected:", streams[0].name(), streams[0].channel_count(), "ch")

filters = make_filters(fs, cfg["bandpass"], cfg["notch_hz"])
Clf = FBCCAClassifier if cfg["use_fbcca"] else CCAClassifier
clf = Clf(fs, cfg["freqs"], n_win, cfg["harmonics"])
selector = Selector(cfg["margin_tau"], cfg["n_consecutive"], cfg["lockout_s"])
to_ui = UDPSender(cfg["ports"]["selection"])
control = UDPReceiver(cfg["ports"]["control"])

buf = deque(maxlen=n_win)
session, recording, paused = [], False, False
next_decision = time.time()

while True:
    chunk, _ = inlet.pull_chunk(timeout=0.05)
    if chunk:
        arr = np.asarray(chunk, dtype=np.float64)
        buf.extend(arr)
        if recording:
            session.append(arr)

    for msg in control.poll():
        cmd = msg.get("cmd")
        if cmd == "start_session":
            session, recording = [], True
        elif cmd == "stop_session":
            recording = False
            out = pathlib.Path(msg.get("path", ROOT / "sessions" / f"session_{int(time.time())}.npy"))
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, np.concatenate(session) if session else np.zeros((0, len(chans))))
            to_ui.send({"type": "session_saved", "path": str(out)})
        elif cmd == "pause":
            paused = True
        elif cmd == "resume":
            paused = False
            buf.clear()
            selector.reset()

    now = time.time()
    if now >= next_decision and len(buf) == n_win:
        next_decision = now + step
        window = preprocess(np.asarray(buf)[:, chans], filters)
        scores = clf.scores(window)
        to_ui.send({"type": "scores", "scores": scores.round(4).tolist(), "t": now})
        if not paused:
            hit = selector.update(scores, now)
            if hit:
                box, margin = hit
                to_ui.send({"type": "select", "box": box, "confidence": round(margin, 4), "t": now})
                buf.clear()   # next decision must use only new gaze data
