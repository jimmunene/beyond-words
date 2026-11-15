"""Reads EEG from LSL, classifies SSVEP every step, sends scores + selections to the UI.
Also records raw EEG for each visitor session (used to draw their postcard art).

    python processing/classifier_node.py                 # the stream named in config.yaml (replay / synthetic)
    python processing/classifier_node.py --headset       # live g.tec Unicorn: finds its LSL stream by itself,
                                                         # decodes from Pz PO7 Oz PO8, reports electrode contact
    python processing/classifier_node.py --headset --stream UN-2021.05.36 --channels 5,6,7

Messages to the UI (UDP, JSON):  scores (every step), select (a pick), session_saved, and once a
second  quality: {names, rms, state: good/noisy/flat per electrode, lost, rate, stream}.

A window with a huge swing in it (a movement, a clenched jaw, an electrode knocked: more than
artifact_uv microvolts after filtering, default 100) is not allowed to count towards a pick.
"""
import argparse
import pathlib
import sys
import time
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import detrend

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, UDPReceiver, UDPSender, load_config  # noqa: E402
from acquisition.unicorn_check import (UNICORN_SSVEP, channel_names, contact_quality, describe,  # noqa: E402
                                       find_stream)
from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess  # noqa: E402
from processing.selector import Selector  # noqa: E402

ap = argparse.ArgumentParser(description="SSVEP classifier: LSL in, scores and selections out")
ap.add_argument("--headset", action="store_true", help="live g.tec Unicorn headset instead of the config.yaml stream")
ap.add_argument("--stream", default=None, help="LSL stream name; 'auto' finds the headset")
ap.add_argument("--channels", default=None, help="channels to decode from, 0-based, e.g. 4,5,6,7")
opts = ap.parse_args()

cfg = load_config()
step = cfg["step_s"]
artifact_uv = float(cfg.get("artifact_uv", 100.0))      # 0 switches the check off
if opts.headset:
    stream_name = opts.stream or cfg.get("headset_stream", "auto")
    chans = list(cfg.get("headset_channels", UNICORN_SSVEP))
else:
    stream_name = opts.stream or cfg["stream_name"]
    chans = list(cfg["eeg_channels"])
if opts.channels:
    chans = [int(c) for c in opts.channels.split(",")]

print(f"Looking for LSL stream '{stream_name}' ...", flush=True)
if opts.headset or stream_name == "auto":
    info = find_stream(stream_name, timeout=40)
else:
    found = resolve_byprop("name", stream_name, timeout=30)
    info = found[0] if found else None
if info is None:
    sys.exit("No EEG stream found. Start the headset's LSL app (Unicorn LSL), or dataset_stream.py / "
             "synthetic_stream.py, first.  See what is there:  python acquisition/unicorn_check.py --list")
if max(chans) >= info.channel_count():
    sys.exit(f"The stream has {info.channel_count()} channels but channels {chans} were asked for. "
             "Fix headset_channels / eeg_channels in config.yaml or pass --channels.")
fs = info.nominal_srate() or cfg["fs"]           # trust the stream: decoding at the wrong rate shifts every frequency
if abs(fs - cfg["fs"]) > 0.5:
    print(f"NOTE: stream runs at {fs:g} Hz (config.yaml says {cfg['fs']}); using {fs:g}.", flush=True)
fs = int(round(fs))
n_win = int(cfg["window_s"] * fs)
names = channel_names(chans, cfg) if opts.headset else [f"ch{c}" for c in chans]
inlet = StreamInlet(info, max_chunklen=32)
print("Connected:", info.name(), info.channel_count(), "ch", flush=True)
print("   ", describe(info), "| decoding from", " ".join(names), flush=True)

filters = make_filters(fs, cfg["bandpass"], cfg["notch_hz"])
Clf = FBCCAClassifier if cfg["use_fbcca"] else CCAClassifier
clf = Clf(fs, cfg["freqs"], n_win, cfg["harmonics"])
selector = Selector(cfg["margin_tau"], cfg["n_consecutive"], cfg["lockout_s"])
to_ui = UDPSender(cfg["ports"]["selection"])
control = UDPReceiver(cfg["ports"]["control"])

buf = deque(maxlen=n_win)
recent = deque(maxlen=2 * fs)                    # last 2 s, for the electrode contact report
session, recording, paused = [], False, False
next_decision = time.time()
last_data, lost, got, next_quality = time.time(), False, 0, time.time() + 1.0

while True:
    chunk, _ = inlet.pull_chunk(timeout=0.05)
    now = time.time()
    if chunk:
        arr = np.nan_to_num(np.asarray(chunk, dtype=np.float64)[:, chans])   # only the electrodes we decode from
        buf.extend(arr)
        recent.extend(arr)
        got += len(arr)
        last_data = now
        if recording:
            session.append(arr)
        if lost:
            lost = False
            print("EEG stream is back.", flush=True)
    elif now - last_data > 1.0 and not lost:     # Bluetooth dropped, headset off, LSL app stopped
        lost = True
        buf.clear()
        recent.clear()
        selector.reset()
        print("EEG stream stopped sending data. Waiting for it to come back ...", flush=True)

    if now >= next_quality:
        rate, got = got / (now - next_quality + 1.0), 0
        next_quality = now + 1.0
        rms, state = contact_quality(np.asarray(recent), filters) if len(recent) >= fs else ([0.0] * len(chans), ["flat"] * len(chans))
        to_ui.send({"type": "quality", "names": names, "rms": rms, "state": state, "lost": lost,
                    "rate": round(rate, 1), "stream": info.name(), "headset": bool(opts.headset)})

    for msg in control.poll():
        cmd = msg.get("cmd")
        if cmd == "start_session":
            session, recording = [], True
        elif cmd == "stop_session":
            recording = False
            out = pathlib.Path(msg.get("path", ROOT / "sessions" / f"session_{int(time.time())}.npy"))
            out.parent.mkdir(parents=True, exist_ok=True)
            data = np.concatenate(session) if session else np.zeros((0, len(chans)))
            np.save(out, data - data.mean(axis=0) if len(data) else data)    # a headset adds a large constant offset
            to_ui.send({"type": "session_saved", "path": str(out)})
        elif cmd == "pause":
            paused = True
        elif cmd == "resume":
            paused = False
            buf.clear()
            selector.reset()

    if now >= next_decision and len(buf) == n_win and not lost:
        next_decision = now + step
        window = np.asarray(buf)
        if opts.headset:                         # live electrodes drift slowly; take the slope out before filtering
            window = detrend(window, axis=0)
        window = preprocess(window, filters)
        scores = clf.scores(window)
        artifact = bool(artifact_uv) and float(np.abs(window).max()) > artifact_uv
        to_ui.send({"type": "scores", "scores": scores.round(4).tolist(), "t": now, "artifact": artifact})
        if artifact:
            selector.last, selector.streak = None, 0     # a pick needs clean windows in a row
        elif not paused:
            hit = selector.update(scores, now)
            if hit:
                box, margin = hit
                to_ui.send({"type": "select", "box": box, "confidence": round(margin, 4), "t": now})
                buf.clear()   # next decision must use only new gaze data
