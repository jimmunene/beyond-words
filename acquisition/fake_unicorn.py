"""A PRACTICE headset: an LSL stream shaped exactly like the g.tec Unicorn's, carrying recorded EEG.

Lets the live-headset path (stream search, channel choice, contact check, `--input eeg` screen)
be rehearsed before the real headset is on the table. It is NOT a headset and nobody's brain is
being read: the stream is named "UN-PRACTICE-RECORDED" and the speller labels it as practice.

What it sends, 250 times a second, 17 channels like Unicorn LSL:
  0-7   EEG  Fz C3 Cz C4 Pz PO7 Oz PO8   (PO7 Oz PO8: the g.tec SSVEP recordings. Pz: the recorded
        POz, weaker. Fz-C4: noise only.) Each with a large constant offset, slow drift and mains hum,
        as a real amplifier has.
  8-13  accelerometer, gyroscope     14 battery %     15 sample counter     16 validation

The "person" looks at box 1, 2, 3, 4 in turn (6 s each, 4 s of looking away in between), or
follows the speller when it is run with --input sim.

    python acquisition/fake_unicorn.py
    python acquisition/fake_unicorn.py --subject 2 --look 6 --rest 4
    python acquisition/fake_unicorn.py --drop 20        # stop sending for 4 s after 20 s (a Bluetooth drop-out)
    python acquisition/fake_unicorn.py --loose Oz       # that electrode has no contact (noisy)
    python acquisition/fake_unicorn.py --jolt 5         # every 5 s the "person" moves: a sudden 300 uV jump on all electrodes
"""
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from acquisition.dataset_stream import load_gtec, option, prepare  # noqa: E402
from acquisition.unicorn_check import UNICORN_EEG  # noqa: E402
from common import ROOT, UDPReceiver, load_config  # noqa: E402

NAME, FS = "UN-PRACTICE-RECORDED", 250
FROM_DATASET = {5: (0, 1.0), 6: (6, 1.0), 7: (4, 1.0), 4: (2, 0.7)}    # Unicorn channel -> (dataset channel, gain)


def main():
    from pylsl import StreamInfo, StreamOutlet
    cfg = load_config()
    freqs = [float(f) for f in cfg["freqs"]]
    look_s, rest_s = float(option("--look", 6)), float(option("--rest", 4))
    drop_at, loose, jolt = float(option("--drop", 0)), option("--loose", ""), float(option("--jolt", 0))
    src_fs, trials, rest = load_gtec(option("--folder", ROOT / "data" / "ssvep"), option("--subject", "1"))
    if any(f not in trials for f in freqs):
        sys.exit(f"config.yaml freqs {freqs} are not all in the g.tec recordings {sorted(trials)}")
    clips = {i: [prepare(t, src_fs, FS) for t in trials[f]] for i, f in enumerate(freqs)}
    clips[-1] = [prepare(r, src_fs, FS) for r in rest]

    outlet = StreamOutlet(StreamInfo(NAME, "Data", 17, FS, "float32", "practice_unicorn"))
    try:
        ctrl = UDPReceiver(cfg["ports"]["sim_target"])        # the speller (--input sim) may steer the gaze
    except OSError:
        ctrl = None
    rng = np.random.default_rng()
    offset = rng.uniform(-40000, 40000, 8)                    # microvolts: electrode offsets
    drift = np.zeros(8)
    print(f"PRACTICE headset streaming as '{NAME}' (17 ch @ {FS} Hz). This is recorded EEG, not a headset.", flush=True)

    target, steered, chunk, sent, t0 = -1, False, 10, 0, time.perf_counter()
    clip, pos, phase_end, turn = clips[-1][0], 0, rest_s, 0
    while True:
        t = sent / FS
        new = target
        for m in (ctrl.poll() if ctrl else []):
            steered, new = True, int(m.get("target", -1))
        if not steered and t >= phase_end:                    # own schedule: rest, box 1, rest, box 2, ...
            new = turn % len(freqs) if target < 0 else -1
            turn += target < 0
            phase_end = t + (look_s if new >= 0 else rest_s)
        if new != target:
            target = new if new in clips else -1
            clip, pos = clips[target][int(rng.integers(len(clips[target])))], 0
            print("look ->", "away" if target < 0 else f"box {target + 1} ({freqs[target]:g} Hz)", flush=True)
        if pos + chunk > len(clip):
            clip, pos = clips[target][int(rng.integers(len(clips[target])))], 0
        ts = (sent + np.arange(chunk)) / FS
        out = np.zeros((chunk, 17), dtype=np.float64)
        out[:, :8] = rng.normal(0, 6.0, (chunk, 8))                                   # front/centre: just noise
        for uni, (src, gain) in FROM_DATASET.items():
            out[:, uni] = gain * clip[pos:pos + chunk, src]
        drift += rng.normal(0, 0.4, 8)                                                # slow wander
        out[:, :8] += offset + drift + 12.0 * np.sin(2 * np.pi * cfg["notch_hz"] * ts)[:, None]
        if jolt:                                                                      # a movement: brief large swing
            out[:, :8] += (300.0 * np.exp(-(ts % jolt) / 0.3))[:, None]              # sudden jump that dies away
        if loose in UNICORN_EEG:                                                      # an electrode in the air
            out[:, UNICORN_EEG.index(loose)] += rng.normal(0, 400.0, chunk)
        out[:, 8:11] = [0.0, 0.0, 1.0] + rng.normal(0, 0.01, (chunk, 3))              # accelerometer (g)
        out[:, 11:14] = rng.normal(0, 0.5, (chunk, 3))                                # gyroscope (deg/s)
        out[:, 14], out[:, 15], out[:, 16] = 87.0, sent + np.arange(chunk), 1.0
        if not (drop_at and drop_at <= t < drop_at + 4.0):
            outlet.push_chunk(out.astype(np.float32).tolist())
        pos += chunk
        sent += chunk
        sleep = t0 + sent / FS - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
