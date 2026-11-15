"""Live headset helpers + a table-side health check for the g.tec Unicorn Hybrid Black.

The Unicorn is streamed into LSL by g.tec's "Unicorn LSL" app (part of Unicorn Suite). What we
expect from it, and what this file checks instead of assuming:
  - 250 samples a second
  - 17 channels: 8 EEG, 3 accelerometer, 3 gyroscope, battery, counter, validation
  - EEG order  Fz C3 Cz C4 Pz PO7 Oz PO8  (numbers 0-7). Only Pz PO7 Oz PO8 sit over the visual
    areas, so those are the ones the decoder uses (config.yaml: headset_channels).
  - the stream is usually named after the device serial ("UN-2021.05.36"), which is why the
    stream is found automatically instead of by a fixed name.
CONFIRM the channel order with g.tec at the booth; if it differs, set headset_channels.

    python acquisition/unicorn_check.py               # find the headset, show contact quality every second
    python acquisition/unicorn_check.py --list        # only list every LSL stream on this computer/network
    python acquisition/unicorn_check.py --stream NAME # use this stream name instead of searching
    python acquisition/unicorn_check.py --seconds 20  # stop after 20 s (default: until Ctrl+C)

Reading the table: "good" = looks like EEG. "noisy" = large swings: electrode not touching the
scalp, hair under it, or the person is moving / clenching. "flat" = no signal at all.
"""
import pathlib
import sys
import time
from collections import deque

import numpy as np
from scipy.signal import detrend

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import load_config  # noqa: E402
from processing.cca import make_filters, preprocess  # noqa: E402

UNICORN_EEG = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]
UNICORN_SSVEP = [4, 5, 6, 7]            # Pz PO7 Oz PO8
FLAT_UV, NOISY_UV = 0.5, 60.0           # band-passed (5-40 Hz) RMS in microvolts: below = flat, above = noisy


def looks_like_unicorn(info):
    name = info.name().lower()
    return name.startswith("un-") or "unicorn" in name


def list_streams(wait=2.0):
    from pylsl import resolve_streams
    return resolve_streams(wait)


def describe(info):
    return f"'{info.name()}'  type {info.type() or '-'}  {info.channel_count()} ch  {info.nominal_srate():g} Hz"


def find_stream(name="auto", timeout=30.0, quiet=False):
    """The LSL stream to read. name = exact stream name, or "auto": prefer a stream that looks
    like a Unicorn, otherwise the only stream with at least 8 channels at 200-300 Hz.
    Returns a pylsl StreamInfo, or None if nothing suitable appeared in time."""
    end, said = time.time() + timeout, False
    while True:
        found = list_streams(1.5)
        if name != "auto":
            hits = [s for s in found if s.name() == name]
        else:
            usable = [s for s in found if s.channel_count() >= 8 and 200 <= s.nominal_srate() <= 300]
            hits = [s for s in usable if looks_like_unicorn(s)] or (usable if len(usable) == 1 else [])
            if len(usable) > 1 and not hits and not quiet:
                print("Several EEG-like streams and none looks like a Unicorn. Pick one with --stream NAME:")
                for s in usable:
                    print("   ", describe(s))
                return None
        if hits:
            return hits[0]
        if time.time() >= end:
            return None
        if not said and not quiet:
            said = True
            print("Waiting for " + ("the headset's LSL stream" if name == "auto" else f"LSL stream '{name}'")
                  + " ... (start Unicorn LSL and press Start there)", flush=True)
            for s in found:
                print("    seen, not used:", describe(s), flush=True)


def contact_quality(x, filters):
    """x: (samples, channels) of raw EEG, about 2 s. -> (rms in microvolts, state per channel).
    States: good / noisy / flat (see the top of this file)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return [0.0] * x.shape[1], ["flat"] * x.shape[1]
    y = preprocess(detrend(x, axis=0), filters)
    edge = len(y) // 8                                     # the filter rings at both ends
    rms = y[edge:len(y) - edge].std(axis=0)
    states = ["flat" if r < FLAT_UV else "noisy" if r > NOISY_UV else "good" for r in rms]
    return [round(float(r), 1) for r in rms], states


def channel_names(chans, cfg):
    names = cfg.get("headset_channel_names", UNICORN_EEG)
    return [names[c] if c < len(names) else f"ch{c}" for c in chans]


def main():
    from pylsl import StreamInlet
    cfg = load_config()
    args = sys.argv[1:]
    if "--list" in args:
        found = list_streams(3.0)
        print(f"{len(found)} LSL stream(s) found:" if found else "No LSL streams found. Is Unicorn LSL started (and its Start pressed)?")
        for s in found:
            print("   ", describe(s), "  <- looks like the Unicorn" if looks_like_unicorn(s) else "")
        return 0
    name = args[args.index("--stream") + 1] if "--stream" in args[:-1] else cfg.get("headset_stream", "auto")
    seconds = float(args[args.index("--seconds") + 1]) if "--seconds" in args[:-1] else 0.0
    info = find_stream(name, timeout=30)
    if info is None:
        print("No headset stream found. Run with --list to see what is there.")
        return 1
    fs = info.nominal_srate() or cfg["fs"]
    n_ch = info.channel_count()
    print("Connected:", describe(info))
    if abs(fs - cfg["fs"]) > 0.5:
        print(f"NOTE: the stream runs at {fs:g} Hz, config.yaml says {cfg['fs']}. The classifier will use {fs:g}.")
    if n_ch != 17:
        print(f"NOTE: {n_ch} channels, a Unicorn normally sends 17 (8 EEG + motion + battery + counter + validation).")
    use = [c for c in cfg.get("headset_channels", UNICORN_SSVEP) if c < n_ch]
    show = list(range(min(8, n_ch)))
    names = channel_names(show, cfg)
    filters = make_filters(fs, cfg["bandpass"], cfg["notch_hz"])
    inlet = StreamInlet(info, max_chunklen=32)
    buf = deque(maxlen=int(2 * fs))
    print("The decoder uses:", " ".join(names[c] for c in use if c < len(names)), "(marked *)")
    print("RMS in microvolts after 5-40 Hz filtering. Aim for 'good' on every * channel.\n")
    print("  rate   " + "".join(f"{('*' if c in use else '') + n:>12s}" for c, n in zip(show, names)) + "   battery")
    t0, t_row, got, bad_rows, rows = time.time(), time.time(), 0, 0, 0
    try:
        while not seconds or time.time() - t0 < seconds:
            chunk, _ = inlet.pull_chunk(timeout=0.2)
            if chunk:
                arr = np.asarray(chunk, dtype=np.float64)
                buf.extend(arr)
                got += len(arr)
            now = time.time()
            if now - t_row >= 1.0:
                rate, got = got / (now - t_row), 0
                t_row = now
                if rate < 1:
                    print("  no data arriving - headset off, out of range, or Unicorn LSL stopped", flush=True)
                    buf.clear()
                    continue
                data = np.asarray(buf)
                rms, states = contact_quality(data[:, show], filters)
                battery = f"{data[-1, 14]:5.0f}%" if n_ch >= 15 else "    -"
                print(f"  {rate:5.0f}  " + "".join(f"{f'{r:.0f} {s}':>12s}" for r, s in zip(rms, states)) + "   " + battery,
                      flush=True)
                rows += 1
                bad_rows += any(states[c] != "good" for c in use if c < len(states)) or abs(rate - fs) > 0.1 * fs
    except KeyboardInterrupt:
        pass
    if rows:
        ok = bad_rows <= 0.2 * rows
        print(f"\n{'READY' if ok else 'NOT READY'}: {rows - bad_rows} of {rows} seconds had good contact on the decoder's "
              "electrodes and a full data rate." + ("" if ok else " Re-seat the electrodes marked noisy/flat (part the hair, "
                                                    "add gel), keep the person still, and run this again."))
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
