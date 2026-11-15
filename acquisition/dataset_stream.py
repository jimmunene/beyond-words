"""Real recorded SSVEP EEG -> LSL stream, steered by the UI. No headset needed.

Run the UI with `--input sim --signal replay`: pressing keys 1-4 says which box you "look at",
and this script plays a REAL recording of a person looking at a light flickering at that box's
frequency. The classifier then has to decode real brain signals, noise and all. Between looks it
plays real EEG recorded while nothing was flickering.

Which recordings to use is chosen with --dataset:
  gtec       ONLY the g.tec SSVEP dataset (default): 2 people, 9 / 10 / 12 / 15 Hz, 256 Hz. data/ssvep/
  both       g.tec subject 1 plus the 35 people of the Tsinghua benchmark. Each new visitor in the
             UI gets the next person's recordings.
  benchmark  only the Tsinghua benchmark: 35 people, 8 to 15 Hz in 1 Hz steps, 250 Hz. data/benchmark/
             (get the files with: python acquisition/prepare_benchmark.py)
All use the same eight electrodes over the visual cortex: PO7 PO3 POz PO4 PO8 O1 Oz O2.

What is real: the EEG, the decoding, the errors. What is not: the EEG is not from the person at
the keyboard, and a key (not their eyes) chooses which recording plays. Say so when you demo it.

    python acquisition/dataset_stream.py                            # g.tec only, subject 1
    python acquisition/dataset_stream.py --dataset gtec --subject all   # g.tec only, both of its people in turn
    python acquisition/dataset_stream.py --dataset both             # g.tec + benchmark, a new person per visitor
    python acquisition/dataset_stream.py --dataset benchmark --subject 7
    python acquisition/dataset_stream.py --dataset both --check     # offline accuracy for every person, then exit
    python acquisition/dataset_stream.py --folder C:/path/to/files  # dataset stored somewhere else

config.yaml must use frequencies the dataset has, e.g.  freqs: [9.0, 10.0, 12.0, 15.0]  (both have these).
eeg_channels [5, 6, 7] = O1 Oz O2 in both.
"""
import pathlib
import sys
import time

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import ROOT, UDPReceiver, load_config  # noqa: E402

SKIP_S = 0.3            # the brain needs a moment to lock on after a light comes on
FADE_S = 0.05           # short fade where two recordings are joined, so there is no jump


def option(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default


def load_gtec(folder, subject):
    """-> (fs, {frequency: [trials (n, 8) in microvolts]}, [resting pieces])."""
    fs = 256
    pattern = "subject_*_fvep_led_training_*.mat" if subject == "all" else f"subject_{subject}_fvep_led_training_*.mat"
    files = sorted(pathlib.Path(folder).glob(pattern))
    if not files:
        sys.exit(f"No g.tec files ({pattern}) in {folder}. Copy the .mat files there or pass --folder.")
    trials, rest = {f: [] for f in (9.0, 10.0, 12.0, 15.0)}, []
    for path in files:
        y = sio.loadmat(str(path))["y"]                 # rows: time, 8 x EEG, trigger, LDA output
        eeg, trig = y[1:9].T, y[9]
        edges = np.flatnonzero(np.diff(trig) != 0) + 1
        for s, e in zip(np.r_[0, edges], np.r_[edges, len(trig)]):
            value = float(trig[s])
            if value in trials:
                trials[value].append(eeg[s + int(SKIP_S * fs):e])
            elif value == 0 and e - s > fs:
                rest.append(eeg[s:e])
    return fs, trials, rest


def load_benchmark(folder, subject):
    """Tsinghua benchmark subject, as saved by prepare_benchmark.py."""
    path = pathlib.Path(folder) / f"S{int(subject):02d}.npz"
    if not path.exists():
        sys.exit(f"{path} not found. Run: python acquisition/prepare_benchmark.py {int(subject)}")
    z = np.load(path)
    fs, eeg = int(z["fs"]), z["eeg"].astype(float)      # (frequency, block, 1500 samples, 8 channels)
    on, off = int(0.5 * fs), int(5.5 * fs)              # each 6 s epoch: 0.5 s cue, 5 s flicker, 0.5 s blank
    trials = {float(f): [block[on + int(SKIP_S * fs):off] for block in eeg[i]] for i, f in enumerate(z["freqs"])}
    # no-flicker EEG: the half seconds before and after each flicker, joined into 8-second pieces
    quiet = [seg - seg.mean(axis=0) for blocks in eeg for block in blocks for seg in (block[:on], block[off:])]
    rest = [np.concatenate(quiet[i:i + 16]) for i in range(0, len(quiet) - 15, 16)]
    return fs, trials, rest


def prepare(segment, src_fs, fs):
    """Resample to the pipeline's rate, remove the offset, fade the ends."""
    x = resample_poly(segment, int(fs), int(src_fs), axis=0) if fs != src_fs else np.array(segment, dtype=float)
    x = x - x.mean(axis=0)
    n = max(1, int(FADE_S * fs))
    ramp = np.linspace(0.0, 1.0, n)[:, None]
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x.astype(np.float32)


def check(trials, src_fs, cfg, freqs):
    """How often does OUR classifier pick the right light, on windows of the configured length?"""
    from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess
    n = int(cfg["window_s"] * src_fs)
    filt = make_filters(src_fs, cfg["bandpass"], cfg["notch_hz"])
    clf = (FBCCAClassifier if cfg["use_fbcca"] else CCAClassifier)(src_fs, freqs, n, cfg["harmonics"])
    ok = total = 0
    for f in freqs:
        right = count = 0
        for seg in trials[f]:
            for a in range(0, len(seg) - n + 1, int(src_fs) // 2):
                scores = clf.scores(preprocess(seg[a:a + n][:, cfg["eeg_channels"]], filt))
                right += freqs[int(np.argmax(scores))] == f
                count += 1
        print(f"  {f:g} Hz: {100 * right / max(count, 1):5.1f}%  ({count} windows)")
        ok, total = ok + right, total + count
    kind = "FBCCA" if cfg["use_fbcca"] else "CCA"
    print(f"  overall: {100 * ok / max(total, 1):.1f}% with {kind}, {cfg['window_s']:g} s windows, "
          f"channels {cfg['eeg_channels']} (chance = {100 / len(freqs):.0f}%)")


def people_for(dataset, subject):
    """The list of (label, loader) this run may play, in order."""
    gtec_dir, bench_dir = ROOT / "data" / "ssvep", ROOT / "data" / "benchmark"
    folder = option("--folder", None)
    people = []
    if dataset in ("gtec", "both"):
        chosen = ["1", "2"] if (dataset == "gtec" and subject == "all") else [subject if dataset == "gtec" else "1"]
        people += [(f"g.tec subject {n}", lambda n=n: load_gtec(folder or gtec_dir, n)) for n in chosen]
    if dataset in ("benchmark", "both"):
        have = sorted(int(f.stem[1:]) for f in pathlib.Path(folder if (folder and dataset == "benchmark") else bench_dir).glob("S??.npz"))
        if dataset == "benchmark" and subject != "all":
            have = [int(subject)]
        if not have:
            sys.exit("No benchmark files in data/benchmark. Run: python acquisition/prepare_benchmark.py")
        people += [(f"benchmark subject {n}", lambda n=n: load_benchmark(folder if (folder and dataset == "benchmark") else bench_dir, n))
                   for n in have]
    return people


def main():
    cfg = load_config()
    dataset, subject = option("--dataset", "gtec"), option("--subject", "1")
    if dataset not in ("gtec", "benchmark", "both"):
        sys.exit("--dataset must be gtec (g.tec recordings only), both (g.tec + benchmark) or benchmark")
    people = people_for(dataset, subject)
    freqs = [float(f) for f in cfg["freqs"]]
    print(f"Dataset: {dataset} -> {len(people)} recorded " + ("person" if len(people) == 1 else "people")
          + (" (g.tec recordings only)" if dataset == "gtec" else ""))

    if "--check" in sys.argv:
        for label, load in people:
            src_fs, trials, _ = load()
            print(label)
            check(trials, src_fs, cfg, freqs)
        return

    from pylsl import StreamInfo, StreamOutlet
    fs = cfg["fs"]
    outlet = StreamOutlet(StreamInfo(cfg["stream_name"], "EEG", 8, fs, "float32", f"{dataset}_ssvep"))
    ctrl = UDPReceiver(cfg["ports"]["sim_target"])
    rng = np.random.default_rng()

    def use(index):
        label, load = people[index]
        src_fs, trials, rest = load()
        if any(f not in trials for f in freqs):
            sys.exit(f"config.yaml has freqs {freqs}, but {label} was recorded at {sorted(trials)} Hz.\n"
                     "Set for example   freqs: [9.0, 10.0, 12.0, 15.0]   in config.yaml and start again.")
        clips = {i: [prepare(t, src_fs, fs) for t in trials[f]] for i, f in enumerate(freqs)}
        clips[-1] = [prepare(r, src_fs, fs) for r in rest]
        print(f"Now playing: {label}", flush=True)
        return clips

    who, first_visitor = 0, True
    clips = use(who)
    print(f"Streaming recorded EEG as '{cfg['stream_name']}' (8 ch @ {fs} Hz). "
          "Now run: python speller/ui.py --input sim --signal replay", flush=True)

    def pick(target, last=-1):
        options = [k for k in range(len(clips[target])) if k != last] or [0]
        k = int(rng.choice(options))
        return clips[target][k], k

    target, chunk, sent, t0 = -1, 10, 0, time.perf_counter()
    clip, k = pick(target)
    pos = 0
    while True:
        for m in ctrl.poll():
            if m.get("person") == "next" and first_visitor:
                first_visitor = False                           # the first visitor gets the first person
            elif m.get("person") == "next" and len(people) > 1:  # each later visitor: the next recorded person
                who = (who + 1) % len(people)
                clips = use(who)
                target = None                                   # forces a fresh recording below
                sent, t0 = 0, time.perf_counter()               # (loading took a moment: restart the clock)
            new = int(m.get("target", -1))
            new = new if new in clips else -1
            if new != target:                                   # the "gaze" moved: start a recording of that light
                target = new
                clip, k = pick(target)
                pos = 0
                print("look ->", "rest" if target < 0 else f"box {target + 1} ({freqs[target]:g} Hz), trial {k + 1}",
                      flush=True)
        if pos + chunk > len(clip):                             # recording finished: continue with another one
            clip, k = pick(target, k)
            pos = 0
        outlet.push_chunk(clip[pos:pos + chunk].tolist())
        pos += chunk
        sent += chunk
        sleep = t0 + sent / fs - time.perf_counter()            # real-time pace without drift
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    main()
