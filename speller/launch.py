"""One command for the whole demo: starts the EEG source and the classifier in the background,
opens the speller, and stops everything again when the speller is closed.

    python speller/launch.py                      # recorded g.tec EEG, automatic demo (the default)
    python speller/launch.py --dataset both       # g.tec + the 35 benchmark people
    python speller/launch.py --dataset gtec --subject all
    python speller/launch.py --source keyboard    # no EEG at all: keys 1-4, boxes do not flash
    python speller/launch.py --source headset     # LIVE g.tec Unicorn: start Unicorn LSL first (see README)
    python speller/launch.py --source headset --check     # headset found? electrodes touching? scores arriving?
    python speller/launch.py --source headset --practice  # rehearse the live path: a fake headset playing recorded EEG
    python speller/launch.py --windowed           # (any other option is passed on to speller/ui.py)
    python speller/launch.py --check              # start the pipeline, confirm scores arrive, stop. No window.

Or double-click "Start BrainType.bat" in the project folder.
The background programs write to logs/launch_stream.log and logs/launch_classifier.log.
"""
import argparse
import pathlib
import socket
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import load_config  # noqa: E402

HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)       # Windows: no extra console windows


def start(name, args, log):
    """Run `python <args>` in the background, output to logs/<log>. Returns (name, process, log path)."""
    path = ROOT / "logs" / log
    path.parent.mkdir(exist_ok=True)
    out = open(path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                            creationflags=HIDDEN)
    return name, proc, path


def tail(path, n=8):
    try:
        return "\n".join("    " + line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return "    (no output)"


def wait_for(part, text, seconds):
    """Wait until `text` shows up in the program's log. False if it stopped or took too long."""
    name, proc, path = part
    end = time.time() + seconds
    while time.time() < end:
        if text in path.read_text(encoding="utf-8", errors="replace"):
            return True
        if proc.poll() is not None:
            break
        time.sleep(0.3)
    why = "stopped" if proc.poll() is not None else f"was not ready after {seconds} s"
    print(f"\nThe {name} {why}. Its last lines:\n{tail(path)}")
    return False


def port_free(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def scores_arrive(port, seconds=25):
    """--check: listen where the speller would and wait for the classifier's first scores.
    -> (scores arrived?, the latest electrode contact report or None)."""
    import json
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", port))
    s.settimeout(1.0)
    end, contact = time.time() + seconds, None
    try:
        while time.time() < end:
            try:
                m = json.loads(s.recv(65536))
            except (socket.timeout, ValueError):
                continue
            if m.get("type") == "quality":
                contact = m
            elif m.get("type") == "scores" and contact:
                return True, contact
        return False, contact
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description="Start the whole BrainType demo with one command.")
    ap.add_argument("--source", choices=["replay", "keyboard", "headset"], default="replay",
                    help="replay = recorded human EEG (default), keyboard = no EEG, headset = live LSL stream")
    ap.add_argument("--dataset", choices=["gtec", "both", "benchmark"], default="gtec")
    ap.add_argument("--subject", default=None, help="which recorded person (see acquisition/dataset_stream.py)")
    ap.add_argument("--check", action="store_true", help="test the pipeline without opening the speller")
    ap.add_argument("--practice", action="store_true",
                    help="with --source headset: no headset, a stand-in stream of recorded EEG in the Unicorn's format")
    ap.add_argument("--stream", default=None, help="headset: LSL stream name (default: found automatically)")
    ap.add_argument("--channels", default=None, help="headset: channels to decode from, e.g. 5,6,7")
    opts, ui_args = ap.parse_known_args()
    cfg = load_config()
    ports = cfg["ports"]
    parts = []

    if opts.source != "keyboard" and not all(port_free(p) for p in (ports["selection"], ports["sim_target"], ports["control"])):
        sys.exit("BrainType already seems to be running (its ports are in use).\n"
                 "Close the other speller / classifier / stream windows first, then start again.")
    try:
        if opts.source == "replay":
            stream = ["acquisition/dataset_stream.py", "--dataset", opts.dataset]
            if opts.subject:
                stream += ["--subject", opts.subject]
            print(f"1/3  Playing recorded EEG ({opts.dataset}) ...", flush=True)
            parts.append(start("EEG stream", stream, "launch_stream.log"))
            if not wait_for(parts[-1], "Streaming recorded EEG", 60):
                return 1
        if opts.source == "headset" and opts.practice:
            print("1/3  PRACTICE: no headset. Playing recorded EEG in the Unicorn's format ...", flush=True)
            parts.append(start("practice headset", ["acquisition/fake_unicorn.py"], "launch_stream.log"))
            if not wait_for(parts[-1], "PRACTICE headset streaming", 60):
                return 1
        elif opts.source == "headset":
            print("1/3  Looking for the headset (Unicorn LSL must be running with Start pressed) ...", flush=True)
        if opts.source != "keyboard":
            print("2/3  Starting the classifier ...", flush=True)
            clf = ["processing/classifier_node.py"]
            if opts.source == "headset":
                clf += ["--headset"] + (["--stream", opts.stream] if opts.stream else []) \
                       + (["--channels", opts.channels] if opts.channels else [])
            parts.append(start("classifier", clf, "launch_classifier.log"))
            if not wait_for(parts[-1], "Connected:", 50):
                if opts.source == "headset":
                    print("\nNo headset stream was found. Open Unicorn LSL, connect the headset, press Start, then try "
                          "again.\nSee every stream on this computer:  python acquisition/unicorn_check.py --list")
                return 1
            print("     " + tail(parts[-1][2], 1).strip(), flush=True)
        if opts.check:
            ok, contact = (True, None) if opts.source == "keyboard" else scores_arrive(ports["selection"])
            if contact:
                print("     Electrodes: " + ",  ".join(f"{n} {s} ({r:g} uV)" for n, s, r in
                                                        zip(contact["names"], contact["state"], contact["rms"])))
                print(f"     Data rate: {contact['rate']:g} samples a second")
                if opts.source == "headset" and any(s != "good" for s in contact["state"]):
                    ok = False
                    print("CHECK FAILED: an electrode has poor contact. Re-seat it, then run "
                          "python acquisition/unicorn_check.py to watch it live.")
                    return 1
            print("CHECK PASSED: the classifier is sending scores." if ok else
                  "CHECK FAILED: no scores arrived. See logs/launch_classifier.log")
            return 0 if ok else 1
        print("3/3  Opening the speller. Close it (ESC on the start screen) to stop everything.", flush=True)
        ui = ["speller/ui.py"] + {"replay": ["--input", "sim", "--signal", "replay"],
                                  "headset": ["--input", "eeg"],
                                  "keyboard": ["--input", "keyboard"]}[opts.source] + ui_args
        return subprocess.call([sys.executable, *ui], cwd=ROOT)
    except KeyboardInterrupt:
        return 0
    finally:
        for name, proc, _ in reversed(parts):               # never leave anything running behind
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if parts:
            print("Stopped:", ", ".join(name for name, _, _ in parts))


if __name__ == "__main__":
    sys.exit(main())
