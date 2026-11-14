"""Run an isolated Part C demo without pygame or a live headset.

Synthetic mode is always clearly labeled. CLI approval is a LOCAL testing
simulation, not evidence that an actual EEG user confirmed the message.

python tools/part_c_demo.py --synthetic --confirm-test --message "HI MOM"
"""

import argparse
import pathlib
import sys
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experience.contracts import ConfirmedMessage  # noqa: E402
from experience.pipeline import create_experience  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Beyond Words Part C verified-output demo")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true", help="Clearly labeled synthetic EEG samples")
    source.add_argument("--eeg-file", type=pathlib.Path, help="Local .npy array: samples by EEG channels")
    p.add_argument("--signal-source", choices=["live", "replay"], default="replay")
    p.add_argument("--sample-rate", type=int, default=250)
    p.add_argument("--message", default="HI MOM")
    p.add_argument("--nickname", default="")
    p.add_argument("--input-mode", choices=["eeg", "keyboard", "synthetic"], default=None)
    p.add_argument("--confirm-test", action="store_true", help="Simulate approval for local integration TEST only")
    p.add_argument("--gallery-consent-test", action="store_true", help="Simulate separate public posting consent; queues locally")
    p.add_argument("--voice", action="store_true", help="Speak approved text, using local offline voice by default")
    p.add_argument("--cloud-voice-consent-test", action="store_true", help="Allow ElevenLabs for this TEST message")
    p.add_argument("--voice-id", default="")
    p.add_argument("--output-dir", type=pathlib.Path, default=None)
    args = p.parse_args()

    if args.synthetic:
        eeg = np.random.default_rng(42).normal(size=(args.sample_rate * 4, 8))
        signal_source = "synthetic"
        input_mode = args.input_mode or "synthetic"
    else:
        eeg = np.load(args.eeg_file, allow_pickle=False)
        signal_source = args.signal_source
        input_mode = args.input_mode or "keyboard"  # never imply EEG-typed by default

    message = ConfirmedMessage(
        text=args.message,
        session_id="local-test-" + uuid.uuid4().hex[:12],
        input_mode=input_mode,
        signal_source=signal_source,
        approved_by_user=args.confirm_test,
        nickname=args.nickname,
        gallery_consent=args.gallery_consent_test,
        cloud_voice_consent=args.cloud_voice_consent_test,
    )
    print("NOTE: --confirm-test simulates a confirmation; it is not real participant approval.")
    result = create_experience(
        message, eeg, args.sample_rate, enable_speech=args.voice,
        voice_id=args.voice_id,
        **({"output_dir": args.output_dir} if args.output_dir else {}),
    )
    print("Image:", result.postcard_path)
    print("Metadata:", result.metadata_path)
    print("Gallery:", result.gallery_status)
    print("Voice worker started:", result.speech_started)


if __name__ == "__main__":
    main()
