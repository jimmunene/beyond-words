"""Part C workflow: a confirmed message becomes voice and EEG-derived artwork.

Part B supplies a ConfirmedMessage created only by a genuine confirmation action.
Part C never infers approval from an EEG classification or AI suggestion.
Part D owns the eventual Base44 API; this module only creates a local queue.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import pathlib
import shutil

from art.postcard import make_postcard
from experience.contracts import ConfirmedMessage
from voice.tts import speak_confirmed

DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "postcards"


@dataclass(frozen=True)
class PartCResult:
    postcard_path: str
    metadata_path: str
    gallery_status: str
    speech_started: bool


def create_experience(
    message: ConfirmedMessage,
    eeg,
    sample_rate_hz: float,
    *,
    output_dir=DEFAULT_OUTPUT,
    queue_dir=None,
    voice_id="",
    enable_speech=False,
    gallery_url=None,
) -> PartCResult:
    """Process only user-confirmed text; export locally by default.

    This is a workflow gate, not an authentication or consent proof: the caller
    (Part B) must obtain and record actual confirmation and separate consent.
    """
    if not isinstance(message, ConfirmedMessage):
        raise TypeError("Expected ConfirmedMessage from the Part B UI")
    message.validate_for_output()

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Don't put private nicknames in filesystem names or use raw EEG in metadata.
    image_path = pathlib.Path(make_postcard(
        eeg,
        sample_rate_hz,
        message.text,
        message.nickname,
        out_dir=output_dir,
        signal_source=message.signal_source,
        input_mode=message.input_mode,
        gallery_url=gallery_url if message.gallery_consent else None,
    ))
    metadata = {
        "message": message.text,
        "nickname": message.nickname,
        "session_id": message.session_id,
        "signal_source": message.signal_source,
        "input_mode": message.input_mode,
        "user_confirmed_message": True,
        "gallery_consent": message.gallery_consent,
        "cloud_voice_consent": message.cloud_voice_consent,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_claim": (
            "Synthetic EEG demonstration, not live brain typing"
            if message.signal_source == "synthetic"
            else "EEG recording replay" if message.signal_source == "replay"
            else "Live EEG recording"
        ),
    }
    meta_path = image_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "local only — no gallery consent"
    if message.gallery_consent:
        # A local upload QUEUE, not a claim that Base44 accepted the postcard.
        queue = pathlib.Path(queue_dir) if queue_dir is not None else DEFAULT_OUTPUT / "to_upload"
        queue.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image_path, queue / image_path.name)
        shutil.copyfile(meta_path, queue / meta_path.name)
        status = "queued for manual Base44 upload (NOT published)"

    started = bool(speak_confirmed(message, voice_id, enabled=True)) if enable_speech else False
    return PartCResult(str(image_path), str(meta_path), status, started)
