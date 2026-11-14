"""Shared contract for the Part B -> Part C hand-off.

IMPORTANT: the Part B speller owns the *actual* user confirmation event. This
class validates the data it receives; it cannot independently establish that a
human approved a message. Part B must construct this object ONLY in its
confirmation handler, after a user action, not when decoding/predicting text.
"""

from dataclasses import dataclass

ALLOWED_SIGNAL_SOURCES = frozenset({"live", "replay", "synthetic"})
ALLOWED_INPUT_MODES = frozenset({"eeg", "keyboard", "synthetic"})


@dataclass(frozen=True)
class ConfirmedMessage:
    text: str
    session_id: str
    input_mode: str
    signal_source: str
    approved_by_user: bool = False
    nickname: str = ""
    cloud_voice_consent: bool = False
    gallery_consent: bool = False

    def validate_for_output(self):
        if not self.approved_by_user:
            raise PermissionError("Message must be explicitly confirmed before Part C outputs")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Confirmed message must not be empty")
        if len(self.text) > 120 or any(ord(c) < 32 and c not in "\n\t" for c in self.text):
            raise ValueError("Message is too long or contains control characters")
        if not self.session_id or len(self.session_id) > 100:
            raise ValueError("Provide an opaque, non-identifying session ID")
        if self.input_mode not in ALLOWED_INPUT_MODES:
            raise ValueError(f"Invalid input mode: {self.input_mode}")
        if self.signal_source not in ALLOWED_SIGNAL_SOURCES:
            raise ValueError(f"Invalid EEG source: {self.signal_source}")
        if len(self.nickname) > 32:
            raise ValueError("Nickname must be 32 characters or fewer")
