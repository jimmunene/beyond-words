# Beyond Words — Part C implementation (Voice + EEG Art + Postcards)

Owner: Part C on `feature/voice-art`. This document describes the delivered code and what remains for other owners.

## Status

- [x] Secure interface object (`experience/contracts.py`) for Part B to populate **only after a confirmed selection**
- [x] TTS that defaults to **no speech without approval**; optional ElevenLabs with explicit cloud consent, offline macOS `say` fallback
- [x] Cache for successfully generated ElevenLabs audio (`voice/cache/`, gitignored)
- [x] EEG art from actual provided samples (`art/postcard.py`); invalid/empty samples raise an error instead of being replaced with fabricated data
- [x] Explicitly marked synthetic postcard mode for demo and tests
- [x] Human-readable source mode on every postcard (live/replay/synthetic and EEG/keyboard/simulated)
- [x] Separate gallery-consent queue with image and JSON metadata (`experience/pipeline.py`)
- [x] Standalone CLI, automated unit tests and GitHub CI (updated core job to install Part C test dependencies)
- [ ] Live Unicorn headset tested by Part A
- [ ] Real confirmation callback connected by Part B in `speller/ui.py`
- [ ] Base44 authenticated publishing configured by Part D
- [ ] Real ElevenLabs key and voice configured/tested by the team; not required for local demo

## 1. Minimal setup — no pygame required

On macOS with Python 3.12:

```bash
brew install python@3.12
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part-c.txt
python -m unittest tests.test_part_c -v
```

On Apple Silicon the executable might be `/opt/homebrew/bin/python3.12`.

If running the *whole speller*, install the repository's full `requirements.txt` with Python 3.12; Python 3.14 can trigger a pygame-from-source build needing SDL headers. To use macOS offline speech, `pyttsx3` is unnecessary: the project invokes the built-in `say` command.

## 2. Run artwork + metadata (clearly labeled synthetic mode)

```bash
python tools/part_c_demo.py --synthetic --confirm-test --message "HI MOM" --nickname "DEMO"
```

`--confirm-test` is an **operator-only simulation** of confirmation; never claim it demonstrates genuine brain-selected approval. The image metadata states the signal is synthetic. No voice or public gallery is used by default.

To test the **manual gallery queue**, add `--gallery-consent-test`. This simulates *separate* consent and generates files under `postcards/to_upload/`. It does **NOT** publish to Base44.

For a real recorded `.npy` file with shape `(samples, channels)`:

```bash
python tools/part_c_demo.py --eeg-file sessions/approved_session.npy \
  --signal-source replay --input-mode keyboard \
  --confirm-test --message "HI MOM"
```

Only declare `--input-mode eeg` when EEG selections genuinely chose the text.

To test voice on macOS, add `--voice`. The default is local `say`. To allow ElevenLabs requests, first separately obtain consent, configure `ELEVENLABS_API_KEY`, choose an available voice ID, install `elevenlabs`, and pass `--cloud-voice-consent-test --voice-id ACTUAL_ID --voice`. Never commit API keys.

## 3. Exact Part B -> Part C contract

```python
from experience.contracts import ConfirmedMessage
from experience.pipeline import create_experience

# This code belongs in Part B's user-confirmation handler, NOT the classifier:
confirmed = ConfirmedMessage(
    text=speller.text,               # the final user-confirmed text
    session_id=session_id,           # non-identifying ID
    input_mode="eeg",               # or keyboard/synthetic (honest labeling)
    signal_source="live",           # or replay/synthetic
    approved_by_user=True,          # set only in a real, explicit user action
    nickname=nickname,
    cloud_voice_consent=False,       # separate opt-in
    gallery_consent=False,           # separate opt-in
)
result = create_experience(
    confirmed, eeg_array, fs, voice_id=voice_id, enable_speech=True
)
```

**Important integration blocker:** The existing `speller/ui.py` still invokes `speak()` immediately when a patient phrase is selected and again at postcard completion, with no confirmation screen. The updated `speak()` intentionally fails closed unless called with `approved=True` or `speak_confirmed(confirmed,...)`. Part B **must modify** the UI to capture a real user confirmation and call `create_experience()`. Part C deliberately does not edit files owned by Part B. This means the original end-to-end UI is not yet fully integrated.

## 4. What Part D receives

If and only if `gallery_consent=True`, the output is queued as:

```
postcards/to_upload/bw_<uuid>.png
postcards/to_upload/bw_<uuid>.json
```

The sidecar has `message`, `nickname`, `signal_source`, `input_mode`, `user_confirmed_message`, `gallery_consent`, and a nonidentifying session ID. **No raw EEG is uploaded**. The queue is gitignored. Part D uploads it manually or implements authenticated Base44 ingest. Never treat queue creation as a published gallery item.

## 5. Safety and privacy notes

- The `ConfirmedMessage` object does not cryptographically verify consent. Part B must create it after a real confirmation interaction.
- Sending text to ElevenLabs transmits it to a third-party service; ask for separate consent. No network request is made by Part C when cloud voice consent is false.
- An EEG-derived art piece is not an emotion detector. Palette choices are visual mappings of frequency-band summaries, not psychological diagnoses.
- Do not treat synthetic EEG data or keyboard input as live brain-controlled spelling. Labels are embedded in the image and metadata.
- Real participant EEG is private and should not be committed or published.

## 6. Verification

```bash
python -m unittest tests.test_part_c -v
python -m unittest discover -s tests -v
python tools/part_c_demo.py --synthetic --confirm-test --message "HELLO"
```

An optional real ElevenLabs network test and full pygame/LSL test still require your own account and hardware respectively.
