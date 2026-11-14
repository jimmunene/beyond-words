# Beyond Words — Part C patch, 2026-11-14

This is a **patch**, not a complete replacement repository. It is designed to be extracted **into the root of the existing `beyond-words` checkout on branch `feature/voice-art`**.

## Files changed/added

- Updated: `voice/tts.py` — approval-gated local/ElevenLabs speech; no third-party TTS call without separate cloud consent.
- Updated: `art/postcard.py` — actual EEG required, clearly labeled synthetic demo mode, source/input labels, QR support.
- New: `experience/contracts.py`, `experience/pipeline.py`, `experience/__init__.py` — typed approval handoff and consent-controlled local gallery queue.
- New: `tools/part_c_demo.py` — standalone demo without headset/pygame.
- New: `tests/test_part_c.py` — 11 deterministic tests.
- New: `docs/PART_C_IMPLEMENTATION.md` — complete integration details with Part B and D boundaries.
- New: `requirements-part-c.txt` — small dependency set.
- New: `.github/workflows/part-c-checks.yml` — Part C CI.
- Updated: `.github/workflows/python-checks.yml` — install additional libraries needed by Part C tests.

## Safe application

```bash
cd ~/Hacknest/beyond-words
# Make sure you are on your feature branch and have a clean working tree.
git status
git switch feature/voice-art
# Place ZIP in Downloads (or adjust the path):
unzip -o ~/Downloads/Beyond_Words_Part_C_Implementation.zip -d .
git status
```

Inspect the changes before committing:

```bash
git diff --stat
git diff -- art/postcard.py voice/tts.py
```

## Minimal test (macOS Python 3.12 recommended)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part-c.txt
python -m unittest discover -s tests -v
python tools/part_c_demo.py --synthetic --confirm-test --message "HI MOM"
```

`--confirm-test` is only a **test harness**; it does NOT demonstrate actual human consent. The synthetic EEG postcard is clearly labeled as a simulation. All postcard output is kept local unless a separate simulated gallery consent flag is passed, in which case files are merely **queued**, not sent to Base44.

## Commit and create PR

```bash
git add voice/tts.py art/postcard.py experience/ tools/part_c_demo.py tests/test_part_c.py docs/PART_C_IMPLEMENTATION.md requirements-part-c.txt .github/workflows/part-c-checks.yml .github/workflows/python-checks.yml PART_C_PATCH_README.md
git commit -m "feat(part-c): consent-gated voice and EEG postcard pipeline"
git push origin feature/voice-art
```

Open `https://github.com/jimmunene/beyond-words/compare/main...feature/voice-art` to create a draft PR.

### Coordination

Part B owns `speller/ui.py` and must supply real user confirmation to create `ConfirmedMessage`. Until this happens, legacy UI voice calls intentionally will not speak. Part D owns the Base44 publishing API; Part C emits a consent-labeled local queue and never claims remote publishing.
