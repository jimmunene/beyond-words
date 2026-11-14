# Beyond Words 🧠 — BrainType EEG Communication Prototype

**An experimental brain–computer communication experience:** detect SSVEP visual selections from EEG, compose short messages, create EEG-derived postcards, and optionally read confirmed text aloud or publish a postcard.

> **Project status:** Hack The Nest starter code. Synthetic EEG and replay entry points are present; real g.tec Unicorn acquisition and full end-to-end behavior still require on-device verification. The proposed **TrustLoop/IntentLock** safeguards and baseline-versus-adaptive evaluation are *planned work*, not implemented features in this starter.
>
> **Safety:** Not a medical device, diagnostic system, or clinically validated assistive communication product. SSVEP uses flickering visual stimuli, which may be unsafe for photosensitive participants. Run live demonstrations only with suitable screening, informed consent, a mentor-approved setup, and a readily accessible stop option. Do not collect or publish raw EEG without explicit consent.

## Quick links

- [Implementation guide](IMPLEMENTATION_GUIDE.md) — original starter's sequence, processes, UI controls, and troubleshooting (time references are event-specific)
- [Original project plan](PLAN.md) — feature vision, UI flows, and role breakdown
- [GitHub/project setup](docs/GITHUB_SETUP.md) — create repo `beyond-words`, push, branches, collaboration
- [Current build status & next steps](docs/STATUS_AND_ROADMAP.md)
- [Contributing](CONTRIBUTING.md) · [Data/privacy](SECURITY.md)

## Technology

Python, numpy/scipy, Lab Streaming Layer (`pylsl`), CCA-based SSVEP decoding, pygame UI, Matplotlib/Pillow art, optional Gemini suggestions, optional ElevenLabs speech, and a Base44 gallery upload queue.

## Install

Recommended: **Python 3.11** with a headset-compatible OS and LSL installation.

```bash
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> `pylsl` requires a compatible liblsl native library on some systems. Pygame fullscreen rendering may not work on a headless machine. Review `config.yaml` to verify the actual EEG channel order, monitor refresh rate, and approved stimulus setup.

## First run: keyboard-only UI

```bash
python speller/ui.py --input keyboard --windowed
```

Keyboard 1–4 selects a box **without EEG**. It is useful for interface development, but must never be described as brain-controlled typing. The implementation guide documents the remaining operator keys.

## Full synthetic signal pipeline (three terminals)

```bash
# Terminal 1: synthetic EEG into LSL
python acquisition/synthetic_stream.py

# Terminal 2: decoder from LSL to local UDP events
python processing/classifier_node.py

# Terminal 3: visual speller, whose 1–4 keys select a simulated gaze target
python speller/ui.py --input sim --windowed
```

Synthetic or replay-mode data is not equivalent to live human EEG. Use the live headset only after verifying its stream and channels with a mentor.

## Live acquisition (requires hardware)

```bash
# Terminal 1: use g.tec's approved Unicorn LSL acquisition tool
# Terminal 2
python processing/classifier_node.py
# Terminal 3
python speller/ui.py --input eeg --windowed
```

The Unicorn LSL tool, hardware setup, and actual channels are **not packaged in this repository**. Frequencies in `config.yaml` are demo defaults and must not be used publicly without a safe, approved setup.

## Tests and source validation

```bash
python -m compileall -q .
python -m unittest discover -s tests -v
```

GitHub Actions runs Python syntax checks and focused unit tests without invoking the graphical UI or requiring EEG hardware.

## Configuration and APIs

- `config.yaml` holds the stream name, sampling rate, selection thresholds, display assumptions, and `use_llm`/`use_tts` flags.
- Keep API credentials in environment variables. See `.env.example` for names; the app does not automatically load `.env`.
- Base44 integration currently has a **manual upload queue**. `upload_via_api()` is a TODO and is intentionally not advertised as connected.
- External datasets and private EEG recordings belong in your own local storage, **not GitHub**.

## What's built versus what's next

| Area | Starter contains | Remaining validation / implementation |
| --- | --- | --- |
| Synthetic/replay acquisition | Source modules | Run and verify on team machines |
| SSVEP CCA + selection | Classifier and selector | Calibrate, evaluate on actual recordings |
| Speller and word suggestions | Keyboard/EEG/sim UI paths | Test full UI and user experience |
| Postcards | EEG-to-image generator | Test artifacts and consent workflow |
| Voice | ElevenLabs/pyttsx3 fallback code | Test credentials, playback, and consent gate |
| Base44 | Manual queue | Connect a real gallery after provider validation |
| TrustLoop / IntentLock | Not in original starter | Implement explicit message approval and reject/wait policies |
| TrustBench | Metric helpers | Create held-out benchmarks and truthful comparative results |

## Team handoff documents

- [Team assignments and workflow](TEAM_TASKS.md) — four owners, branch conventions, interface contracts and priorities.
- [Dataset links and rationale](docs/DATASETS.md) — Kalunga2016, Dong2023, optional BETA, and consented live Unicorn EEG.
- [GitHub initialization](docs/GITHUB_SETUP.md) — how to create `jimmunene/beyond-words` and push safely.
- [Current status and roadmap](docs/STATUS_AND_ROADMAP.md) — verified functionality versus incomplete TrustLoop work.

## Team structure

1. **Signal & data:** LSL/unicorn acquisition, CCA, filtering, dataset replay.
2. **Speller & TrustLoop:** UI state machine, suggestion consent, safe message confirmation.
3. **Experience:** speech, brainwave art, safety/consent UX.
4. **Integration/QA:** Base44, testing, CI, documentation, judging/demo operations.

## Responsible claims

Never describe keyboard mode, a synthetic EEG stream, or prerecorded replay as live brain-controlled output. Report measured accuracy only for the data and evaluation protocol actually used. The prototype is not a substitute for established augmentative and alternative communication devices.

## License

**No license has been selected yet.** Until the team chooses one, the repository is not automatically licensed for third-party reuse. Before making the repository public, confirm that the original starter's authors agree to the license and that external data/code rights permit publication.
