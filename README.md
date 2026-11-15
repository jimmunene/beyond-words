# Beyond Words 🧠 — BrainType EEG Communication Prototype

**An experimental brain–computer communication experience:** detect SSVEP visual selections from EEG, compose short messages, create EEG-derived postcards, and optionally read confirmed text aloud or publish a postcard.

> **Project status:** Hack The Nest prototype. The full loop (EEG stream → decoder → speller → voice, phone alert, caregiver web app, postcard) runs end to end on **recorded human EEG** from public datasets, decoded live. A real-time path for the g.tec Unicorn headset is written and tested on a stand-in stream of recorded EEG, but **has not yet been run on a real headset**. The proposed **TrustLoop/IntentLock** safeguards and baseline-versus-adaptive evaluation are *planned work*, not implemented features in this starter.
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

## Quick start: one command

```bash
python speller/launch.py                      # simulation: recorded human EEG, decoded live
python speller/launch.py --source headset     # real time: g.tec Unicorn (start Unicorn LSL first)
python speller/launch.py --source keyboard    # no EEG: keys 1-4 pick a box
python speller/launch.py --check              # start the pipeline, confirm scores arrive, stop
```

The launcher starts the EEG source and the classifier in the background, opens the speller and stops
everything when the speller is closed. On Windows the same options are double-click files in the
project folder (`Start BrainType.bat`, `Start BrainType - Live Headset.bat`,
`Start BrainType - Keyboard.bat`, `Check Headset.bat`); they expect the environment in `venv`.

The simulation needs the g.tec SSVEP dataset in `data/ssvep/` (and, for `--dataset both`, the
Tsinghua benchmark via `python acquisition/prepare_benchmark.py`). Datasets are not in this repository.
Details, keys and every option: [speller/README.md](speller/README.md).

## What it does today

- **Patient mode:** one look says a need out loud (WATER, PAIN, COME HERE, EMERGENCY, more pages, or
  spell a sentence with word prediction). The voice can speak eleven languages.
- **Reaching a caregiver:** COME HERE and EMERGENCY ring a bell or siren and can alert a phone;
  every spoken message appears in a Base44 web app, where family can reply and the reply is spoken
  on the patient's screen. A "Live EEG" page shows the decoder's scores (never raw EEG).
- **Postcard mode:** spell a short message, get a postcard drawn from the EEG recorded while typing.
- **Decoder:** filter-bank CCA on a 3 s window, no per-person training; a pick needs a clear margin
  three times in a row; windows containing a large movement artifact cannot count towards a pick.
- **Live headset support:** finds the Unicorn's LSL stream by itself, decodes from Pz PO7 Oz PO8,
  reports electrode contact (good / noisy / flat) on screen, and pauses picking if the stream drops.
- **Gentler flicker:** 30% contrast around a dim grey, flicker only while a choice is being made,
  instant pause. This lowers, but does not remove, the risk for photosensitive people.

## Measured results (recorded EEG, offline)

| Electrodes, 3 s window, FBCCA | Mean accuracy | People at 90% or better |
|---|---|---|
| 8 over the visual cortex, Tsinghua benchmark (35 people) | 99.0% | 34 of 35 |
| PO7 Oz PO8 only (the ones a Unicorn has), same 35 people | 96.7% | 31 of 35 |
| 8 electrodes, g.tec dataset (2 people) | 91.5% (98.6% and 84.4%) | 1 of 2 |

Four targets, chance = 25%. Full grid of window lengths, methods and electrode sets:
[docs/offline_results.md](docs/offline_results.md), made by `python eval/offline_accuracy.py`.
These are public recordings, not people wearing our headset, and nothing has been tested with patients.

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
# 1. In Unicorn Suite: open Unicorn LSL, connect the headset, press Start
python acquisition/unicorn_check.py           # 2. electrode contact, once a second, until all "good"
python speller/launch.py --source headset     # 3. classifier + speller  (T on the typing screen = accuracy test)

python speller/launch.py --source headset --practice   # no headset: same path on a stand-in stream of recorded EEG
```

The channel order (Fz C3 Cz C4 Pz PO7 Oz PO8) and the stream name are assumptions to confirm with g.tec.

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
| Synthetic/replay acquisition | Synthetic stream, dataset replay (g.tec + Tsinghua benchmark), one-command launcher | Run and verify on other team machines |
| Live headset | Stream discovery, electrode contact check, drop-out handling, practice stream | Run on a real Unicorn; confirm channel order; tune thresholds per person |
| SSVEP CCA + selection | FBCCA classifier, selector, artifact rejection, offline accuracy grid | Evaluate on live recordings |
| Speller and word suggestions | Postcard + patient modes, word prediction, start screen, signal-strength display | User testing |
| Caregiver link | Phone alerts (ntfy; Twilio/Telegram untested), Base44 feed with replies, live decoder dashboard | Test Twilio/Telegram; access control on the public feed |
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
