# Beyond Words (BrainType TrustLoop): Team Tasks & GitHub Workflow

Each person **owns specific folders**. Prefer editing the folders you own to reduce conflicts; coordinate any shared changes.
The starter contains synthetic/replay EEG paths and a UI, but the full graphical demo still needs verification on the team laptop. Your job is to **finish, harden, test, and improve your part** using the tasks below.

| Person | Role | Owns (only you edit these) | Git branch |
|---|---|---|---|
| **A** | Signal lead | `acquisition/` · `processing/` · `data/convert_gtec.py` · `eval/offline_accuracy.py` | `feature/a-signal` |
| **B** | Speller UI + prediction | `speller/` · `data/custom_words.txt` | `feature/b-speller` |
| **C** | Art + voice | `art/` · `voice/` | `feature/c-art-voice` |
| **D** | Gallery + demo + metrics | `upload/` · `eval/metrics.py` · `eval/report.py` · `tools/` · `docs/` | `feature/d-gallery-demo` |

**Shared files** (`config.yaml`, `common.py`, `requirements.txt`): announce in team chat before changing, change only your own section, commit them separately.

---

## GitHub workflow (everyone, read once)

### One-time setup (repo owner prepares the empty repo; see docs/GITHUB_SETUP.md)
1. Create GitHub repo `beyond-words` under `jimmunene` (or a designated team organization). Keep private until the team confirms judging visibility requirements and dataset/code rights. Push the starter to `main`.
2. Add the other three as collaborators (Settings → Collaborators).
3. Everyone:
```bash
git clone https://github.com/jimmunene/beyond-words.git
cd beyond-words
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
git checkout -b <your-branch>                        # e.g. feature/a-signal
```

### Every time you work
```bash
git fetch origin                     # get everyone's latest merged work
git merge origin/main                # while on your feature branch, sync with main
# ... code ...
python speller/ui.py --input keyboard --windowed    # quick smoke test, must still run
git add <your files only>
git commit -m "A: add offline accuracy grid search"  # start message with your letter
git push origin <your-branch>
```

### Merging into `main` (at check-ins: 9:30 PM, 1:30 AM, 6:30 AM)
1. Open a Pull Request: your branch → `main`.
2. **Before merging, run the full loop** (fake EEG, 3 terminals, Implementation Guide Step 0.2): nickname → type "HI" → DONE → postcard.
3. The designated integration lead merges after review. If it breaks `main`, revert first and fix after.

### Never commit
- **API keys.** Set them as environment variables in your terminal:
  `export ELEVENLABS_API_KEY=...` / `export GEMINI_API_KEY=...` (Windows: `set NAME=value`)
- Visitor data: `sessions/*.npy`, `logs/*.csv`, `postcards/*.png` (already in `.gitignore`)
- Big datasets (share them via USB stick or Google Drive instead)

### Interfaces: do NOT change without telling everyone
| Interface | Format |
|---|---|
| Classifier → UI (UDP 5005) | `{"type":"scores","scores":[4 floats]}`, `{"type":"select","box":0-3,"confidence":float}`, `{"type":"session_saved","path":str}` |
| UI → classifier (UDP 5007) | `{"cmd":"start_session"\|"stop_session"\|"pause"\|"resume"}` (stop_session has `"path"`) |
| UI → fake EEG (UDP 5006) | `{"target":0-3 or -1}` |
| Postcard | `make_postcard(eeg, fs, message, nickname) -> png_path` |
| Voice | `speak(text, voice_id, enabled)` (never blocks, never crashes) |
| Upload | `upload(png_path, message, nickname, typing_seconds, consent) -> status_str` |
| Prediction | `Predictor.suggest(typed) -> WORD or None` |

---

## Person A: Signal lead

**Goal:** the headset reliably tells the speller which box the person is looking at, and we have numbers to prove it.

**Already exists:** `synthetic_stream.py`, `replay_stream.py`, `cca.py` (CCA + FBCCA), `selector.py`, `classifier_node.py`.

### What you need to program

**A1. `data/convert_gtec.py`: dataset converter** *(6:30–8:00)*
- Load the g.tec SSVEP dataset (whatever format it comes in: `.mat`, `.csv`, `.hdf5`, …)
- Save `data/gtec_ssvep.npy`, shape `(n_samples, n_channels)` in µV
- Save `data/gtec_trials.csv`: `start_sample, freq_hz` per trial
- ✅ Done when: shapes and trial count print correctly

**A2. `eval/offline_accuracy.py`: find the best settings** *(8:00–9:30)*
- For every trial: cut a window → `preprocess` → classifier → argmax → compare to true frequency
- Grid search: window `1.0 / 1.5 / 2.0 / 3.0 s` × `CCA / FBCCA` × channel sets `[Oz]`, `[PO7,Oz,PO8]`, `[Pz,PO7,Oz,PO8]`, all 8
- Print a table of accuracy and ITR (use `itr_bits_per_min` from `eval/metrics.py`)
- Put the winning settings in `config.yaml`
- ✅ Done when: the table is saved to `docs/offline_results.md` (goes on the Devpost)

**A3. `acquisition/unicorn_check.py`: headset health check** *(before 11:00)*
- Connect to the Unicorn LSL stream; print channel count + sample rate
- Every second print per-channel standard deviation and flag ⚠ for flat (< 1 µV) or noisy (> 100 µV) channels
- ✅ Done when: used at the table to check electrode contact in < 20 s

**A4. Live tuning** *(11:00 PM–1:30 AM)*: no new files; edit `config.yaml` following the tuning table in the Implementation Guide Step 3.5. Record each test in `docs/live_tuning_log.md` (person, settings, accuracy).

**A5. *(Stretch)* `processing/classifier_node.py` improvements**
- Ignore windows with huge artifacts (blinks/movement: amplitude > threshold), so they don't create false clicks
- Optional per-user margin: during the T test, set `margin_tau` from that person's score distribution

---

## Person B: Speller UI + word prediction (integration lead)

**Goal:** a speller that's easy to use, smooth and fast, with smart word suggestions.

**Already exists:** `state_machine.py` (postcard + patient mode), `ui.py`, `predictor.py`, `build_corpus.py`.

### What you need to program

**B1. Corpus** *(before 6:30)*: download Norvig `count_1w.txt` + `count_2w.txt` and the Vertanen AAC sentences (one sentence per line → `data/aac_sentences.txt`), run `python speller/build_corpus.py`. ✅ `python speller/predictor.py` gives sensible words.

**B2. `ui.py` polish** *(6:30–8:00)*
- FPS readout top-left (`self.clock.get_fps()`); turn red if below the refresh rate − 2
- Click sound on selection (`pygame.mixer.Sound`, a short `.wav` in `speller/assets/`)
- Bigger, clearer green flash on the selected box
- Layout check on the real external monitor
- ✅ Done when: FPS steady, labels readable from 60 cm

**B3. Quick-message lane in `state_machine.py`** *(8:00–9:30)*
- When nothing is typed yet, box 4 shows **"★ QUICK"** → a screen of preset postcards: `HI MOM`, `HELLO EMU`, `I LOVE YOU`, `MORE ▶` → `THANK YOU`, `HACK THE NEST`, `MIND = BLOWN`, `← BACK`
- Picking one fills the text and finishes the postcard (2 looks total). For long queues.
- ✅ Done when: a visitor can make a postcard in ~6 seconds

**B4. Integration** *(9:30–11:00)*: run the full loop with A (replay data), C (postcard + voice), D (upload). Fix glue bugs. Merge everything to `main`.

**B5. Gemini prediction** *(1:30–3:30)*: `pip install google-genai`, set `GEMINI_API_KEY`, `use_llm: true`, check the current model name and set `GEMINI_MODEL`. Improve the prompt in `predictor.py`. ✅ Wi-Fi off → local suggestions still work.

**B6. Voice picker hook** *(after C finishes C4)*: on the intro screen, key **V** cycles voices from `voice/voices.py`; pass the chosen voice ID into `speak(...)`.

---

## Person C: Art + voice

**Goal:** every visitor gets a beautiful, unique postcard, and the system speaks with a natural voice that never fails.

**Already exists:** `art/postcard.py`, `voice/tts.py` (cache → ElevenLabs → offline voice → text), `voice/precache.py`.

### What you need to program

**C1. Better art in `art/postcard.py`** *(6:30–8:30)*
- More palettes (5–6 moods), smoother gradients
- Draw the frequencies the visitor used as concentric rings
- Add typing time to the card ("typed in 47 s")
- Make sure text is readable on every palette
- ✅ Done when: 5 different EEG recordings give 5 clearly different, good-looking cards

**C2. Real EEG test** *(8:30–9:30)*: run `make_postcard` on slices of `data/gtec_ssvep.npy` (from A). Tune feature → color mapping so real EEG looks good.

**C3. Voice setup** *(9:30–11:00)*: ElevenLabs key, choose a warm voice, put its ID in `config.yaml`, `use_tts: true`, run `python voice/precache.py`. ✅ Wi-Fi off → WATER still speaks.

**C4. `voice/voices.py`: voice picker for the ElevenLabs track** *(3:30–6:30)*
```python
VOICES = [("Warm", "<voice_id>"), ("Bright", "<voice_id>"), ("Calm", "<voice_id>")]
def get_voice(i): return VOICES[i % len(VOICES)]
```
Pre-cache patient phrases for each voice. Tell B when ready (B6).

**C5. Backup video clips** *(during live testing + visitor table)*: phone clips of live typing, postcard appearing, voice, patient mode. Hand to D.

---

## Person D: Gallery + demo + metrics

**Goal:** postcards reach the public gallery, we have hard numbers, and the submission and pitch are perfect.

**Already exists:** `upload/base44_upload.py` (manual-upload queue), `eval/metrics.py`.

### What you need to build

**D1. Base44 gallery** *(before 6:30 → 9:30)*: paste the prompt from `PLAN.md` Section 9, refine on a phone. Add 3 test postcards.

**D2. `upload/base44_upload.py`** *(by 9:30)*: if Base44 supports API/webhook uploads, implement `upload_via_api()` and set `USE_API = True`. If not, keep the queue and upload manually from `postcards/to_upload/` (each image has a `.json` with message/nickname/time).

**D3. `tools/make_qr.py`** *(by 9:30)*: reads `gallery_url` from `config.yaml` → saves `docs/qr.png`. Print it / put it on the table.

**D4. `eval/report.py`** *(11:00 PM–1:30 AM)*: reads `logs/selections.csv` → makes `docs/results.png` (accuracy bar per person, chars/min, ITR) for the demo slide and Devpost.

**D5. Domain** *(1:30–3:30)*: redeem the MLH GoDaddy Registry code, point the domain at the gallery, update `config.yaml` + QR.

**D6. Docs + submission**
- Update existing `README.md` for the repo (judges read it): how to run, screenshots, tested results
- Devpost draft from 11 PM; one paragraph per sponsor tool (g.tec/PiEEG, Base44, ElevenLabs, Gemini, GoDaddy)
- Demo slides (3–4 slides max) + the 3-minute script from the Implementation Guide
- Edit the backup video from C's clips
- ✅ Submitted by ~10:00 AM

---

## Check-in checklist (9:30 PM · 1:30 AM · 6:30 AM, 5 minutes)

- [ ] Each person: what works, what's broken, what's next
- [ ] Merge PRs into `main`; full loop passes
- [ ] Anything behind? Cut in this order: quantum → ElevenLabs extras (C4/B6) → Gemini (B5) → FBCCA → fancy art (C1)
