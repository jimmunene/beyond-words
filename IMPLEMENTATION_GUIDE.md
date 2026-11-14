# BrainType Postcards: Implementation Guide

Step by step, from "nothing installed" to a winning demo. Follow it in order.
Every step ends with a **✅ Check**. Don't move on until it passes.

> **Status of the starter code in this repo:** the full pipeline runs end to end on a laptop with **synthetic EEG**: classifier, speller (postcard + patient mode), word prediction, postcard art, upload queue and metrics. It has **not** been run on a real Unicorn headset yet. That's Phase 3, tonight.

---

## Contents

1. [What we're building (60-second version)](#1-what-were-building)
2. [Everything we need](#2-everything-we-need)
3. [How the code fits together](#3-how-the-code-fits-together)
4. [Phase 0: Setup (before 6:30 PM)](#phase-0-setup-before-630-pm)
5. [Phase 1: Own your module (6:30–9:30 PM)](#phase-1-own-your-module-630930-pm)
6. [Phase 2: Integrate without the headset (9:30–11:00 PM)](#phase-2-integrate-without-the-headset-930-pm1100-pm)
7. [Phase 3: Live headset (11:00 PM–1:30 AM)](#phase-3-live-headset-1100-pm130-am)
8. [Phase 4: Polish + add-on tracks (1:30–6:30 AM)](#phase-4-polish--add-on-tracks-130630-am)
9. [Phase 5: Visitor table (6:30–8:30 AM)](#phase-5-visitor-table-630830-am)
10. [Phase 6: Submit + demo (8:30–10:00 AM)](#phase-6-submit--demo-8301000-am)
11. [What makes this the best project](#11-what-makes-this-the-best-project)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What we're building

- A visitor wears the **g.tec Unicorn** EEG headset.
- They look at **4 flickering boxes** (7, 9, 11, 13 Hz). The box they look at shows up as that rhythm in their visual cortex (**SSVEP**). **CCA** detects which one, with no training per person.
- Three looks pick a letter (group → subgroup → letter). One look picks the **predicted word** in box 4.
- **Postcard mode:** they type a short message → their EEG becomes **generative art** → the postcard goes to a **Base44 gallery** → it's read aloud by **ElevenLabs**.
- **Patient mode:** one-look needs (**WATER, PAIN, TOILET, …**) spoken aloud in ~6 seconds, plus full spelling. This is the "real impact" part of the demo.

**Tracks:** Mind and Machine · Best Use of Base44 · Hack The Nest · ElevenLabs · (MLH Gemini, MLH GoDaddy domain, Future of Intelligence, if offered)

---

## 2. Everything we need

### Hardware
| Item | Source | Notes |
|---|---|---|
| g.tec Unicorn Hybrid Black headset | g.tec booth | **Book time slots tonight.** Ask about electrode gel/saline and fitting |
| 2+ laptops | Team | One runs the stimulus screen (must hold a steady frame rate) |
| External monitor, **ideally 60 Hz** | Venue / team | Check the refresh rate. Flicker frequencies depend on it |
| Speaker (or laptop speakers) | Team | For ElevenLabs voice |
| Power strip | Team | Long night |
| Phone | Team | Test the gallery + QR code |
| *(Optional)* PiEEG Octopus board | PiEEG | Only if Unicorn time is scarce |

### Software (all free)
| What | Why |
|---|---|
| Python 3.10+ | Everything |
| `pip install -r requirements.txt` | numpy, scipy, pylsl, pygame, matplotlib, … |
| **Unicorn Suite** + its **LSL interface** (from g.tec) | Streams the headset into Lab Streaming Layer. **Ask g.tec** which tool/version to use |
| Git + GitHub repo | Team code |
| *(Optional)* LabRecorder | Recording raw sessions for later analysis |

### Accounts & keys
| What | For | Get it from |
|---|---|---|
| Base44 account | Gallery app | base44 (ask booth about Builder credits) |
| ElevenLabs API key + voice ID | Voice | ElevenLabs booth / site (ask for credits) |
| Gemini API key | Word prediction | Google AI Studio (ask help desk about credits) |
| MLH GoDaddy Registry domain code | Custom domain | MLH table, if offered |
| Devpost account | Submission | devpost.com |

### Data (all free)
| File → put in `data/` | Source | Used for |
|---|---|---|
| g.tec SSVEP dataset | g.tec (event) | Testing CCA on real brain signals before the headset |
| `count_1w.txt`, `count_2w.txt` | norvig.com/ngrams | General word + next-word frequencies |
| `aac_sentences.txt` | Vertanen AAC collection (keithv.com), one sentence per line | Makes predictions sound like real AAC users |
| `custom_words.txt` | Us | Names, event words (already in repo) |

### Questions to ask before 6:30 PM (write answers in team chat)
- [ ] **g.tec:** channel order in the LSL stream? (We assume EEG = first 8 channels: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8; **verify**.) Total channel count (the stream may include accelerometer/gyro/battery)? Sample rate (we assume 250 Hz)? Best electrode prep? Headset schedule?
- [ ] **Base44:** can an app receive uploads by API/webhook? How does community voting work?
- [ ] **Help desk:** submission deadline? How many tracks per project? Which MLH prizes (Gemini, GoDaddy)?

---

## 3. How the code fits together

```
 TERMINAL 1                 TERMINAL 2                      TERMINAL 3
┌────────────────────┐    ┌──────────────────────────┐    ┌───────────────────────────┐
│ EEG SOURCE (pick 1)│    │ processing/              │    │ speller/ui.py             │
│ • Unicorn LSL app  │LSL │   classifier_node.py     │UDP │  • flicker boxes (pygame) │
│ • replay_stream.py │───▶│  filter → CCA → selector │───▶│  • state machine          │
│ • synthetic_stream │    │  records session EEG     │5005│  • word prediction        │
└────────▲───────────┘    └────────────▲─────────────┘    │  • postcard + voice +     │
         │ UDP 5006 (sim gaze)         │ UDP 5007          │    upload                 │
         └─────────────────────────────┴───────────────────┤  • CSV logging            │
                                                           └───────────────────────────┘
```

| File | Does | Owner |
|---|---|---|
| `config.yaml` | **All tunable numbers** (channels, freqs, thresholds, ports, feature flags) | Everyone |
| `acquisition/synthetic_stream.py` | Fake SSVEP EEG; keys 1–4 in the UI choose the "gaze" | A |
| `acquisition/replay_stream.py` | Streams a recorded `.npy`/`.csv` as if live | A |
| `processing/cca.py` | Filters, CCA, FBCCA | A |
| `processing/selector.py` | Turns scores into clicks (margin, N in a row, lockout) | A |
| `processing/classifier_node.py` | LSL in → scores/selections out, session recording | A |
| `speller/state_machine.py` | Screens, letters, BACK/DELETE/DONE, patient-mode needs | B |
| `speller/predictor.py` | Local corpus prediction + Gemini, merged | B |
| `speller/build_corpus.py` | Builds `data/corpus.json` from Norvig + AAC sentences | B |
| `speller/ui.py` | The pygame app the visitor sees | B |
| `art/postcard.py` | EEG → art PNG with message | C |
| `voice/tts.py` | ElevenLabs speech (background thread, fails silently) | C |
| `upload/base44_upload.py` | Queue for manual upload, or API upload | D |
| `eval/metrics.py` | Accuracy, ITR, chars/min from the log | D |

**Messages (UDP, JSON, localhost):**
- classifier → UI (5005): `{"type":"scores","scores":[...]}`, `{"type":"select","box":0-3,"confidence":x}`, `{"type":"session_saved","path":...}`
- UI → classifier (5007): `{"cmd":"start_session"|"stop_session"|"pause"|"resume"}`
- UI → synthetic stream (5006): `{"target":0-3 or -1}`

---

## Phase 0: Setup (before 6:30 PM)

**Step 0.1: Get the code on every laptop**
```bash
git clone <your-repo-url> braintype && cd braintype
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
✅ Check: `python -c "import pygame, pylsl, scipy; print('ok')"` prints `ok`.

**Step 0.2: Run the whole system with fake EEG (proves your laptop works)**
```bash
# Terminal 1
python acquisition/synthetic_stream.py
# Terminal 2
python processing/classifier_node.py
# Terminal 3
python speller/ui.py --input sim --windowed
```
In the UI: type a nickname, press **ENTER**. Press **1–4** to "look" at a box. The box gets selected by the *classifier* after ~1–2 s. Gaze stays on that box until you press another key, so press **0** ("look away") when you want it to stop selecting.
✅ Check: you can spell "HI" and press ✓ DONE (box 3 → box 4 → box 2), and a postcard appears.

**Step 0.3: Keyboard-only mode (UI without any EEG)**
```bash
python speller/ui.py --input keyboard --windowed
```
✅ Check: keys 1–4 select instantly. This is also the **clearly labelled backup** if the headset fails during judging.

**Step 0.4: Download the data** into `data/` (Section 2), then:
```bash
python speller/build_corpus.py
python speller/predictor.py         # prints sample suggestions
```
✅ Check: `'I WANT ' -> WATER` (or similar sensible words).

**Step 0.5: Measure the monitor refresh rate.** Windows: Display settings → Advanced display. macOS: System Settings → Displays. Put it in `config.yaml → refresh_rate`.
✅ Check: the UI's flicker looks smooth, not stuttering.

**Step 0.6: Team huddle (6:00–6:30 PM)**
- Roles: **A** signal, **B** speller/UI, **C** art/voice, **D** Base44/demo/metrics
- Everyone has run Step 0.2
- Base44 gallery started (Person D can start right away)

---

## Phase 1: Own your module (6:30–9:30 PM)

The starter code works. Each person's job is to **make their part solid on real data** and improve it.

### Person A: Signal

**A1. Convert the g.tec SSVEP dataset to `.npy` `(n_samples, n_channels)` in µV**, plus a list of trials (start sample, target frequency). Write a small `data/convert_gtec.py` for its format.
✅ Check: `np.load(...).shape` looks right; you know which samples belong to which target frequency.

**A2. Offline accuracy on real EEG.** For each trial: take a 2 s window after onset, `preprocess`, `CCAClassifier.scores`, argmax. If the dataset's frequencies differ from ours, temporarily use theirs.
```python
from processing.cca import CCAClassifier, FBCCAClassifier, make_filters, preprocess
filt = make_filters(fs, (5, 40), 60)   # use 50 Hz notch if the dataset was recorded in Europe
clf = CCAClassifier(fs, dataset_freqs, int(2.0 * fs))
pred = clf.scores(preprocess(eeg[s:s + int(2*fs)][:, chans], filt)).argmax()
```
✅ Check: accuracy **written down** for window 1 s / 2 s / 3 s, CCA vs FBCCA, channel sets (Oz only; PO7+Oz+PO8; all 8). Pick the best for `config.yaml`. These numbers go on the demo slide.

**A3. Replay test:** `python acquisition/replay_stream.py data/<file>.npy` + classifier node.
✅ Check: classifier prints no errors; scores move when the replayed target changes.

**A4. Write `acquisition/unicorn_check.py`:** connect to the Unicorn LSL stream and print channel count, rate, and the per-channel standard deviation every second. You'll use this at the table to check electrode contact.

### Person B: Speller UI + prediction

**B1. Run** `python -m speller.state_machine` (text version) and spell "I WANT WATER". Fix anything awkward.

**B2. Layout pass on the real monitor:** big boxes, dark background, labels readable from 60 cm. Edit sizes in `ui.py → App.__init__`.
✅ Check: someone who hasn't seen it can read every label instantly.

**B3. Frame timing:** add an FPS readout (top-left, `self.clock.get_fps()`). Flicker accuracy depends on hitting the refresh rate every frame.
✅ Check: FPS stays at the refresh rate (±1) with all 4 boxes flickering.

**B4. Prediction quality:** add names and words to `data/custom_words.txt`, rebuild the corpus, test typical messages: "HI MOM", "I LOVE YOU", "THANK YOU", "I WANT WATER".
✅ Check: suggestions appear by the 1st–2nd letter of common words.

**B5. Feedback:** add a short click sound on selection (`pygame.mixer`), and make the selected-box flash obvious.

### Person C: Art + voice

**C1. Run** `python art/postcard.py`, open `postcards/*.png`, and iterate on the look. Ideas: more vivid palettes, the SSVEP frequency the visitor used most drawn as rings, the visitor's typing time on the card.
✅ Check: two different EEG recordings give clearly different cards. Text is readable on every palette.

**C2. Real EEG:** run `make_postcard` on a slice of the g.tec dataset.

**C3. Voice:** set `ELEVENLABS_API_KEY`, pick a warm voice, put its ID in `config.yaml`, set `use_tts: true`.
```bash
python -c "from voice.tts import _speak; _speak('Hello from BrainType', 'YOUR_VOICE_ID')"
```
✅ Check: it speaks. With the key removed, it falls back to the offline voice (pyttsx3) and nothing crashes.

**C4. Pre-cache patient phrases:** `python voice/precache.py` saves ElevenLabs audio for every needs phrase into `voice/cache/`. Run it while Wi-Fi works. Afterwards those phrases play instantly, even offline.
✅ Check: turn off Wi-Fi → WATER still speaks in the ElevenLabs voice.

**How voice fails safely:** cached ElevenLabs audio → live ElevenLabs → offline computer voice → text on screen only. The demo never goes silent *and* blank, and never crashes.

### Person D: Base44 gallery + demo assets

**D1.** Paste the gallery prompt (`PLAN.md`, Section 9) into Base44 and refine it on a phone.
**D2.** Upload 3 test postcards through the admin page.
**D3.** Ask Base44 about API upload. If yes, implement `upload_via_api()` and set `USE_API = True`.
**D4.** Put the gallery link in `config.yaml → gallery_url`. Make a QR code:
```bash
python -c "import qrcode; qrcode.make('YOUR_GALLERY_URL').save('qr.png')"
```
**D5.** Table sign: *"Type your name with your mind → get a postcard. You choose letters with your eyes; your brainwaves tell the computer which one."*
✅ Check: gallery opens on a phone over mobile data; QR code scans.

---

## Phase 2: Integrate without the headset (9:30 PM–11:00 PM)

**Step 2.1: Real-data replay through the full app.** Terminal 1: `replay_stream.py <dataset>.npy`. Terminal 2: classifier. Terminal 3: `ui.py --input eeg`.
✅ Check: selections arrive from real recorded EEG.

**Step 2.2: Full visitor loop on one laptop (sim mode):** nickname → spell "HI MOM" → DONE → postcard → spoken → **Y** → file appears in `postcards/to_upload/` → Person D uploads it → visible in the gallery.
✅ Check: **one complete visitor, start to gallery, in under 3 minutes.**

**Step 2.3: Patient mode:** F2 on the intro screen → WATER → spoken "I would like some water, please."
✅ Check: 1 look → spoken phrase. MORE → MORE → ✎ SPELL → spell → ✓ DONE speaks the typed text.

**Step 2.4: Metrics pipeline:** in a spell session press **T** (cued test; a red outline shows which box to pick), do 20 selections, then:
```bash
python eval/metrics.py logs/selections.csv
```
✅ Check: accuracy, seconds/selection and ITR print.

**Step 2.5: Commit + tag** `v0-integrated`. From now on, never leave `main` broken.

---

## Phase 3: Live headset (11:00 PM–1:30 AM)

**Step 3.1: Fit the headset** with g.tec's help. Occipital electrodes (PO7, Oz, PO8) matter most: good contact on the back of the head.

**Step 3.2: Stream:** start g.tec's LSL interface. Run `unicorn_check.py` (A4).
✅ Check: stream found; channel count + rate match `config.yaml`; occipital channels are not flat and not huge.

**Step 3.3: Point the config at the real stream:** set `stream_name` to the Unicorn's LSL name, confirm `eeg_channels` = PO7, Oz, PO8 indices.

**Step 3.4: Verification (no training):** UI `--input eeg`, press **T**, wearer follows the red outline for 20 selections.
✅ Check: accuracy well above 25% (chance). Aim for 80%+.

**Step 3.5: Tune: change ONE thing at a time, re-test with T, write results down:**

| Symptom | Change in `config.yaml` |
|---|---|
| Wrong box selected often | ↑ `margin_tau` (0.05 → 0.08), ↑ `n_consecutive` (3 → 4) |
| Correct but very slow | ↓ `n_consecutive`, ↓ `window_s` (2.0 → 1.5) |
| Scores barely differ between boxes | Check electrode contact; try `use_fbcca: true`; `window_s: 3.0`; try channels `[4,5,6,7]` |
| One box always wins | That frequency may sit on the wearer's alpha. Swap it (e.g. 9 → 8 Hz) |
| Selections fire while looking away | ↑ `margin_tau`; use **P** to pause between visitors |

Also tune the **environment:** dim room lighting around the screen, sit ~60 cm away, bigger boxes, ask the wearer to stay still and relax their jaw.

**Step 3.6: Test on all four teammates.** Log accuracy per person.
✅ **1:30 AM checkpoint:** 3 of 4 teammates can type "HI". If not: freeze features, use the quick-message/patient screens (fewer looks), lengthen the window, and focus on reliability.

---

## Phase 4: Polish + add-on tracks (1:30–6:30 AM)

Sleep in shifts: **2 on, 2 off.** Only work on this list, in this order:

1. **Reliability fixes from Phase 3** (always first)
2. **Gemini prediction (B):** `pip install google-genai`, set `GEMINI_API_KEY`, `use_llm: true`. Check the current model name in Google's docs and set `GEMINI_MODEL`. ✅ Check: unplug Wi-Fi and suggestions still appear (local fallback).
3. **ElevenLabs on postcards + patient phrases (C):** already wired; just `use_tts: true`.
4. **Domain (D):** redeem the MLH GoDaddy code, point the domain at the gallery, regenerate the QR.
5. **Live score bars:** already in the UI. Make them prettier, and label them with frequencies.
6. *(Only with a spare teammate)* **Quantum comparison:** QSVM vs CCA offline on the dataset; one results table for Devpost. Never in the live demo.

✅ Check after each item: a full visitor loop still works. Commit.

---

## Phase 5: Visitor table (6:30–8:30 AM)

**Per visitor (~2–3 min):**
1. Operator: nickname → ENTER (postcard mode)
2. Fit headset, quick contact check
3. 30-second explanation: "Look at the box with the letters you want. Keep looking until it flashes green."
4. Type (≤ 12 characters, suggestions help)
5. Postcard appears + is spoken → ask **"OK to post it publicly?"** → Y/N
6. Show them the QR → they find their card → *"If you liked it, vote for us!"*
7. Press **P** (pause) while swapping visitors

Every hour: post in Discord, *"N people have typed with their minds today 🧠"* + gallery link.

✅ Check: 15–20+ postcards in the gallery; `metrics.py` numbers updated.

**Privacy:** never upload raw EEG; consent before posting; no personal info in messages.

---

## Phase 6: Submit + demo (8:30–10:00 AM)

**6.1 Backup video (60–90 s):** live typing, score bars lighting up, postcard appearing, voice, gallery, patient-mode "WATER" in 6 seconds.

**6.2 Numbers slide:** accuracy (with chance = 25%), seconds/selection, ITR (bits/min), chars/min, visitors served, postcards made.

**6.3 Devpost:** one project; select all eligible tracks. Sections:
- Inspiration (people with ALS / locked-in syndrome)
- What it does (postcard + patient mode)
- How we built it (SSVEP, CCA, AAC-corpus + Gemini prediction, Base44, ElevenLabs)
- **One paragraph per sponsor tool** explaining exactly how it's used
- Results (the numbers) · Challenges · What's next
- Links: gallery, domain, repo, video

**6.4 Demo script (3 min):**
1. **0:00** "Imagine you can't speak or move. How do you ask for water?"
2. **0:20** Patient mode: teammate looks at WATER → spoken in ~6 s
3. **0:50** Postcard mode: judge or teammate types a short word live; point at the score bars
4. **1:50** Postcard + voice; scan QR → it's in the gallery with N other visitors' cards
5. **2:20** Numbers, and "no training needed: works on anyone in seconds"
6. **2:40** Vision: same speller at home, in the user's own banked voice

**6.5 Rehearse twice with a timer.** Decide who wears the headset and who talks.

✅ Check: submitted before the deadline with all links.

---

## 11. What makes this the best project

- **It works live, on anyone, without training.** Most BCI hackathon demos are videos or only work on one person.
- **Real numbers.** Accuracy, ITR, chars/min from logged trials. Very few teams measure anything.
- **Two layers of impact:** a fun postcard for everyone *and* patient mode showing the real use case.
- **Research-backed prediction:** word suggestions trained on AAC-style text (how non-speaking people actually communicate), plus an LLM.
- **Robust engineering:** works offline, every add-on fails gracefully, keyboard backup, replay mode, pause button.
- **Community vote built in:** every visitor walks away with something they made, and a link to vote.
- **Honest science:** you choose with your eyes and the brain signal tells the computer which box. Say it clearly; g.tec judges will respect it.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `No EEG stream found` | Start the source first; check `stream_name` matches; same network/firewall allows LSL |
| UI gets no selections in sim/eeg mode | Classifier running? Ports 5005–5007 free? Only one UI instance open |
| `Address already in use` | An old process is still running: close it (Task Manager / `pkill -f classifier_node`) |
| Flicker stutters | Close other apps, use the external monitor as main display, disable laptop power saving, check FPS |
| Emoji show as boxes | Expected: `clean()` converts labels to ASCII. Use a font with symbols if you want icons |
| Postcard looks the same for everyone | EEG not recorded: check `session_saved` arrives; in keyboard mode there's no EEG (noise art is used) |
| Gemini/ElevenLabs errors | Check keys/credits; app keeps working without them; check current SDK docs |
| Selections fire when nobody's looking | Press **P** between visitors; raise `margin_tau` |
| Very low live accuracy | Electrode contact first (back of head), then Step 3.5 table, then ask g.tec booth |
