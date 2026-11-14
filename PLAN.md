# BrainType Postcards 🧠✉️

**Type with your mind. Keep the postcard.**

Visitors put on a g.tec Unicorn EEG headset, type a short message using only their eyes and brain signals (SSVEP), and get a unique postcard: generative art made from their own brainwaves with their brain-typed message on top. Every postcard goes into a public gallery built with Base44.

The same speller is a real communication tool for people with ALS, locked-in syndrome or severe paralysis, who cannot speak or use their hands.

**Core tracks:** Mind and Machine (g.tec + PiEEG) · Best Use of Base44 (community vote) · Hack The Nest (overall)
**Add-on tracks:** Best Use of ElevenLabs · MLH Best Use of Gemini API* · MLH Best Domain (GoDaddy Registry)* · Future of Intelligence*
<sub>*Confirm at the help desk that these are offered and how many tracks one project can enter.</sub>

> 👉 **Building it? Jump to [Section 16: Step-by-Step Build Guide](#16-step-by-step-build-guide).**

> **Pitch line:** "Everyone here today typed with their mind. For some people, this is the only way they can talk."

---

## 1. Visitor Experience (~2 minutes)

1. **Put on the headset** (we check signal quality on a live dashboard, ~20 s)
2. **Quick check:** look at each flickering box for 3 s; the system confirms it sees you (~15 s)
3. **Type:** spell a name or short message by looking at boxes; AI word suggestions speed it up (~60–90 s)
4. **Art:** EEG recorded during typing becomes a unique generative art piece
5. **Postcard:** message printed on the art → uploaded to the Base44 gallery
6. **Scan the QR code**, find your postcard, share it, and vote 😉

**Fast lane for long queues:** a "Quick message" box offers preset phrases ("HELLO FROM HACK THE NEST") so a visitor finishes in a few selections.

---

## 2. How the Speller Works

### Screen layout: 4 flickering boxes

```
┌────────────────────────────────────────────┐
│  Typed:  H E L _                           │
│                                            │
│   ┌────────┐               ┌────────┐      │
│   │ A – I  │               │ J – R  │      │
│   │ 7 Hz   │               │ 9 Hz   │      │
│   └────────┘               └────────┘      │
│                                            │
│   ┌────────┐               ┌────────┐      │
│   │ S – Z  │               │ HELLO  │      │
│   │ ␣ ⌫    │               │(suggest│      │
│   │ 11 Hz  │               │ 13 Hz) │      │
│   └────────┘               └────────┘      │
│                                            │
│   [confidence bars]  [live EEG spectrum]   │
└────────────────────────────────────────────┘
```

### Selection tree (3 looks per letter, 1 look per suggested word)

| Level | Box 1 | Box 2 | Box 3 | Box 4 |
|---|---|---|---|---|
| **1. Group** | A–I | J–R | S–Z, ␣, ⌫ | **Suggested word** |
| **2. Subgroup** (e.g. after A–I) | A B C | D E F | G H I | ← Back |
| **3. Letter** (e.g. after G H I) | G | H | I | ← Back |

- Picking the suggested word at Level 1 types the whole word + space in **one look**
- Rough speed: ~3 s per look → ~9 s per letter, much faster with suggestions
- "HI MOM" ≈ 45–60 s without suggestions

### Word suggestions
- **Default:** a local word list ranked by frequency (instant, works offline). Add names and hackathon words ("HELLO", "LOVE", "MOM", "EMU", "HACK")
- **Stretch:** LLM next-word prediction (OpenAI or Gemini), with the word list as fallback if Wi-Fi or the API fails

---

## 3. Signal Processing

### Stimuli: sampled sinusoidal flicker
Use **sampled sine luminance modulation**: each frame, set box brightness to `0.5 + 0.5·sin(2π·f·t)`. This makes any frequency possible on a 60 Hz monitor, unlike simple on/off toggling.

| Box | Frequency | Why |
|---|---|---|
| 1 | 7 Hz | |
| 2 | 9 Hz | Avoids exactly 10 Hz (alpha rhythm) |
| 3 | 11 Hz | |
| 4 | 13 Hz | |

No frequency is a harmonic of another (harmonics 14, 18, 22, 26 Hz never collide with fundamentals). Use `t = frame_index / refresh_rate`, never wall-clock time. **Check the venue monitor's refresh rate** and lock vsync.

### CCA classifier (no calibration needed)
1. Channels: occipital/parietal (check Unicorn channel map; likely PO7, Oz, PO8 plus neighbours)
2. Filter: bandpass 5–40 Hz, notch 60 Hz
3. Window: 2.0 s, sliding every 0.25 s
4. References per frequency f: sin/cos at f, 2f, 3f
5. Score = canonical correlation between EEG window and each reference set
6. **Select** when the same box wins with margin > τ for N consecutive windows (start N = 3)
7. After each selection: brief **lockout (~1 s)** + visual flash so the user re-targets

### Optional: filter-bank CCA (FBCCA)
If accuracy is low, split into 3–5 sub-bands, run CCA on each, and combine weighted scores. It's a well-known accuracy boost and ~30 lines of code.

---

## 4. EEG → Art

Record all EEG during the typing session, then compute:

| Feature | Controls |
|---|---|
| Alpha / beta / theta band power | Colour palette (calm/high alpha → cool blues; high beta → warm oranges) |
| Signal variability | Shape density, line thickness |
| Per-channel power | Position/weight of shapes across the canvas |
| A raw EEG trace segment | Drawn as a flowing "brainwave line" signature |
| Session hash | Random seed → every postcard unique |

Render with `matplotlib` (or `p5.js`/`PIL`) as a 1200×800 PNG, overlay the message in a bold font, plus a small footer: *"Typed with brain signals · Hack The Nest 2026"*.

---

## 5. Architecture

```
┌──────────────────────┐
│  g.tec Unicorn EEG   │  (live)  ─┐
│  PiEEG Cloud sim     │  (dev)    ├─ same LSL stream
│  g.tec SSVEP dataset │  (replay) ┘
└──────────┬───────────┘
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│ CCA classifier       │────▶│ Speller state machine│
│ (+ confidence)       │     │ group → sub → letter │
└──────────────────────┘     │ + word suggestions   │
           ▲                 └──────────┬───────────┘
           │ EEG buffer                 │ typed message
           ▼                            ▼
┌──────────────────────┐     ┌──────────────────────┐
│ Art generator        │────▶│ Postcard PNG         │
└──────────────────────┘     └──────────┬───────────┘
                                        ▼
                             ┌──────────────────────┐
                             │ Base44 gallery app   │
                             │ (API upload, or      │
                             │  manual upload form) │
                             └──────────────────────┘
```

---

## 6. Repository Structure

```
braintype/
├── README.md
├── requirements.txt
├── config.yaml               # freqs, channels, thresholds, refresh rate
├── acquisition/
│   ├── unicorn_stream.py     # live headset → LSL
│   └── replay_stream.py      # dataset / PiEEG sim → LSL (same format)
├── processing/
│   ├── cca.py                # CCA / FBCCA scoring
│   └── selector.py           # debounce, margin threshold, lockout
├── speller/
│   ├── state_machine.py      # group → subgroup → letter, back, delete
│   ├── suggestions.py        # word list (+ optional LLM)
│   └── ui.py                 # pygame: sine-flicker boxes, typed text, feedback
├── art/
│   └── postcard.py           # EEG features → PNG + message overlay
├── upload/
│   └── base44_upload.py      # API upload if available; else save to folder
├── dashboard/
│   └── live_panel.py         # signal quality, spectrum, confidence bars
├── eval/
│   └── log_session.py        # selections, accuracy, chars/min
└── postcards/                # generated PNGs
```

```bash
pip install numpy scipy scikit-learn pylsl pygame pyqtgraph PyQt5 matplotlib pillow qrcode
```

---

## 7. Team Split (4 people)

| Person | Owns | First deliverable |
|---|---|---|
| **A: Signal lead** | LSL, CCA/FBCCA, selector, live headset tuning | Correct box predicted from replayed SSVEP dataset |
| **B: Speller UI** | pygame flicker boxes, state machine, suggestions, feedback | Type a word with keyboard keys 1–4 standing in for EEG |
| **C: Art + postcard** | EEG features → art, overlay, QR codes, upload script | Postcard PNG from a recorded EEG file |
| **D: Base44 + demo** | Base44 gallery, visitor flow, table signage, Devpost, video | Live gallery link with test postcards |

**Interface contract (agree in the first 30 min):**
- Classifier emits: `{"box": 0-3, "confidence": float, "t": timestamp}`
- Speller consumes the same messages from **keyboard 1–4** (dev/backup) or **EEG** (live). One flag switches them.
- Postcard function: `make_postcard(eeg_array, fs, message, nickname) -> png_path`

---

## 8. Timeline

*Build starts **Sat 6:30 PM**. Assumes submission around **Sun 10:00 AM** (~15.5 hours of build time). Adjust once the real deadline is confirmed.*

### Before 6:30 PM: prep (no coding needed)

| Time | Task |
|---|---|
| **Now–5:00 PM** | **g.tec booth:** channel map, electrode tips, how headsets are shared (book time slots tonight), Unicorn Python API / LSL. **Base44 booth:** API/upload options, how voting works, Devpost rules |
| **5:00–6:00 PM** | Install all packages on every laptop. Download the g.tec SSVEP dataset. Get PiEEG Cloud access. Check the venue monitor refresh rate |
| **6:00–6:30 PM** | Team huddle: roles, interface contract (section 7), repo created. **Base44 person can start the gallery right away** (it's just prompting) |

### Build

| Time | Goal | Done when |
|---|---|---|
| **6:30–9:30 PM** | Build in parallel: A on dataset replay + CCA, B on speller with keyboard input, C on art from recorded EEG, D on Base44 gallery | Each part works on its own |
| **9:30–11:00 PM** | **Integrate:** replay → CCA → speller → postcard → gallery | Full pipeline runs end to end without the headset |
| **11:00 PM–1:30 AM** | **Live headset:** tune channels, window, τ, N; test on all teammates | A teammate types "HI" with EEG |
| **1:30 AM checkpoint** | Live typing reliable? If not: 2–3 boxes, 3 s window, higher N, lean on quick-message lane. **No new features** | Scope locked |
| **1:30–5:30 AM** | Sleep in shifts (2 on, 2 off). Polish: suggestions, feedback sounds, dashboard, FBCCA if needed. **Gemini word prediction** (Person B) | Reliable typing for 3 of 4 teammates |
| **5:30–6:30 AM** | **ElevenLabs** voice for postcards (Person C), **GoDaddy domain** → gallery (Person D) | Postcard speaks; domain live |
| **6:30–8:30 AM** | **Open the table:** real visitors, real postcards, QR codes, push the gallery link on Discord. Log accuracy + chars/min | 15–20+ postcards in gallery |
| **8:30–10:00 AM** | Record backup video, write Devpost (all tracks), rehearse 3-min demo | Submitted with buffer |
| **Judging** | Live demo + gallery + numbers | 🏆 |

**Cut list if behind (drop in this order):** Quantum comparison → ElevenLabs → Gemini suggestions → FBCCA → API upload (use manual upload) → fancy art (simple gradient + message is fine) → live dashboard polish.

---

## 9. Base44 Gallery App

**Prompt to paste into Base44:**

> Build a mobile-friendly public gallery app called **BrainType Postcards**. Each postcard has an image, a short message (typed using a brain-computer interface), a first name or nickname, a typing time in seconds, and a timestamp.
> 1. Home: a bold banner: "Every message here was typed using only brain signals — the same technology that lets people with ALS communicate." Below it, a grid gallery, newest first, with live count of total postcards.
> 2. Postcard detail page: large image, message, nickname, typing time, share button, download button, heart/like button with count.
> 3. "Most loved" tab sorted by hearts.
> 4. "About BCI communication" page explaining SSVEP spellers simply, and who they help (ALS, locked-in syndrome, paralysis).
> 5. An admin-only upload page to add new postcards (image, message, nickname, typing time).
> 6. Clean, bold, high-contrast design, large text, great on phones.

**Getting postcards in:** API upload from `base44_upload.py` if Base44 supports it; otherwise Person D uploads each one through the admin page (fine at queue speed).

**For the vote:** QR code at the table, link in the hackathon Discord, and a one-liner: *"Type your name with your mind, get a postcard."*

---

## 10. Privacy & Consent

- Store only the **postcard image, message and nickname**. Don't upload raw EEG.
- Ask before posting: "OK to add your postcard to the public gallery?"
- Optional, separate opt-in to save anonymized EEG for research. Default is off.
- Never let visitors type personal info (phone numbers, addresses) into the public gallery.

---

## 11. Metrics to Report

- **Selection accuracy:** % correct over cued test selections
- **Characters per minute** (with and without suggestions)
- **Information transfer rate (ITR)** in bits/min
- **Visitors served** and **postcards created**
- Average time per postcard

---

## 12. Demo Script (3 minutes)

1. **(0:00)** "Imagine you can't speak or move your hands. How do you say 'I love you'?"
2. **(0:30)** A judge or teammate wears the headset and types a short word live; dashboard shows the SSVEP peak lighting up for the box they look at
3. **(1:30)** The postcard appears from their own brainwaves; scan the QR code, it's in the gallery
4. **(2:00)** Show the gallery: "X people typed with their minds today." Show accuracy and chars/min
5. **(2:30)** Vision: the same speller, at home, for people with ALS, with LLM prediction and their own voice

---

## 13. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| Limited headset time | Everything developed on dataset replay / PiEEG sim first |
| Low live accuracy | FBCCA, longer window (3 s), higher N, fewer boxes; electrode tips from g.tec |
| Flicker timing jitter | Lock vsync, frame-indexed sine; move stimuli to PsychoPy if needed |
| Visitor queue too slow | Quick-message lane; cap messages at ~8 characters |
| Wi-Fi / API down | Local word list; postcards saved locally, uploaded later |
| Live demo fails | Backup video + replay mode through the same pipeline; keyboard mode clearly labelled |
| Base44 can't receive uploads by API | Manual admin upload page |

---

## 14. Track Integrations & Stretch Goals (only after the core is reliable)

| Track | Integration | Time | When |
|---|---|---|---|
| **Hack The Nest** | Nothing extra, just submit | 0 | Submission |
| **Best Use of ElevenLabs** | Speak each finished postcard message aloud in a natural voice ("giving people a voice") | ~1 h | 5:30 AM polish |
| **MLH Gemini API*** | Gemini next-word suggestions in Box 4, local word list as fallback | ~1–2 h | After 1:30 AM checkpoint |
| **Future of Intelligence*** | Frame the Gemini prediction as "AI + brain working together"; read track description first | Overlaps Gemini | Devpost write-up |
| **MLH GoDaddy Domain*** | Register e.g. `typewithyourmind.tech`, point it at the Base44 gallery | ~15 min | Anytime (Person D) |
| *Optional:* **Code of Life / CS Dept** | Submit if the track description fits (ask organizers) | 0 | Submission |
| *Optional:* **Quantum Computing** | Offline only: quantum SVM vs. CCA on the g.tec SSVEP dataset. Never in the live demo | 2–3 h | Only with a spare teammate |

**Skip:** Future of Money, Beyond the Horizon, Capture the Flag, Best Beginner, SSOJet (Base44 already handles login).

**Other stretch goals:**
1. **Steering mode:** reuse the same pipeline to drive a pygame wheelchair (the NeuroDrive idea)
2. Print physical postcards if a printer is available

---

## 15. Right-Now Checklist

- [ ] g.tec booth: channel map, electrode tips, headset schedule, Unicorn Python API / LSL
- [ ] Base44 booth: API or upload options, how voting works, Devpost submission
- [ ] Download the g.tec SSVEP dataset + PiEEG Cloud access
- [ ] Check venue monitor refresh rate
- [ ] Repo created, interfaces agreed, roles assigned
- [ ] Confirm the submission deadline and fix the timeline above
- [ ] Help desk: how many tracks per project, which MLH prizes are offered (Gemini? GoDaddy?), whether sponsor prizes need separate Devpost entries
- [ ] Get API keys/credits: ElevenLabs, Gemini, GoDaddy Registry domain code (MLH)

---

## 16. Step-by-Step Build Guide

Follow the steps in order. Each step ends with a **✅ Check**: don't move on until it passes.

### Phase 0: Setup (before 6:30 PM, everyone)

**Step 0.1: Environment**
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install numpy scipy scikit-learn pylsl pygame pyqtgraph PyQt5 matplotlib pillow qrcode pyyaml
```
✅ Check: `python -c "import pygame, pylsl, sklearn; print('ok')"` prints `ok` on every laptop.

**Step 0.2: Repo + folders.** One person creates the repo with the structure in Section 6 and pushes it; everyone clones.
✅ Check: everyone can pull and push.

**Step 0.3: `config.yaml`** (everyone reads from this, no hard-coded numbers)
```yaml
fs: 250                    # Unicorn sample rate (confirm)
refresh_rate: 60           # measure the venue monitor!
freqs: [7, 9, 11, 13]
channels: [5, 6, 7]        # occipital indices, set from g.tec channel map
window_s: 2.0
step_s: 0.25
margin_tau: 0.05
n_consecutive: 3
lockout_s: 1.0
```

**Step 0.4: Booth answers.** Write down: channel map, how to stream Unicorn to LSL (g.tec reportedly provides an LSL tool in Unicorn Suite; ask), headset time slots, Base44 upload options.
✅ Check: answers written in the team chat.

**Step 0.5: Agree the interface contract** (Section 7). Everyone codes against it.

---

### Phase 1: Build in parallel (6:30–9:30 PM)

#### Person A: Signal (replay → CCA → selections)

**A1. Replay stream.** Load the g.tec SSVEP dataset, push samples into an LSL stream at the real sample rate, in chunks.
```python
from pylsl import StreamInfo, StreamOutlet
info = StreamInfo('BrainType_EEG', 'EEG', n_channels, fs, 'float32', 'replay01')
outlet = StreamOutlet(info)
# loop: outlet.push_chunk(data[i:i+chunk].tolist()); time.sleep(chunk/fs)
```
✅ Check: a second script with `StreamInlet` receives samples at ~250 Hz.

**A2. Filters.** Bandpass 5–40 Hz + notch 60 Hz (`scipy.signal.butter`, `iirnotch`, `filtfilt` on each window).

**A3. CCA scoring.**
```python
import numpy as np
from sklearn.cross_decomposition import CCA

def refs(f, n, fs, harmonics=3):
    t = np.arange(n) / fs
    return np.column_stack([fn(2*np.pi*h*f*t) for h in range(1, harmonics+1)
                            for fn in (np.sin, np.cos)])

def cca_scores(window, freqs, fs):          # window: (n_samples, n_channels)
    scores = []
    for f in freqs:
        cca = CCA(n_components=1)
        u, v = cca.fit_transform(window, refs(f, len(window), fs))
        scores.append(np.corrcoef(u[:, 0], v[:, 0])[0, 1])
    return np.array(scores)
```
✅ Check: on dataset trials with known targets, the correct frequency wins most of the time. **Write the accuracy down.**

**A4. Selector.** Sliding window every 0.25 s → winner + margin (top1 − top2). Emit `{"box", "confidence", "t"}` only when the same box wins N times in a row with margin > τ, then lock out for 1 s.
✅ Check: replaying one long trial produces one selection, not a burst.

**A5. Output.** Send selections over a local socket/queue (or a second LSL marker stream) to the speller.

#### Person B: Speller UI (keyboard first)

**B1. pygame window + 4 boxes** in a 2×2 layout, fullscreen, vsync on.

**B2. Sine flicker driven by frame index, never by time.**
```python
frame += 1
t = frame / REFRESH
for box, f in zip(boxes, FREQS):
    lum = int(255 * (0.5 + 0.5*np.sin(2*np.pi*f*t)))
    pygame.draw.rect(screen, (lum, lum, lum), box.rect)
```
✅ Check: print the measured FPS; it must stay at the refresh rate with no drops.

**B3. State machine.** Level 1 groups → Level 2 subgroups → Level 3 letters, Back, Space, Delete (Section 2). Box 4 = suggested word at Level 1.

**B4. Keyboard input.** Keys `1–4` produce the same `{"box": ...}` messages as the EEG.
✅ Check: type "HELLO WORLD" with keys only.

**B5. Word suggestions.** Local list sorted by frequency + hackathon words; show the top match for the current prefix in Box 4.

**B6. Feedback.** Flash the selected box, short click sound, show typed text big at the top. A "Done" path finishes the message (e.g. Space twice, or a "✓" option in the S–Z group).

**B7. Input switch.** `--input keyboard` or `--input eeg` flag.

#### Person C: Art + postcard

**C1. Load a recorded EEG file** (from the dataset) as `(n_samples, n_channels)`.

**C2. Features.** Welch PSD per channel → theta (4–8), alpha (8–13), beta (13–30) power; variance; seed = hash of the data.

**C3. Render art** (1200×800): palette from the alpha/beta ratio, shapes/lines from per-channel power, one raw EEG segment drawn as a flowing line.

**C4. Overlay.** Message in a big bold font, nickname, footer *"Typed with brain signals · Hack The Nest 2026"*.

**C5. Function** `make_postcard(eeg, fs, message, nickname) -> png_path`, saved to `postcards/`.
✅ Check: two different EEG files give visibly different postcards.

**C6. QR code** for the gallery link (`qrcode` package) → print/display at the table.

#### Person D: Base44 + demo

**D1.** Paste the Section 9 prompt into Base44. Refine by chatting until it looks right on a phone.

**D2.** Add 3–5 test postcards via the admin upload page.

**D3.** Ask Base44 about API upload. If yes, write `upload/base44_upload.py`; if no, manual upload is the plan.
✅ Check: the gallery opens from a phone on mobile data, and a test postcard appears.

**D4.** Start the Devpost draft and table signage ("Type your name with your mind → get a postcard").

---

### Phase 2: Integrate without the headset (9:30–11:00 PM)

**Step 2.1.** Run: replay stream → Person A's selector → Person B's speller (`--input eeg`).
✅ Check: the speller reacts to replayed selections.

**Step 2.2.** Speller buffers all raw EEG during a session; on "Done", call `make_postcard()`.

**Step 2.3.** Postcard → upload (API or folder for manual upload).
✅ Check: **one full run, replay to gallery, with no human help except pressing Start.**

**Step 2.4.** Session logger: every selection, intended box (for test runs), timestamps → CSV for accuracy and chars/min.

---

### Phase 3: Live headset (11:00 PM–1:30 AM)

**Step 3.1. Fit the headset.** Use g.tec's tips; check signal quality on the dashboard (flat, low noise, no 60 Hz domination). Fix contact before touching code.

**Step 3.2. Stream live.** Unicorn → LSL (same stream format as replay). Run with `--source live`.

**Step 3.3. Calibration check (no training, just verification).** User stares at each box for 5 s while you log CCA scores.
✅ Check: the correct box has the highest score for each of the 4 boxes.

**Step 3.4. Tune, one change at a time:**
1. Channel selection (try only Oz/PO7/PO8 vs. all occipital)
2. Window 2 s → 3 s if unstable
3. τ and N (raise if false selections, lower if too slow)
4. Box size, brightness, spacing (bigger boxes, dark background, sit ~60 cm away)
5. FBCCA if still weak

**Step 3.5. Test on every teammate.** Type "HI" and log accuracy.
✅ Check at **1:30 AM:** at least 3 of 4 teammates can type "HI". If not, apply the checkpoint rules and cut list.

---

### Phase 4: Polish + add-on tracks (1:30–6:30 AM, in sleep shifts)

**Step 4.1. Gemini word prediction (Person B).** Run in a background thread; fall back to the local list on any error or if it takes >500 ms.
```python
from google import genai            # check current SDK docs at the event
client = genai.Client()             # GEMINI_API_KEY in env
resp = client.models.generate_content(
    model="<fast Gemini model>",
    contents=f"Typed so far: '{typed}'. Reply with ONE likely next word, uppercase, nothing else.")
```

**Step 4.2. ElevenLabs voice (Person C).** When a postcard is created, speak the message aloud.
```python
from elevenlabs.client import ElevenLabs
from elevenlabs import play
tts = ElevenLabs()                  # ELEVENLABS_API_KEY in env
play(tts.text_to_speech.convert(voice_id="<voice_id>", model_id="eleven_flash_v2_5", text=message))
```
Run in a background thread so the UI never freezes.

**Step 4.3. Domain (Person D).** Redeem the MLH GoDaddy Registry code, register the domain, point/redirect it to the Base44 gallery. Update the QR code.

**Step 4.4. Live dashboard polish:** spectrum with the 4 target frequencies marked, confidence bars.

**Step 4.5. Optional quantum comparison (only a spare teammate):** offline QSVM vs. CCA on the dataset; one results table for the Devpost. Never in the live loop.

✅ Check: all add-ons fail gracefully. Unplug the Wi-Fi and the speller still works.

---

### Phase 5: Visitor table (6:30–8:30 AM)

**Step 5.1. Table setup:** monitor at eye level, headset + saline/gel supplies (per g.tec), signage, QR code, consent line.

**Step 5.2. Per visitor (~2 min):** fit headset → signal check → 4-box verification → type (cap ~8 characters, or use quick-message lane) → postcard → ask "OK to post publicly?" → upload → show QR.

**Step 5.3. Log every session** (accuracy, chars/min, time). Keep the best postcards for the demo.

**Step 5.4. Push the gallery link** in Discord every hour with a live count ("27 people have typed with their minds today").

✅ Check: 15–20+ postcards in the gallery.

---

### Phase 6: Submit + demo (8:30–10:00 AM)

**Step 6.1. Backup video** (60–90 s): live typing, dashboard peak, postcard appearing, voice reading it, gallery.

**Step 6.2. Metrics slide:** accuracy, chars/min, ITR, visitors served.

**Step 6.3. Devpost:** one project page; select every eligible track. In the description, add a short paragraph per sponsor tool (g.tec/PiEEG, Base44, ElevenLabs, Gemini, domain) explaining exactly how it's used. Judges for sponsor prizes look for this.

**Step 6.4. Rehearse** the Section 12 demo script twice, with a timer. Decide who wears the headset and who talks.

✅ Check: submitted before the deadline, with the gallery link, domain, repo and video included.
