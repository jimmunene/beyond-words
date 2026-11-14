# BrainType TrustLoop — Dataset Links, Purpose & Access Plan

**Hack The Nest 2026 | Team of 4 | Prepared November 14, 2026**

## The short answer

Download **Kalunga2016 first**, then **one subject from Dong2023**, and record **your own consented Unicorn Hybrid Black EEG sessions** as soon as the headset is available. Use BETA only if there is time. These sources answer different questions; none alone validates the complete BrainType communication system.

| Priority | Data source | Direct link | Why we need it | Download now? |
|---|---|---|---|---|
| **P0 — Essential** | **Kalunga2016: SSVEP + reject/rest condition** | [NEMAR dataset](https://nemar.org/dataset/nm000123) · [MOABB documentation](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kalunga2016.html) · [Original Zenodo files](https://zenodo.org/records/2392979) | Test the **SELECT vs WAIT/ABSTAIN** decision, especially accidental selections when no target is intended. | **Yes**; about 0.1 GB on NEMAR. |
| **P0 — Essential** | **Dong2023: 40-target SSVEP** | [NEMAR dataset](https://nemar.org/dataset/nm000128) · [Zenodo files, including individual subjects](https://zenodo.org/records/18847318) · [Original paper](https://doi.org/10.26599/BSA.2023.9050020) | Test and benchmark **CCA frequency/target decoding** and window-size tradeoffs on a comparatively hardware-similar 8-channel dataset. | **Yes**, start with one subject. |
| **P0 — Essential** | **Team's own live Unicorn sessions** | [g.tec Unicorn technical specifications](https://www.gtec.at/product/unicorn-hybrid-black-bci-platform/) · [g.tec LSL/streaming integration guide](https://www.gtec.at/2026/11/05/eeg-interoperability/) | Validate **the actual screen, electrodes, stimulus timing, signal quality and user interaction**. Create genuine postcard art and end-to-end demo evidence. | **Collect when hardware is ready.** |
| **P1 — Optional** | **BETA/Liu2020 SSVEP benchmark** | [NEMAR dataset](https://nemar.org/dataset/nm000129) · [Source paper](https://doi.org/10.3389/fnins.2020.00627) | Evaluate whether decoding/rejection generalizes beyond one dataset. **Not needed** for the hackathon MVP. | **No**, unless core demo already works. |

> **Critical distinction:** A public cued SSVEP dataset tests classification under its *own* stimulus frequencies, hardware, and timing. It cannot prove live performance with your team's 7/9/11/13 Hz proposal or with the Unicorn headset. Your locally recorded sessions provide that evidence.

---

## 1. Kalunga2016 — most important for TrustLoop's safety claim

- **Verified links:** [NEMAR nm000123](https://nemar.org/dataset/nm000123), [MOABB Kalunga2016](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kalunga2016.html), [Zenodo archive](https://zenodo.org/records/2392979).
- **What's inside:** SSVEP data from **12 healthy subjects**, **8 EEG channels**, sampled at **256 Hz**. Targets are **13, 17, and 21 Hz** plus a **reject/rest** condition (fixation away from the three targets). MOABB's summary and event representation may distinguish recorded trial duration from the benchmark's processed epoch length; inspect actual annotations rather than assuming the same epoch duration in every tool.
- **Why the team needs it:** Without a rest/reject class, you cannot meaningfully test whether the speller **refuses to select** when no target is intended. Build your uncertainty policy using this dataset first.
- **Specific tasks:** Preprocess EEG; compute CCA scores for the *actual three stimulus frequencies*; compare (A) always-select-the-top-score against (B) select/wait/abstain rules. Report false selections on reject trials, selected-target accuracy, coverage and latency.
- **Expected deliverable:** `results/kalunga_rest_rejection.csv` and a baseline-vs-TrustLoop figure.
- **Limits:** This is not an ALS or clinical population dataset. Its LED stimulus and g.tec MobiLab acquisition differ from a computer-screen Unicorn demo. The MOABB page notes a historical correction involving switched 17 Hz/21 Hz event notation; use current, documented labels.

### Easiest way to access Kalunga2016

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install moabb mne numpy scipy scikit-learn pandas matplotlib
```

```python
from moabb.datasets import Kalunga2016

dataset = Kalunga2016()
recordings = dataset.get_data(subjects=[1])  # one subject first
print(recordings.keys())
```

Run the data inspection before writing a custom epoch parser. MOABB will download/cached supported raw data; the user may need to accept download prompts. If MOABB has compatibility issues, use the original Zenodo archive or NEMAR instead.

---

## 2. Dong2023 — most important for SSVEP spelling performance

- **Verified links:** [NEMAR nm000128](https://nemar.org/dataset/nm000128), [Zenodo record with subject `.mat` files](https://zenodo.org/records/18847318), [original journal DOI](https://doi.org/10.26599/BSA.2023.9050020).
- **What's inside:** **59 participants**, **8 EEG channels**, **250 Hz**, a **40-target SSVEP** paradigm with visual stimulation between **8.0 and 15.8 Hz** and frequency/phase coding. NEMAR hosts an approximately **1 GB** converted set; the Zenodo listing includes individual subject files around **12 MB** each.
- **Why the team needs it:** It provides labeled target selections for testing CCA decoding and the impact of short vs longer evidence windows. The channel count/sample rate are closer to Unicorn than a 64-channel benchmark, but that **does not mean the electrode positions are identical**.
- **Specific tasks:** Load 1–2 subjects; verify label-to-frequency mapping and channel metadata; baseline CCA on labeled windows; vary decision window size (e.g. 1.0, 1.5, 2.0, 3.0 s where data permits); test the TrustLoop confirmation policy on classifier outputs.
- **Expected deliverable:** `results/dong_window_tradeoff.csv` showing target accuracy, abstention/coverage, and decision time by window size.
- **Limits:** Cued target-choice data do **not** include free-form spelling, full user consent interaction, or representative no-intent behavior. Do not claim this dataset tests the full IntentLock workflow.
- **License:** The NEMAR copy is marked **CC BY-NC 4.0** (non-commercial); attribute original authors and comply with the source terms.

### Download a small, practical subset

**Simplest hackathon route:** Open [Zenodo Dong2023](https://zenodo.org/records/18847318) and download `S1.mat` first, instead of the entire archive. Inspect its MATLAB field names in Python with `scipy.io.loadmat` before designing preprocessing.

Or, if using the NEMAR CLI:

```bash
npm install -g nemar-cli
nemar dataset download nm000128 --subjects sub-01
```

NEMAR's dataset page documents the command. For a common Python interface, MOABB also exposes `Dong2023` (check your installed MOABB release):

```python
from moabb.datasets import Dong2023
recordings = Dong2023().get_data(subjects=[1])
print(recordings.keys())
```

---

## 3. Live Unicorn Hybrid Black EEG — essential for a credible final demo

- **Verified links:** [Official headset page](https://www.gtec.at/product/unicorn-hybrid-black-bci-platform/), [g.tec's EEG/LSL interoperability guide](https://www.gtec.at/2026/11/05/eeg-interoperability/), [g.Pype device reference](https://gpype.gtec.at/content/7_sdk_reference/backend_sources/hybrid_black.html), [g.Pype acquisition tutorial](https://gpype.gtec.at/content/2_gpype_training/season3_gearing_up/s3e2_hybrid_black.html).
- **Hardware:** 8 EEG channels, 250 Hz. g.tec documents live **LSL output through the Unicorn LSL app**; native Python acquisition paths may have separate licensing/driver requirements. **Ask the sponsor mentor which setup is enabled at the venue before coding around a specific SDK.**
- **Why the team needs it:** Proves the system actually works with the *headset, display refresh rate, stimulus schedule, wiring, timing and actual participant* used in judging. These same recordings drive personalized postcard art.

### Minimum local protocol (with informed consent and mentor approval)

| Recording | Procedure | Why it matters |
|---|---|---|
| Cued targets | Display a small set of mentor-approved visual targets, record which was intentionally attended, and capture synchronized EEG/visual event markers. | Train/calibrate and test real headset decoding. |
| No-selection | Ask a participant to fixate on a designated non-target point during an otherwise matched trial. | Measure false-selection rate when there should be no output. |
| Message composition | Have a participant select a very short message, correct mistakes and confirm it before output. | Test full latency, corrections and IntentLock behavior. |
| Postcard session | Use the **same consented session** to derive EEG-driven colors/shapes and place *only approved text* on art. | Makes the postcard authentic, not decorative random generation. |

### Metadata that must be saved

`participant_code`, `trial_id`, `sampling_rate`, `channel_names`, `stimulus_frequencies`, `stimulus_onset_timestamp`, `stimulus_offset_timestamp`, `expected_target_or_rest`, `predicted_target`, `decision` (`SELECT`/`WAIT`/`ABSTAIN`), `score_margin`, `selection_time_ms`, `user_confirmation`, `input_mode` (`live_eeg` / `dataset_replay` / `keyboard_demo`).

**Privacy:** Use pseudonyms. No raw EEG, real names or personally identifiable participation data in a public Base44 gallery or GitHub repo. Ask permission separately for participation, recording and public postcard publishing. Keep consent records private. Flickering visual stimuli can be unsafe for photosensitive participants; use mentor-approved safety procedures and an immediate stop control.

---

## 4. BETA SSVEP (optional) — secondary benchmark only

- **Links:** [NEMAR nm000129](https://nemar.org/dataset/nm000129), [original paper](https://doi.org/10.3389/fnins.2020.00627).
- **Contains:** 70 participants, 64 EEG channels, 250 Hz, 40-target cued SSVEP spelling; NEMAR lists approximately 7.72 GB.
- **Why:** External robustness check across a different collection procedure and participant set. It can help show your work is not tailored to a single subject or dataset.
- **Why not today:** Large download and different channel layout make it lower priority than reliable live decoding and IntentLock.
- **License:** NEMAR marks this dataset for **non-commercial research use only**. Verify terms before publishing derived data, using in commercial products or redistributing recordings.

Subset access (only if needed):

```bash
npm install -g nemar-cli
nemar dataset download nm000129 --subjects sub-01
```

---

## 5. What does *not* require a public EEG dataset?

| Component | What you need instead | Explanation |
|---|---|---|
| **IntentLock** | Synthetic test event sequences | Test unauthorized AI rewrites, duplicate events, backspace, confirmation, and blocked output with unit tests. No patient EEG is needed. |
| **AI word suggestions** | A small local frequency-ranked word list and hand-written phrases | Avoid new large NLP training data; suggestions must remain separate from confirmed text. |
| **ElevenLabs voice** | User-confirmed message text and an API key, or browser speech fallback | Text-to-speech does not require training on raw EEG. |
| **EEG postcard art** | Consented raw EEG from a genuine local session | Generate artwork from actual signal features; don't claim color represents medically validated mental states. |
| **Base44 gallery** | Approved postcard PNG, approved text, optional nickname, consent state | Do not publish raw brain signals or identifying information. |

---

## 6. TrustBench: one fair experiment that judges can understand

**Question:** Does TrustLoop reduce unintended committed selections compared with a fixed-window baseline?

1. **Baseline:** For each same held-out EEG trial, choose the maximum CCA target score at a fixed window (e.g. 2 s).
2. **TrustLoop:** On those same trials, require a calibrated margin/consistency criterion, otherwise WAIT or ABSTAIN; do not silently change any confirmed text.
3. **Compare:** (i) accuracy among selected trials, (ii) false selections on rest trials, (iii) abstention rate, (iv) decision latency, (v) message corrections, and (vi) blocked unauthorized speech/publication from synthetic IntentLock tests.
4. **Avoid leakage:** Pick thresholds and channels on calibration data; evaluate on distinct trials/blocks. If possible, separately test on another subject. Do not tune parameters after seeing final test outcomes.
5. **Be honest:** Record whether results come from `dataset_replay`, `live_eeg`, or `keyboard_demo`; these are different types of evidence.

**Important:** A raw CCA correlation or score margin is **not a calibrated 90% confidence probability**. Report it as a score unless probabilities have actually been calibrated and validated.

### Suggested output files

```text
beyond-words/
├── data/
│   ├── external/                 # gitignored public dataset downloads
│   │   ├── kalunga2016/
│   │   ├── dong2023/
│   │   └── beta/                 # optional
│   ├── raw/unicorn/              # gitignored private recordings
│   ├── processed/                # gitignored trials/features
│   └── test_scenarios/           # synthetic, safe to commit
├── results/
│   ├── kalunga_rest_rejection.csv
│   ├── dong_window_tradeoff.csv
│   ├── live_selection_summary.csv
│   └── trustbench_comparison.csv
├── src/
│   ├── acquisition/
│   ├── preprocessing/
│   ├── classifier/
│   ├── trustloop/
│   └── postcard/
└── docs/
    └── DATASETS.md
```

Use `.gitignore` for `data/external/`, `data/raw/`, `data/processed/`, participant-level results, and any API secrets.

---

## 7. Assign the dataset work to the team

| Person | Immediate task | Definition of done |
|---|---|---|
| **1 — EEG/ML** | Download Kalunga and Dong; inspect labels, channels and events; run fixed CCA baseline. | One working classification plot plus recorded metrics. |
| **2 — TrustLoop/experiments** | Implement SELECT/WAIT/ABSTAIN and evaluate on Kalunga's reject class. | Fair baseline-vs-TrustLoop CSV and test script. |
| **3 — Interface/live capture** | Coordinate approved headset access, configure stream + event timestamps, UI safety/stop controls. | Genuine labeled EEG sample saved with metadata. |
| **4 — Postcards/gallery/demo** | Build art generator, ConsentCanvas, voice/gallery path and backup replay demonstration. | Approved message becomes a postcard; no raw EEG uploaded. |

### Do these in order

- **First hour:** Confirm headset access, safety, monitor configuration, and streaming interface; download **Kalunga** and **one Dong participant**.
- **Next few hours:** Get deterministic CCA classification and reject handling working **before** adding AI text suggestions.
- **Next:** Record live EEG, compare a fixed policy with TrustLoop, and test explicit approval end to end.
- **Only then:** Complete postcard art, Base44 gallery, voice output and video/presentation polish.

---

## 8. Source and licensing checklist

These links were checked against the dataset/project pages on **November 14, 2026**. Read individual dataset licenses before redistributing data or turning the prototype into a commercial product.

1. Kalunga MOABB record: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Kalunga2016.html
2. Kalunga NEMAR record: https://nemar.org/dataset/nm000123
3. Kalunga Zenodo: https://zenodo.org/records/2392979
4. Dong NEMAR record (CC BY-NC 4.0): https://nemar.org/dataset/nm000128
5. Dong Zenodo (subject MAT files): https://zenodo.org/records/18847318
6. Dong original paper: https://doi.org/10.26599/BSA.2023.9050020
7. BETA NEMAR record (non-commercial research): https://nemar.org/dataset/nm000129
8. BETA paper: https://doi.org/10.3389/fnins.2020.00627
9. Unicorn product/hardware: https://www.gtec.at/product/unicorn-hybrid-black-bci-platform/
10. Unicorn LSL/interop guide: https://www.gtec.at/2026/11/05/eeg-interoperability/
11. g.Pype acquisition guide: https://gpype.gtec.at/content/2_gpype_training/season3_gearing_up/s3e2_hybrid_black.html
12. MOABB SSVEP dataset summary: https://moabb.neurotechx.com/docs/dataset_summary.html

**Bottom line:** **Kalunga = no-selection safety**; **Dong = spelling signal decoding**; **Unicorn = credible live proof**; **BETA = optional robustness**. Put your remaining effort into a measured TrustLoop comparison, an authentic EEG selection, and a consent-based message-to-postcard demonstration.
