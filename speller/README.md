# Speller UI + word prediction (Part B)

The screen a person looks at: four flickering boxes, a typed-text bar, and everything that
happens when a box is picked. Works with the keyboard (demo), simulated EEG, or the live classifier.

## Run

```bash
python speller/ui.py --input keyboard --windowed    # keys 1-4 or mouse pick a box
python speller/ui.py --input sim --windowed         # with synthetic_stream.py + classifier_node.py
python speller/ui.py --input eeg                    # live headset (add --signal replay for replay_stream.py)
python speller/test_speller.py                      # logic tests, no pygame or EEG needed
```

Press **TAB** in the app for every operator key.

## Two modes

**Postcard (F1)**: spell a short message (group, smaller group, letter: 3 looks). Box 4 on the
first screen finishes the most likely word in one look, or opens ready-made messages when
nothing is typed yet.

**Patient (F2)**: one-look needs and phrases.

| Page | Boxes |
|---|---|
| 1 | WATER, PAIN, EMERGENCY, MORE |
| 2 | COME HERE (call bell), BODY, TALK, MORE |
| 3 | SPELL, REST, MY PHRASES, START |

- **EMERGENCY** sounds a siren until a helper presses SPACE. A look can never cancel it.
- **PAIN** is spoken at once, then asks where it hurts.
- **REST** blanks the choices; waking takes two looks at two different corners.
- **Q** (helper key) shows YES / NO / NOT SURE after a spoken question. **R** repeats the last phrase.
- **F4** on the start screen: two-box easy mode (one option + NEXT) for a weak signal.
- Personal phrases: copy `speller/my_phrases.example.txt` to `data/my_phrases.txt` and edit it.
- `python speller/care_report.py` summarises the day from `logs/selections.csv` (stays local).

## Word prediction

```bash
python speller/build_corpus.py --download   # fetch AAC text + build data/corpus.json (not committed)
python speller/eval_predictor.py            # looks saved on held-out AAC sentences
```

Without `data/corpus.json` the predictor falls back to a small built-in vocabulary.
`needs_vocabulary.txt` (care words) and `seed_sentences.txt` (postcard and care sentences) are
ours; add to them freely. Patient mode also offers clinical words that are never suggested on
public postcards.

Credits: AAC text collections by Keith Vertanen and collaborators, https://www.aactext.org
(licences in `data/aac_raw/*readme.txt` after download). Word counts: norvig.com/ngrams.

## Gemini (optional: `use_llm: true` + `GEMINI_API_KEY`)

| Feature | What happens | What is sent to Gemini |
|---|---|---|
| Suggested answers | A helper presses **Q**, types the question they asked, ENTER. Three likely answers become one-look boxes (YES / NO is one look further). | The helper's question |
| Fuller sentence | After spelling key words and DONE, the person chooses SAY IT (as typed) or SAY GEMINI'S (one natural sentence). | What the patient spelled: only with `llm_in_patient_mode: true` |
| Word prediction | Box 4 may show Gemini's next word (postcard mode; one request per word). | The postcard text so far |

Gemini only proposes; nothing it writes is spoken until the person picks it. Without a key,
internet or quota, the built-in pages and the local corpus are used.
Try it: `python speller/assistant.py "What would you like for dinner?"`

## Spoken languages (ElevenLabs multilingual voice)

The boxes stay in English; the voice can speak Spanish, Hindi, French, German, Portuguese,
Arabic, Chinese, Japanese or Tamil. **F5** on the start screen (or **L** while typing) picks the language.

- Built-in phrases come from `speller/translations.json` (made with Gemini, works offline).
- Spelled sentences, suggested answers and personal phrases are translated live by Gemini;
  without Gemini they are spoken in English.
- `python speller/languages.py --precache es hi` pre-generates the audio for those languages.
- The translations are machine-made and use masculine forms where the language needs a gender.
  Have a native speaker check them before real use.

## Reaching a caregiver who is not in the room (`speller/notify.py`)

COME HERE ("I need you. Please come to me.") and EMERGENCY can also alert a phone: an urgent push
alert (ntfy, free), a real call plus text message (Twilio), or a Telegram message. Each channel is
switched on by environment variables; see the top of `speller/notify.py`. Nothing is sent when
none is set, and never during the automatic demo. An unanswered EMERGENCY alerts again once a
minute, five times at most. Test: `python speller/notify.py "Test: I need you"`.

## Caregiver web app on Base44 (`speller/care_sync.py`)

With `care_app_url` + `care_app_id` in `config.yaml` and a `BASE44_TOKEN` environment variable:

- every spoken need, spelled sentence and emergency appears on the app's live feed;
- a caregiver's reply in the app is shown and spoken on the patient's screen;
- tapping "I'm coming" on an emergency (acknowledged) stops the alarm;
- a postcard the visitor agreed to share (Y) is uploaded to the app's public gallery.

Check the connection: `python speller/care_sync.py`. The token is a secret; never commit it.

## Not done yet

- **No separate confirmation step.** `voice/tts.py` asks callers to state that a message was
  approved and that cloud voice is allowed. The speller states both on every box pick: the pick
  is treated as the person's approval, and `use_tts: true` as consent to send the text to
  ElevenLabs. With a live headset a misread look is therefore spoken at once (THAT WAS WRONG
  takes it back). Spelled sentences do get a say-it page when `llm_in_patient_mode` is on.
- Keyboard mode records no EEG, so no postcard art is made there.
- Not yet tested with a live headset.

Safety: the boxes flash at 7-13 Hz. Not for anyone with photosensitive epilepsy.

## Live EEG dashboard (web)

With the caregiver app set up, the speller also writes one `LiveStatus` record about once a second:
state (detecting / selected / speaking / paused / emergency), the four frequencies and their scores,
the detected frequency, confidence and threshold, the last pick. Decoder scores only, never raw EEG.
The Base44 app needs a `LiveStatus` entity and a page that reads it; until then nothing is sent.
Switch it off with `live_dashboard: false` in `config.yaml`.

## Gentler flicker

The boxes pulse around a dim middle grey instead of black to white, hold still for a moment after
each pick, while a sentence is spoken, during the alarm and when paused, and fade back in.
Optional `config.yaml` keys: `flicker_contrast` (default 0.3, 1.0 = black to white),
`flicker_brightness` (default 0.36), `flicker_rest_s` (default 1.5). This lowers the risk for
light-sensitive people; it does not remove it. Test accuracy on the real headset before relying on it.
