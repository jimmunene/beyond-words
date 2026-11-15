"""Quick checks for the speller logic and word prediction (no pygame, no EEG).

    python speller/test_speller.py          # or: python -m pytest speller/test_speller.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import Predictor, valid_word  # noqa: E402
from speller.state_machine import SCREENS, Speller  # noqa: E402

PRED = Predictor()

# box sequence (0-based) for each letter: group, subgroup, letter
PATH = {}
for gi, group in enumerate(["AI", "JR", "SZ"]):
    for si, (_, action) in enumerate(SCREENS[group][:3]):
        for li, item in enumerate(SCREENS[action[1]]):
            if item[1][0] == "char":
                PATH[item[1][1]] = (gi, si, li)
DELETE, DONE = (2, 3, 0), (2, 3, 1)


def spell(sp, text):
    for ch in text:
        for box in PATH[ch]:
            sp.select(box)


def press(sp, boxes):
    for box in boxes:
        sp.select(box)


def test_spell_letters_and_done():
    sp = Speller(None)
    spell(sp, "HI MOM")
    assert sp.text == "HI MOM" and sp.screen == "root"
    press(sp, DONE)
    assert sp.finished and sp.text == "HI MOM"


def test_every_letter_reachable():
    assert set(PATH) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")


def test_no_double_or_leading_space():
    sp = Speller(None)
    spell(sp, " A  B")
    assert sp.text == "A B"


def test_quick_lane_two_looks():
    sp = Speller(PRED)
    assert sp.labels()[3] == "★ QUICK"
    press(sp, (3, 0))
    assert sp.finished and sp.text == "HI MOM"
    sp = Speller(PRED)
    press(sp, (3, 3, 1))                     # QUICK > MORE > second message (longer than 12: allowed)
    assert sp.finished and sp.text == "HACK THE NEST"
    sp = Speller(PRED)
    press(sp, (3, 3, 3))                     # QUICK > MORE > BACK leaves the quick lane
    assert sp.screen == "root" and not sp.finished and not sp.stack


def test_quick_lane_only_when_empty():
    sp = Speller(PRED)
    spell(sp, "H")
    assert "QUICK" not in sp.labels()[3] and sp.labels()[3].startswith("★ ")


def test_custom_quick_messages():
    sp = Speller(PRED, quick_messages=["go emu", "hi dad"])
    press(sp, (3, 1))
    assert sp.finished and sp.text == "HI DAD"


def test_suggestion_types_word_and_space():
    sp = Speller(PRED)
    spell(sp, "H")
    word = sp.suggestion
    assert word and word.startswith("H")
    sp.select(3)
    assert sp.text == word + " "


def test_suggestion_never_overflows():
    sp = Speller(PRED, max_chars=6)
    spell(sp, "I W")                         # 3 characters of room left for the word
    s = sp.suggestion
    assert s is None or len(s) <= 4
    sp = Speller(PRED, max_chars=5)
    spell(sp, "HELL")
    if sp.suggestion:
        sp.select(3)
    assert len(sp.text) <= 5


def test_char_limit():
    sp = Speller(None, max_chars=3)
    spell(sp, "ABCDE")
    assert sp.text == "ABC" and sp.full


def test_delete_undoes_whole_suggestion():
    sp = Speller(PRED)
    spell(sp, "H")
    sp.select(3)                             # accept e.g. HELLO
    press(sp, DELETE)
    assert sp.text == "H"
    press(sp, DELETE)
    assert sp.text == ""
    press(sp, DELETE)                        # deleting nothing is harmless
    assert sp.text == ""


def test_done_with_nothing_typed_is_ignored():
    sp = Speller(PRED)
    press(sp, DONE)
    assert not sp.finished and sp.screen == "root"


def test_empty_box_is_ignored():
    sp = Speller(None)                       # no predictor: box 4 on the root screen is the quick lane
    spell(sp, "A")
    assert sp.labels()[3] == "" and sp.select(3) == "empty" and sp.text == "A"


def test_back_returns_one_level():
    sp = Speller(None)
    press(sp, (0, 0))                        # A-I > ABC
    sp.select(3)                             # BACK
    assert sp.screen == "AI"
    sp.select(3)
    assert sp.screen == "root"


def test_patient_phrase_one_look():
    sp = Speller(PRED, "patient")
    sp.select(0)
    assert sp.spoken == "I would like some water, please." and sp.screen == "needs1"


def test_patient_spelling_speaks_and_returns():
    sp = Speller(PRED, "patient", max_chars=12)
    press(sp, (3, 3, 0))                     # MORE > MORE > SPELL
    assert sp.screen == "root" and sp.spoken == "I am typing a message. Please wait."
    spell(sp, "I AM COLD AND TIRED")         # longer than the postcard limit
    assert sp.text == "I AM COLD AND TIRED"
    press(sp, DONE)
    assert sp.spoken == "I am cold and tired." and sp.screen == "needs1" and sp.text == ""


def test_patient_empty_done_and_needs_button():
    sp = Speller(PRED, "patient")
    press(sp, (3, 3, 0))
    sp.spoken = None                         # (the "I am typing" notice)
    press(sp, DONE)
    assert sp.spoken is None and sp.screen == "needs1"
    press(sp, (3, 3, 0, 2, 3, 3))            # SPELL > S-Z > controls > NEEDS
    assert sp.screen == "needs1"
    press(sp, (3, 3, 3))                     # last needs page > START
    assert sp.screen == "needs1" and not sp.stack


def test_emergency_one_look_and_only_a_helper_clears_it():
    sp = Speller(PRED, "patient")
    assert "EMERGENCY" in sp.labels()[2]
    sp.select(2)
    assert sp.alarm and sp.spoken == "Emergency! I need help right now."
    assert sp.labels() == [""] * 4
    for box in range(4):                     # stray looks cannot cancel or change anything
        assert sp.select(box) == "empty" and sp.alarm
    sp.clear_alarm()
    assert not sp.alarm and sp.screen == "needs1" and sp.labels()[0]
    sp.set_mode("postcard")                  # no emergency box in postcard mode
    assert not sp.alarm and not any("EMERGENCY" in lab for lab in sp.labels())


def test_predictor_basics():
    assert PRED.suggest("") and PRED.suggest("H").startswith("H")
    assert PRED.suggest("I WANT WA") == "WATER"
    assert PRED.suggest("I LOVE ") == "YOU"
    assert PRED.suggest("HACK THE ") == "NEST"
    assert PRED.suggest("i want wa") == "WATER"                 # case does not matter
    assert PRED.suggest("XQZJ") is None


def test_predictor_max_len_and_complete_word():
    assert len(PRED.suggest("I WANT WA", max_len=3) or "") <= 3
    assert PRED.suggest("HI") == "HI"        # finished word: box 4 offers "HI + space"
    sp = Speller(PRED)
    spell(sp, "HI")
    assert sp.labels()[3] == "★ HI ␣"
    sp.select(3)
    assert sp.text == "HI "


def test_predictor_is_clean():
    assert not valid_word("SHIT") and not valid_word("WWW") and not valid_word("X")
    assert valid_word("I") and valid_word("HI") and valid_word("WATER")
    for prefix in ["F", "FU", "S", "SH", "SE", "A", "AS", "B", "BI", "P", "PO", "D", "DA", "C", "CR"]:
        w = PRED.suggest(prefix)
        assert w is None or valid_word(w), (prefix, w)


def test_care_words_are_predictable_and_private_words_stay_out_of_postcards():
    if PRED.source != "corpus":
        return                                   # the small built-in vocabulary has no care words
    sp = Speller(PRED, "patient")                # patient mode: clinical words allowed
    for typed, word in [("I NEED SU", "SUCTION"), ("MY CATH", "CATHETER"), ("THE VENTI", "VENTILATOR"),
                        ("I AM CHO", "CHOKING"), ("MY TAILB", "TAILBONE"), ("I FEEL NAUS", "NAUSEOUS")]:
        assert PRED.suggest(typed) == word, (typed, PRED.suggest(typed))
    assert PRED.suggest("I AM DYI") == "DYING"
    sp.set_mode("postcard")                      # public postcards: never offered
    assert PRED.suggest("I AM DYI") != "DYING"
    assert PRED.suggest("H") in ("HELLO", "HI")  # booth words come first on a postcard


def test_second_word_on_letter_group_page():
    if PRED.source != "corpus":
        return
    sp = Speller(PRED, "patient")                # default: BACK stays on every letter page
    press(sp, (3, 3, 0, 0))
    assert "BACK" in sp.labels()[3]
    sp = Speller(PRED, "patient", group_words=True)
    press(sp, (3, 3, 0))                         # SPELL
    top = sp.suggestion
    sp.select(0)                                 # A-I page: box 4 offers another word starting with A-I
    label = sp.labels()[3]
    assert label.startswith("★ ") and label[2] in "ABCDEFGHI" and label[2:] != top
    sp.select(3)
    assert sp.text == label[2:] + " " and sp.screen == "root"
    sp.select(0); sp.select(0)                   # A-I > ABC: BACK skips the page that has no BACK
    sp.select(3)
    assert sp.screen == "root"


def test_pain_follow_up_talk_and_answer_pages():
    sp = Speller(PRED, "patient", my_phrases=[("my glasses", "Please put my glasses on.")])
    sp.select(1)                                 # PAIN: spoken at once, then "where?"
    assert sp.spoken == "I am in pain. Please help me." and sp.screen == "pain1" and sp.in_menu
    sp.select(1)
    assert sp.spoken == "The pain is in my chest." and sp.screen == "needs1"
    sp.select(1); sp.select(3)                   # PAIN > START skips the follow-up
    assert sp.screen == "needs1"
    press(sp, (3, 2))                            # MORE > TALK
    assert sp.screen == "talk1" and "YES" in sp.labels()[0]
    press(sp, (3, 3, 3))                         # MORE > MORE > START
    press(sp, (3, 3, 2))                         # MORE > MORE > MY PHRASES
    assert sp.screen == "my1" and sp.labels()[0] == "MY GLASSES"
    sp.select(0)
    assert sp.spoken == "Please put my glasses on." and sp.last_spoken == sp.spoken
    sp.ask()                                     # helper asked a question
    assert sp.screen == "answer"
    sp.select(1)
    assert sp.spoken == "No." and sp.screen == "needs1"
    sp.set_mode("postcard"); sp.ask()            # only in patient mode
    assert sp.screen == "root"


def test_call_bell_rest_and_mistake():
    sp = Speller(PRED, "patient")
    press(sp, (3, 0))                            # MORE > COME HERE: a bell, not the alarm
    assert sp.spoken == "Could you come here, please?" and sp.chime and not sp.alarm
    press(sp, (3, 3, 1))                         # MORE > MORE > REST
    assert sp.spoken == "I am going to rest now." and sp.screen == "rest1" and not sp.chime
    assert sp.labels()[:3] == ["", "", ""]
    for box in (0, 1, 2):                        # stray looks do nothing while resting
        assert sp.select(box) == "empty" and sp.screen == "rest1"
    sp.select(3)                                 # WAKE UP asks once more, on a different box
    assert sp.screen == "rest2"
    sp.select(3)                                 # still looking at the same corner: keep resting
    assert sp.screen == "rest1"
    press(sp, (3, 0))
    assert sp.screen == "needs1"
    press(sp, (3, 3, 2))                         # no personal phrases here: THAT WAS WRONG
    assert sp.spoken.startswith("Sorry, that was a mistake")


def test_two_box_easy_mode():
    sp = Speller(PRED, "patient", easy=True)
    assert sp.screen == "easy" and "WATER" in sp.labels()[0] and sp.labels()[1:3] == ["", ""]
    sp.select(0)
    assert sp.spoken == "I would like some water, please." and sp.screen == "easy"
    seen = []
    for _ in range(len(sp.easy_list)):           # NEXT walks through every option and wraps around
        seen.append(sp.labels()[0])
        sp.select(3)
    assert len(set(seen)) == len(seen) and "WATER" in sp.labels()[0]
    press(sp, (3, 0))                            # PAIN: spoken, no follow-up page in this mode
    assert sp.spoken == "I am in pain. Please help me." and sp.screen == "easy"
    press(sp, (3, 3, 0))                         # EMERGENCY still raises the alarm
    assert sp.alarm
    sp.clear_alarm()
    assert sp.screen == "easy"
    sp.ask()                                     # a question jumps straight to YES
    assert "YES" in sp.labels()[0]


class FakeAssistant:
    enabled = True

    def __init__(self, sentence):
        self.sentence, self.asked = sentence, []

    def expand(self, words):
        self.asked.append(words)

    def get(self, kind, text):
        return self.sentence


def test_suggested_answers_page():
    sp = Speller(PRED, "patient")
    sp.ask("What would you like for dinner?", [])            # waiting for suggestions: nothing to pick
    assert sp.screen == "asked" and sp.labels() == [""] * 4 and sp.hint().startswith("Q: What")
    sp.ask("What would you like for dinner?", ["PIZZA", "SOUP", "NOTHING"])
    assert sp.labels()[:3] == ["PIZZA", "SOUP", "NOTHING"] and "YES / NO" in sp.labels()[3]
    sp.select(1)
    assert sp.spoken == "Soup." and sp.screen == "needs1" and sp.question == ""
    sp.ask("Are you cold?", ["YES", "NO", "A LITTLE"])       # already has yes/no: box 4 leaves the page
    assert sp.labels()[3] == "SOMETHING ELSE"
    sp.ask("Are you cold?", ["PIZZA", "SOUP", "NOTHING"]); sp.select(3)
    assert sp.screen == "answer" and "YES" in sp.labels()[0]  # YES / NO page is one look away


def test_gemini_sentence_is_only_offered_never_spoken_by_itself():
    sp = Speller(PRED, "patient")
    sp.assistant = FakeAssistant("My feet are cold, may I have a blanket?")
    press(sp, (3, 3, 0)); sp.spoken = None
    spell(sp, "COLD FEET")
    press(sp, DONE)
    assert sp.screen == "say" and sp.spoken is None and sp.assistant.asked == ["COLD FEET"]
    assert sp.labels() == ["SAY IT", "SAY GEMINI'S", "KEEP TYPING", "CANCEL"]
    sp.select(1)
    assert sp.spoken == "My feet are cold, may I have a blanket?" and sp.text == "" and sp.screen == "needs1"
    press(sp, (3, 3, 0)); spell(sp, "HI"); press(sp, DONE)
    sp.select(0)                                             # the person's own words
    assert sp.spoken == "Hi."
    press(sp, (3, 3, 0)); spell(sp, "HI"); press(sp, DONE)
    sp.select(2)                                             # KEEP TYPING: text is kept
    assert sp.screen == "root" and sp.text == "HI"
    press(sp, DONE); sp.spoken = None; sp.select(3)          # CANCEL: nothing is said
    assert sp.spoken is None and sp.text == "" and sp.screen == "needs1"
    sp.assistant = FakeAssistant(False)                      # Gemini had nothing: only the person's words
    press(sp, (3, 3, 0)); spell(sp, "HI"); press(sp, DONE)
    assert sp.labels()[1] == ""


def test_every_builtin_phrase_has_a_translation_and_unknown_text_falls_back():
    from speller.languages import LANGUAGES, Translator, builtin_phrases
    tr = Translator()                            # no Gemini: only the built-in table
    phrases = builtin_phrases()
    assert "Emergency! I need help right now." in phrases and len(phrases) >= 25
    for code, name in LANGUAGES[1:]:
        missing = [p for p in phrases if not tr.lookup(p, code)]
        assert not missing, (name, missing)
    assert tr.lookup("Yes.", "es") == "Sí." and tr.lookup("Yes.", "en") == "Yes."
    heard = []
    tr.translate("I would like some water, please.", "es", heard.append)
    tr.translate("My cat is called Biscuit.", "es", heard.append)      # not in the table, no Gemini
    assert heard[0].startswith("Quisiera") and heard[1] == "My cat is called Biscuit."


def test_person_picks_mode_and_language_by_looking():
    langs = [("en", "English"), ("es", "Spanish"), ("hi", "Hindi"), ("fr", "French"), ("sw", "Swahili")]
    sp = Speller(PRED, languages=langs)
    sp.welcome()
    assert sp.labels() == ["POSTCARD", "PATIENT", "VOICE: ENGLISH", ""] and sp.in_menu
    sp.select(2)                                 # languages, three per page
    assert sp.labels() == ["ENGLISH", "SPANISH", "HINDI", "MORE ▶"]
    sp.select(3)
    assert sp.labels()[:2] == ["FRENCH", "SWAHILI"] and "BACK" in sp.labels()[3]
    sp.select(1)
    assert sp.language == "sw" and sp.screen == "welcome" and sp.labels()[2] == "VOICE: SWAHILI"
    sp.select(1)                                 # PATIENT
    assert sp.mode == "patient" and sp.screen == "needs1" and sp.language == "sw"
    sp.welcome(); sp.select(0)                   # POSTCARD
    assert sp.mode == "postcard" and sp.screen == "root" and "QUICK" in sp.labels()[3]
    sp.welcome(); sp.select(3)                   # the empty box does nothing
    assert sp.screen == "welcome"


def test_llm_answer_must_match_and_arrive_in_time():
    p = Predictor(llm_deadline_s=0.0)        # deadline already passed for late answers
    local = p.suggest("I WANT W")
    p.llm_cache["I WANT W"] = "WAFFLES"      # Gemini answers late: box must not change under the user's eyes
    assert p.suggest("I WANT W") == local
    p.suggest("I WANT ")                     # ...but next time this text comes up, the answer is used
    assert p.suggest("I WANT W") == "WAFFLES" and p.last_source == "llm"
    p.llm_cache["I WANT WA"] = "COFFEE"      # does not match the typed letters: ignored
    assert p.suggest("I WANT WA") == "WATER" and p.last_source == "local"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   ", name)
        except Exception as e:
            failed += 1
            print("FAIL ", name, "->", repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
