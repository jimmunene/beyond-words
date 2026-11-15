"""Speller logic: which 4 labels are on screen, and what happens when a box is selected.

Two modes:
  postcard : spell a short message (letters via 3 looks, suggested word via 1 look),
             or pick a ready-made one from the quick lane (2 looks, for long queues)
  patient  : one-look needs (WATER, PAIN, ...) + full spelling when needed, and a one-look
             EMERGENCY box that raises an alarm until a helper acknowledges it.
             PAIN is spoken at once and then asks where it hurts. TALK holds everyday social
             phrases and the person's own phrases (data/my_phrases.txt). A helper can ask a
             question and call up a YES / NO / NOT SURE screen (Speller.ask()).
Word prediction: box 4 on the first screen finishes the most likely word in one look.
(Optional, off by default: Speller(group_words=True) also offers a second word on the A-I / J-R pages.)
Pure Python, no pygame, so it can be tested with the keyboard or unit tests.

    python -m speller.state_machine            # text version, postcard mode
    python -m speller.state_machine patient
"""

# A screen = list of 4 (label, action) tuples. None = empty box (not shown, not selectable).
BACK = ("← BACK", ("back",))

# Quick lane: ready-made postcards. First three on page 1, next three on page 2.
QUICK_MESSAGES = ["HI MOM", "HELLO EMU", "I LOVE YOU", "THANK YOU", "HACK THE NEST", "MIND = BLOWN"]

PATIENT_MAX_CHARS = 80      # patients spell real sentences; the postcard limit does not apply

# Spoken (and repeated) while the emergency alarm is on. It is a normal "phrase" action with an
# extra "alarm" marker, so voice/precache.py caches it like every other phrase.
EMERGENCY_PHRASE = "Emergency! I need help right now."
MISTAKE = "Sorry, that was a mistake. Please ignore it."
TYPING_NOTICE = "I am typing a message. Please wait."       # spoken when spelling starts (patient mode)


def _letters(chars):
    items = [(c if c != " " else "SPACE", ("char", c)) for c in chars]
    return items + [BACK] * (4 - len(items)) if len(items) < 4 else items


def phrase_screens(phrases, name="my"):
    """The person's own phrases, [(LABEL, sentence to speak)]: up to 6 on two pages."""
    items = [(lab.strip().upper(), ("phrase", text.strip())) for lab, text in phrases
             if lab.strip() and text.strip()][:6]
    page1, page2 = items[:3], items[3:6]
    page1 += [None] * (3 - len(page1))
    page2 += [None] * (3 - len(page2))
    page1.append(("MORE ▶", ("goto", f"{name}2")) if page2[0] else ("← START", ("needs",)))
    page2.append(("← START", ("needs",)))
    return {f"{name}1": page1, f"{name}2": page2}


def load_phrases(path):
    """Read 'LABEL | sentence' lines (# = comment). Missing file = no personal phrases."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                label, sep, text = line.partition("|")
                if sep and not line.lstrip().startswith("#"):
                    out.append((label.strip(), text.strip()))
    except OSError:
        pass
    return out


def quick_screens(messages):
    """Two pages of ready-made messages: 3 + MORE, then 3 + BACK (BACK leaves the quick lane,
    so nobody gets stuck there: QUICK > MORE > BACK is back at the letters)."""
    msgs = [m.strip().upper() for m in messages if m and m.strip()][:6]
    page1 = [(m, ("quick", m)) for m in msgs[:3]]
    page2 = [(m, ("quick", m)) for m in msgs[3:6]]
    page1 += [None] * (3 - len(page1))
    page2 += [None] * (3 - len(page2))
    page1.append(("MORE ▶", ("goto", "quick2")) if msgs[3:] else ("← BACK", ("home",)))
    page2.append(("← BACK", ("home",)))
    return {"quick1": page1, "quick2": page2}


SCREENS = {
    # Level 1: groups. Box 4 is filled live: quick lane (nothing typed yet) or the suggested word.
    "root": [("A – I", ("goto", "AI")), ("J – R", ("goto", "JR")),
             ("S – Z", ("goto", "SZ")), ("", ("suggest",))],
    # Level 2
    "AI": [("A B C", ("goto", "ABC")), ("D E F", ("goto", "DEF")), ("G H I", ("goto", "GHI")), BACK],
    "JR": [("J K L", ("goto", "JKL")), ("M N O", ("goto", "MNO")), ("P Q R", ("goto", "PQR")), BACK],
    "SZ": [("S T U", ("goto", "STU")), ("V W X", ("goto", "VWX")), ("Y Z SPACE", ("goto", "YZ_")),
           ("EDIT", ("goto", "CTRL"))],
    # Level 3
    "ABC": _letters("ABC"), "DEF": _letters("DEF"), "GHI": _letters("GHI"),
    "JKL": _letters("JKL"), "MNO": _letters("MNO"), "PQR": _letters("PQR"),
    "STU": _letters("STU"), "VWX": _letters("VWX"), "YZ_": _letters("YZ "),
    # Box 4 is empty in postcard mode, "NEEDS" in patient mode (see Speller.items)
    "CTRL": [("DELETE", ("delete",)), ("DONE", ("done",)), BACK, None],
    # Patient mode
    "needs1": [("💧 WATER", ("phrase", "I would like some water, please.")),
               ("😣 PAIN", ("phrase", "I am in pain. Please help me.", "then", "pain1")),
               ("🚨 EMERGENCY", ("phrase", EMERGENCY_PHRASE, "alarm")),
               ("MORE ▶", ("goto", "needs2"))],
    "needs2": [("🔔 COME HERE", ("phrase", "Could you come here, please?", "bell")),   # call bell, not an alarm
               ("🛏 BODY", ("goto", "body1")),
               ("💬 TALK", ("goto", "talk1")),
               ("MORE ▶", ("goto", "needs3"))],
    "needs3": [("✎ SPELL", ("goto", "root")),
               ("😴 REST", ("phrase", "I am going to rest now.", "then", "rest1")),
               ("✗ THAT WAS WRONG", ("phrase", MISTAKE)),     # replaced by MY PHRASES when there are any
               ("← START", ("needs",))],
    "body1": [("🚻 TOILET", ("phrase", "I need to use the toilet.")),
              ("🍽 FOOD", ("phrase", "I am hungry.")),
              ("🛏 MOVE ME", ("phrase", "Please help me change position.")),
              ("MORE ▶", ("goto", "body2"))],
    "body2": [("🥶 COLD / HOT", ("phrase", "I am uncomfortable. Please check my temperature.")),
              ("📞 CALL SOMEONE", ("phrase", "Please call my family.")),
              ("✗ THAT WAS WRONG", ("phrase", MISTAKE)),
              ("← START", ("needs",))],
    # Resting: nothing can be picked by accident. Waking takes two looks at two different boxes.
    "rest1": [None, None, None, ("WAKE UP", ("goto", "rest2"))],
    "rest2": [("YES, WAKE UP", ("needs",)), None, None, ("KEEP RESTING", ("goto", "rest1"))],
    "_spoken": [("", ("phrase", TYPING_NOTICE)), None, None, None],   # never shown: lets precache.py find it
    # After PAIN (already spoken): where does it hurt? Every answer is spoken; START skips it.
    "pain1": [("HEAD", ("phrase", "The pain is in my head.")),
              ("CHEST", ("phrase", "The pain is in my chest.")),
              ("MORE ▶", ("goto", "pain2")),
              ("← START", ("needs",))],
    "pain2": [("STOMACH", ("phrase", "The pain is in my stomach.")),
              ("MY BACK", ("phrase", "The pain is in my back.")),
              ("ARMS / LEGS", ("phrase", "The pain is in my arms or legs.")),
              ("IT IS SEVERE", ("phrase", "The pain is severe. I need help now."))],
    # Everyday social phrases: people want to talk, not only to ask for things.
    "talk1": [("👍 YES", ("phrase", "Yes.")), ("👎 NO", ("phrase", "No.")),
              ("🙏 THANK YOU", ("phrase", "Thank you.")), ("MORE ▶", ("goto", "talk2"))],
    "talk2": [("👋 HELLO", ("phrase", "Hello!")), ("❤ I LOVE YOU", ("phrase", "I love you.")),
              ("HOW ARE YOU?", ("phrase", "How are you?")), ("MORE ▶", ("goto", "talk3"))],
    "talk3": [("I AM OKAY", ("phrase", "I am okay.")), ("SORRY", ("phrase", "I am sorry.")),
              ("✗ THAT WAS WRONG", ("phrase", MISTAKE)), ("← START", ("needs",))],
    # A helper asked a question (Speller.ask()): the fastest possible conversation.
    "answer": [("👍 YES", ("phrase", "Yes.")), ("👎 NO", ("phrase", "No.")),
               ("NOT SURE", ("phrase", "I am not sure.")), ("SOMETHING ELSE", ("needs",))],
    **quick_screens(QUICK_MESSAGES),
}
# Emoji may not render in every pygame font; ui.py strips them if needed.

HINTS = {
    "root": "Look at the box that holds your letter",
    "AI": "Now the smaller group", "JR": "Now the smaller group", "SZ": "Now the smaller group",
    "CTRL": "Delete, finish, or go back",
    "quick1": "Pick a ready-made message", "quick2": "Pick a ready-made message",
    "needs1": "Look at what you need", "needs2": "Look at what you need", "needs3": "Look at what you need",
    "body1": "Look at what you need", "body2": "Look at what you need",
    "rest1": "Resting. Look at WAKE UP when you are ready", "rest2": "Wake up now?",
    "talk3": "Pick something to say",
    "pain1": "Where does it hurt?", "pain2": "Where does it hurt?",
    "talk1": "Pick something to say", "talk2": "Pick something to say",
    "my1": "Pick something to say", "my2": "Pick something to say",
    "answer": "Answer the question",
    "welcome": "Look at what you would like to do",
}
MENUS = ("needs", "pain", "talk", "my", "answer", "body", "rest", "easy", "asked", "say", "welcome", "lang")          # screens that are not part of spelling
GROUP_LETTERS = {"AI": "ABCDEFGHI", "JR": "JKLMNOPQR"}     # letter-group pages that can offer a word


def easy_options(screens):
    """Two-box mode for a weak signal: one option at a time (top-left) and NEXT (bottom-right).
    The most urgent things come first; follow-up pages are skipped."""
    order = [("needs1", 0), ("needs1", 1), ("needs1", 2), ("needs2", 0), ("talk1", 0), ("talk1", 1),
             ("body1", 0), ("body1", 1), ("body1", 2), ("body2", 0), ("talk1", 2)]
    out = []
    for screen, i in order:
        label, action = screens[screen][i]
        out.append((label, tuple(a for a in action if a != "then" and not str(a).startswith("pain"))))
    return out


class Speller:
    def __init__(self, predictor, mode="postcard", max_chars=12, quick_messages=None,
                 patient_max_chars=PATIENT_MAX_CHARS, my_phrases=None, easy=False, group_words=False,
                 languages=None):
        self.predictor = predictor
        self.postcard_max, self.patient_max = max_chars, patient_max_chars
        self.screens = dict(SCREENS)
        if quick_messages:
            self.screens.update(quick_screens(quick_messages))
        self.screens.update(phrase_screens(my_phrases or []))
        if self.screens["my1"][0]:
            self.screens["needs3"] = self.screens["needs3"][:2] + [("★ MY PHRASES", ("goto", "my1"))] + \
                self.screens["needs3"][3:]
        # Off by default: box 4 on the A-I / J-R pages stays BACK, like every other page. Turning it
        # on offers a second word there (fewer looks, but BACK disappears whenever a word shows).
        self.group_words = group_words
        self.easy = easy                        # patient mode with two boxes only (weak signal)
        self.easy_list, self.easy_i = easy_options(self.screens), 0
        self.assistant = None                   # speller.assistant.Assistant: offers a fuller sentence (optional)
        self.question = ""                      # what the helper asked (shown while answering)
        self.say_text = ""                      # spelled sentence waiting on the "say it?" page
        self.chime = False                      # the phrase just picked wants the call bell (UI plays it)
        self.last_spoken = ""
        # The person can choose the mode and the spoken language themselves (welcome page).
        self.languages = list(languages or [("en", "English")])     # [(code, name)]
        self.language = self.languages[0][0]
        pages = [self.languages[i:i + 3] for i in range(0, len(self.languages), 3)]
        for n, page in enumerate(pages, 1):
            items = [(name.upper(), ("language", code)) for code, name in page] + [None] * (3 - len(page))
            items.append(("MORE ▶", ("goto", f"lang{n + 1}")) if n < len(pages) else ("← BACK", ("welcome",)))
            self.screens[f"lang{n}"] = items
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        if self.predictor is not None:          # care words (body, end of life) only in patient mode
            self.predictor.clinical = mode == "patient"
        self.text = ""
        self.screen = self._menu_home() if mode == "patient" else "root"
        self.stack = []
        self.history = []         # earlier texts, so DELETE undoes a whole suggested word
        self.spoken = None        # phrase to show/speak (patient mode)
        self.finished = False     # DONE pressed (postcard mode)
        self.alarm = False        # EMERGENCY picked: stays on until a helper calls clear_alarm()
        self._group_words = {}    # (text, screen) -> second suggestion shown on a letter-group page

    # ------------------------------------------------------------ text helpers
    @property
    def max_chars(self):
        return self.patient_max if self.mode == "patient" else self.postcard_max

    @property
    def full(self):
        return len(self.text) >= self.max_chars

    @property
    def suggestion(self):
        """Best word for box 4. Only words that still fit in the message are offered."""
        if not self.predictor:
            return None
        room = self.max_chars - (len(self.text) - len(self.text.split(" ")[-1]))
        if room <= 0:
            return None
        try:
            word = self.predictor.suggest(self.text, max_len=room)
        except TypeError:                       # a predictor without the max_len option
            word = self.predictor.suggest(self.text)
        return word if word and len(word) <= room else None

    @property
    def in_menu(self):
        """True on the needs / talk / answer pages (anything that is not spelling)."""
        return self.screen.startswith(MENUS)

    def group_word(self, screen):
        """A-I and J-R pages: the likeliest word (other than box 4's) whose next letter is in this
        group. The user came here for a letter of this group, so this is the word they may want."""
        if not self.group_words or screen not in GROUP_LETTERS or not hasattr(self.predictor, "candidates"):
            return None
        key = (self.text, screen)
        if key not in self._group_words:
            words = self.text.split(" ")
            room, n, top = self.max_chars - (len(self.text) - len(words[-1])), len(words[-1]), self.suggestion
            found = None
            if room > 0:
                for w in self.predictor.candidates(self.text, 40, max_len=room):
                    if w != top and len(w) > n and w[n] in GROUP_LETTERS[screen]:
                        found = w
                        break
            self._group_words[key] = found
        return self._group_words[key]

    @property
    def expanded(self):
        """Gemini's fuller version of the spelled words: a sentence, None (waiting) or False (none)."""
        if self.screen != "say" or not self.assistant:
            return False
        return self.assistant.get("expand", self.text)

    def welcome(self):
        """First page for a new person: they pick postcard or patient mode, and the voice's
        language, by looking, without a helper touching the keyboard."""
        self.text, self.history, self.stack, self.finished, self.spoken = "", [], [], False, None
        self.screen = "welcome"

    def ask(self, question="", options=None):
        """A helper asked a question (patient mode). Without options: YES / NO / NOT SURE.
        options=[] shows an empty page while suggestions are fetched; options=[...] shows up to
        three suggested answers as one-look boxes."""
        if self.mode == "patient" and not self.alarm:
            self.screen, self.stack, self.question = "answer", [], question.strip()
            if options is not None and not self.easy:
                picks = [(o, ("phrase", o.capitalize() + ".")) for o in options[:3]]
                last = None if not picks else ("SOMETHING ELSE", ("needs",)) if any(o in ("YES", "NO") for o in options)                     else ("YES / NO ▶", ("goto", "answer"))
                self.screens["asked"] = picks + [None] * (3 - len(picks)) + [last]
                self.screen = "asked"
            if self.easy:                                   # two-box mode: step through YES, NO, ...
                self.screen = "easy"
                self.easy_i = next(i for i, it in enumerate(self.easy_list) if it[1][1] == "Yes.")

    def hint(self):
        """One short line telling the user what to do on this screen."""
        if self.alarm:
            return "Help is being called"
        if self.screen in ("asked", "answer") and self.question:
            return "Q: " + self.question
        if self.screen == "say":
            return "Say it?"
        if self.screen.startswith("lang"):
            return "Which language should the voice speak?"
        if self.screen == "easy":
            return f"Look left to say it, or NEXT for another  ({self.easy_i + 1} of {len(self.easy_list)})"
        if self.screen in HINTS:
            if self.screen == "root" and self.full:
                return "Message is full: S - Z, EDIT, then DONE"
            if self.screen == "root" and self.text:       # typed text is only spoken / sent after DONE
                return ("To say it aloud: S - Z, then EDIT, then DONE" if self.mode == "patient"
                        else "Finished? S - Z, then EDIT, then DONE")
            if self.group_word(self.screen):
                return "Now the smaller group, or the whole word"
            return HINTS[self.screen]
        return "Now pick the letter"

    # ------------------------------------------------------------ screen content
    def items(self):
        """The 4 (label, action) items on screen right now (None = empty box)."""
        if self.alarm:                          # nothing can be picked (or cancelled) by a stray look
            return [None] * 4
        if self.screen == "easy":
            return [self.easy_list[self.easy_i], None, None, ("NEXT ▶", ("next",))]
        if self.screen == "welcome":
            name = dict(self.languages).get(self.language, "English")
            return [("POSTCARD", ("mode", "postcard")), ("PATIENT", ("mode", "patient")),
                    (f"VOICE: {name.upper()}", ("goto", "lang1")) if len(self.languages) > 1 else None, None]
        if self.screen == "say":                            # spelled words: say them, or Gemini's sentence
            better = self.expanded
            return [("SAY IT", ("say", self.say_text)),
                    ("SAY GEMINI'S", ("say", better)) if better else None,
                    ("KEEP TYPING", ("home",)), ("CANCEL", ("cancel",))]
        items = list(self.screens[self.screen])
        if self.screen == "root":
            if self.mode == "postcard" and not self.text and self.screens["quick1"][0]:
                items[3] = ("★ QUICK", ("goto", "quick1"))
            else:
                s = self.suggestion
                if not s:
                    items[3] = None
                elif s == self.text.split(" ")[-1]:          # the word is complete: one look adds the space
                    items[3] = (f"★ {s} ␣", ("suggest", s))
                else:
                    items[3] = (f"★ {s}", ("suggest", s))
        elif self.screen == "CTRL" and self.mode == "patient":
            items[3] = ("☰ NEEDS", ("needs",))
        elif self.screen in GROUP_LETTERS:
            w = self.group_word(self.screen)
            if w:                                           # takes the place of BACK on this page
                items[3] = (f"★ {w}", ("suggest", w))
        return items

    def labels(self):
        return [it[0] if it else "" for it in self.items()]

    # ------------------------------------------------------------ selection
    def _set_text(self, new):
        if new != self.text:
            self.history.append(self.text)
            self.text = new

    def select(self, box):
        """Apply a selection (box 0-3). Returns a short event string for logging."""
        item = self.items()[box]
        if item is None:
            return "empty"
        label, action = item
        kind = action[0]
        if kind == "goto":
            self.stack.append(self.screen)
            self.screen = action[1]
            if action[1] == "root" and self.mode == "patient" and not self.text:
                self.spoken = TYPING_NOTICE                 # so people wait instead of walking away
        elif kind == "next":
            self.easy_i = (self.easy_i + 1) % len(self.easy_list)
        elif kind == "back":
            self.screen = self.stack.pop() if self.stack else self._home()
            if self.screen != "root" and not any(it and it[1][0] == "back" for it in self.items()):
                self._go_home()                             # that page has no BACK of its own: go to the start
        elif kind == "char":
            c = action[1]
            if not self.full and not (c == " " and (not self.text or self.text.endswith(" "))):
                self._set_text(self.text + c)
            self._go_home()
        elif kind == "delete":
            # undo the last thing typed: one letter, or a whole suggested word
            self.text = self.history.pop() if self.history else self.text[:-1]
            self._go_home()
        elif kind == "suggest":
            word = action[1] if len(action) > 1 else self.suggestion
            if word:
                parts = self.text.split(" ")
                parts[-1] = word
                new = " ".join(parts)
                self._set_text(new + " " if len(new) < self.max_chars else new)
            self._go_home()
        elif kind == "quick":
            self._set_text(action[1])
            self.stack = []
            if self.mode == "patient":
                self.spoken, self.text, self.history, self.screen = action[1].capitalize() + ".", "", [], self._menu_home()
            else:
                self.screen, self.finished = "root", True
        elif kind == "done":
            self.text = self.text.strip()
            if self.mode == "patient":
                if self.text and getattr(self.assistant, "enabled", False):
                    self.say_text = self.text.capitalize() + "."     # ask Gemini for a fuller sentence and
                    self.assistant.expand(self.text)                  # let the person choose what is said
                    self.screen, self.stack = "say", []
                    return "say:?"
                if self.text:
                    self.spoken = self.last_spoken = self.text.capitalize() + "."
                self.text, self.history = "", []
                self.screen, self.stack = self._menu_home(), []
            elif self.text:
                self.finished = True
            else:                                # nothing typed yet: DONE does nothing
                self._go_home()
                return "empty_done"
        elif kind == "say":
            self.spoken = self.last_spoken = action[1]
            self.text, self.history = "", []
            self.screen, self.stack = self._menu_home(), []
        elif kind == "cancel":
            self.text, self.history = "", []
            self.screen, self.stack = self._menu_home(), []
        elif kind == "home":
            self._go_home()
        elif kind == "mode":
            self.set_mode(action[1])
        elif kind == "language":
            self.language = action[1]
            self.welcome()
        elif kind == "welcome":
            self.welcome()
        elif kind == "needs":
            self.screen, self.stack = self._menu_home(), []
        elif kind == "phrase":
            self.spoken = self.last_spoken = action[1]
            self.screen, self.stack = self._menu_home(), []
            if "alarm" in action[2:]:
                self.alarm = True
            elif "then" in action[2:]:                      # spoken at once, then a follow-up page
                self.screen = action[action.index("then") + 1]
            self.chime = "bell" in action[2:]
        return f"{self.screen}:{label}"

    def clear_alarm(self):
        """A helper has seen the alarm (keyboard, never the patient's gaze): back to the needs."""
        self.alarm = False
        self.screen, self.stack = self._menu_home(), []

    def _menu_home(self):
        """First patient page: the needs, or the single-option page in two-box mode."""
        self.easy_i, self.question = 0, ""
        return "easy" if getattr(self, "easy", False) else "needs1"

    def _home(self):
        return "root"

    def _go_home(self):
        self.screen, self.stack = "root", []


if __name__ == "__main__":
    import sys

    from speller.predictor import Predictor  # run as: python -m speller.state_machine
    sp = Speller(Predictor(), sys.argv[1] if len(sys.argv) > 1 else "postcard")
    print("Type 1-4 to select, q to quit")
    while not sp.finished:
        if sp.spoken:
            print(f'\n  SPEAKS: "{sp.spoken}"')
            sp.spoken = None
        if sp.alarm:
            input("  *** EMERGENCY ALARM *** helper presses Enter to acknowledge > ")
            sp.clear_alarm()
        labels = [lab.encode("ascii", "replace").decode() for lab in sp.labels()]
        print(f"\n[{sp.text}]   {sp.hint()}\n   " + " | ".join(f"{i + 1}:{lab}" for i, lab in enumerate(labels)))
        k = input("> ").strip()
        if k == "q":
            break
        if k and k in "1234":
            sp.select(int(k) - 1)
    print("Final:", sp.text)
