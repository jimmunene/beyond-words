"""Speller logic: which 4 labels are on screen, and what happens when a box is selected.

Two modes:
  postcard : spell a short message (letters via 3 looks, suggested word via 1 look)
  patient  : one-look needs (WATER, PAIN, ...) + full spelling when needed
Pure Python, no pygame, so it can be tested with the keyboard or unit tests.
"""

# A screen = list of 4 (label, action) tuples. None = empty box (not shown, not selectable).
BACK = ("← BACK", ("back",))


def _letters(chars):
    items = [(c if c != " " else "␣ SPACE", ("char", c)) for c in chars]
    return items + [BACK] * (4 - len(items)) if len(items) < 4 else items


SCREENS = {
    # Level 1: groups. Box 4 is filled live with the suggested word.
    "root": [("A – I", ("goto", "AI")), ("J – R", ("goto", "JR")),
             ("S – Z  ␣ ⌫ ✓", ("goto", "SZ")), ("", ("suggest",))],
    # Level 2
    "AI": [("A B C", ("goto", "ABC")), ("D E F", ("goto", "DEF")), ("G H I", ("goto", "GHI")), BACK],
    "JR": [("J K L", ("goto", "JKL")), ("M N O", ("goto", "MNO")), ("P Q R", ("goto", "PQR")), BACK],
    "SZ": [("S T U", ("goto", "STU")), ("V W X", ("goto", "VWX")), ("Y Z ␣", ("goto", "YZ_")),
           ("⌫  ✓  ←", ("goto", "CTRL"))],
    # Level 3
    "ABC": _letters("ABC"), "DEF": _letters("DEF"), "GHI": _letters("GHI"),
    "JKL": _letters("JKL"), "MNO": _letters("MNO"), "PQR": _letters("PQR"),
    "STU": _letters("STU"), "VWX": _letters("VWX"), "YZ_": _letters("YZ "),
    "CTRL": [("⌫ DELETE", ("delete",)), ("✓ DONE", ("done",)), BACK, None],
    # Patient mode
    "needs1": [("💧 WATER", ("phrase", "I would like some water, please.")),
               ("😣 PAIN", ("phrase", "I am in pain. Please help me.")),
               ("🚻 TOILET", ("phrase", "I need to use the toilet.")),
               ("MORE ▶", ("goto", "needs2"))],
    "needs2": [("🍽 FOOD", ("phrase", "I am hungry.")),
               ("🥶 COLD / HOT", ("phrase", "I am uncomfortable. Please check my temperature.")),
               ("🛏 MOVE ME", ("phrase", "Please help me change position.")),
               ("MORE ▶", ("goto", "needs3"))],
    "needs3": [("👍 YES", ("phrase", "Yes.")), ("👎 NO", ("phrase", "No.")),
               ("📞 CALL SOMEONE", ("phrase", "Please call my family.")),
               ("✎ SPELL", ("goto", "root"))],
}
# Emoji may not render in every pygame font; ui.py strips them if needed.


class Speller:
    def __init__(self, predictor, mode="postcard", max_chars=12):
        self.predictor, self.max_chars = predictor, max_chars
        self.set_mode(mode)

    def set_mode(self, mode):
        self.mode = mode
        self.text = ""
        self.screen = "needs1" if mode == "patient" else "root"
        self.stack = []
        self.spoken = None        # phrase to show/speak (patient mode)
        self.finished = False     # DONE pressed (postcard mode)

    @property
    def suggestion(self):
        return self.predictor.suggest(self.text) if self.predictor else None

    def labels(self):
        items = list(SCREENS[self.screen])
        if self.screen == "root":
            s = self.suggestion
            items[3] = (f"★ {s}", ("suggest",)) if s else None
        return [it[0] if it else "" for it in items]

    def select(self, box):
        """Apply a selection (box 0-3). Returns a short event string for logging."""
        items = list(SCREENS[self.screen])
        if self.screen == "root" and not self.suggestion:
            items[3] = None
        item = items[box]
        if item is None:
            return "empty"
        label, action = item
        kind = action[0]
        if kind == "goto":
            self.stack.append(self.screen)
            self.screen = action[1]
        elif kind == "back":
            self.screen = self.stack.pop() if self.stack else self._home()
        elif kind == "char":
            if len(self.text) < self.max_chars:
                if not (action[1] == " " and (not self.text or self.text.endswith(" "))):
                    self.text += action[1]
            self._go_home()
        elif kind == "delete":
            self.text = self.text[:-1]
            self._go_home()
        elif kind == "suggest":
            word = self.suggestion
            parts = self.text.split(" ")
            parts[-1] = word
            self.text = (" ".join(parts) + " ")[: self.max_chars + 1]
        elif kind == "done":
            self.text = self.text.strip()
            if self.mode == "patient":
                self.spoken = self.text.capitalize() + "."
                self.text = ""
                self.screen, self.stack = "needs1", []
            else:
                self.finished = True
        elif kind == "phrase":
            self.spoken = action[1]
            self.screen, self.stack = "needs1", []
        return f"{self.screen}:{label}"

    def _home(self):
        return "root"

    def _go_home(self):
        self.screen, self.stack = "root", []


if __name__ == "__main__":
    from speller.predictor import Predictor  # run as: python -m speller.state_machine
    sp = Speller(Predictor())
    print("Type 1-4 to select, q to quit")
    while not sp.finished:
        print(f"\n[{sp.text}]   " + " | ".join(f"{i + 1}:{lab}" for i, lab in enumerate(sp.labels())))
        k = input("> ").strip()
        if k == "q":
            break
        if k in "1234" and k:
            sp.select(int(k) - 1)
    print("Final:", sp.text)
