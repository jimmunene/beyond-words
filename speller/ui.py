"""BrainType speller UI (pygame).

    python speller/ui.py --input keyboard   # keys 1-4 select directly (UI dev, demo backup)
    python speller/ui.py --input sim        # keys 1-4 = where you "look"; synthetic EEG -> classifier -> UI
    python speller/ui.py --input sim --signal replay   # same, with REAL recorded EEG (acquisition/dataset_stream.py)
    python speller/ui.py --input eeg        # live headset (or replay) via classifier_node.py
    add --windowed for a window instead of fullscreen, --display 1 for the second monitor, --mute for no sounds

Operator keys:
  Intro screen : type nickname, ENTER = start visitor, F1 postcard mode, F2 patient mode, F3 next voice
                 (person_chooses: true in config.yaml: the person picks mode and language by looking instead)
  While typing : P pause/resume, T cued accuracy test on/off, ESC back to intro
                 keyboard mode does not flash the boxes (F toggles it, --flicker starts with it on)
                 patient mode: Q = you asked a yes/no question (shows YES / NO / NOT SURE),
                 R = say the last phrase again, SPACE = acknowledge an emergency alarm
                 (sim mode: 1-4 = look at a box, 0 = look away)
  Result screen: Y = post to gallery, N = keep private, SPACE = next visitor
                 L = next spoken language (start screen: F5)
                 sim / dataset replay: A = automatic demo on / off (types demo_message, or says
                 demo_phrases in patient mode). Dataset replay starts in automatic demo.
  Anywhere     : TAB = show all keys. Start screen: F4 = two-box easy mode (patient, weak signal)
"""
import argparse
import csv
import inspect
import json
import os
import pathlib
import random
import sys
import threading
import time

import numpy as np
import pygame

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from art.postcard import make_postcard  # noqa: E402
from common import ROOT, UDPReceiver, UDPSender, load_config  # noqa: E402
from speller.assistant import Assistant  # noqa: E402
from speller.care_sync import CareSync  # noqa: E402
from speller.languages import LANGUAGES, V3_LANGS, Translator, prefetch  # noqa: E402
from speller.notify import PhoneAlert  # noqa: E402
from speller.predictor import Predictor  # noqa: E402
from speller.sounds import SoundBank  # noqa: E402
from speller.state_machine import SCREENS, TYPING_NOTICE, Speller, load_phrases  # noqa: E402
from upload.base44_upload import upload  # noqa: E402
from voice.tts import speak  # noqa: E402

try:                                            # optional voice picker (Person C, voice/voices.py)
    from voice.voices import VOICES  # noqa: E402
except Exception:
    VOICES = []

# Symbols the chosen font may not have -> plain replacements, so nothing renders as an empty box.
ASCII = {"←": "<", "★": "*", "⌫": "DEL", "✓": "OK", "␣": "_", "–": "-", "▶": ">", "✎": "", "—": "-",
         "☰": "", "·": "-", "…": "..."}

BG = (9, 10, 16)
PANEL = (22, 24, 36)
LINE = (52, 56, 78)
WHITE = (245, 246, 250)
MUTED = (172, 176, 198)                         # secondary text: about 9:1 against the background
DIM = (128, 132, 156)                           # quiet hints: still above 5:1, readable from a distance
YELLOW = (255, 214, 10)
GREEN = (40, 230, 130)
CYAN = (70, 200, 255)
RED = (255, 84, 84)
AMBER = (255, 170, 60)
VIOLET = (150, 112, 255)
PINK = (255, 96, 162)
BRAND = (CYAN, VIOLET, PINK)                    # title / accent gradient

# Box 4 (and anything that only moves between pages) is drawn smaller and quieter than a real choice.
NAV_WORDS = ("MORE", "BACK", "START", "NEXT", "KEEP RESTING", "NEEDS", "SOMETHING ELSE", "YES / NO",
             "KEEP TYPING", "CANCEL")
# One short line under a box that opens another page, so nobody has to guess what is inside.
SUBTITLES = {
    "S - Z": "space, delete, done", "EDIT": "delete, done, back", "QUICK": "ready-made messages",
    "BODY": "toilet, food, position", "TALK": "yes, no, thank you", "SPELL": "write your own words",
    "REST": "pause the screen", "MY PHRASES": "your own sayings", "COME HERE": "rings a bell",
    "PAIN": "then: where it hurts", "EMERGENCY": "sounds the alarm",
    "SAY IT": "as you typed it", "SAY GEMINI'S": "the fuller sentence",
    "POSTCARD": "spell a short message", "PATIENT": "say what you need",
}


class App:
    def __init__(self, args):
        self.cfg = load_config()
        self.args = args
        self.refresh = float(self.cfg["refresh_rate"])
        pygame.init()
        self._open_display()
        pygame.display.set_caption("BrainType Postcards")
        self.W, self.H = self.screen.get_size()
        self.clock = pygame.time.Clock()

        self.font_name = next((n for n in ("segoeui", "arial", "dejavusans", "helvetica", "liberationsans")
                               if pygame.font.match_font(n)), None)
        self.sym_name = next((n for n in ("segoeuisymbol", "dejavusans", "applesymbols", "notosanssymbols2",
                                          "symbola", "arialunicodems") if pygame.font.match_font(n)), None)
        self._fonts, self._glyphs, self._surfs = {}, {}, {}
        # Windows hands out Segoe UI *Light* when a non-bold Segoe UI is asked for: thin strokes that
        # are hard to read at small sizes from a distance. Use the Regular file for all small text.
        heavy = pygame.font.match_font(self.font_name, bold=True) if self.font_name == "segoeui" else None
        regular = pathlib.Path(heavy).with_name("segoeui.ttf") if heavy else None
        self.regular_path = str(regular) if regular and regular.exists() else None
        H = self.H
        self.f_title = self.font(H * 0.10)
        self.f_big = self.font(H * 0.085)
        self.f_mid = self.font(H * 0.042)
        self.f_small = self.font(H * 0.029, bold=False)
        self.f_tiny = self.font(H * 0.0225, bold=False)

        self.predictor = Predictor(use_llm=self.cfg.get("use_llm", False))
        self.predictor.llm_patient = bool(self.cfg.get("llm_in_patient_mode", False))   # default: postcards only
        self.speller = Speller(self.predictor, "postcard", self.cfg["max_postcard_chars"],
                               quick_messages=self.cfg.get("quick_messages"),
                               my_phrases=load_phrases(ROOT / "data" / "my_phrases.txt"),
                               languages=LANGUAGES)
        # Normal flow: the helper picks the mode on the start screen (F1 / F2 / click) and the language
        # with the picker. With person_chooses: true in config.yaml no mode is fixed, and the person
        # picks mode and language by looking at a first page of boxes instead.
        self.person_chooses = bool(self.cfg.get("person_chooses", False))
        self.preset = None if self.person_chooses else "postcard"
        self.from_clf = UDPReceiver(self.cfg["ports"]["selection"]) if args.input != "keyboard" else None
        self.to_clf = UDPSender(self.cfg["ports"]["control"])
        self.to_sim = UDPSender(self.cfg["ports"]["sim_target"])
        self.sounds = SoundBank(enabled=not args.mute)

        # Four big stimulus boxes in the corners, far apart; text and feedback live in the middle column.
        bw, bh = int(self.W * 0.335), int(self.H * 0.325)
        xs, ys = (0.195 * self.W, 0.805 * self.W), (0.40 * self.H, 0.78 * self.H)
        self.boxes = [pygame.Rect(0, 0, bw, bh) for _ in range(4)]
        for r, (x, y) in zip(self.boxes, [(xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])]):
            r.center = (int(x), int(y))
        self.mid = pygame.Rect(int(self.W * 0.37), int(self.H * 0.25), int(self.W * 0.26), int(self.H * 0.70))

        self.state, self.nickname = "intro", ""
        self.frame, self.paused = 0, False
        self.test, self.cued, self.test_n, self.test_ok = False, -1, 0, 0
        self.quality, self.quality_t = {}, 0.0      # electrode contact report from the classifier (live headset)
        self.flash = (-1, 0.0, "")                 # (box, time, label that was picked)
        self.scores, self.scores_t = [0.0, 0.0, 0.0, 0.0], 0.0
        self.last_sel_t = time.time()
        self.session_t0, self.session_secs, self.n_sel = 0.0, 0.0, 0
        self.message, self.postcard, self.postcard_img = "", None, None
        self.consent, self.upload_msg = None, ""
        self.banner = ("", 0.0, GREEN)
        self.spoken = ("", 0.0)
        self.alarm_t0 = self.alarm_said = 0.0        # emergency alarm: when it started / last spoken
        self.sim_target = -1
        self.voice_i = 0
        self.making_t0, self.saved_path, self.worker, self.made = 0.0, None, None, None
        self.fps_warn, self.fps_samples = "", []
        self.qr = self._make_qr()
        self.confetti = (0.0, [])
        self.help = False                           # TAB: operator key overview
        # Gemini helpers (patient mode). Suggested answers use the HELPER's typed question; the fuller
        # sentence sends what the PATIENT spelled, so it needs llm_in_patient_mode: true as well.
        self.assistant = Assistant(enabled=self.cfg.get("use_llm", False))
        if self.cfg.get("llm_in_patient_mode", False):
            self.speller.assistant = self.assistant
        self.translator = Translator(self.assistant)
        self.alert = PhoneAlert()                   # call / text / push to a caregiver who is not in the room
        self.care = CareSync(self.cfg)              # caregiver web app: live feed of what is said, replies back
        self.live_t, self.live_trail, self.live_pick = 0.0, [], ("", 0, 0.0)   # web dashboard feed
        # gentler flicker: how far the brightness swings (1 = black to white), and the still time after a pick
        self.contrast = max(0.05, min(1.0, float(self.cfg.get("flicker_contrast", 0.3))))
        self.bright = max(0.15, min(0.5, float(self.cfg.get("flicker_brightness", 0.36))))   # the middle grey
        self.rest_s = max(0.0, float(self.cfg.get("flicker_rest_s", 1.5)))
        self.still_t = 0.0
        self.care_msg = ("", 0.0)                 # latest reply from a caregiver, shown on the patient's screen
        self.alert_note = ("", GREEN, 0.0)          # what to show about the last phone alert
        self.alert_t, self.alarm_calls = 0.0, 0     # last COME HERE alert; alerts sent for this alarm
        # --input sim --signal replay: acquisition/dataset_stream.py plays real recorded EEG (g.tec dataset)
        self.dataset = args.input == "sim" and getattr(args, "signal", "live") == "replay"
        self.practice = False        # --input eeg fed by acquisition/fake_unicorn.py: recorded EEG, said so everywhere
        self.auto, self.auto_t = "", 0.0             # automatic demo: the message being typed, last look
        self.auto_say = []                          # automatic demo in patient mode: phrases still to be said
        self.auto_spell = ""                        # ...and then one sentence to spell out and speak
        # which demo message comes next. Each launch starts at a random place in the list (it used to start
        # at the first one every time), then every visitor gets the next message.
        self.auto_n = {"postcard": random.randrange(1000), "patient": random.randrange(1000)}
        self.letter_path = {}                       # letter -> (group box, subgroup box, letter box)
        for gi, group in enumerate(["AI", "JR", "SZ"]):
            for si, (_, action) in enumerate(SCREENS[group][:3]):
                for li, item in enumerate(SCREENS[action[1]]):
                    if item[1][0] == "char":
                        self.letter_path[item[1][1]] = (group, gi, action[1], si, li)
        self.click_zones = []                       # start screen: (rect, what a click there does)
        self.lang_i = 0                             # language the voice speaks (index into LANGUAGES)
        self.said_in = ("", "", 0.0)                # (language name, translated sentence, when)
        self.asking = None                          # the question a helper is typing (None = not typing)
        self.ask_wait = ("", 0.0)                   # (question sent to Gemini, when)
        # Backgrounds are painted once. The typing screen gets the calm one: nothing colourful or
        # moving may compete with the four flickering boxes.
        self.bg = pygame.Surface((self.W, self.H)).convert()     # every screen: the same flat dark page
        self.bg.fill(BG)
        self.bg_home = self.bg                      # start screen: the same page, plus the picture if there is one
        self.bg_show = self.bg                      # postcard screens: the picture, large and dim, behind everything
        self.hero = self._add_hero()                # optional picture on the start screen
        self.logo = None                            # team logo on the start screen: speller/assets/logo.png
        try:
            raw = pygame.image.load(str(pathlib.Path(__file__).resolve().parent / "assets" / "logo.png")).convert_alpha()
            h = int(self.H * 0.142)
            self.logo = pygame.transform.smoothscale(raw, (int(raw.get_width() * h / raw.get_height()), h))
        except (OSError, pygame.error):
            pass                                    # no file: the title is written as text instead
        # What the screen shows about the signal is steadied (display only; the classifier decides alone):
        self.show_scores, self.sig_t = [0.0, 0.0, 0.0, 0.0], 0.0     # bars glide instead of jumping
        self.lock_box, self.lock_cand, self.lock_cand_t = -1, -1, 0.0   # "locking on" outline with hysteresis
        self.pick_info = ("", None, 0.0)            # last pick: (label, decoder margin or None for a key, when)
        self.key_t = 0.0                            # last key / click pick (double-press guard)
        # Keyboard demo: no EEG is read, so the boxes have no reason to flash. --flicker or F shows it.
        self.still_mode = args.input == "keyboard" and not getattr(args, "flicker", False)
        self.bg_calm = pygame.Surface((self.W, self.H)).convert()
        self.bg_calm.fill(BG)
        # Typing screen backdrop: speller/assets/play.jpg (or .png / .jpeg / .webp) if it is there, else the
        # start-screen picture. Either way very dim and perfectly still, so it never competes with the boxes.
        assets = pathlib.Path(__file__).resolve().parent / "assets"
        play = next((assets / f"play{ext}" for ext in (".jpg", ".png", ".jpeg", ".webp")
                     if (assets / f"play{ext}").exists()), None)
        faint = None
        try:
            if play:
                img = pygame.image.load(str(play)).convert()
                k = max(self.W / img.get_width(), self.H / img.get_height())       # fill the whole page
                faint = pygame.transform.smoothscale(img, (int(img.get_width() * k) + 1, int(img.get_height() * k) + 1))
                faint.fill((84, 84, 84), special_flags=pygame.BLEND_RGB_MULT)       # about a third of its brightness
            elif self.hero and getattr(self, "hero_big", None):
                faint = self.hero_big.copy()
                faint.fill((235, 235, 235), special_flags=pygame.BLEND_RGB_MULT)
        except Exception as e:
            print("[ui] could not use the typing-screen picture ->", e)
        if faint:
            self.bg_calm.blit(faint, faint.get_rect(center=(self.W // 2, self.H // 2)), special_flags=pygame.BLEND_RGB_ADD)
            veil = pygame.Surface((self.mid.w + 16, int(self.H * 0.74)), pygame.SRCALPHA)   # keeps the middle text readable
            pygame.draw.rect(veil, (9, 10, 16, 170 if play else 150), veil.get_rect(), border_radius=26)
            self.bg_calm.blit(veil, veil.get_rect(midtop=(self.mid.centerx, int(self.H * 0.225))))
        self.typed_bar = pygame.Rect(int(self.W * 0.14), int(self.H * 0.045), int(self.W * 0.72), int(self.H * 0.135))
        real, self.screen = self.screen, self.bg_calm       # paint the fixed text bar into the background once
        self.card(self.typed_bar, 24, border=(70, 76, 108))
        for rect in self.boxes:                             # ...and a black frame around each flicker box
            pygame.draw.rect(self.screen, (2, 2, 5), rect.inflate(14, 14), border_radius=32)
        self.screen = real

        logp = ROOT / "logs" / "selections.csv"
        logp.parent.mkdir(exist_ok=True)
        new = not logp.exists()
        self.logf = open(logp, "a", newline="", encoding="utf-8")
        self.log = csv.writer(self.logf)
        if new:
            self.log.writerow(["time", "mode", "event", "screen", "box", "label", "cued_box", "confidence",
                               "seconds_since_last", "text", "session_seconds"])

    # ---------------------------------------------------------------- setup helpers
    def _open_display(self):
        a = self.args
        try:
            sizes = pygame.display.get_desktop_sizes()
        except Exception:
            sizes = [(1280, 800)]
        disp = a.display if 0 <= a.display < len(sizes) else 0
        # A window must fit this screen with its title bar and the taskbar: a fixed 1280 x 800 is taller
        # than many laptop screens (1366 x 768, or 1920 x 1080 shown at 150%), so size it from the desktop.
        size = (int(sizes[disp][0] * 0.94), int(sizes[disp][1] * 0.85)) if a.windowed else sizes[disp]
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        flags = pygame.SCALED | (0 if a.windowed else pygame.FULLSCREEN)
        self.vsync = True
        try:
            self.screen = pygame.display.set_mode(size, flags, display=disp, vsync=1)
        except pygame.error as e:
            print("[ui] vsync display not available (", e, "): flicker timing may be less exact")
            self.vsync = False
            try:
                self.screen = pygame.display.set_mode(size, flags & ~pygame.SCALED, display=disp)
            except pygame.error:
                self.screen = pygame.display.set_mode((1280, 800))

    def _make_qr(self):
        url = str(self.cfg.get("gallery_url", ""))
        if not url.startswith("http") or "YOUR-GALLERY-LINK" in url:
            return None
        try:
            import qrcode
            from PIL import Image
            px = int(self.H * 0.26)
            img = qrcode.make(url, border=2).convert("RGB").resize((px, px), Image.NEAREST)
            return pygame.image.fromstring(img.tobytes(), img.size, "RGB")
        except Exception as e:
            print("[ui] no QR code:", e)
            return None

    def voice_id(self):
        return VOICES[self.voice_i % len(VOICES)][1] if VOICES else self.cfg.get("elevenlabs_voice_id", "")

    def say(self, text):
        """Speak in the chosen language (the ElevenLabs voice speaks whatever language it is given)."""
        code, name = LANGUAGES[self.lang_i]

        def out(sentence):
            if code != "en":
                self.said_in = (name if sentence != text else "English (no translation available)",
                                sentence if sentence != text else "", time.time())
            voice, on = self.voice_id(), self.cfg.get("use_tts", False)
            # Team decision (Part B): picking a box IS the person's approval to say it, and switching
            # use_tts on is the operator's consent to send the text to ElevenLabs. The newer
            # voice/tts.py asks for both explicitly; the older one takes no such arguments.
            ok = {"approved": True, "allow_cloud": True} if "approved" in inspect.signature(speak).parameters else {}
            if on and code in V3_LANGS:                      # this language needs the slower Eleven v3 model:
                threading.Thread(target=lambda: (prefetch(sentence, code, voice), speak(sentence, voice, on, **ok)),
                                 daemon=True).start()        # fetch its audio first, off the UI thread
            else:
                speak(sentence, voice, on, **ok)

        self.translator.translate(text, code, out)

    def next_language(self, step=1):
        self.lang_i = (self.lang_i + step) % len(LANGUAGES)
        self.speller.language = LANGUAGES[self.lang_i][0]
        self.said_in = ("", "", 0.0)

    # ---------------------------------------------------------------- look and feel
    def _background(self, glows):
        """Dark vertical gradient with optional soft colour glows: (x, y, radius, colour, strength),
        positions as fractions of the screen. Built small and scaled up, which blurs it for free."""
        W, H = self.W, self.H
        w, h = max(2, W // 8), max(2, H // 8)
        x = np.linspace(0, 1, w)[:, None, None]
        y = np.linspace(0, 1, h)[None, :, None]
        top, bottom = np.array([17.0, 19.0, 36.0]), np.array([6.0, 7.0, 12.0])
        img = top + (bottom - top) * y + 0 * x
        for gx, gy, radius, color, strength in glows:
            d = np.sqrt(((x - gx) * W / H) ** 2 + (y - gy) ** 2)
            img = img + np.array(color, float) * (strength * np.clip(1 - d / radius, 0, 1) ** 2)
        surf = pygame.surfarray.make_surface(np.clip(img, 0, 255).astype(np.uint8))
        return pygame.transform.smoothscale(surf, (W, H)).convert()

    def _add_hero(self):
        """Start-screen picture: speller/assets/brain.png (or .jpg / .jpeg / .webp), if the file is
        there. It is painted into the background once, on the right, with its black areas left
        transparent, so it costs nothing per frame. Returns True when a picture was found."""
        folder = pathlib.Path(__file__).resolve().parent / "assets"
        path = next((folder / f"brain{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp")
                     if (folder / f"brain{ext}").exists()), None)
        if not path:
            return False
        try:
            img = pygame.image.load(str(path)).convert()
            side = int(self.H * 0.92)
            scale = side / max(img.get_size())
            img = pygame.transform.smoothscale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            w, h = img.get_size()                               # fade the edges so it melts into the page
            fx = np.clip(np.minimum(np.arange(w), w - 1 - np.arange(w)) / (0.16 * w), 0, 1)
            fy = np.clip(np.minimum(np.arange(h), h - 1 - np.arange(h)) / (0.10 * h), 0, 1)
            px = pygame.surfarray.pixels3d(img)
            px[:] = (px * (0.85 * fx[:, None, None] * fy[None, :, None])).astype(np.uint8)
            del px
            big = pygame.image.load(str(path)).convert()        # postcard screens: fills the page, kept dim
            k = 1.12 * self.H / big.get_height()
            big = pygame.transform.smoothscale(big, (int(big.get_width() * k), int(big.get_height() * k)))
            bw_, bh_ = big.get_size()
            gx = np.clip(np.minimum(np.arange(bw_), bw_ - 1 - np.arange(bw_)) / (0.22 * bw_), 0, 1)
            bpx = pygame.surfarray.pixels3d(big)
            bpx[:] = (bpx * (0.34 * gx[:, None, None])).astype(np.uint8)
            del bpx
            self.hero_big = big                                 # kept for the typing screen's backdrop
            self.bg_show = self.bg.copy()
            self.bg_show.blit(big, big.get_rect(center=(self.W // 2, int(self.H * 0.47))), special_flags=pygame.BLEND_RGB_ADD)
            self.bg_home = self.bg.copy()               # only the start screen carries the bright picture
            self.bg_home.blit(img, img.get_rect(center=(int(self.W * 0.79), self.H // 2)), special_flags=pygame.BLEND_RGB_ADD)
            return True
        except Exception as e:
            print("[ui] could not use", path.name, "->", e)
            return False

    def soft(self, rect, radius=22, spread=14, color=(0, 0, 0), alpha=130, dy=8):
        """Blurred rounded shape behind a rect: a drop shadow (black) or a glow (colour). Cached."""
        key = ("soft", rect.w, rect.h, radius, spread, color, alpha)
        if key not in self._surfs:
            w, h = rect.w + 4 * spread, rect.h + 4 * spread
            sharp = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(sharp, (*color, alpha), (2 * spread, 2 * spread, rect.w, rect.h), border_radius=radius)
            small = pygame.transform.smoothscale(sharp, (max(1, w // 12), max(1, h // 12)))
            self._surfs[key] = pygame.transform.smoothscale(small, (w, h))
        self.screen.blit(self._surfs[key], (rect.x - 2 * spread, rect.y - 2 * spread + dy))

    def card(self, rect, radius=22, fill=PANEL, border=LINE, width=2, glow=None):
        """Rounded panel with a drop shadow, or a coloured glow when it is the active one."""
        if glow:
            self.soft(rect, radius, 16, glow, 110, dy=0)
        else:
            self.soft(rect, radius)
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        if width:
            pygame.draw.rect(self.screen, glow or border, rect, width, border_radius=radius)

    def gradient_text(self, s, px, colors=BRAND):
        """Text filled with a left-to-right colour gradient. Cached."""
        key = ("grad", s, int(px), colors)
        if key not in self._surfs:
            base = self.font(px).render(s, True, (255, 255, 255)).convert_alpha()
            w, h = base.get_size()
            pos = np.linspace(0, 1, len(colors))
            ramp = np.stack([np.interp(np.linspace(0, 1, w), pos, [c[i] for c in colors]) for i in range(3)], axis=1)
            grad = pygame.surfarray.make_surface(np.repeat(ramp[:, None, :], h, axis=1).astype(np.uint8))
            base.blit(grad, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
            self._surfs[key] = base
        return self._surfs[key]

    def waves(self, now, cy, amp, colors=BRAND, dim=0.55, speed=1.0):
        """Slow 'brainwave' ribbons across the screen (intro and waiting screens only)."""
        u = np.linspace(0, 1, 110)
        fade = np.exp(-((u - 0.5) ** 2) / 0.10)                    # calm at the edges, lively in the middle
        for i, col in enumerate(colors):
            ph = now * speed * (0.7 + 0.3 * i)
            y = cy + amp * fade * (0.62 * np.sin(2 * np.pi * (1.4 + 0.6 * i) * u + ph)
                                   + 0.30 * np.sin(2 * np.pi * (5.0 + 1.5 * i) * u - 1.7 * ph)
                                   + 0.10 * np.sin(2 * np.pi * (13.0 + 3 * i) * u + 2.9 * ph))
            c = tuple(int(v * dim) for v in col)
            pts = np.column_stack([u * self.W, y])
            pygame.draw.aalines(self.screen, c, False, pts)
            pygame.draw.aalines(self.screen, c, False, pts + (0, 1))

    def keycap(self, label, center, color=WHITE):
        """A little keyboard key, for operator hints."""
        surf = self.f_small.render(label, True, color)
        r = surf.get_rect(center=(int(center[0]), int(center[1]))).inflate(26, 12)
        pygame.draw.rect(self.screen, (34, 37, 56), r, border_radius=9)
        pygame.draw.rect(self.screen, (84, 90, 124), r, 2, border_radius=9)
        self.screen.blit(surf, surf.get_rect(center=r.center))
        return r

    def chip(self, label, color, center, font=None):
        """Status line with a coloured dot in front."""
        font = font or self.f_tiny
        surf = font.render(self.clean(label), True, MUTED if color == GREEN else color)
        w = surf.get_width() + 22
        x, y = int(center[0] - w / 2), int(center[1])
        pygame.draw.circle(self.screen, color, (x + 6, y + 1), 5)
        self.screen.blit(surf, surf.get_rect(midleft=(x + 22, y)))

    def stat(self, value, caption, center, color=WHITE):
        """Big number with a small caption underneath (result screen)."""
        r = pygame.Rect(0, 0, int(self.W * 0.15), int(self.H * 0.105))
        r.center = (int(center[0]), int(center[1]))
        self.card(r, 18)
        self.text(value, self.f_mid, color, (r.centerx, r.y + r.h * 0.38))
        self.text(caption, self.f_tiny, MUTED, (r.centerx, r.y + r.h * 0.76))

    def start_confetti(self, now):
        rng = np.random.default_rng()
        n = 140
        parts = np.column_stack([
            rng.uniform(0.05, 0.95, n) * self.W,                   # x0
            rng.uniform(-0.25, 0.0, n) * self.H,                   # y0 (above the screen)
            rng.uniform(-0.05, 0.05, n) * self.W,                  # sideways speed
            rng.uniform(0.25, 0.55, n) * self.H,                   # falling speed
            rng.integers(0, 5, n), rng.uniform(5, 12, n), rng.uniform(0, 6.3, n)])
        self.confetti = (now, parts)

    def draw_confetti(self, now):
        t0, parts = self.confetti
        t = now - t0
        if not len(parts) or t > 5.0:
            return
        colors = (YELLOW, CYAN, PINK, VIOLET, GREEN)
        for x0, y0, vx, vy, ci, size, ph in parts:
            x = x0 + vx * t + 14 * np.sin(3 * t + ph)
            y = y0 + vy * t + 0.10 * self.H * t * t
            if 0 <= y < self.H:
                w = max(2, int(size * abs(np.cos(4 * t + ph))))    # flutter
                pygame.draw.rect(self.screen, colors[int(ci)], (int(x), int(y), w, int(size)))

    # ---------------------------------------------------------------- text rendering
    def font(self, px, bold=True, symbol=False):
        key = (max(10, int(px)), bold, symbol)
        if key not in self._fonts and not bold and not symbol and self.regular_path:
            self._fonts[key] = pygame.font.Font(self.regular_path, key[0])     # true Regular, not Light
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont(self.sym_name if symbol else self.font_name, key[0], bold=bold)
        return self._fonts[key]

    def glyph_source(self, c):
        """Which font can draw this character: 'main', 'symbol' or None. (A font draws every
        character it does not have as the same 'missing' box, so compare against that box.)"""
        if ord(c) < 128:
            return "main"
        if c not in self._glyphs:
            found = None
            if ord(c) <= 0xFFFF:
                for kind in ("main", "symbol"):
                    if kind == "symbol" and not self.sym_name:
                        continue
                    try:
                        f = self.font(40, True, kind == "symbol")
                        a, b = f.render(c, True, WHITE), f.render("\u0378", True, WHITE)
                        if a.get_width() > 0 and (a.get_size() != b.get_size() or
                                                  pygame.image.tostring(a, "RGBA") != pygame.image.tostring(b, "RGBA")):
                            found = kind
                            break
                    except Exception:
                        pass
            self._glyphs[c] = found
        return self._glyphs[c]

    def clean(self, label, font=None):
        """Keep every character we can draw; swap the rest for ASCII so nothing shows as a box."""
        out = "".join(c if self.glyph_source(c) else ASCII.get(c, "") for c in label)
        return " ".join(out.split())

    def render(self, s, px, color, bold=True):
        """Render one line, drawing symbols the main font lacks with the symbol font."""
        if all(ord(c) < 128 for c in s):
            return self.font(px, bold).render(s, True, color)
        runs, cur, cur_kind = [], "", "main"
        for c in s:
            kind = self.glyph_source(c) or "main"
            if kind != cur_kind and cur:
                runs.append((cur, cur_kind))
                cur = ""
            cur, cur_kind = cur + c, kind
        runs.append((cur, cur_kind))
        parts = [self.font(px * (0.9 if k == "symbol" else 1), bold, k == "symbol").render(t, True, color)
                 for t, k in runs if t]
        h = self.font(px, bold).get_linesize()
        surf = pygame.Surface((sum(x.get_width() for x in parts), h), pygame.SRCALPHA)
        x = 0
        for part in parts:
            surf.blit(part, (x, (h - part.get_height()) // 2))
            x += part.get_width()
        return surf

    def text(self, s, font, color, center):
        surf = font.render(s, True, color)
        self.screen.blit(surf, surf.get_rect(center=(int(center[0]), int(center[1]))))
        return surf.get_width()

    def wrap(self, s, font, max_w):
        lines, cur = [], ""
        for word in s.split():
            trial = (cur + " " + word).strip()
            if cur and font.size(trial)[0] > max_w:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        return lines + [cur] if cur else lines

    def paragraph(self, s, font, color, cx, top, max_w):
        """Centered, word-wrapped text. Returns the y below the last line."""
        y = top
        for line in self.wrap(s, font, max_w):
            surf = font.render(line, True, color)
            self.screen.blit(surf, surf.get_rect(midtop=(int(cx), int(y))))
            y += font.get_linesize()
        return y

    def fitted(self, s, max_w, max_px, color, bold=True):
        """Single line, shrunk until it fits max_w."""
        px = int(max_px)
        while px > 12 and self.render(s, px, color, bold).get_width() > max_w:
            px = int(px * 0.92)
        return self.render(s, px, color, bold)

    def label_surface(self, raw, max_w, max_h, color=YELLOW):
        """Box label: as large as fits, on one or two lines, with a dark outline so it stays
        readable while the box flickers between dark and bright. Cached (rendered once)."""
        key = (raw, max_w, max_h, color)
        if key in self._surfs:
            return self._surfs[key]
        if len(self._surfs) > 400:
            self._surfs.clear()
        px = int(self.H * 0.075)
        while True:
            font = self.font(px)
            s = self.clean(raw)

            def width(x, px=px):
                return self.render(x, px, color).get_width()

            options = [[s]]
            words = s.split(" ")
            if len(words) > 1:                                  # best two-line split = most even widths
                splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
                options.append(list(min(splits, key=lambda ab: max(width(ab[0]), width(ab[1])))))
            fit = next((ls for ls in options if max(width(x) for x in ls) <= max_w
                        and len(ls) * font.get_linesize() <= max_h), None)
            if fit or px <= 14:
                lines = fit or options[0]
                break
            px = int(px * 0.9)
        o = max(2, px // 14)
        lh = font.get_linesize()
        w = max(width(x) for x in lines) + 2 * o
        surf = pygame.Surface((w, lh * len(lines) + 2 * o), pygame.SRCALPHA)
        for i, line in enumerate(lines):
            dark = self.render(line, px, (0, 0, 0))
            x = (w - dark.get_width()) // 2
            for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o), (-o, -o), (o, o), (-o, o), (o, -o)):
                surf.blit(dark, (x + dx, o + i * lh + dy))
            surf.blit(self.render(line, px, color), (x, o + i * lh))
        self._surfs[key] = surf
        return surf

    # ---------------------------------------------------------------- input
    def write_log(self, now, event, screen="", box="", label="", cued="", conf="", dt="", text="", secs=""):
        self.log.writerow([round(now, 3), self.speller.mode, event, screen, box, label, cued, conf, dt, text, secs])
        self.logf.flush()

    def new_cue(self):
        options = [i for i, lab in enumerate(self.speller.labels()) if lab]
        self.cued = random.choice(options) if options else -1

    def on_select(self, box, confidence=1.0, decoded=False):
        """decoded=True: the pick came from the classifier and `confidence` is its score margin."""
        if self.state != "spell" or self.paused or self.speller.alarm or not 0 <= box <= 3:
            return
        now = time.time()
        item = self.speller.items()[box]
        label = self.clean(item[0]) if item else ""
        self.pick_info = (label, float(confidence) if decoded else None, now)
        dt = round(now - self.last_sel_t, 3)
        self.last_sel_t = now
        if self.args.input == "sim":                 # like a real user: look away after a selection
            self.sim_target = -1
            self.to_sim.send({"target": -1})

        if self.test:                                # accuracy test: score the look, do not type anything
            self.test_n += 1
            self.test_ok += box == self.cued
            self.write_log(now, "test", self.speller.screen, box, label, self.cued, confidence, dt, self.speller.text)
            self.flash = (box, now, label)
            self.sounds.play("click" if box == self.cued else "back")
            self.new_cue()
            return

        screen = self.speller.screen
        self.speller.select(box)
        self.write_log(now, "select", screen, box, label, -1, confidence, dt, self.speller.text)
        if item is None:
            return
        self.n_sel += 1
        self.flash = (box, now, label)
        kind = item[1][0]
        self.sounds.play({"char": "type", "suggest": "type", "back": "back", "delete": "back"}.get(kind, "click"))
        if self.speller.spoken:                      # patient-mode phrase
            self.spoken = (self.speller.spoken, now)
            self.say(self.speller.spoken)
            self.write_log(now, "spoken", screen, text=self.speller.spoken)     # for speller/care_report.py
            if self.speller.spoken != TYPING_NOTICE:     # the caregiver app shows what was said, as it is said
                self.care.say(self.speller.spoken, "emergency" if self.speller.alarm else
                              "spelled" if kind in ("done", "say") else "need", self.nickname)
            self.speller.spoken = None
        if self.speller.language != LANGUAGES[self.lang_i][0]:      # the person picked a language by looking
            self.lang_i = [c for c, _ in LANGUAGES].index(self.speller.language)
            self.said_in = ("", "", 0.0)
            self.spoken = ("Hello!", now)
            self.say("Hello!")                           # greet in the new language, as confirmation
        if self.speller.chime:                       # call bell: "come here", heard across the room
            self.sounds.play("bell")
            self.speller.chime = False
            self.phone_alert(now)                        # ...and on their phone, if nobody is in the room
        if self.speller.alarm:                       # EMERGENCY: siren + repeated voice until a helper answers
            self.alarm_t0 = self.alarm_said = now
            self.alarm_calls = 1
            self.phone_alert(now, urgent=True)
            self.sounds.play("alarm", loop=True)
            self.write_log(now, "emergency", screen, box, label, -1, confidence, dt, "")
        if self.speller.finished:
            self.begin_postcard()

    def stop_alarm(self, now=None):
        """A helper acknowledged the emergency (keyboard only: a stray look can never cancel it)."""
        if self.speller.alarm:
            self.speller.clear_alarm()
            self.sounds.stop("alarm")
            self.spoken = ("", 0.0)
            if now:
                self.write_log(now, "emergency_ack", secs=round(now - self.alarm_t0, 1))
                self.last_sel_t = now
                self.to_clf.send({"cmd": "resume"})  # start from a clean EEG window

    def show_answers(self, question, options=None):
        """Put the answer boxes up (suggested ones, or YES / NO / NOT SURE) for the helper's question."""
        self.speller.ask(question, options)
        self.flash, self.spoken = (-1, 0.0, ""), ("", 0.0)
        self.last_sel_t = time.time()
        self.to_clf.send({"cmd": "resume"})                  # judge the answer on fresh EEG only

    def key_question(self, e):
        """The helper is typing the question they just asked out loud."""
        if e.key == pygame.K_ESCAPE:
            self.asking = None
        elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            question, self.asking = self.asking.strip(), None
            if question:                                     # boxes stay empty until the answers are ready
                self.assistant.answers(question)
                self.ask_wait = (question, time.time())
                self.show_answers(question, [])
            else:
                self.show_answers("")
        elif e.key == pygame.K_BACKSPACE:
            self.asking = self.asking[:-1]
        elif e.unicode and e.unicode.isprintable() and len(self.asking) < 70:
            self.asking += e.unicode

    def poll_assistant(self, now):
        question, t0 = self.ask_wait
        if not question:
            return
        got = self.assistant.get("answers", question)
        if got or got is False or now - t0 > 5.0:            # ready, failed, or too slow: plain yes/no
            self.ask_wait = ("", 0.0)
            if self.state == "spell" and self.speller.screen == "asked":
                self.show_answers(question, got or None)

    def auto_box(self):
        """Automatic demo: which box must be looked at next to type self.auto and finish?"""
        sp, goal = self.speller, self.auto
        if sp.screen == "say":                                # patient mode, after DONE: say it as typed
            return 0
        if sp.in_menu:                                        # patient menu pages: find the way to SPELL
            return self.route(lambda action: action == ("goto", "root"))
        text = sp.text
        want = "done" if text.strip() == goal else "delete" if not goal.startswith(text) else goal[len(text)]
        if want in ("done", "delete"):
            route = {"root": 2, "SZ": 3, "CTRL": 1 if want == "done" else 0}
        else:
            group, gi, sub_screen, si, li = self.letter_path[want]
            word = goal[len(text) - len(text.split(" ")[-1]):].split(" ")[0]      # the word being typed
            if sp.screen == "root" and text and sp.suggestion == word:
                return 3                                                       # box 4 finishes the word
            route = {"root": gi, group: si, sub_screen: li}
        if sp.screen in route:
            return route[sp.screen]
        for i, item in enumerate(sp.items()):                 # somewhere unexpected (a misread look): go back
            if item and item[1][0] in ("back", "home", "needs"):
                return i
        return 3

    def route(self, wanted):
        """Automatic demo: the box that leads, by the shortest way through the menu pages, to an
        item for which wanted(action) is true. None if there is no way. Never goes via EMERGENCY."""
        sp = self.speller

        def edges(screen):
            for i, item in enumerate(sp.items() if screen == sp.screen else sp.screens.get(screen, [])):
                if not item:
                    continue
                kind = item[1][0]
                if wanted(item[1]):
                    yield i, None
                elif kind == "goto":
                    yield i, item[1][1]
                elif kind in ("needs", "home", "back", "welcome"):
                    yield i, "needs1"

        seen, queue = {sp.screen}, [(sp.screen, None)]
        while queue:
            screen, first = queue.pop(0)
            for i, nxt in edges(screen):
                step = i if first is None else first
                if nxt is None:
                    return step
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, step))
        return None

    def auto_phrase_box(self):
        """Automatic demo, patient mode: the box on the way to the next phrase in self.auto_say."""
        target = self.auto_say[0]
        return self.route(lambda action: action[0] == "phrase" and action[1] == target)

    def start_auto(self):
        """Begin the automatic demo for the current mode (sim / dataset replay only). Each run takes
        the next message from the list, so every visitor sees something different."""
        self.auto, self.auto_say, self.auto_spell = "", [], ""
        if self.args.input != "sim":
            return

        def clean(text, limit):
            return " ".join("".join(c for c in str(text).upper() if "A" <= c <= "Z" or c == " ").split())[:limit].strip()

        mode = self.speller.mode
        n = self.auto_n[mode]
        self.auto_n[mode] += 1
        if mode == "postcard":
            messages = self.cfg.get("demo_messages") or ([self.cfg["demo_message"]] if self.cfg.get("demo_message") else
                                                         ["HI MOM", "HELLO EMU", "I LOVE YOU", "THANK YOU", "BRAIN POWER",
                                                          "SEE YOU SOON", "GO EAGLES", "MIND BLOWN"])
            self.auto = clean(messages[n % len(messages)], self.speller.max_chars)
        else:
            self.auto_say = list(self.cfg.get("demo_phrases") or [
                "I would like some water, please.", "I am in pain. Please help me.", "The pain is in my chest.",
                "Could you come here, please?", "Thank you."])
            sentences = self.cfg.get("demo_sentences") or ["I AM COLD", "CALL MY SON", "I AM HUNGRY", "I LOVE YOU ALL"]
            self.auto_spell = clean(sentences[n % len(sentences)], self.speller.max_chars)

    def auto_step(self, now):
        """Automatic demo (sim / dataset replay): the app chooses where to "look", so the recorded
        brain signals type a message, or say a few needs, with no key presses. Decoding is still real."""
        if self.state != "spell" or self.paused or self.test or self.speller.alarm or not (self.auto or self.auto_say):
            return
        patient = self.speller.mode == "patient"
        if self.auto_say and self.speller.last_spoken == self.auto_say[0]:
            self.speller.last_spoken = ""                     # that phrase was said: on to the next one
            self.auto_say.pop(0)
            if not self.auto_say:
                self.auto, self.auto_spell = self.auto_spell, ""      # then spell a sentence and speak it
                return
        elif patient and self.auto and not self.auto_say and self.speller.last_spoken == self.auto.capitalize() + ".":
            self.auto = ""                                    # the spelled sentence was spoken: demo finished
            return
        if self.sim_target >= 0:
            if now - self.auto_t > 9.0:                       # nothing was decoded: look away and try again
                self.sim_target = -1
                self.to_sim.send({"target": -1})
                self.last_sel_t = now
        elif now - self.last_sel_t > (4.5 if now - self.spoken[1] < 4.5 else 1.5):   # let a phrase be heard first
            box = self.auto_phrase_box() if self.auto_say else self.auto_box()
            if box is None:                                   # no way there from here: skip it
                if self.auto_say:
                    self.auto_say.pop(0)
                else:
                    self.auto = ""
                return
            self.sim_target, self.auto_t = box, now
            self.to_sim.send({"target": box})

    def poll_care(self, now):
        """Replies from the caregiver app: show and speak them; "I'm coming" stops the alarm."""
        for text in self.care.take_replies():
            self.care_msg = (text, now)
            self.sounds.play("type")
            self.say(text)
            self.write_log(now, "caregiver_reply", text=text)
        if self.speller.alarm and self.care.emergency_answered():
            self.stop_alarm(now)
            self.care_msg = ("Your caregiver saw the alarm and is coming", now)
            self.say("Your caregiver is coming.")

    def push_live(self, now):
        """Feed the web dashboard: what the decoder sees and what the system is doing right now.
        Scores and state only (never raw EEG), sampled five times a second; care_sync sends the newest."""
        if not self.care.live_on or now - self.live_t < 0.2:
            return
        self.live_t = now
        fresh = self.from_clf is not None and now - self.scores_t < 1.5 and not self.paused
        scores = [round(max(0.0, min(1.0, float(s))), 3) if fresh else 0.0 for s in self.scores]
        self.live_trail = (self.live_trail + [scores])[-40:]             # the last 8 seconds
        order = np.argsort(scores)[::-1]
        margin = float(scores[order[0]] - scores[order[1]])
        tau = float(self.cfg["margin_tau"])
        lead = int(order[0]) if fresh and margin > tau else -1
        speaking = bool(self.spoken[0]) and now - self.spoken[1] < 3.5
        if self.speller.alarm:
            state = "emergency"
        elif self.paused:
            state = "paused"
        elif speaking:
            state = "speaking"
        elif self.state != "spell":
            state = "idle"
        elif now - self.flash[1] < 1.0 and self.flash[0] >= 0:
            state = "selected"
        else:
            state = "detecting" if fresh else "waiting"
        if self.flash[0] >= 0 and self.flash[1] != self.live_pick[2]:   # a new pick: remember it for the page
            self.live_pick = (self.flash[2], float(self.cfg["freqs"][self.flash[0]]), self.flash[1])
        self.care.live({
            "state": state, "mode": self.speller.mode,
            "source": ("Keyboard (no EEG)" if self.args.input == "keyboard" else
                       "Recorded human EEG (dataset replay)" if self.dataset or self.practice else
                       "Simulated EEG" if self.args.input == "sim" else "Live EEG headset"),
            "freqs": [float(f) for f in self.cfg["freqs"]], "scores": scores,
            "labels": [self.clean(lab or "") for lab in self.speller.labels()] if self.state == "spell" else ["", "", "", ""],
            "detected_hz": float(self.cfg["freqs"][lead]) if lead >= 0 else 0,
            "confidence": round(margin, 3), "threshold": tau,
            "trail": json.dumps(self.live_trail),
            "speaking_text": self.spoken[0] if speaking else "",
            "last_pick": self.live_pick[0], "last_pick_hz": self.live_pick[1],
            "last_pick_at": int(self.live_pick[2] * 1000),
        })

    def sync_alert(self):
        """Pick up the outcome of the last phone alert (it is sent in the background)."""
        if self.alert.last[0] and self.alert.last[2] > self.alert_note[2]:
            self.alert_note = ("Phone alert: " + self.alert.last[0], GREEN if self.alert.last[1] else RED,
                               self.alert.last[2])

    def phone_alert(self, now, urgent=False):
        """COME HERE or EMERGENCY: also reach the caregiver's phone, if that is set up.
        Not during the automatic demo, so a booth demo never calls anybody (unless
        alert_in_demo: true is set in config.yaml, e.g. to record a video)."""
        if not self.alert.enabled:
            return
        if (self.auto or self.auto_say) and not self.cfg.get("alert_in_demo", False):
            self.alert_note = ("Phone alert skipped", MUTED, now)
            return
        if not urgent and now - self.alert_t < 30:           # a second look within 30 s does not call again
            return
        who = (self.nickname or str(self.cfg.get("alert_name", ""))).strip()
        text = ("EMERGENCY! I need help right now." if urgent else
                str(self.cfg.get("alert_message", "I need you. Please come to me.")))
        if self.alert.send((f"{who}: " if who else "") + text, urgent=urgent):
            self.alert_t = now
            self.alert_note = ("Calling your caregiver's phone..." if self.alert.twilio else
                               "Alerting your caregiver's phone...", CYAN, now)
            self.write_log(now, "phone_alert", text="emergency" if urgent else "come here")

    def handle_key(self, e):
        k = e.key
        if k == pygame.K_TAB:
            self.help = not self.help
            return True
        if self.help:                                # any other key closes the overview first
            self.help = False
            return True
        if self.state == "spell" and self.speller.alarm:
            if k in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
                self.stop_alarm(time.time())
            return True
        if self.asking is not None:
            self.key_question(e)
            return True
        if k == pygame.K_ESCAPE:
            if self.state == "intro":
                return False
            if self.state == "making":
                return True
            self.to_intro()
            return True
        if self.state == "intro":
            if k == pygame.K_F1:
                self.choose_preset("postcard")
            elif k == pygame.K_F2:
                self.choose_preset("patient")
            elif k in (pygame.K_F5, pygame.K_RIGHT):         # language the voice speaks
                self.next_language()
            elif k == pygame.K_LEFT:
                self.next_language(-1)
            elif k == pygame.K_F4:                           # two-box easy mode (patient), for a weak signal
                self.speller.easy = not self.speller.easy
                self.preset = "patient"                          # (two boxes cannot show the choice page)
                self.speller.set_mode("patient")
            elif k == pygame.K_F3 and VOICES:
                self.voice_i = (self.voice_i + 1) % len(VOICES)
            elif k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start_visitor()
            elif k == pygame.K_BACKSPACE:
                self.nickname = self.nickname[:-1]
            elif e.unicode and e.unicode.isalnum() and e.unicode.isascii() and len(self.nickname) < 16:
                self.nickname += e.unicode
        elif self.state == "result":
            if k in (pygame.K_y, pygame.K_n) and self.consent is None:
                self.consent = k == pygame.K_y
                try:
                    self.upload_msg = upload(self.postcard, self.message, self.nickname,
                                             self.session_secs, consent=self.consent) if self.postcard else ""
                except Exception as ex:
                    self.upload_msg = f"upload failed: {ex}"
                if self.consent and self.postcard and self.care.enabled:   # the visitor said yes: public gallery
                    self.care.postcard(self.postcard, self.message, self.nickname, self.session_secs)
                    self.upload_msg = "Sent to the online gallery"
            elif k == pygame.K_SPACE:
                self.to_intro()
        elif self.state == "spell":
            if k == pygame.K_p:
                self.paused = not self.paused
                self.to_clf.send({"cmd": "pause" if self.paused else "resume"})
                if self.paused:                                  # everything stops: no look, no moving signal bars
                    self.scores = [0.0, 0.0, 0.0, 0.0]
                    if self.args.input == "sim":
                        self.sim_target = -1
                        self.to_sim.send({"target": -1})
                else:
                    self.last_sel_t = time.time()                # (the automatic demo waits a moment, then goes on)
            elif k == pygame.K_q and self.speller.mode == "patient":
                if self.assistant.enabled and not self.speller.easy:
                    self.asking = ""                             # type the question: Gemini suggests answers
                else:
                    self.show_answers("")                        # plain YES / NO / NOT SURE
            elif k == pygame.K_l:                            # change the spoken language on the fly
                self.next_language()
                self.banner = (f"Voice now speaks {LANGUAGES[self.lang_i][1]}", time.time(), CYAN)
            elif k == pygame.K_r and self.speller.last_spoken:
                self.spoken = (self.speller.last_spoken, time.time())
                self.say(self.speller.last_spoken)               # "say that again"
            elif k == pygame.K_a and self.args.input == "sim":     # automatic demo on / off
                if self.auto or self.auto_say:
                    self.auto, self.auto_say, self.auto_spell = "", [], ""
                else:
                    self.start_auto()
                self.sim_target = -1
                self.to_sim.send({"target": -1})
            elif k == pygame.K_t:
                self.test = not self.test
                self.test_n = self.test_ok = 0
                self.new_cue() if self.test else setattr(self, "cued", -1)
            elif k == pygame.K_f and self.args.input == "keyboard":   # show / hide the flicker in the keyboard demo
                self.still_mode = not self.still_mode
                self.banner = ("Boxes do not flash (keyboard demo)" if self.still_mode else
                               "Boxes flash, as they do with the headset", time.time(), CYAN)
            else:
                n = {pygame.K_0: -1, pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
                     pygame.K_KP0: -1, pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2, pygame.K_KP4: 3}.get(k)
                if n is None:
                    return True
                if self.args.input == "keyboard" and n >= 0:
                    if self.fresh_press():
                        self.on_select(n)
                elif self.args.input == "sim":
                    self.sim_target = n                                  # 0 key = look away
                    self.to_sim.send({"target": n})
        return True

    def handle_click(self, pos):
        """Start screen: click a mode card or the language arrows.
        Typing screen (demo modes only): clicking a box works like pressing its number key."""
        if self.state in ("intro", "result"):
            for rect, action in self.click_zones:
                if rect.collidepoint(pos):
                    action()
                    return
        if self.state != "spell" or self.speller.alarm or self.args.input == "eeg":
            return
        for i, rect in enumerate(self.boxes):
            if rect.collidepoint(pos):
                if self.args.input == "keyboard":
                    if self.fresh_press():
                        self.on_select(i)
                else:
                    self.sim_target = i
                    self.to_sim.send({"target": i})

    def fresh_press(self):
        """False for a second key press or click within a fifth of a second: one press, one pick."""
        now = time.time()
        if now - self.key_t < 0.2:
            return False
        self.key_t = now
        return True

    def to_intro(self):
        self.stop_alarm()
        self.auto, self.auto_say = "", []
        self.asking, self.ask_wait = None, ("", 0.0)
        self.state, self.nickname = "intro", ""
        self.test, self.cued = False, -1
        if self.args.input == "sim":
            self.sim_target = -1
            self.to_sim.send({"target": -1})

    def choose_preset(self, mode):
        """Helper shortcut on the start screen: fix the mode (again = let the person choose)."""
        self.preset = None if (self.person_chooses and self.preset == mode) else mode
        self.speller.set_mode(self.preset or "postcard")

    def start_visitor(self):
        self.speller.set_mode(self.preset or "postcard")
        if not self.preset:                              # the person picks mode and language by looking
            self.speller.language = LANGUAGES[self.lang_i][0]
            self.speller.welcome()
        now = time.time()
        self.state, self.session_t0, self.last_sel_t, self.n_sel = "spell", now, now, 0
        self.flash, self.spoken = (-1, 0.0, ""), ("", 0.0)
        self.to_clf.send({"cmd": "start_session"})
        self.paused = False
        self.to_clf.send({"cmd": "resume"})          # also clears the classifier's old EEG window
        if self.dataset:                             # dataset_stream.py --dataset both: next recorded person
            self.sim_target = -1
            self.to_sim.send({"target": -1, "person": "next"})
        if self.dataset and self.cfg.get("auto_demo", True) and not self.speller.easy:
            self.start_auto()                        # dataset replay runs by itself; A switches to manual

    # ---------------------------------------------------------------- postcard
    def begin_postcard(self):
        now = time.time()
        self.message = self.speller.text or "HELLO"
        self.session_secs = now - self.session_t0
        path = ROOT / "sessions" / f"session_{int(now)}.npy"
        self.to_clf.send({"cmd": "stop_session", "path": str(path)})
        self.write_log(now, "postcard_done", text=self.message, secs=round(self.session_secs, 1))
        self.sounds.play("done")
        self.state, self.making_t0 = "making", now
        self.saved_path, self.worker, self.made = None, None, None
        self.postcard, self.postcard_img, self.consent, self.upload_msg = None, None, None, ""

    def _make(self, eeg_path):
        try:
            eeg = None
            if eeg_path:
                try:
                    eeg = np.load(eeg_path)
                except Exception as e:
                    print("[ui] could not read the session EEG:", e)
            labelled = "signal_source" in inspect.signature(make_postcard).parameters
            if eeg is None or eeg.ndim != 2 or len(eeg) < self.cfg["fs"]:
                if labelled:             # Part C's postcard never invents data: no recording, no art
                    self.made = ("noart", "")
                    return
                eeg = np.zeros((0, 8))                   # (older make_postcard draws from noise)
            extra = {}
            if labelled:                 # say honestly where the signal and the selections came from
                signal = getattr(self.args, "signal", "live")
                source = {"keyboard": signal, "sim": "replay" if self.dataset else "synthetic",
                          "eeg": "replay" if self.practice else signal}[self.args.input]
                # dataset replay: the EEG is real but a key chose which recording played, so not "eeg"
                mode = {"keyboard": "keyboard", "sim": "keyboard" if self.dataset else "synthetic",
                        "eeg": "keyboard" if self.practice else "eeg"}[self.args.input]
                extra = {"signal_source": source, "input_mode": mode}
            self.made = ("ok", make_postcard(eeg, self.cfg["fs"], self.message, self.nickname, **extra))
        except Exception as e:
            self.made = ("error", str(e))

    def update_making(self, now):
        if self.worker is None:
            wait = 1.5 if self.from_clf else 0.0         # give the classifier time to save the recording
            if self.saved_path or now - self.making_t0 >= wait:
                self.worker = threading.Thread(target=self._make, args=(self.saved_path,), daemon=True)
                self.worker.start()
        elif self.made is not None:
            status, value = self.made
            if status == "ok":
                self.postcard = value
                try:
                    img = pygame.image.load(value).convert()
                    scale = min(self.W * 0.52 / img.get_width(), self.H * 0.49 / img.get_height())
                    img = pygame.transform.smoothscale(
                        img, (int(img.get_width() * scale), int(img.get_height() * scale)))
                    pad = max(8, int(self.H * 0.012))                # white photo border, slightly tilted
                    framed = pygame.Surface((img.get_width() + 2 * pad, img.get_height() + 2 * pad), pygame.SRCALPHA)
                    framed.fill(WHITE)
                    framed.blit(img, (pad, pad))
                    self.postcard_img = pygame.transform.rotozoom(framed, 1.6, 1.0)
                    self.start_confetti(now)
                except Exception as e:
                    print("[ui] could not show the postcard:", e)
            elif status == "noart":
                self.consent, self.upload_msg = False, "No EEG was recorded, so no postcard art was made"
            else:
                print("[ui] postcard failed:", value)
                self.banner = ("Postcard art failed - the message is still saved in the log", now, RED)
            self.say(self.message)
            self.state = "result"

    # ---------------------------------------------------------------- drawing
    def link_status(self, now):
        """(text, color) describing where selections come from."""
        if self.args.input == "keyboard":
            return "KEYBOARD DEMO MODE - not using EEG", AMBER
        # alive = scores, or the classifier's once-a-second contact report (it sends no scores for
        # about 3 s after each pick while it collects a fresh window)
        if now - max(self.scores_t, self.quality_t) < 2.2:
            if self.dataset:
                return "Recorded human EEG (SSVEP dataset replay) - decoded live", GREEN
            if self.args.input == "sim":
                return "Simulated EEG - classifier live", GREEN
            return self.headset_status(now) or ("EEG classifier live", GREEN)
        return self.headset_status(now) or ("Waiting for classifier (start classifier_node.py)", RED)

    def headset_status(self, now):
        """(text, color) from the classifier's electrode report, or None when there is nothing to add.
        Only for the live headset screen: says when the stream stopped or an electrode lost contact."""
        q = self.quality
        if self.args.input != "eeg" or not q or now - self.quality_t > 3.0:
            return None
        if q.get("lost"):
            return "No data from the headset - check Bluetooth", RED
        bad = [n for n, st in zip(q.get("names", []), q.get("state", [])) if st != "good"]
        if bad:
            return "Check electrode" + ("s " if len(bad) > 1 else " ") + ", ".join(bad) + ": poor contact", AMBER
        if "PRACTICE" in str(q.get("stream", "")).upper():        # acquisition/fake_unicorn.py, not a person
            return "Practice headset - recorded EEG", GREEN
        return "Live headset - electrodes good", GREEN

    def draw_header(self, now):
        fps = self.clock.get_fps()
        bad = self.frame > 120 and abs(fps - self.refresh) > 2
        if bad or self.help:                        # a technical number: only shown when it matters
            self.screen.blit(self.f_tiny.render(f"{fps:5.1f} fps", True, RED if bad else DIM), (12, 8))
        if self.lang_i:
            self.screen.blit(self.f_tiny.render(f"speaks {LANGUAGES[self.lang_i][1]}", True, CYAN), (110, 8))
        if self.speller.mode == "patient":
            surf = self.f_tiny.render("PATIENT MODE", True, CYAN)
            r = surf.get_rect(topright=(self.W - 22, 10)).inflate(24, 10)
            pygame.draw.rect(self.screen, (16, 40, 56), r, border_radius=r.h // 2)
            pygame.draw.rect(self.screen, CYAN, r, 1, border_radius=r.h // 2)
            self.screen.blit(surf, surf.get_rect(center=r.center))

    def draw_typed(self):
        sp = self.speller
        bar = self.typed_bar                                # (the bar itself is part of the background)
        if sp.text or not sp.in_menu:
            surf = self.fitted(sp.text or " ", bar.w - 170, self.H * 0.085, WHITE)
            r = surf.get_rect(center=bar.center)
            if sp.text:
                self.screen.blit(surf, r)
            else:                                               # empty: say what this bar is for
                ghost = self.f_mid.render("Your message appears here" if sp.mode == "postcard"
                                          else "Your sentence appears here", True, (84, 88, 112))
                r = ghost.get_rect(center=bar.center)
                self.screen.blit(ghost, r)
                r = pygame.Rect(r.x - 22, r.y, 0, r.h)          # caret in front of the placeholder
            if int(time.time() * 1.6) % 2 == 0:                 # slow caret blink (far from the flicker rates)
                h = int(self.H * 0.062)
                pygame.draw.rect(self.screen, CYAN, (r.right + 8, bar.centery - h // 2, 5, h), border_radius=2)
            if sp.mode == "postcard":
                count = self.f_tiny.render(f"{len(sp.text)}/{sp.max_chars}", True, AMBER if sp.full else DIM)
                self.screen.blit(count, count.get_rect(midright=(bar.right - 20, bar.centery)))
                track = pygame.Rect(bar.x + 28, bar.bottom - 12, bar.w - 56, 4)        # how full the postcard is
                pygame.draw.rect(self.screen, (40, 44, 64), track, border_radius=2)
                fill = track.copy()
                fill.w = int(track.w * min(1.0, len(sp.text) / max(1, sp.max_chars)))
                if fill.w:
                    pygame.draw.rect(self.screen, AMBER if sp.full else CYAN, fill, border_radius=2)
        else:
            self.text(sp.hint(), self.f_mid,
                      WHITE if sp.alarm else MUTED, bar.center)

    def resting(self, now):
        """Should the boxes hold still? The flicker runs only while a choice is being made: not for
        a moment after each pick, not while a sentence is spoken, not during the alarm or a pause.
        Less time under flicker is easier on the eyes and lowers the risk for light-sensitive people."""
        return (self.paused or self.speller.alarm or now - self.last_sel_t < self.rest_s
                or (bool(self.spoken[0]) and now - self.spoken[1] < 3.5))

    def track_signal(self, now):
        """Steady what the screen shows about the signal. Display only: selections are still decided
        by the classifier alone. Without this the "locking on" outline blinked on and off whenever
        the lead hovered at the threshold, and the bars jumped with every classifier update."""
        dt, self.sig_t = min(0.1, max(0.0, now - self.sig_t)), now
        fresh = self.from_clf is not None and now - self.scores_t < 1.5 and not self.paused
        target = [max(0.0, min(1.0, float(s))) if fresh else 0.0 for s in self.scores]
        k = 1.0 - float(np.exp(-dt / 0.12))                              # bars glide to the new value
        self.show_scores = [a + (b - a) * k for a, b in zip(self.show_scores, target)]
        order = np.argsort(target)[::-1]
        lead, margin, tau = int(order[0]), target[order[0]] - target[order[1]], float(self.cfg["margin_tau"])
        cand = lead if fresh and margin > tau else -1
        if self.lock_box >= 0 and fresh and lead == self.lock_box and margin > 0.6 * tau:
            cand = self.lock_box                                         # once shown, it takes a real drop to lose it
        if cand != self.lock_cand:
            self.lock_cand, self.lock_cand_t = cand, now
        if now - self.lock_cand_t >= 0.25:                               # ...and any change must last a moment
            self.lock_box = cand

    def draw_strength(self, now, cx, y):
        """How clearly the last pick stood out. The classifier's number is a score margin, not a
        probability, so it is shown as signal strength against the threshold, never as a percentage."""
        label, margin, t = self.pick_info
        if margin is None or now - t > 5.0 or self.args.input == "keyboard":
            return
        tau = float(self.cfg["margin_tau"])
        level = 1 if margin < 1.5 * tau else 2 if margin < 3 * tau else 3
        word = {1: "Clear signal", 2: "Strong signal", 3: "Very strong signal"}[level]
        surf = self.f_small.render(word, True, GREEN)
        bw, bh, gap = 9, int(self.H * 0.026), 5
        x = int(cx - (3 * bw + 2 * gap + 12 + surf.get_width()) / 2)
        for i in range(3):
            h = int(bh * (0.45 + 0.275 * i))
            pygame.draw.rect(self.screen, GREEN if i < level else (50, 54, 76),
                             (x + i * (bw + gap), int(y + bh / 2 - h), bw, h), border_radius=2)
        self.screen.blit(surf, surf.get_rect(midleft=(x + 3 * bw + 2 * gap + 12, int(y))))
        self.text(f"decoder margin {margin:.2f}  -  needs {tau:.2f}", self.f_tiny, DIM, (cx, y + self.H * 0.032))

    def draw_boxes(self, now):
        labels = self.speller.labels()
        still = self.resting(now) or self.still_mode
        if still:
            self.still_t = now
        depth = self.contrast * min(1.0, (now - self.still_t) / 0.4)     # the flicker fades in, never snaps on
        t = self.frame / self.refresh                       # frame-indexed: exact flicker frequency
        if self.args.input == "eeg":                        # a real brain is watching: follow the clock, so a
            t = time.perf_counter()                         # dropped frame cannot shift the frequencies
        order = np.argsort(self.scores)[::-1]
        live = self.from_clf is not None and now - self.scores_t < 1.0 and not self.paused
        margin = self.scores[order[0]] - self.scores[order[1]]
        for i, (rect, f, lab) in enumerate(zip(self.boxes, self.cfg["freqs"], labels)):
            flashing = self.flash[0] == i and now - self.flash[1] < 0.45
            if not lab and not flashing:
                pygame.draw.rect(self.screen, (24, 26, 38), rect, 2, border_radius=26)
                continue
            if flashing:                                    # clear "you picked this" confirmation
                age = (now - self.flash[1]) / 0.45          # 0..1: a ring grows out of the box and fades
                pygame.draw.rect(self.screen, GREEN, rect, border_radius=26)
                grow = int(14 + 44 * age)
                ring = tuple(int(c * (1 - age) + b * age) for c, b in zip(WHITE, BG))
                pygame.draw.rect(self.screen, ring, rect.inflate(grow, grow), max(2, int(8 * (1 - age))),
                                 border_radius=26 + grow // 2)
                lab = self.flash[2] or lab
            else:
                # brightness swings gently around a steady middle grey (flicker_contrast), not black to white
                # and that middle grey is kept dim (flicker_brightness): large bright areas are what tire the eyes
                lum = 0.30 if self.paused else max(0.0, self.bright + 0.5 * depth * np.sin(2 * np.pi * f * t))
                g = int(30 + 225 * lum)
                pygame.draw.rect(self.screen, (g, g, g), rect, border_radius=26)
            surf = self.option_surface(lab, rect.w, rect.h)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
            if self.args.input != "eeg":                    # demo modes: show which key picks this box
                pygame.draw.circle(self.screen, (12, 13, 20), (rect.x + 28, rect.y + 28), 15)
                self.text(str(i + 1), self.f_tiny, MUTED, (rect.x + 28, rect.y + 28))
            if self.test and self.cued == i:
                pygame.draw.rect(self.screen, RED, rect.inflate(30, 30), 8, border_radius=38)
            elif live and not flashing and self.lock_box == i:
                pygame.draw.rect(self.screen, CYAN, rect.inflate(18, 18), 4, border_radius=32)   # "locking on"
            if self.args.input == "sim" and self.sim_target == i:
                self.text("playing a recording of this light" if self.dataset else "sim gaze", self.f_tiny, CYAN,
                          (rect.centerx, rect.bottom + 14))

    def option_surface(self, raw, w, h):
        """What is written on a box: the words on a dark plate (readable at every flicker
        brightness), an optional small line saying what is inside, and a style:
        a choice (white), a word suggestion (cyan edge), EMERGENCY (red), or navigation (quiet)."""
        key = ("opt", raw, w, h)
        if key in self._surfs:
            return self._surfs[key]
        main = self.clean("".join(c for c in raw if c not in "★␣✗✎☰"))
        name = main.strip(" <>-" + "←▶").replace("–", "-")
        nav = any(word in main for word in NAV_WORDS)
        danger = "EMERGENCY" in main
        word = raw.startswith("★") and name not in SUBTITLES          # a predicted word
        sub = ("adds a space" if "␣" in raw else "finish the word") if word else SUBTITLES.get(name, "")
        if name.startswith("VOICE:"):
            sub = "look here to change the language"
        if name == "COME HERE" and self.alert.enabled:
            sub = "bell + alerts their phone"
        color = (176, 180, 200) if nav else WHITE
        px = self.H * (0.052 if nav else 0.078)             # the word is what the eyes rest on: make it large
        text = self.fitted(main, w - 84, px, color)
        lines = [text]
        if len(main.split()) > 1 and text.get_height() < self.H * 0.05 and not nav:   # long: use two lines
            words = main.split()
            cut = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
            two = [self.fitted(" ".join(part), w - 84, px, color) for part in (words[:cut], words[cut:])]
            if min(t.get_height() for t in two) > text.get_height():
                lines = two
        small = self.f_tiny.render(sub, True, (150, 200, 235) if word else (168, 172, 196)) if sub else None
        tw = max([t.get_width() for t in lines] + ([small.get_width()] if small else []))
        th = sum(t.get_height() for t in lines) + (small.get_height() + 2 if small else 0)
        pad_x, pad_y = (30, 12) if nav else (40, 18)
        plate = pygame.Rect(0, 0, min(w - 24, tw + 2 * pad_x), min(h - 20, th + 2 * pad_y))
        surf = pygame.Surface(plate.size, pygame.SRCALPHA)
        fill = (176, 24, 36) if danger else (26, 28, 42) if nav else (12, 13, 22)
        pygame.draw.rect(surf, fill, plate, border_radius=18)
        if word or danger:
            pygame.draw.rect(surf, WHITE if danger else CYAN, plate, 3, border_radius=18)
        y = (plate.h - th) // 2
        for t in lines:
            surf.blit(t, t.get_rect(midtop=(plate.w // 2, y)))
            y += t.get_height()
        if small:
            surf.blit(small, small.get_rect(midtop=(plate.w // 2, y + 2)))
        if len(self._surfs) > 400:
            self._surfs.clear()
        self._surfs[key] = surf
        return surf

    def draw_page_dots(self, cx, y):
        """Small dots: which page of this menu is showing (page 2 of 3)."""
        name = self.speller.screen
        if not self.speller.in_menu or not name[-1].isdigit():
            return
        pages = sorted(k for k in self.speller.screens if k[:-1] == name[:-1] and k[-1].isdigit())
        if len(pages) < 2:
            return
        x = cx - (len(pages) - 1) * 13
        for k in pages:
            pygame.draw.circle(self.screen, WHITE if k == name else (60, 64, 88), (x, y), 6 if k == name else 4)
            x += 26

    def draw_signal(self, now):
        """Mini-map of the 4 boxes: how strongly each flicker frequency shows up in the EEG right now."""
        m = self.mid
        tw, th, gap = int(m.w * 0.40), int(self.H * 0.085), int(m.w * 0.06)
        x0, y0 = m.centerx - tw - gap // 2, int(self.H * 0.615)
        fresh = now - self.scores_t < 1.5 and not self.paused
        best = self.lock_box                                # a winner is highlighted only when it clearly leads
        for i in range(4):
            r = pygame.Rect(x0 + (i % 2) * (tw + gap), y0 + (i // 2) * (th + gap), tw, th)
            pygame.draw.rect(self.screen, PANEL, r, border_radius=10)
            sc = self.show_scores[i]
            if sc > 0.01:
                fill = pygame.Rect(r.x, r.y, max(8, int(r.w * sc)), r.h)
                pygame.draw.rect(self.screen, GREEN if i == best else (44, 92, 128), fill, border_radius=10)
            pygame.draw.rect(self.screen, LINE, r, 2, border_radius=10)
            self.text(f"{self.cfg['freqs'][i]:g} Hz", self.f_tiny, WHITE if i == best else MUTED, r.center)
        # one line that says what the decoder is doing right now
        if self.paused:
            state, col = "Paused", DIM
        elif not fresh:
            state, col = "No signal from the classifier", AMBER
        elif now - self.last_sel_t < self.rest_s:
            state, col = "Picked - boxes rest a moment", GREEN
        elif best >= 0:
            state, col = f"Locking on {self.cfg['freqs'][best]:g} Hz", CYAN
        else:
            state, col = "Reading the brain signal...", MUTED
        self.text(state, self.f_tiny, col, (m.centerx, y0 + 2 * th + gap + self.H * 0.024))   # under the bars

    def draw_spell(self, now):
        sp, m = self.speller, self.mid
        self.draw_header(now)
        self.draw_typed()
        self.track_signal(now)
        self.draw_boxes(now)

        # middle column: what to do, what was just picked, live signal, status
        y = int(self.H * 0.265)
        if sp.alarm:
            self.draw_alarm(now)
            return
        if self.paused:
            self.text("PAUSED", self.f_big, RED, (m.centerx, y + self.H * 0.04))
            self.text("press P to continue", self.f_small, MUTED, (m.centerx, y + self.H * 0.11))
            if self.args.input == "keyboard":
                self.paragraph("Letters are not typed directly: pick boxes with keys 1 2 3 4, like the eyes would",
                               self.f_tiny, AMBER, m.centerx, y + self.H * 0.16, m.w)
        elif self.test:
            self.text("ACCURACY TEST", self.f_mid, RED, (m.centerx, y + 10))
            pct = f"  {100 * self.test_ok / self.test_n:.0f}%" if self.test_n else ""
            self.text(f"{self.test_ok} / {self.test_n} correct{pct}", self.f_mid, WHITE, (m.centerx, y + self.H * 0.075))
            self.paragraph("Look at the red-outlined box", self.f_small, MUTED, m.centerx, y + self.H * 0.115, m.w)
        else:
            if sp.text or not sp.in_menu:                        # (the top bar already says it on menu pages)
                y = self.paragraph(sp.hint(), self.f_small, MUTED, m.centerx, y, m.w)
            if sp.screen == "say":                           # (this page shows Gemini's sentence here instead)
                pass
            elif self.spoken[0] and now - self.spoken[1] < 7:
                y2 = self.paragraph(f"\"{self.spoken[0]}\"", self.f_mid, GREEN, m.centerx, y + 14, m.w)
                name, sentence, t = self.said_in
                if name and now - t < 7:                     # what the listener actually heard
                    shown = sentence if sentence and all(self.glyph_source(c) == "main" or c == " "
                                                         for c in sentence) else ""
                    self.paragraph(f"spoken in {name}" + (f": {shown}" if shown else ""), self.f_tiny, CYAN,
                                   m.centerx, y2 + 6, m.w)
            elif self.flash[2] and now - self.flash[1] < 2.0:
                self.screen.blit(*self._centered(self.fitted(self.clean("✓ " + self.flash[2].lstrip("★* ")), m.w,
                                                             self.H * 0.05, GREEN), m.centerx, y + self.H * 0.06))
            self.draw_page_dots(m.centerx, int(self.H * 0.59))
            if not (self.care_msg[0] and now - self.care_msg[1] < 20):
                self.draw_strength(now, m.centerx, self.H * 0.485)
            # (the automatic demo is not announced here and its sentence is never shown in advance:
            #  the screen shows only what the decoder works out. The signal-source label stays.)
            if sp.screen == "say":                           # Gemini's fuller sentence, for the person to judge
                better = sp.expanded
                if better:
                    self.paragraph(f"Gemini: \"{better}\"", self.f_small, CYAN, m.centerx, y + self.H * 0.03, m.w)
                elif better is None:
                    self.paragraph("Gemini is writing a fuller sentence...", self.f_tiny, DIM, m.centerx,
                                   y + self.H * 0.03, m.w)
            elif sp.screen == "asked" and self.ask_wait[0]:
                self.paragraph("Gemini is finding answers...", self.f_tiny, DIM, m.centerx, y + self.H * 0.03, m.w)
            if self.care_msg[0] and now - self.care_msg[1] < 20:        # a caregiver answered from their phone
                self.paragraph("Caregiver: " + self.clean(self.care_msg[0]), self.f_mid, CYAN, m.centerx,
                               self.H * 0.40, m.w)
            self.sync_alert()
            if self.alert_note[0] and now - self.alert_note[2] < 12:
                self.paragraph(self.alert_note[0], self.f_tiny, self.alert_note[1], m.centerx, self.H * 0.545, m.w)
        if self.from_clf:
            self.draw_signal(now)
        status, color = self.link_status(now)
        status = {"KEYBOARD DEMO MODE - not using EEG": "Keyboard demo - no EEG" + (", no flashing" if self.still_mode else ""),         # short forms that
                  "Recorded human EEG (SSVEP dataset replay) - decoded live": "Recorded human EEG - decoded live",
                  "Simulated EEG - classifier live": "Simulated EEG - decoded live",        # fit the column
                  "Waiting for classifier (start classifier_node.py)": "Waiting for the classifier"}.get(status, status)
        self.chip(status, color, (m.centerx, self.H * 0.875))
        keys = "TAB all keys   -   ESC exit"                 # (the full list lives on the TAB card)
        if self.args.input != "eeg":
            keys = "1 2 3 4 or click to pick   -   " + keys
        self.text(keys, self.f_tiny, DIM, (m.centerx, self.H * 0.975))      # clear of the boxes above

    def draw_alarm(self, now):
        """Full-width red alert. It stays (siren + repeated voice) until a helper presses SPACE."""
        if now - self.alarm_said > 6.0:
            self.alarm_said = now
            self.say(self.spoken[0] or "Emergency! I need help right now.")
        if self.alarm_calls < 5 and now - self.alarm_t0 > 60 * self.alarm_calls:   # nobody came: call again,
            self.alarm_calls += 1                                                  # once a minute, 5 times at most
            self.phone_alert(now, urgent=True)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * (now - self.alarm_t0))       # slow 1 Hz pulse
        panel = pygame.Rect(int(self.W * 0.04), int(self.H * 0.24), int(self.W * 0.92), int(self.H * 0.68))
        pygame.draw.rect(self.screen, (int(120 + 110 * pulse), 18, 24), panel, border_radius=30)
        pygame.draw.rect(self.screen, WHITE, panel, 6, border_radius=30)
        cx = panel.centerx
        surf = self.fitted("EMERGENCY", panel.w * 0.8, self.H * 0.19, WHITE)
        self.screen.blit(surf, surf.get_rect(center=(cx, int(panel.y + panel.h * 0.28))))
        self.text("I NEED HELP RIGHT NOW", self.f_big, WHITE, (cx, panel.y + panel.h * 0.55))
        self.text(f"calling for {int(now - self.alarm_t0)} s", self.f_small, WHITE, (cx, panel.y + panel.h * 0.70))
        if self.alert.enabled:
            self.sync_alert()
            self.text(self.alert_note[0] or f"Phone alert: {self.alert.describe()}", self.f_small, WHITE,
                      (cx, panel.y + panel.h * 0.775))
        self.text("Helper: press SPACE when you are with the person", self.f_mid, YELLOW,
                  (cx, panel.y + panel.h * 0.86))

    def _centered(self, surf, cx, cy):
        return surf, surf.get_rect(center=(int(cx), int(cy)))

    def mode_card(self, key, title, blurb, rect, active):
        """Big selectable card on the start screen, in the style of the typing screen's panels."""
        self.card(rect, 24, fill=(27, 33, 54) if active else PANEL, border=(70, 76, 108))
        if active:
            pygame.draw.rect(self.screen, CYAN, rect, 3, border_radius=24)
            tag = self.f_tiny.render("selected", True, CYAN)
            self.screen.blit(tag, tag.get_rect(topright=(rect.right - 22, rect.y + 18)))
        self.keycap(key, (rect.x + 46, rect.y + 32), WHITE if active else MUTED)
        t = self.font(self.H * 0.058).render(title, True, WHITE if active else MUTED)
        self.screen.blit(t, t.get_rect(center=(rect.centerx, int(rect.y + rect.h * 0.50))))
        self.paragraph(blurb, self.f_small, MUTED if active else DIM, rect.centerx, int(rect.y + rect.h * 0.70),
                       rect.w - 50)
        self.click_zones.append((rect, lambda m=title.lower(): self.choose_preset(m)))

    def language_picker(self, center, width=None):
        """Start screen: which language the voice speaks. Arrow keys, F5 or a click change it."""
        code, name = LANGUAGES[self.lang_i]
        r = pygame.Rect(0, 0, width or int(self.W * 0.42), int(self.H * 0.072))
        r.center = (int(center[0]), int(center[1]))
        self.card(r, 18, border=CYAN if self.lang_i else (70, 76, 108))
        left = self.render("◀", self.H * 0.034, WHITE)
        right = self.render("▶", self.H * 0.034, WHITE)
        self.screen.blit(left, left.get_rect(center=(r.x + 34, r.centery)))
        self.screen.blit(right, right.get_rect(center=(r.right - 34, r.centery)))
        label = self.f_small.render("Voice speaks", True, MUTED)
        value = self.f_mid.render(name, True, CYAN if self.lang_i else WHITE)
        x = r.centerx - (label.get_width() + 14 + value.get_width()) // 2
        self.screen.blit(label, label.get_rect(midleft=(x, r.centery)))
        self.screen.blit(value, value.get_rect(midleft=(x + label.get_width() + 14, r.centery)))
        half = r.w // 2
        self.click_zones.append((pygame.Rect(r.x, r.y, half, r.h), lambda: self.next_language(-1)))
        self.click_zones.append((pygame.Rect(r.x + half, r.y, half, r.h), lambda: self.next_language(1)))
        cover = self.translator.ready(code)
        if cover != "table":
            self.text("No translations for this language: it will speak English" if cover == "none"
                      else "Translated live by Gemini", self.f_tiny, AMBER if cover == "none" else MUTED,
                      (r.centerx, r.bottom + 12))

    def draw_intro(self, now):
        """Start screen in the look of the typing screen: flat dark page, quiet blue panels, and
        nothing that moves (only the caret in the nickname field blinks)."""
        W, H = self.W, self.H
        self.click_zones = []
        cx = int(W * 0.335) if self.hero else W // 2            # with a picture: words left, picture right
        col = int(W * (0.60 if self.hero else 0.72))            # width of the content column
        edge = (70, 76, 108)
        if self.logo:
            self.screen.blit(self.logo, self.logo.get_rect(center=(cx, int(H * 0.084))))
            self.text("Look at a box to say what you need, or to write a postcard.", self.f_tiny, MUTED,
                      (cx, H * 0.176))
        else:
            self.text("BrainType Postcards", self.font(H * 0.072), WHITE, (cx, H * 0.085))
            self.text("Look at a box to say what you need, or to write a postcard.", self.f_small, MUTED,
                      (cx, H * 0.152))

        patient, postcard = self.preset == "patient", self.preset == "postcard"
        gap, top = int(W * 0.022), int(H * 0.200)
        cw, ch = (col - gap) // 2, int(H * 0.205)
        left = pygame.Rect(cx - col // 2, top, cw, ch)
        self.mode_card("F1", "Postcard", "Spell a short message, keep the art",
                       left, postcard)
        self.mode_card("F2", "Patient", "Say a need with a single look",
                       pygame.Rect(left.right + gap, top, cw, ch), patient)
        if not self.preset:
            self.text("No mode picked: the person chooses the mode and the language by looking",
                      self.f_tiny, GREEN, (cx, top + ch + H * 0.022))

        fw = int(col * 0.62)
        field = pygame.Rect(0, 0, fw, int(H * 0.078))
        field.center = (cx, int(H * 0.492))
        self.card(field, 20, border=edge)
        if self.nickname:
            surf = self.fitted(self.nickname, field.w - 60, field.h * 0.62, WHITE)   # always inside the field
            r = surf.get_rect(center=field.center)
            self.screen.blit(surf, r)
            if int(now * 2) % 2 == 0:
                pygame.draw.rect(self.screen, CYAN, (r.right + 6, field.centery - field.h * 0.3, 4, field.h * 0.6))
        else:
            self.text("Type a nickname", self.f_mid, DIM, field.center)
        self.language_picker((cx, H * 0.585), fw)

        btn = pygame.Rect(0, 0, int(col * 0.34), int(H * 0.070))                 # Start: ENTER or a click
        btn.center = (cx, int(H * 0.690))
        self.card(btn, btn.h // 2, fill=(30, 74, 110), border=CYAN)
        label = self.f_mid.render("Start", True, WHITE)
        hint = self.f_tiny.render("ENTER", True, (190, 225, 245))
        x = btn.centerx - (label.get_width() + 14 + hint.get_width()) // 2
        self.screen.blit(label, label.get_rect(midleft=(x, btn.centery - 1)))
        self.screen.blit(hint, hint.get_rect(midleft=(x + label.get_width() + 14, btn.centery + 2)))
        self.click_zones.append((btn, self.start_visitor))

        if self.still_mode:
            self.text("Keyboard demo: the boxes do not flash.", self.f_tiny, MUTED, (cx, H * 0.755))
        else:
            self.text("This screen flashes. Not for anyone with photosensitive epilepsy, or who feels unwell "
                      "from flashing lights.", self.f_tiny, AMBER, (cx, H * 0.755))

        # compact status panel: where the input comes from and which helpers are connected
        status, color = self.link_status(now)
        rows = [("Input: " + ("keyboard demo, no EEG" if self.args.input == "keyboard" else status), color),
                (f"Words: {self.predictor.describe()}", AMBER if self.predictor.source == "built-in" else GREEN)]
        if self.cfg.get("use_llm"):
            rows.append(("Gemini: connected" if self.assistant.enabled else "Gemini: GEMINI_API_KEY is not set",
                         GREEN if self.assistant.enabled else AMBER))
        if self.cfg.get("use_tts"):
            keyed = bool(os.environ.get("ELEVENLABS_API_KEY"))
            name = f"{VOICES[self.voice_i % len(VOICES)][0]} (F3 = next)" if VOICES else "ElevenLabs"
            rows.append((f"Voice: {name}" if keyed else "Voice: built-in (ElevenLabs key not found)",
                         GREEN if keyed else AMBER))
        if self.care.enabled or self.cfg.get("care_app_url"):
            ok = self.care.enabled and not self.care.last_error
            rows.append(("Caregiver app: connected" if ok else "Caregiver app: not connected ("
                         + (self.care.last_error or "BASE44_TOKEN is not set") + ")", GREEN if ok else AMBER))
        if not postcard:
            rows.append((f"Phone alerts: {self.alert.describe()}", GREEN if self.alert.enabled else DIM))
            rows.append(("Easy mode ON: two boxes at a time (F4 = off)" if self.speller.easy else
                         "Weak signal? F4 = two-box easy mode", CYAN if self.speller.easy else DIM))
        if not self.vsync:
            rows.append(("Vsync is off: flicker timing is less exact", AMBER))
        if self.fps_warn:
            rows.append((self.fps_warn, AMBER))
        panel = pygame.Rect(cx - col // 2, int(H * 0.785), col, int(H * 0.158))
        self.card(panel, 18, border=edge)
        per = (len(rows) + 1) // 2
        step = min(H * 0.036, (panel.h - 16) / per)
        y0 = panel.centery - per * step / 2
        for i, (line, c) in enumerate(rows):
            x = panel.x + 28 + (i // per) * (panel.w // 2)
            y = int(y0 + (i % per + 0.5) * step)
            pygame.draw.circle(self.screen, c, (x, y + 1), 5)
            surf = self.fitted(self.clean(line), panel.w // 2 - 60, H * 0.0225, MUTED if c in (GREEN, DIM) else c,
                               bold=False)
            self.screen.blit(surf, surf.get_rect(midleft=(x + 16, y)))
        self.text("TAB all keys   -   ESC quit", self.f_tiny, DIM, (cx, H * 0.972))

    def draw_question(self, now):
        """The helper types the question they asked; ENTER puts suggested answers on the boxes."""
        card = pygame.Rect(0, 0, int(self.W * 0.62), int(self.H * 0.30))
        card.center = (self.W // 2, self.H // 2)
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))
        self.card(card, 24, fill=(20, 22, 34), glow=CYAN)
        self.text("What did you ask?", self.f_mid, WHITE, (card.centerx, card.y + card.h * 0.20))
        caret = "|" if int(now * 2) % 2 == 0 else " "
        surf = self.fitted(self.asking + caret, card.w - 80, self.H * 0.05, YELLOW, bold=False)
        self.screen.blit(surf, surf.get_rect(center=(card.centerx, int(card.y + card.h * 0.50))))
        self.text("ENTER = Gemini suggests three answers   -   empty + ENTER = YES / NO   -   ESC cancel",
                  self.f_tiny, MUTED, (card.centerx, card.y + card.h * 0.82))

    def draw_help(self):
        """TAB: every operator key in one place, so nobody has to remember them."""
        rows = [("Start screen", ""), ("F1 / F2", "postcard mode / patient mode"),
                ("F3", "next voice"), ("F4", "two-box easy mode for a weak signal (patient)"),
                ("F5 or arrows", "language the voice speaks (or click it)"),
                ("ENTER", "start"),
                ("While typing", ""), ("1 2 3 4", "pick a box, or click it (keyboard and sim modes)"),
                ("P", "pause / continue"), ("T", "accuracy test: look at the red box"),
                ("F", "keyboard demo: boxes flash / do not flash"),
                ("A", "automatic demo on / off (sim and dataset replay)"),
                ("Q", "you asked a question: type it, get answer boxes (patient)"),
                ("R  /  L", "say the last phrase again  /  next spoken language"),
                ("SPACE", "acknowledge an emergency alarm"), ("ESC", "back to the start screen"),
                ("Postcard ready", ""), ("Y / N", "post to the gallery / keep private"), ("SPACE", "next visitor")]
        card = pygame.Rect(0, 0, int(self.W * 0.66), int(self.H * 0.86))
        card.center = (self.W // 2, self.H // 2)
        dim = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 170))
        self.screen.blit(dim, (0, 0))
        self.card(card, 26, fill=(20, 22, 34))
        self.text("Keys", self.f_mid, WHITE, (card.centerx, card.y + self.H * 0.05))
        y, step = card.y + self.H * 0.105, (card.h - self.H * 0.16) / len(rows)
        for key, what in rows:
            if not what:                                       # section title
                t = self.f_tiny.render(key.upper(), True, CYAN)
                self.screen.blit(t, t.get_rect(midleft=(card.x + card.w * 0.07, y + step * 0.55)))
            else:
                k = self.f_small.render(key, True, YELLOW)
                self.screen.blit(k, k.get_rect(midright=(card.x + card.w * 0.24, y + step * 0.5)))
                d = self.f_small.render(what, True, MUTED)
                self.screen.blit(d, d.get_rect(midleft=(card.x + card.w * 0.28, y + step * 0.5)))
            y += step
        self.text("press any key to close", self.f_tiny, DIM, (card.centerx, card.bottom - self.H * 0.03))

    def draw_making(self, now):
        cx = self.W // 2
        self.waves(now, self.H * 0.30, self.H * 0.13, dim=0.8, speed=2.2)
        dots = "." * (1 + int((now - self.making_t0) * 3) % 3)
        title = self.gradient_text("Painting your brainwaves", self.H * 0.085)
        r = title.get_rect(center=(cx, int(self.H * 0.52)))
        self.screen.blit(title, r)
        self.screen.blit(self.f_big.render(dots, True, PINK), (r.right + 6, r.y))
        surf = self.fitted(f"\"{self.message}\"", self.W * 0.8, self.H * 0.07, WHITE)
        self.screen.blit(surf, surf.get_rect(center=(cx, int(self.H * 0.65))))
        note = ("This postcard is drawn from the recorded EEG that was just decoded" if self.dataset or self.practice else
                "This postcard is drawn from the simulated EEG" if self.args.input == "sim" else
                "Every postcard is drawn from the EEG recorded while you typed" if self.from_clf
                else "Keyboard demo: no EEG was recorded, so this art comes from random noise")
        self.text(note, self.f_small, MUTED, (cx, self.H * 0.74))

    def draw_result(self, now):
        W, H = self.W, self.H
        cx = W // 2 if not self.qr else int(W * 0.40)
        if self.postcard_img:
            r = self.postcard_img.get_rect(center=(cx, int(H * 0.325)))
            self.soft(r.inflate(-24, -24), 6, 22, (0, 0, 0), 170, dy=14)
            self.screen.blit(self.postcard_img, r)
        else:
            surf = self.fitted(self.message, W * 0.6, H * 0.14, WHITE)
            self.screen.blit(surf, surf.get_rect(center=(cx, int(H * 0.315))))
        if self.qr:
            q = self.qr.get_rect(center=(int(W * 0.85), int(H * 0.30)))
            self.card(q.inflate(20, 20), 14, fill=WHITE, width=0)
            self.screen.blit(self.qr, q)
            self.text("Scan to find your postcard", self.f_small, WHITE, (q.centerx, q.bottom + 34))

        self.click_zones = []
        panel = pygame.Rect(0, 0, int(W * 0.74), int(H * 0.345))        # facts and the question, on one quiet panel
        panel.midtop = (W // 2, int(H * 0.612))
        self.card(panel, 24, border=(70, 76, 108))
        who = f"{self.nickname} typed" if self.nickname else "Typed"
        how = ("in keyboard demo mode" if not self.from_clf else
               "decoded from recorded brain signals (dataset replay)" if self.dataset or self.practice else
               "with simulated brain signals" if self.args.input == "sim" else "using only brain signals")
        line = self.fitted(f"{who} \"{self.message}\" {how}", W * 0.70, H * 0.04, WHITE)
        self.screen.blit(line, line.get_rect(center=(W // 2, int(H * 0.655))))
        letters = len(self.message.replace(" ", ""))
        per_min = 60 * letters / self.session_secs if self.session_secs > 0 else 0
        self.stat(f"{self.session_secs:.0f} s", "typing time", (W * 0.33, H * 0.745), CYAN)
        self.stat(str(self.n_sel), "look" + ("" if self.n_sel == 1 else "s"), (W * 0.50, H * 0.745), YELLOW)
        self.stat(f"{per_min:.0f}", "letters per minute", (W * 0.67, H * 0.745), PINK)

        y = H * 0.855
        if self.consent is None:
            self.text("OK to add your postcard to the public gallery?", self.f_mid, WHITE, (W // 2, y))
            self.button("Y", "Yes, post it", (W * 0.40, y + H * 0.062), pygame.K_y, GREEN)
            self.button("N", "Keep it private", (W * 0.60, y + H * 0.062), pygame.K_n, MUTED)
        else:
            self.text("Posting to the gallery" if self.consent else "Kept private", self.f_mid,
                      GREEN if self.consent else MUTED, (W // 2, y - H * 0.012))
            self.text(self.upload_msg, self.f_tiny, DIM, (W // 2, y + H * 0.026))
            self.button("SPACE", "Next visitor", (W // 2, y + H * 0.070), pygame.K_SPACE, CYAN)
        url = str(self.cfg.get("gallery_url", ""))
        if "YOUR-GALLERY-LINK" not in url and self.consent is None:
            self.text(f"Gallery: {url}", self.f_tiny, MUTED, (W // 2, H * 0.978))
        self.draw_confetti(now)

    def button(self, key, label, center, key_code, accent):
        """A pill button that shows its key and can also be clicked (it presses that key)."""
        cap = self.f_tiny.render(key, True, accent)
        text = self.f_small.render(label, True, WHITE)
        r = pygame.Rect(0, 0, cap.get_width() + text.get_width() + 62, int(self.H * 0.056))
        r.center = (int(center[0]), int(center[1]))
        pygame.draw.rect(self.screen, (27, 33, 54), r, border_radius=r.h // 2)
        pygame.draw.rect(self.screen, accent, r, 2, border_radius=r.h // 2)
        self.screen.blit(cap, cap.get_rect(midleft=(r.x + 22, r.centery + 1)))
        self.screen.blit(text, text.get_rect(midleft=(r.x + 22 + cap.get_width() + 18, r.centery)))
        self.click_zones.append((r, lambda: self.handle_key(pygame.event.Event(pygame.KEYDOWN, key=key_code,
                                                                                unicode=""))))
        return r

    def _left(self, surf, key_rect):
        """Place a caption right after a keycap."""
        return surf, surf.get_rect(midleft=(key_rect.right + 12, key_rect.centery))

    # ---------------------------------------------------------------- main loop
    def poll_classifier(self, now):
        for m in self.from_clf.poll():
            kind = m.get("type")
            if kind == "scores" and len(m.get("scores", [])) == 4:
                if not self.paused:                              # paused: the display stays still
                    self.scores, self.scores_t = m["scores"], now
            elif kind == "select" and now - self.last_sel_t >= self.rest_s:   # boxes were still: not a real look
                self.on_select(int(m.get("box", -1)), m.get("confidence", 0), decoded=True)
            elif kind == "session_saved" and self.state == "making":
                self.saved_path = m.get("path")
            elif kind == "quality":
                self.quality, self.quality_t = m, now
                self.practice = self.args.input == "eeg" and "PRACTICE" in str(m.get("stream", "")).upper()

    def check_refresh(self):
        """The flicker frequencies are only right if we know the display's real refresh rate.
        Measured on the intro screen; if it clearly is another standard rate than config.yaml
        says, use the measured one (and say so) instead of flickering at wrong frequencies."""
        fps = self.clock.get_fps()
        if self.vsync and fps > 300:                        # vsync was requested but is not active
            self.vsync = False
        if not self.vsync or self.frame < 60 or self.frame % 20:
            return
        self.fps_samples = (self.fps_samples + [fps])[-9:]
        if len(self.fps_samples) < 5:
            return
        fps = sorted(self.fps_samples)[len(self.fps_samples) // 2]     # median: ignore single hiccups
        configured = float(self.cfg["refresh_rate"])
        rate = min((50, 60, 72, 75, 90, 100, 120, 144, 165, 240), key=lambda r: abs(r - fps))
        if abs(rate - fps) > 1.5:                           # unstable or unusual: keep what we have, but warn
            if abs(fps - self.refresh) > 3:
                self.fps_warn = (f"Display runs at about {fps:.0f} fps, expected {self.refresh:g} - "
                                 "close other apps / check refresh_rate in config.yaml")
            return
        self.refresh = float(rate)
        self.fps_warn = "" if rate == configured else (
            f"Display runs at {rate} Hz: using {rate} Hz for flicker timing (config.yaml says {configured:g})")

    def run(self):
        running = True
        while running:
            now = time.time()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    running = self.handle_key(e) and running
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.handle_click(e.pos)
            if self.from_clf:
                self.poll_classifier(now)
            self.poll_assistant(now)
            self.poll_care(now)
            self.auto_step(now)
            self.push_live(now)
            if self.state == "making":
                self.update_making(now)
            self.screen.blit(self.bg_calm if self.state == "spell" else self.bg_home if self.state == "intro"
                             else self.bg_show, (0, 0))
            {"intro": self.draw_intro, "spell": self.draw_spell, "making": self.draw_making,
             "result": self.draw_result}[self.state](now)
            if self.asking is not None:
                self.draw_question(now)
            if self.help:
                self.draw_help()
            msg, t0, color = self.banner
            if msg and now - t0 < 6:
                self.text(self.clean(msg), self.f_small, color, (self.W // 2, self.H * 0.985 - 14))
            pygame.display.flip()
            self.frame += 1
            self.clock.tick(0 if self.vsync else self.refresh)   # vsync does the real pacing
            if self.state == "intro":
                self.check_refresh()
        self.logf.close()
        pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="BrainType speller UI")
    ap.add_argument("--input", choices=["keyboard", "sim", "eeg"], default="keyboard")
    ap.add_argument("--windowed", action="store_true", help="1280x800 window instead of fullscreen")
    ap.add_argument("--display", type=int, default=0, help="monitor number for fullscreen (0 = main)")
    ap.add_argument("--mute", action="store_true", help="no feedback sounds")
    ap.add_argument("--flicker", action="store_true",
                    help="keyboard mode only: flash the boxes anyway (by default the keyboard demo does not flash)")
    ap.add_argument("--signal", choices=["live", "replay"], default="live",
                    help="what feeds the classifier: the headset (live) or a recording (replay). With "
                         "--input sim, replay means dataset_stream.py: keys 1-4 play real recorded EEG")
    App(ap.parse_args()).run()
