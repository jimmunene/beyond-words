"""BrainType speller UI (pygame).

    python speller/ui.py --input keyboard   # keys 1-4 select directly (UI dev, demo backup)
    python speller/ui.py --input sim        # keys 1-4 = where you "look"; synthetic EEG -> classifier -> UI
    python speller/ui.py --input eeg        # live headset (or replay) via classifier_node.py
    add --windowed for a window instead of fullscreen, --display 1 for the second monitor, --mute for no sounds

Operator keys:
  Intro screen : type nickname, ENTER = start visitor, F1 postcard mode, F2 patient mode, F3 next voice
  While typing : P pause/resume, T cued accuracy test on/off, ESC back to intro
                 patient mode: Q = you asked a yes/no question (shows YES / NO / NOT SURE),
                 R = say the last phrase again, SPACE = acknowledge an emergency alarm
                 (sim mode: 1-4 = look at a box, 0 = look away)
  Result screen: Y = post to gallery, N = keep private, SPACE = next visitor
                 L = next spoken language (start screen: F5)
  Anywhere     : TAB = show all keys. Start screen: F4 = two-box easy mode (patient, weak signal)
"""
import argparse
import csv
import inspect
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
from speller.languages import LANGUAGES, V3_LANGS, Translator, prefetch  # noqa: E402
from speller.predictor import Predictor  # noqa: E402
from speller.sounds import SoundBank  # noqa: E402
from speller.state_machine import Speller, load_phrases  # noqa: E402
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
MUTED = (150, 154, 176)
DIM = (96, 100, 122)
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
        H = self.H
        self.f_title = self.font(H * 0.10)
        self.f_big = self.font(H * 0.085)
        self.f_mid = self.font(H * 0.042)
        self.f_small = self.font(H * 0.028, bold=False)
        self.f_tiny = self.font(H * 0.021, bold=False)

        self.predictor = Predictor(use_llm=self.cfg.get("use_llm", False))
        self.predictor.llm_patient = bool(self.cfg.get("llm_in_patient_mode", False))   # default: postcards only
        self.speller = Speller(self.predictor, "postcard", self.cfg["max_postcard_chars"],
                               quick_messages=self.cfg.get("quick_messages"),
                               my_phrases=load_phrases(ROOT / "data" / "my_phrases.txt"))
        self.from_clf = UDPReceiver(self.cfg["ports"]["selection"]) if args.input != "keyboard" else None
        self.to_clf = UDPSender(self.cfg["ports"]["control"])
        self.to_sim = UDPSender(self.cfg["ports"]["sim_target"])
        self.sounds = SoundBank(enabled=not args.mute)

        # Four big stimulus boxes in the corners, far apart; text and feedback live in the middle column.
        bw, bh = int(self.W * 0.31), int(self.H * 0.28)
        xs, ys = (0.195 * self.W, 0.805 * self.W), (0.40 * self.H, 0.78 * self.H)
        self.boxes = [pygame.Rect(0, 0, bw, bh) for _ in range(4)]
        for r, (x, y) in zip(self.boxes, [(xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]), (xs[1], ys[1])]):
            r.center = (int(x), int(y))
        self.mid = pygame.Rect(int(self.W * 0.37), int(self.H * 0.25), int(self.W * 0.26), int(self.H * 0.70))

        self.state, self.nickname = "intro", ""
        self.frame, self.paused = 0, False
        self.test, self.cued, self.test_n, self.test_ok = False, -1, 0, 0
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
        self.click_zones = []                       # start screen: (rect, what a click there does)
        self.lang_i = 0                             # language the voice speaks (index into LANGUAGES)
        self.said_in = ("", "", 0.0)                # (language name, translated sentence, when)
        self.asking = None                          # the question a helper is typing (None = not typing)
        self.ask_wait = ("", 0.0)                   # (question sent to Gemini, when)
        # Backgrounds are painted once. The typing screen gets the calm one: nothing colourful or
        # moving may compete with the four flickering boxes.
        self.bg = self._background([(0.12, 0.05, 0.95, VIOLET, 0.20), (0.92, 1.0, 0.95, (0, 150, 175), 0.17)])
        self.hero = self._add_hero()                # optional picture on the start screen
        self.bg_calm = pygame.Surface((self.W, self.H)).convert()
        self.bg_calm.fill(BG)
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
        size = (1280, 800) if a.windowed else sizes[disp]
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
            self.bg = self.bg.copy()
            self.bg.blit(img, img.get_rect(center=(int(self.W * 0.775), self.H // 2)), special_flags=pygame.BLEND_RGB_ADD)
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

    def on_select(self, box, confidence=1.0):
        if self.state != "spell" or self.paused or self.speller.alarm or not 0 <= box <= 3:
            return
        now = time.time()
        item = self.speller.items()[box]
        label = self.clean(item[0]) if item else ""
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
            self.speller.spoken = None
        if self.speller.chime:                       # call bell: "come here", heard across the room
            self.sounds.play("bell")
            self.speller.chime = False
        if self.speller.alarm:                       # EMERGENCY: siren + repeated voice until a helper answers
            self.alarm_t0 = self.alarm_said = now
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
                self.speller.set_mode("postcard")
            elif k == pygame.K_F2:
                self.speller.set_mode("patient")
            elif k in (pygame.K_F5, pygame.K_RIGHT):         # language the voice speaks
                self.next_language()
            elif k == pygame.K_LEFT:
                self.next_language(-1)
            elif k == pygame.K_F4:                           # two-box easy mode (patient), for a weak signal
                self.speller.easy = not self.speller.easy
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
            elif k == pygame.K_SPACE:
                self.to_intro()
        elif self.state == "spell":
            if k == pygame.K_p:
                self.paused = not self.paused
                self.to_clf.send({"cmd": "pause" if self.paused else "resume"})
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
            elif k == pygame.K_t:
                self.test = not self.test
                self.test_n = self.test_ok = 0
                self.new_cue() if self.test else setattr(self, "cued", -1)
            else:
                n = {pygame.K_0: -1, pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3,
                     pygame.K_KP0: -1, pygame.K_KP1: 0, pygame.K_KP2: 1, pygame.K_KP3: 2, pygame.K_KP4: 3}.get(k)
                if n is None:
                    return True
                if self.args.input == "keyboard" and n >= 0:
                    self.on_select(n)
                elif self.args.input == "sim":
                    self.sim_target = n                                  # 0 key = look away
                    self.to_sim.send({"target": n})
        return True

    def handle_click(self, pos):
        """Start screen: click a mode card or the language arrows.
        Typing screen (demo modes only): clicking a box works like pressing its number key."""
        if self.state == "intro":
            for rect, action in self.click_zones:
                if rect.collidepoint(pos):
                    action()
                    return
        if self.state != "spell" or self.speller.alarm or self.args.input == "eeg":
            return
        for i, rect in enumerate(self.boxes):
            if rect.collidepoint(pos):
                if self.args.input == "keyboard":
                    self.on_select(i)
                else:
                    self.sim_target = i
                    self.to_sim.send({"target": i})

    def to_intro(self):
        self.stop_alarm()
        self.asking, self.ask_wait = None, ("", 0.0)
        self.state, self.nickname = "intro", ""
        self.test, self.cued = False, -1
        if self.args.input == "sim":
            self.sim_target = -1
            self.to_sim.send({"target": -1})

    def start_visitor(self):
        self.speller.set_mode(self.speller.mode)
        now = time.time()
        self.state, self.session_t0, self.last_sel_t, self.n_sel = "spell", now, now, 0
        self.flash, self.spoken = (-1, 0.0, ""), ("", 0.0)
        self.to_clf.send({"cmd": "start_session"})
        self.paused = False
        self.to_clf.send({"cmd": "resume"})          # also clears the classifier's old EEG window

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
                source = {"keyboard": signal, "sim": "synthetic", "eeg": signal}[self.args.input]
                mode = {"keyboard": "keyboard", "sim": "synthetic", "eeg": "eeg"}[self.args.input]
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
        if now - self.scores_t < 1.5:
            return ("Simulated EEG - classifier live" if self.args.input == "sim" else "EEG classifier live"), GREEN
        return "Waiting for classifier (start classifier_node.py)", RED

    def draw_header(self, now):
        fps = self.clock.get_fps()
        bad = self.frame > 120 and abs(fps - self.refresh) > 2
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
        if sp.text or sp.mode == "postcard" or not sp.in_menu:
            surf = self.fitted(sp.text or " ", bar.w - 170, self.H * 0.085, WHITE)
            r = surf.get_rect(center=bar.center)
            if sp.text:
                self.screen.blit(surf, r)
            else:
                r.width = 0
                r.centerx = bar.centerx
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

    def draw_boxes(self, now):
        labels = self.speller.labels()
        t = self.frame / self.refresh                       # frame-indexed: exact flicker frequency
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
                lum = 0.5 + 0.5 * np.sin(2 * np.pi * f * t) if not self.paused else 0.30
                g = int(30 + 225 * lum)
                pygame.draw.rect(self.screen, (g, g, g), rect, border_radius=26)
            surf = self.option_surface(lab, rect.w, rect.h)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
            if self.args.input != "eeg":                    # demo modes: show which key picks this box
                pygame.draw.circle(self.screen, (12, 13, 20), (rect.x + 28, rect.y + 28), 15)
                self.text(str(i + 1), self.f_tiny, MUTED, (rect.x + 28, rect.y + 28))
            if self.test and self.cued == i:
                pygame.draw.rect(self.screen, RED, rect.inflate(30, 30), 8, border_radius=38)
            elif live and not flashing and int(order[0]) == i and margin > self.cfg["margin_tau"]:
                pygame.draw.rect(self.screen, CYAN, rect.inflate(18, 18), 4, border_radius=32)   # "locking on"
            if self.args.input == "sim" and self.sim_target == i:
                self.text("sim gaze", self.f_tiny, CYAN, (rect.centerx, rect.bottom + 14))

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
        color = (176, 180, 200) if nav else WHITE
        px = self.H * (0.046 if nav else 0.066)
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
        pad_x, pad_y = (26, 10) if nav else (34, 16)
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
        fresh = now - self.scores_t < 1.5
        best = int(np.argmax(self.scores)) if fresh else -1
        for i in range(4):
            r = pygame.Rect(x0 + (i % 2) * (tw + gap), y0 + (i // 2) * (th + gap), tw, th)
            pygame.draw.rect(self.screen, PANEL, r, border_radius=10)
            sc = max(0.0, min(1.0, float(self.scores[i]))) if fresh else 0.0
            if sc > 0.01:
                fill = pygame.Rect(r.x, r.y, max(8, int(r.w * sc)), r.h)
                pygame.draw.rect(self.screen, GREEN if i == best else (44, 92, 128), fill, border_radius=10)
            pygame.draw.rect(self.screen, LINE, r, 2, border_radius=10)
            self.text(f"{self.cfg['freqs'][i]:g} Hz", self.f_tiny, WHITE if i == best else MUTED, r.center)
        self.text("Live brain signal", self.f_tiny, DIM, (m.centerx, y0 - 16))

    def draw_spell(self, now):
        sp, m = self.speller, self.mid
        self.draw_header(now)
        self.draw_typed()
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
            if sp.screen == "say":                           # Gemini's fuller sentence, for the person to judge
                better = sp.expanded
                if better:
                    self.paragraph(f"Gemini: \"{better}\"", self.f_small, CYAN, m.centerx, y + self.H * 0.03, m.w)
                elif better is None:
                    self.paragraph("Gemini is writing a fuller sentence...", self.f_tiny, DIM, m.centerx,
                                   y + self.H * 0.03, m.w)
            elif sp.screen == "asked" and self.ask_wait[0]:
                self.paragraph("Gemini is finding answers...", self.f_tiny, DIM, m.centerx, y + self.H * 0.03, m.w)
        if self.from_clf:
            self.draw_signal(now)
        status, color = self.link_status(now)
        self.chip(status, color, (m.centerx, self.H * 0.875))
        keys = "TAB all keys   -   ESC exit"                 # (the full list lives on the TAB card)
        if self.args.input != "eeg":
            keys = "1 2 3 4 or click to pick   -   " + keys
        self.text(keys, self.f_tiny, DIM, (m.centerx, self.H * 0.955))

    def draw_alarm(self, now):
        """Full-width red alert. It stays (siren + repeated voice) until a helper presses SPACE."""
        if now - self.alarm_said > 6.0:
            self.alarm_said = now
            self.say(self.spoken[0] or "Emergency! I need help right now.")
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 1.0 * (now - self.alarm_t0))       # slow 1 Hz pulse
        panel = pygame.Rect(int(self.W * 0.04), int(self.H * 0.24), int(self.W * 0.92), int(self.H * 0.68))
        pygame.draw.rect(self.screen, (int(120 + 110 * pulse), 18, 24), panel, border_radius=30)
        pygame.draw.rect(self.screen, WHITE, panel, 6, border_radius=30)
        cx = panel.centerx
        surf = self.fitted("EMERGENCY", panel.w * 0.8, self.H * 0.19, WHITE)
        self.screen.blit(surf, surf.get_rect(center=(cx, int(panel.y + panel.h * 0.28))))
        self.text("I NEED HELP RIGHT NOW", self.f_big, WHITE, (cx, panel.y + panel.h * 0.55))
        self.text(f"calling for {int(now - self.alarm_t0)} s", self.f_small, WHITE, (cx, panel.y + panel.h * 0.70))
        self.text("Helper: press SPACE when you are with the person", self.f_mid, YELLOW,
                  (cx, panel.y + panel.h * 0.86))

    def _centered(self, surf, cx, cy):
        return surf, surf.get_rect(center=(int(cx), int(cy)))

    def mode_card(self, key, title, blurb, center, active, accent):
        """Big selectable card on the intro screen."""
        r = pygame.Rect(0, 0, int(self.W * 0.285), int(self.H * 0.125))
        r.center = (int(center[0]), int(center[1]))
        self.card(r, 20, fill=(28, 31, 48) if active else PANEL, glow=accent if active else None)
        self.keycap(key, (r.x + r.w * 0.13, r.centery), WHITE if active else MUTED)
        x = r.x + int(r.w * 0.23)
        t = self.f_mid.render(title, True, WHITE if active else MUTED)
        self.screen.blit(t, t.get_rect(midleft=(x, r.y + r.h * 0.36)))
        b = self.f_tiny.render(blurb, True, MUTED if active else DIM)
        self.screen.blit(b, b.get_rect(midleft=(x, r.y + r.h * 0.70)))
        self.click_zones.append((r, lambda m=title.lower(): self.speller.set_mode(m)))

    def language_picker(self, center):
        """Start screen: which language the voice speaks. Arrow keys, F5 or a click change it."""
        code, name = LANGUAGES[self.lang_i]
        r = pygame.Rect(0, 0, int(self.W * 0.42), int(self.H * 0.072))
        r.center = (int(center[0]), int(center[1]))
        self.card(r, 18, glow=CYAN if self.lang_i else None)
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
        W, H = self.W, self.H
        self.click_zones = []
        cx = int(W * 0.335) if self.hero else W // 2            # with a picture: words left, picture right
        up = H * 0.055 if self.hero else 0                      # the stacked layout needs the room
        if not self.hero:
            self.waves(now, H * 0.165, H * 0.085)
        title = self.gradient_text("BrainType Postcards", H * (0.088 if self.hero else 0.105))
        self.screen.blit(title, title.get_rect(center=(cx, int(H * 0.165 - up))))
        self.text("Type with your mind. Keep the postcard.", self.f_mid, WHITE, (cx, H * 0.275 - 1.6 * up))
        for i, line in enumerate(["You choose letters with your eyes -", "your brainwaves tell the computer which one."]
                                 if self.hero else
                                 ["You choose letters with your eyes - your brainwaves tell the computer which one."]):
            self.text(line, self.f_small, MUTED, (cx, H * 0.325 - 1.7 * up + i * H * 0.036))
        patient = self.speller.mode == "patient"
        dx, top = W * 0.152, H * 0.445
        if self.hero:                                           # narrower column: cards under each other
            self.mode_card("F1", "Postcard", "Spell a message, take home the art", (cx, H * 0.375), not patient, YELLOW)
            self.mode_card("F2", "Patient", "Speak a need with a single look", (cx, H * 0.512), patient, CYAN)
        else:
            self.mode_card("F1", "Postcard", "Spell a message, take home the art", (cx - dx, top), not patient, YELLOW)
            self.mode_card("F2", "Patient", "Speak a need with a single look", (cx + dx, top), patient, CYAN)

        field = pygame.Rect(0, 0, int(W * 0.42), int(H * 0.088))
        field.center = (cx, int(H * (0.635 if self.hero else 0.585)))
        self.card(field, 20, glow=CYAN if patient else YELLOW)
        if self.nickname:
            surf = self.f_big.render(self.nickname, True, WHITE)
            r = surf.get_rect(center=field.center)
            self.screen.blit(surf, r)
            if int(now * 2) % 2 == 0:
                pygame.draw.rect(self.screen, CYAN, (r.right + 6, field.centery - field.h * 0.3, 4, field.h * 0.6))
        else:
            self.text("type a nickname", self.f_mid, DIM, field.center)
        hint = self.f_small.render("to start", True, MUTED)
        self.language_picker((cx, H * (0.725 if self.hero else 0.682)))
        key = self.keycap("ENTER", (cx - hint.get_width() / 2 - 8, H * (0.792 if self.hero else 0.755)))
        self.screen.blit(hint, hint.get_rect(midleft=(key.right + 12, key.centery)))

        low = H * 0.085 if self.hero else H * 0.053
        self.text("This screen flashes. Not for anyone with photosensitive epilepsy, or who feels unwell "
                  "from flashing lights.", self.f_tiny, AMBER, (cx, H * 0.752 + low))
        status, color = self.link_status(now)
        rows = [(status, color), (f"Word prediction: {self.predictor.describe()}",
                                  AMBER if self.predictor.source == "built-in" else GREEN)]
        if patient and self.cfg.get("use_llm"):
            rows.append(("Gemini: Q = suggested answers" + (", fuller sentences after DONE" if self.speller.assistant
                         else "") if self.assistant.enabled else
                         "Gemini is on in config.yaml but GEMINI_API_KEY is not set in this window",
                         GREEN if self.assistant.enabled else AMBER))
        if patient:
            rows.append(("Two-box easy mode is ON: one choice at a time (F4 = off)" if self.speller.easy else
                         "Weak signal? F4 = two-box easy mode", CYAN if self.speller.easy else GREEN))
        if self.cfg.get("use_llm") and not self.predictor.use_llm:
            rows.append(("Gemini is on in config.yaml but GEMINI_API_KEY is not set - local words only", AMBER))
        if self.cfg.get("use_tts") and not os.environ.get("ELEVENLABS_API_KEY"):
            rows.append(("ElevenLabs key not found in this window: new sentences use the computer's built-in "
                         "voice (open a new terminal)", AMBER))
        if VOICES:
            rows.append((f"Voice: {VOICES[self.voice_i % len(VOICES)][0]}   (F3 = next voice)", GREEN))
        if not self.vsync:
            rows.append(("Vsync is off - flicker timing is less exact on this display", AMBER))
        if self.fps_warn:
            rows.append((self.fps_warn, AMBER))
        for i, (line, col) in enumerate(rows):
            self.chip(line, col, (cx, H * 0.79 + low + i * H * 0.029))
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
        note = ("Every postcard is drawn from the EEG recorded while you typed" if self.from_clf
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

        who = f"{self.nickname} typed" if self.nickname else "Typed"
        how = "using only brain signals" if self.from_clf else "in keyboard demo mode"
        line = self.fitted(f"{who} \"{self.message}\" {how}", W * 0.9, H * 0.04, WHITE)
        self.screen.blit(line, line.get_rect(center=(W // 2, int(H * 0.655))))
        letters = len(self.message.replace(" ", ""))
        per_min = 60 * letters / self.session_secs if self.session_secs > 0 else 0
        self.stat(f"{self.session_secs:.0f} s", "typing time", (W * 0.33, H * 0.745), CYAN)
        self.stat(str(self.n_sel), "look" + ("" if self.n_sel == 1 else "s"), (W * 0.50, H * 0.745), YELLOW)
        self.stat(f"{per_min:.0f}", "letters per minute", (W * 0.67, H * 0.745), PINK)

        y = H * 0.855
        if self.consent is None:
            self.text("OK to add your postcard to the public gallery?", self.f_mid, WHITE, (W // 2, y))
            a = self.keycap("Y", (W * 0.37, y + H * 0.06), GREEN)
            self.screen.blit(*self._left(self.f_small.render("yes, post it", True, MUTED), a))
            b = self.keycap("N", (W * 0.55, y + H * 0.06), AMBER)
            self.screen.blit(*self._left(self.f_small.render("keep it private", True, MUTED), b))
        else:
            self.text("Posting to the gallery" if self.consent else "Kept private", self.f_mid,
                      GREEN if self.consent else MUTED, (W // 2, y))
            self.text(self.upload_msg, self.f_tiny, DIM, (W // 2, y + H * 0.04))
            hint = self.f_small.render("next visitor", True, MUTED)
            k = self.keycap("SPACE", (W // 2 - hint.get_width() / 2 - 8, y + H * 0.085), YELLOW)
            self.screen.blit(*self._left(hint, k))
        url = str(self.cfg.get("gallery_url", ""))
        if "YOUR-GALLERY-LINK" not in url and self.consent is None:
            self.text(f"Gallery: {url}", self.f_tiny, MUTED, (W // 2, H * 0.972))
        self.draw_confetti(now)

    def _left(self, surf, key_rect):
        """Place a caption right after a keycap."""
        return surf, surf.get_rect(midleft=(key_rect.right + 12, key_rect.centery))

    # ---------------------------------------------------------------- main loop
    def poll_classifier(self, now):
        for m in self.from_clf.poll():
            kind = m.get("type")
            if kind == "scores" and len(m.get("scores", [])) == 4:
                self.scores, self.scores_t = m["scores"], now
            elif kind == "select":
                self.on_select(int(m.get("box", -1)), m.get("confidence", 0))
            elif kind == "session_saved" and self.state == "making":
                self.saved_path = m.get("path")

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
            if self.state == "making":
                self.update_making(now)
            self.screen.blit(self.bg_calm if self.state == "spell" else self.bg, (0, 0))
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
    ap.add_argument("--signal", choices=["live", "replay"], default="live",
                    help="what feeds the classifier: the headset (live) or replay_stream.py (replay); "
                         "printed on the postcard")
    App(ap.parse_args()).run()
