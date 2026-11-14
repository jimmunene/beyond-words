"""BrainType speller UI (pygame).

    python speller/ui.py --input keyboard   # keys 1-4 select directly (UI dev, demo backup)
    python speller/ui.py --input sim        # keys 1-4 = where you "look"; synthetic EEG -> classifier -> UI
    python speller/ui.py --input eeg        # live headset (or replay) via classifier_node.py
    add --windowed for a window instead of fullscreen

Operator keys:
  Intro screen : type nickname, ENTER = start visitor, F1 postcard mode, F2 patient mode
  Anytime      : P pause/resume, T cued accuracy test on/off, ESC back/quit
  Result screen: Y = post to gallery, N = keep private, SPACE = next visitor
"""
import argparse
import csv
import pathlib
import random
import sys
import time

import numpy as np
import pygame

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from art.postcard import make_postcard  # noqa: E402
from common import ROOT, UDPReceiver, UDPSender, load_config  # noqa: E402
from speller.predictor import Predictor  # noqa: E402
from speller.state_machine import Speller  # noqa: E402
from upload.base44_upload import upload  # noqa: E402
from voice.tts import speak  # noqa: E402

ASCII = {"←": "<", "★": "*", "⌫": "DEL", "✓": "OK", "␣": "_", "–": "-", "▶": ">", "✎": "", "—": "-"}


def clean(label):
    """Default pygame fonts lack emoji/symbols: map to ASCII so nothing renders as boxes."""
    out = "".join(ASCII.get(c, c) for c in label)
    return "".join(c for c in out if ord(c) < 128).strip()


class App:
    def __init__(self, args):
        self.cfg = load_config()
        self.args = args
        pygame.init()
        flags = 0 if args.windowed else pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((0, 0) if not args.windowed else (1280, 800),
                                                  flags | pygame.SCALED, vsync=1)
        except pygame.error:
            self.screen = pygame.display.set_mode((1280, 800))
        pygame.display.set_caption("BrainType Postcards")
        self.W, self.H = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.SysFont("arial", int(self.H * 0.07), bold=True)
        self.f_box = pygame.font.SysFont("arial", int(self.H * 0.055), bold=True)
        self.f_small = pygame.font.SysFont("arial", int(self.H * 0.028))

        self.predictor = Predictor(use_llm=self.cfg["use_llm"])
        self.speller = Speller(self.predictor, "postcard", self.cfg["max_postcard_chars"])
        self.from_clf = UDPReceiver(self.cfg["ports"]["selection"]) if args.input != "keyboard" else None
        self.to_clf = UDPSender(self.cfg["ports"]["control"])
        self.to_sim = UDPSender(self.cfg["ports"]["sim_target"])

        s = int(min(self.W, self.H) * 0.30)
        cx, cy = (0.28 * self.W, 0.72 * self.W), (0.43 * self.H, 0.76 * self.H)
        self.boxes = [pygame.Rect(0, 0, s, int(s * 0.62)) for _ in range(4)]
        for r, (x, y) in zip(self.boxes, [(cx[0], cy[0]), (cx[1], cy[0]), (cx[0], cy[1]), (cx[1], cy[1])]):
            r.center = (int(x), int(y))

        self.state, self.nickname = "intro", ""
        self.frame, self.paused, self.test = 0, False, False
        self.cued, self.flash = -1, (-1, 0.0)
        self.scores = [0, 0, 0, 0]
        self.last_sel_t = time.time()
        self.session_t0, self.postcard, self.postcard_img, self.banner = 0, None, None, ("", 0.0)
        self.sim_target = -1

        logp = ROOT / "logs" / "selections.csv"
        logp.parent.mkdir(exist_ok=True)
        new = not logp.exists()
        self.logf = open(logp, "a", newline="")
        self.log = csv.writer(self.logf)
        if new:
            self.log.writerow(["time", "mode", "event", "screen", "box", "label", "cued_box", "confidence",
                               "seconds_since_last", "text", "session_seconds"])

    # ---------------------------------------------------------------- input
    def on_select(self, box, confidence=1.0):
        if self.state != "spell" or self.paused:
            return
        now = time.time()
        label = self.speller.labels()[box]
        self.speller.select(box)
        self.flash = (box, now)
        self.log.writerow([round(now, 3), self.speller.mode, "select", self.speller.screen, box, clean(label),
                           self.cued if self.test else -1, confidence, round(now - self.last_sel_t, 3),
                           self.speller.text, ""])
        self.logf.flush()
        self.last_sel_t = now
        if self.test:
            self.cued = random.randrange(4)
        if self.speller.spoken:                      # patient-mode phrase
            self.banner = (self.speller.spoken, now)
            speak(self.speller.spoken, self.cfg["elevenlabs_voice_id"], self.cfg["use_tts"])
            self.speller.spoken = None
        if self.speller.finished:
            self.finish_postcard()

    def handle_key(self, e):
        k = e.key
        if k == pygame.K_ESCAPE:
            if self.state == "intro":
                return False
            self.state = "intro"
            return True
        if k == pygame.K_p:
            self.paused = not self.paused
            self.to_clf.send({"cmd": "pause" if self.paused else "resume"})
            return True
        if k == pygame.K_t and self.state == "spell":
            self.test = not self.test
            self.cued = random.randrange(4) if self.test else -1
            return True
        if self.state == "intro":
            if k == pygame.K_F1:
                self.speller.set_mode("postcard")
            elif k == pygame.K_F2:
                self.speller.set_mode("patient")
            elif k == pygame.K_RETURN:
                self.start_visitor()
            elif k == pygame.K_BACKSPACE:
                self.nickname = self.nickname[:-1]
            elif e.unicode.isalnum() and len(self.nickname) < 16:
                self.nickname += e.unicode
            return True
        if self.state == "result":
            if k in (pygame.K_y, pygame.K_n):
                msg = upload(self.postcard, self.speller.text, self.nickname,
                             time.time() - self.session_t0, consent=(k == pygame.K_y))
                self.banner = (msg, time.time())
            elif k == pygame.K_SPACE:
                self.state, self.nickname = "intro", ""
            return True
        if self.state == "spell" and pygame.K_0 <= k <= pygame.K_4:
            n = k - pygame.K_1
            if self.args.input == "keyboard" and 0 <= n <= 3:
                self.on_select(n)
            elif self.args.input == "sim":
                self.sim_target = n if 0 <= n <= 3 else -1      # 0 key = rest
                self.to_sim.send({"target": self.sim_target})
        return True

    def start_visitor(self):
        self.speller.set_mode(self.speller.mode)
        self.state, self.session_t0, self.last_sel_t = "spell", time.time(), time.time()
        self.to_clf.send({"cmd": "start_session"})
        if self.paused:
            self.paused = False
            self.to_clf.send({"cmd": "resume"})

    def finish_postcard(self):
        text = self.speller.text or "HELLO"
        path = ROOT / "sessions" / f"session_{int(time.time())}.npy"
        self.to_clf.send({"cmd": "stop_session", "path": str(path)})
        eeg, deadline = None, time.time() + 1.5
        while time.time() < deadline and self.from_clf:            # wait for the recording
            for m in self.from_clf.poll():
                if m.get("type") == "session_saved":
                    eeg = np.load(m["path"])
                    deadline = 0
            time.sleep(0.02)
        if eeg is None or eeg.ndim != 2 or len(eeg) == 0:
            eeg = np.zeros((0, 8))
        self.postcard = make_postcard(eeg, self.cfg["fs"], text, self.nickname)
        img = pygame.image.load(self.postcard)
        h = int(self.H * 0.62)
        self.postcard_img = pygame.transform.smoothscale(img, (int(h * img.get_width() / img.get_height()), h))
        secs = time.time() - self.session_t0
        self.log.writerow([round(time.time(), 3), "postcard", "postcard_done", "", "", "", "", "", "",
                           text, round(secs, 1)])
        self.logf.flush()
        speak(text, self.cfg["elevenlabs_voice_id"], self.cfg["use_tts"])
        self.state = "result"

    # ---------------------------------------------------------------- drawing
    def text(self, s, font, color, center):
        surf = font.render(s, True, color)
        self.screen.blit(surf, surf.get_rect(center=center))

    def draw_spell(self, now):
        typed = self.speller.text
        mode = "PATIENT MODE" if self.speller.mode == "patient" else "Type with your eyes"
        self.text(mode, self.f_small, (150, 150, 170), (self.W // 2, int(self.H * 0.04)))
        self.text((typed + "_") if self.speller.mode == "postcard" or typed else " ", self.f_big,
                  (255, 255, 255), (self.W // 2, int(self.H * 0.13)))
        labels = self.speller.labels()
        t = self.frame / self.cfg["refresh_rate"]
        for i, (rect, f, lab) in enumerate(zip(self.boxes, self.cfg["freqs"], labels)):
            if not lab:
                continue
            lum = 0.5 + 0.5 * np.sin(2 * np.pi * f * t) if not self.paused else 0.35
            g = int(40 + 215 * lum)
            pygame.draw.rect(self.screen, (g, g, g), rect, border_radius=18)
            label = clean(lab)
            self.text(label, self.f_box, (0, 0, 0), (rect.centerx + 2, rect.centery + 2))
            self.text(label, self.f_box, (255, 210, 0), rect.center)
            if self.flash[0] == i and now - self.flash[1] < 0.35:
                pygame.draw.rect(self.screen, (0, 255, 120), rect.inflate(24, 24), 8, border_radius=24)
            if self.test and self.cued == i:
                pygame.draw.rect(self.screen, (255, 60, 60), rect.inflate(40, 40), 6, border_radius=28)
            if self.args.input == "sim" and self.sim_target == i:
                self.text("(sim gaze)", self.f_small, (120, 200, 255), (rect.centerx, rect.bottom + 22))
        # live confidence bars
        bw = int(self.W * 0.05)
        for i, sc in enumerate(self.scores):
            x = int(self.W * 0.40) + i * (bw + 12)
            h = int(self.H * 0.10 * max(0.0, min(1.0, sc)))
            pygame.draw.rect(self.screen, (60, 200, 255), (x, self.H - 30 - h, bw, h))
            self.text(f"{self.cfg['freqs'][i]:g}Hz", self.f_small, (150, 150, 170), (x + bw // 2, self.H - 14))
        if self.paused:
            self.text("PAUSED (press P)", self.f_big, (255, 80, 80), (self.W // 2, int(self.H * 0.6)))
        if self.test:
            self.text("TEST: look at the red-outlined box", self.f_small, (255, 120, 120),
                      (self.W // 2, int(self.H * 0.22)))

    def draw_intro(self):
        self.text("BrainType Postcards", self.f_big, (255, 255, 255), (self.W // 2, int(self.H * 0.25)))
        self.text("Choose letters with your eyes - your brainwaves tell the computer which one.",
                  self.f_small, (190, 190, 210), (self.W // 2, int(self.H * 0.35)))
        mode = "PATIENT mode (F1 for postcard)" if self.speller.mode == "patient" else "POSTCARD mode (F2 for patient)"
        self.text(mode, self.f_small, (255, 210, 0), (self.W // 2, int(self.H * 0.45)))
        self.text(f"Nickname: {self.nickname}_", self.f_box, (255, 255, 255), (self.W // 2, int(self.H * 0.58)))
        self.text("Operator: type nickname, press ENTER to start   ·   input: " + self.args.input,
                  self.f_small, (130, 130, 150), (self.W // 2, int(self.H * 0.72)))

    def draw_result(self):
        r = self.postcard_img.get_rect(center=(self.W // 2, int(self.H * 0.40)))
        self.screen.blit(self.postcard_img, r)
        self.text("Post to the public gallery?  Y = yes   N = keep private   SPACE = next visitor",
                  self.f_small, (255, 210, 0), (self.W // 2, int(self.H * 0.80)))
        self.text(f"Gallery: {self.cfg['gallery_url']}", self.f_small, (190, 190, 210),
                  (self.W // 2, int(self.H * 0.86)))

    def run(self):
        running = True
        while running:
            now = time.time()
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type == pygame.KEYDOWN:
                    running = self.handle_key(e)
            if self.from_clf:
                for m in self.from_clf.poll():
                    if m.get("type") == "scores":
                        self.scores = m["scores"]
                    elif m.get("type") == "select":
                        self.on_select(m["box"], m.get("confidence", 0))
            self.screen.fill((10, 10, 18))
            {"intro": self.draw_intro, "spell": lambda: self.draw_spell(now),
             "result": self.draw_result}[self.state]()
            msg, t0 = self.banner
            if msg and now - t0 < 6:
                self.text(clean(msg), self.f_big, (120, 255, 160), (self.W // 2, int(self.H * 0.93)))
            pygame.display.flip()
            self.frame += 1
            self.clock.tick(self.cfg["refresh_rate"] + 5)   # vsync does the real pacing
        self.logf.close()
        pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["keyboard", "sim", "eeg"], default="keyboard")
    ap.add_argument("--windowed", action="store_true")
    App(ap.parse_args()).run()
