"""Short feedback sounds for the speller. The .wav files live in speller/assets/ and are
re-created automatically if they are missing. Sound never crashes the UI: with no audio
device, play() simply does nothing.

    python speller/sounds.py        # (re)generate the .wav files
"""
import pathlib
import wave

import numpy as np

ASSETS = pathlib.Path(__file__).resolve().parent / "assets"
RATE = 44100


def _tone(freq, secs, decay=18.0, volume=0.5):
    t = np.arange(int(RATE * secs)) / RATE
    wave_ = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 2 * freq * t)
    env = np.exp(-decay * t) * np.minimum(1.0, t / 0.004)        # soft attack, no click artifacts
    return volume * wave_ * env


def _mix(notes, gap):
    """notes: [(freq, secs)] started `gap` seconds apart."""
    total = int(RATE * (gap * (len(notes) - 1) + max(s for _, s in notes))) + 1
    out = np.zeros(total)
    for i, (f, s) in enumerate(notes):
        tone = _tone(f, s, decay=9.0, volume=0.4)
        start = int(RATE * gap * i)
        out[start:start + len(tone)] += tone
    return out


def _siren():
    """One second of a two-tone alarm; the UI loops it while the emergency alarm is on."""
    t = np.arange(RATE // 2) / RATE
    fade = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.01)          # no clicks where the tones meet
    tone = lambda f: np.sign(np.sin(2 * np.pi * f * t)) * 0.35 + np.sin(2 * np.pi * f * t) * 0.45
    return 0.9 * np.concatenate([tone(960) * fade, tone(720) * fade])


SOUNDS = {
    "click": lambda: _tone(880, 0.09, decay=45.0),                              # every selection
    "type": lambda: _mix([(660, 0.16), (990, 0.2)], 0.06),                      # a letter or word was typed
    "back": lambda: _tone(330, 0.12, decay=30.0),                               # back / delete
    "done": lambda: _mix([(523, 0.35), (659, 0.35), (784, 0.35), (1047, 0.6)], 0.09),   # postcard finished
    "bell": lambda: _mix([(1319, 0.9), (1047, 1.2)], 0.28),                    # call bell: "come here"
    "alarm": _siren,                                                            # emergency (looped)
}


def generate(force=False):
    ASSETS.mkdir(exist_ok=True)
    for name, make in SOUNDS.items():
        path = ASSETS / f"{name}.wav"
        if path.exists() and not force:
            continue
        pcm = (np.clip(make(), -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())


class SoundBank:
    def __init__(self, enabled=True):
        self.sounds = {}
        if not enabled:
            return
        try:
            import pygame
            generate()
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            for name in SOUNDS:
                self.sounds[name] = pygame.mixer.Sound(str(ASSETS / f"{name}.wav"))
        except Exception as e:
            print("[sound] no feedback sounds:", e)
            self.sounds = {}

    def play(self, name, loop=False):
        snd = self.sounds.get(name)
        if snd:
            try:
                snd.play(loops=-1 if loop else 0)
            except Exception:
                pass

    def stop(self, name):
        snd = self.sounds.get(name)
        if snd:
            try:
                snd.stop()
            except Exception:
                pass


if __name__ == "__main__":
    generate(force=True)
    print("Wrote", ", ".join(f"{n}.wav" for n in SOUNDS), "to", ASSETS)
