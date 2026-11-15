"""Reach a caregiver who is not in the room: a phone call, a text message, or a loud push alert.

Used by the speller when the person picks COME HERE (message: "I need you") or EMERGENCY.
Nothing is sent unless a channel below is set up; the speller then only rings its local bell.

Channels (use one or several; each is switched on by its environment variables):

  ntfy (free, no account)   NTFY_TOPIC = a long secret word, e.g. my-long-secret-topic
        The caregiver installs the "ntfy" app, subscribes to that topic, and gets an urgent push
        alert that keeps ringing. Anyone who knows the topic can read it, so keep it secret.
  Twilio (real call + SMS)  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (your Twilio number),
        CAREGIVER_PHONE (e.g. +17345550123). The phone rings and a voice reads the message (a missed
        call if nobody answers), and a text message is sent as well. Trial accounts can only reach
        phone numbers verified in the Twilio console.
  Telegram (free)           TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID: a message from your own bot.

Set them like the other keys, then open a new terminal:   setx NTFY_TOPIC "my-long-secret-topic"
Never put these values in a file that is committed.

    python speller/notify.py                      # show which channels are set up
    python speller/notify.py "Test: I need you"   # send a test alert now
"""
import base64
import json
import os
import pathlib
import sys
import threading
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import load_saved_keys  # noqa: E402

NAMES = ("NTFY_TOPIC", "NTFY_SERVER", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "CAREGIVER_PHONE",
         "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def _post(url, data, headers=None, timeout=15):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


class PhoneAlert:
    def __init__(self):
        load_saved_keys(NAMES)
        env = os.environ.get
        self.ntfy = env("NTFY_TOPIC", "").strip()
        self.ntfy_server = env("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
        self.twilio = all(env(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM", "CAREGIVER_PHONE"))
        self.telegram = bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))
        self.last = ("", True, 0.0)             # (what happened, all ok, when) for the screen
        self.busy = False

    @property
    def enabled(self):
        return bool(self.ntfy or self.twilio or self.telegram)

    def describe(self):
        parts = (["push alert (ntfy)"] if self.ntfy else []) + \
                ([f"call + text to ...{os.environ['CAREGIVER_PHONE'][-4:]}"] if self.twilio else []) + \
                (["Telegram message"] if self.telegram else [])
        return ", ".join(parts) if parts else "not set up"

    def send(self, text, urgent=False, wait=False):
        """Send on every channel that is set up. Runs in the background unless wait=True.
        Returns immediately; the outcome lands in self.last."""
        if not self.enabled:
            return False
        worker = threading.Thread(target=self._send, args=(text, urgent), daemon=True)
        worker.start()
        if wait:
            worker.join(40)
        return True

    def _send(self, text, urgent):
        import time
        self.busy = True
        results = []
        for name, on, fn in (("push alert", self.ntfy, self._ntfy), ("phone call", self.twilio, self._call),
                             ("text message", self.twilio, self._sms), ("Telegram", self.telegram, self._telegram)):
            if not on:
                continue
            try:
                fn(text, urgent)
                results.append((name, True, ""))
            except Exception as e:              # no internet, wrong key, unverified number...
                results.append((name, False, str(e)[:90]))
        ok = [n for n, good, _ in results if good]
        bad = [f"{n} ({why})" for n, good, why in results if not good]
        summary = ("Sent: " + ", ".join(ok) if ok else "") + ("  FAILED: " + "; ".join(bad) if bad else "")
        self.last = (summary.strip(), not bad, time.time())
        self.busy = False
        print("[notify]", self.last[0])

    # ------------------------------------------------------------ channels
    def _ntfy(self, text, urgent):
        _post(f"{self.ntfy_server}/{urllib.parse.quote(self.ntfy)}", text.encode(), {
            "Title": "EMERGENCY" if urgent else "I need you",
            "Priority": "urgent" if urgent else "high",
            "Tags": "rotating_light" if urgent else "bell"})

    def _twilio(self, resource, fields):
        sid, token = os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"]
        auth = base64.b64encode(f"{sid}:{token}".encode()).decode()
        fields = {"To": os.environ["CAREGIVER_PHONE"], "From": os.environ["TWILIO_FROM"], **fields}
        _post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/{resource}.json",
              urllib.parse.urlencode(fields).encode(), {"Authorization": "Basic " + auth})

    def _call(self, text, urgent):
        safe = text.replace("&", "and").replace("<", "").replace(">", "")
        twiml = f"<Response><Pause length=\"1\"/><Say>{safe}</Say><Pause length=\"1\"/><Say>{safe}</Say></Response>"
        self._twilio("Calls", {"Twiml": twiml, "Timeout": "25"})

    def _sms(self, text, urgent):
        self._twilio("Messages", {"Body": text})

    def _telegram(self, text, urgent):
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        _post(f"https://api.telegram.org/bot{token}/sendMessage",
              json.dumps({"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": ("🚨 " if urgent else "🔔 ") + text}).encode(),
              {"Content-Type": "application/json"})


if __name__ == "__main__":
    alert = PhoneAlert()
    print("Phone alerts:", alert.describe())
    if len(sys.argv) > 1:
        if not alert.enabled:
            sys.exit("Nothing is set up, so nothing was sent. See the top of speller/notify.py.")
        alert.send(" ".join(sys.argv[1:]), wait=True)
