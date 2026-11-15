"""Link to the caregiver web app (built on Base44): what the person says appears on the family's
phones, and their replies come back to the speller.

  speller -> app   every spoken need / spelled sentence / emergency becomes a Message
                   a postcard the visitor agreed to share becomes a Postcard (image uploaded)
                   the decoder's scores and state go to one LiveStatus record about once a second,
                   for the "Live EEG" dashboard page (no raw EEG; switch off with live_dashboard: false)
  app -> speller   a caregiver's reply is shown and spoken on the patient's screen
                   tapping "I'm coming" on an emergency (acknowledged = true) stops the alarm

Set up (the token is a secret: environment variable only, never in a file that is committed):
    setx BASE44_TOKEN "b44u_..."                 # a Base44 personal access token
    config.yaml:  care_app_url: "https://your-app.base44.app"
                  care_app_id:  "the app id"

Everything runs in background threads and nothing here can stop the speller: with no token, no
internet or an error, messages are simply not sent and the reason is kept in last_error.

    python speller/care_sync.py                  # show whether the app can be reached
    python speller/care_sync.py "Test message"   # send one test message to the feed
"""
import json
import os
import pathlib
import queue
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from speller.predictor import load_saved_keys  # noqa: E402


class CareSync:
    def __init__(self, cfg, poll_s=3.0):
        load_saved_keys(("BASE44_TOKEN",))
        self.url = str(cfg.get("care_app_url", "")).rstrip("/")
        self.app_id = str(cfg.get("care_app_id", ""))
        self.token = os.environ.get("BASE44_TOKEN", "")
        self.enabled = bool(self.url and self.app_id and self.token)
        self.base = f"{self.url}/api/apps/{self.app_id}"
        self.sent, self.last_error = 0, ""
        self.replies = []                       # caregiver replies not yet shown: [text]
        self.emergency_id, self.emergency_seen = None, False
        self._seen, self._since = set(), time.time()
        self._lock = threading.Lock()
        self._jobs = queue.Queue()
        # live dashboard: one LiveStatus record that is overwritten about once a second
        self.live_on = self.enabled and bool(cfg.get("live_dashboard", True))
        self.station = str(cfg.get("live_station", "booth"))
        self.live_error, self.live_sent = "", 0
        self._live, self._live_id = None, None
        if self.enabled:
            threading.Thread(target=self._worker, daemon=True).start()
            threading.Thread(target=self._poller, args=(poll_s,), daemon=True).start()
        if self.live_on:
            threading.Thread(target=self._live_loop, daemon=True).start()

    # ------------------------------------------------------------ what the speller calls
    def say(self, text, kind="need", name=""):
        """Put something the person said on the feed. kind: need / spelled / emergency."""
        if self.enabled and text:
            self._jobs.put(("message", {"text": text, "kind": kind, "from": "patient", "patient_name": name}))

    def postcard(self, png_path, message, nickname, typing_seconds):
        if self.enabled and png_path:
            self._jobs.put(("postcard", (str(png_path), message, nickname, typing_seconds)))

    def live(self, status):
        """Latest snapshot for the live dashboard (decoder scores and state, never raw EEG).
        Only the newest one is kept; a background thread sends it about once a second."""
        if self.live_on:
            self._live = status

    def take_replies(self):
        with self._lock:
            out, self.replies = self.replies, []
        return out

    def emergency_answered(self):
        """True once, when a caregiver tapped "I'm coming" on the current emergency."""
        if self.emergency_seen:
            self.emergency_seen, self.emergency_id = False, None
            return True
        return False

    # ------------------------------------------------------------ HTTP
    def _call(self, method, path, body=None, raw=None, content_type=None):
        headers = {"Authorization": "Bearer " + self.token}
        data = raw
        if body is not None:
            headers["Content-Type"], data = "application/json", json.dumps(body).encode()
        if content_type:
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=20) as r:
            text = r.read().decode(errors="replace")
        return json.loads(text) if text.strip().startswith(("{", "[")) else text

    def _upload(self, path):
        boundary = uuid.uuid4().hex
        name = pathlib.Path(path).name
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                "Content-Type: image/png\r\n\r\n").encode() + pathlib.Path(path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        out = self._call("POST", "/integration-endpoints/Core/UploadFile", raw=body,
                         content_type=f"multipart/form-data; boundary={boundary}")
        return out["file_url"]

    def _worker(self):
        while True:
            kind, payload = self._jobs.get()
            try:
                if kind == "message":
                    made = self._call("POST", "/entities/Message", payload)
                    if payload["kind"] == "emergency":
                        self.emergency_id, self.emergency_seen = made.get("id"), False
                else:
                    path, message, nickname, secs = payload
                    self._call("POST", "/entities/Postcard", {"image": self._upload(path), "message": message,
                                                               "nickname": nickname or "anonymous",
                                                               "typing_seconds": round(float(secs))})
                self.sent += 1
                self.last_error = ""
            except Exception as e:
                self.last_error = str(e)[:120]
                print("[care app] could not send:", self.last_error)

    def _live_loop(self, every=1.0):
        sent_t, last = 0.0, None
        while True:
            time.sleep(every)
            status = self._live
            if status is None or (status is last and time.time() - sent_t < 3):
                continue
            try:
                body = {**status, "station": self.station, "updated_at": int(time.time() * 1000)}
                if self._live_id is None:                    # reuse this station's record, or make it once
                    found = self._call("GET", "/entities/LiveStatus?station=" + urllib.parse.quote(self.station) + "&limit=1")
                    self._live_id = found[0]["id"] if found else self._call("POST", "/entities/LiveStatus", body)["id"]
                self._call("PUT", f"/entities/LiveStatus/{self._live_id}", body)
                sent_t, last = time.time(), status
                self.live_sent += 1
                self.live_error = ""
            except Exception as e:                           # page not built yet, no internet, rate limit...
                if not self.live_error:
                    print("[care app] live dashboard not updated:", str(e)[:120])
                self.live_error, self._live_id = str(e)[:120], None
                time.sleep(15)

    def _poller(self, every):
        while True:
            time.sleep(every)
            try:
                for m in self._call("GET", "/entities/Message?from=caregiver&sort=-created_date&limit=5"):
                    if m.get("id") in self._seen:
                        continue
                    self._seen.add(m.get("id"))
                    if _when(m.get("created_date")) >= self._since - 2 and m.get("text"):     # not old history
                        with self._lock:
                            self.replies.append(str(m["text"])[:120])
                if self.emergency_id:
                    if self._call("GET", f"/entities/Message/{self.emergency_id}").get("acknowledged"):
                        self.emergency_seen = True
                self.last_error = ""
            except Exception as e:
                self.last_error = str(e)[:120]


def _when(stamp):
    """'2026-11-15T08:29:05.919945Z' -> seconds since 1970 (0 if unreadable)."""
    try:
        import datetime
        t = datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if t.tzinfo is None:                                 # the app's lists give UTC times without a zone
            t = t.replace(tzinfo=datetime.timezone.utc)
        return t.timestamp()
    except Exception:
        return 0.0


if __name__ == "__main__":
    from common import load_config
    care = CareSync(load_config())
    if not care.enabled:
        sys.exit("Caregiver app is not set up: needs care_app_url and care_app_id in config.yaml and BASE44_TOKEN.")
    try:
        app = care._call("GET", "")
        print("Connected to:", app.get("name"), "at", care.url)
    except Exception as e:
        sys.exit(f"Could not reach the app: {e}")
    if len(sys.argv) > 1:
        care.say(" ".join(sys.argv[1:]), "need", "test")
        time.sleep(4)
        print("sent" if care.sent else f"NOT sent: {care.last_error}")
