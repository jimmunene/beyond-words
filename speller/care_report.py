"""Care report: what the person said and asked for, from logs/selections.csv.
Something to show a nurse, a doctor or the family at the end of the day.

    python speller/care_report.py              # today
    python speller/care_report.py all          # everything in the log
    python speller/care_report.py 2026-11-15   # one day

Only patient-mode phrases are listed. The log stays on this computer; nothing is uploaded.
"""
import collections
import csv
import datetime
import pathlib
import sys

LOG = pathlib.Path(__file__).resolve().parents[1] / "logs" / "selections.csv"


def load(day):
    rows = []
    if not LOG.exists():
        return rows
    with open(LOG, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.datetime.fromtimestamp(float(r["time"]))
            except (ValueError, KeyError, TypeError):
                continue
            if r.get("mode") == "patient" and (day is None or t.date() == day):
                rows.append((t, r))
    return rows


def report(rows):
    spoken = [(t, r["text"]) for t, r in rows if r["event"] == "spoken" and r["text"]
              and not r["text"].startswith("I am typing a message")]          # (a notice, not a request)
    alarms = [t for t, r in rows if r["event"] == "emergency"]
    acks = [(t, r["session_seconds"]) for t, r in rows if r["event"] == "emergency_ack"]
    out = [f"Care report: {len(spoken)} phrases spoken, {len(alarms)} emergency call(s)", ""]
    if alarms:
        out.append("EMERGENCY CALLS")
        for t in alarms:
            wait = next((s for at, s in acks if at >= t), None)
            out.append(f"  {t:%a %H:%M:%S}   " + (f"answered after {wait} s" if wait else "NOT acknowledged in the log"))
        out.append("")
    pain = [(t, text) for t, text in spoken if "pain" in text.lower() or "hurt" in text.lower()]
    if pain:
        out.append("PAIN REPORTS")
        out += [f"  {t:%a %H:%M}   {text}" for t, text in pain]
        out.append("")
    counts = collections.Counter(text for _, text in spoken)
    if counts:
        out.append("MOST OFTEN SAID")
        out += [f"  {n:3d} x  {text}" for text, n in counts.most_common(12)]
        out.append("")
        out.append("FULL TIMELINE")
        out += [f"  {t:%a %H:%M}   {text}" for t, text in spoken]
    else:
        out.append("Nothing was spoken in patient mode in this period.")
    return "\n".join(out)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "today"
    day = None if arg == "all" else datetime.date.today() if arg == "today" else datetime.date.fromisoformat(arg)
    print(report(load(day)))
