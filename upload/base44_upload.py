"""Send a finished postcard to the Base44 gallery.

Default: copies the postcard into postcards/to_upload/ and writes a sidecar .json, so the
Base44 teammate can add it through the gallery's admin upload page.

If the Base44 booth confirms an API / webhook, fill in upload_via_api() and set
USE_API = True. Keep the manual path as the fallback.
"""
import json
import pathlib
import shutil

USE_API = False
QUEUE = pathlib.Path(__file__).resolve().parents[1] / "postcards" / "to_upload"


def upload_via_api(png_path, meta):
    raise NotImplementedError("Fill in with the endpoint the Base44 team gives you.")


def upload(png_path, message, nickname, typing_seconds, consent):
    if not consent:
        return "kept locally (no consent to post)"
    meta = {"message": message, "nickname": nickname, "typing_seconds": round(typing_seconds)}
    if USE_API:
        try:
            upload_via_api(png_path, meta)
            return "uploaded via API"
        except Exception as e:
            print("[upload] API failed, queuing for manual upload:", e)
    QUEUE.mkdir(parents=True, exist_ok=True)
    dst = QUEUE / pathlib.Path(png_path).name
    shutil.copy(png_path, dst)
    json.dump(meta, open(dst.with_suffix(".json"), "w"), indent=2)
    return f"queued for manual upload: {dst.name}"
