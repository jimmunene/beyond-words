"""Deterministic Part C tests; no key, hardware, Wi-Fi or pygame required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from art.postcard import make_postcard, make_demo_postcard, validate_eeg
from experience.contracts import ConfirmedMessage
from experience.pipeline import create_experience
from voice.tts import speak, speak_confirmed


class PartCTests(unittest.TestCase):
    def setUp(self):
        self.eeg = np.random.default_rng(23).normal(size=(500, 8))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def message(self, **kwargs):
        fields = dict(
            text="HI MOM", session_id="session-test-01", input_mode="eeg",
            signal_source="replay", approved_by_user=True,
        )
        fields.update(kwargs)
        return ConfirmedMessage(**fields)

    def test_no_output_before_confirmation(self):
        with self.assertRaises(PermissionError):
            create_experience(self.message(approved_by_user=False), self.eeg, 250,
                              output_dir=self.folder)
        self.assertFalse(list(self.folder.glob("*.png")))

    def test_no_postcard_fabricated_from_missing_eeg(self):
        with self.assertRaises(ValueError):
            make_postcard(np.zeros((0, 8)), 250, "HELLO", out_dir=self.folder)
        with self.assertRaises(ValueError):
            make_postcard(np.full((500, 8), np.nan), 250, "HELLO", out_dir=self.folder)

    def test_valid_eeg_input_and_png(self):
        self.assertEqual(validate_eeg(self.eeg, 250).shape, (500, 8))
        path = make_postcard(self.eeg, 250, "HI MOM", "Ava", self.folder,
                             signal_source="replay", input_mode="keyboard")
        with Image.open(path) as img:
            self.assertEqual(img.format, "PNG")
            self.assertEqual(img.size, (1200, 800))

    def test_explicitly_labeled_synthetic_demo(self):
        path = make_demo_postcard(out_dir=self.folder)
        self.assertTrue(Path(path).exists())

    def test_no_gallery_without_separate_consent(self):
        queue = self.folder / "queue"
        result = create_experience(self.message(), self.eeg, 250,
                                   output_dir=self.folder, queue_dir=queue)
        self.assertIn("local only", result.gallery_status)
        self.assertFalse(queue.exists())
        meta = json.loads(Path(result.metadata_path).read_text())
        self.assertFalse(meta["gallery_consent"])
        self.assertEqual(meta["input_mode"], "eeg")
        self.assertNotIn("raw_eeg", meta)

    def test_gallery_queue_requires_distinct_consent(self):
        queue = self.folder / "queue"
        result = create_experience(self.message(gallery_consent=True), self.eeg, 250,
                                   output_dir=self.folder, queue_dir=queue)
        self.assertIn("NOT published", result.gallery_status)
        self.assertEqual(len(list(queue.glob("*.png"))), 1)
        self.assertEqual(len(list(queue.glob("*.json"))), 1)
        queued = json.loads(next(queue.glob("*.json")).read_text())
        self.assertEqual(queued["signal_source"], "replay")
        self.assertTrue(queued["gallery_consent"])

    def test_synthetic_label_in_queue(self):
        result = create_experience(
            self.message(signal_source="synthetic", input_mode="synthetic", gallery_consent=True),
            self.eeg, 250, output_dir=self.folder, queue_dir=self.folder / "queue")
        meta = json.loads(Path(result.metadata_path).read_text())
        self.assertIn("Synthetic", meta["public_claim"])

    def test_legacy_voice_fails_closed(self):
        with patch("voice.tts.threading.Thread") as worker:
            self.assertFalse(speak("UNCONFIRMED", "voice-id", enabled=True))
            worker.assert_not_called()

    def test_voice_cloud_opt_in_is_separate(self):
        with patch("voice.tts.speak") as mocked:
            mocked.return_value = True
            result = speak_confirmed(self.message(cloud_voice_consent=False), "voice-id")
            self.assertTrue(result)
            self.assertFalse(mocked.call_args.kwargs["allow_cloud"])
            speak_confirmed(self.message(cloud_voice_consent=True), "voice-id")
            self.assertTrue(mocked.call_args.kwargs["allow_cloud"])

    def test_invalid_message_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.message(input_mode="unknown").validate_for_output()
        with self.assertRaises(ValueError):
            self.message(text=" ").validate_for_output()

    def test_voice_is_only_triggered_for_approved_flow(self):
        with patch("experience.pipeline.speak_confirmed") as mock_voice:
            mock_voice.return_value = True
            output = create_experience(self.message(), self.eeg, 250,
                                       output_dir=self.folder, enable_speech=True)
            self.assertTrue(output.speech_started)
            mock_voice.assert_called_once()
            with self.assertRaises(PermissionError):
                create_experience(self.message(approved_by_user=False), self.eeg, 250,
                                  output_dir=self.folder, enable_speech=True)
            mock_voice.assert_called_once()


if __name__ == "__main__":
    unittest.main()
