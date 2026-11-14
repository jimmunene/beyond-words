import unittest

from speller.state_machine import Speller


class MockPredictor:
    def suggest(self, typed):
        return 'HELLO' if typed.startswith('H') else None


class SpellerTests(unittest.TestCase):
    def test_letter_then_delete(self):
        s = Speller(predictor=None, max_chars=12)
        s.select(0)  # A-I
        s.select(0)  # ABC
        s.select(0)  # A
        self.assertEqual(s.text, 'A')
        self.assertEqual(s.screen, 'root')
        s.select(2)  # controls via S-Z
        s.select(3)  # CTRL
        s.select(0)  # delete
        self.assertEqual(s.text, '')

    def test_patient_phrase_is_not_full_intentlock(self):
        s = Speller(predictor=None, mode='patient')
        s.select(0)
        self.assertEqual(s.spoken, 'I would like some water, please.')
        # Existing starter sets spoken phrase immediately: confirmation gate is future work.

    def test_suggestion_requires_decoded_selection(self):
        s = Speller(predictor=MockPredictor())
        self.assertEqual(s.suggestion, None)
        s.select(0)  # A-I
        s.select(2)  # GHI
        s.select(1)  # H
        self.assertEqual(s.text, 'H')
        self.assertEqual(s.suggestion, 'HELLO')
        s.select(3)
        self.assertEqual(s.text, 'HELLO ')


if __name__ == "__main__":
    unittest.main()
