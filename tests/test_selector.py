import unittest

from processing.selector import Selector


class SelectorTests(unittest.TestCase):
    def test_requires_consecutive_evidence(self):
        s = Selector(margin_tau=0.1, n_consecutive=3, lockout_s=1)
        scores = [0.8, 0.2, 0.1, 0.0]
        self.assertIsNone(s.update(scores, 1.0))
        self.assertIsNone(s.update(scores, 1.25))
        self.assertEqual(s.update(scores, 1.5)[0], 0)
        self.assertIsNone(s.update(scores, 1.75))

    def test_ambiguous_evidence_abstains(self):
        s = Selector(margin_tau=0.1, n_consecutive=2, lockout_s=0)
        self.assertIsNone(s.update([0.6, 0.59, 0.2, 0.1], 1.0))
        self.assertIsNone(s.update([0.6, 0.59, 0.2, 0.1], 1.25))

    def test_change_of_target_resets_streak(self):
        s = Selector(margin_tau=0.1, n_consecutive=2, lockout_s=0)
        self.assertIsNone(s.update([0.8, 0.1, 0.0, 0.0], 1.0))
        self.assertIsNone(s.update([0.1, 0.8, 0.0, 0.0], 1.2))
        self.assertEqual(s.update([0.1, 0.8, 0.0, 0.0], 1.4)[0], 1)


if __name__ == "__main__":
    unittest.main()
