"""Checks for the public run-level analysis, using hand-computable examples."""
import unittest
from scripts.reproduce_paper import exact_test, score_bfi


class RunLevelInferenceTest(unittest.TestCase):
    def test_three_vs_three_has_exact_resolution(self):
        delta, p, count = exact_test([4, 5, 6], [1, 2, 3])
        self.assertEqual((delta, count), (3, 20))
        self.assertAlmostEqual(p, 0.1)

    def test_null_and_reversal(self):
        self.assertEqual(exact_test([1, 1], [1, 1])[1], 1)
        self.assertEqual(exact_test([1, 2, 3], [4, 5, 6])[:2], (-3, 0.1))

    def test_reverse_scoring_and_range(self):
        self.assertEqual(score_bfi([1, 5, 1, 1, 1, 5, 1, 5, 5, 5]), [5]*5)
        self.assertEqual(score_bfi([5, 1, 5, 5, 5, 1, 5, 1, 1, 1]), [1]*5)
        with self.assertRaises(ValueError):
            score_bfi([0]*10)
        with self.assertRaises(ValueError):
            score_bfi([3]*9)
