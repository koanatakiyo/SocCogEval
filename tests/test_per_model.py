"""The per-model (C2)-(C4) tests recompute from the bundled raw BFI ratings."""
import tempfile
import unittest
from pathlib import Path
from scripts.reproduce_per_model import reproduce


class PerModelTest(unittest.TestCase):
    def test_reproduces_reference_table(self):
        with tempfile.TemporaryDirectory() as d:
            rows = reproduce(Path(d))
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(r['n_c2'] == 8 and r['n_c4'] == 5 for r in rows))
