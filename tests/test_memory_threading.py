"""Regression tests for concurrent C6 memory embedding."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cogclaw import memory


class MemoryThreadingTest(unittest.TestCase):
    def test_encoder_initializes_once_and_encode_is_serialized(self):
        state_lock = threading.Lock()
        state = {"initializations": 0, "active": 0, "max_active": 0}

        class FakeEncoder:
            def encode(self, texts, **_kwargs):
                with state_lock:
                    state["active"] += 1
                    state["max_active"] = max(
                        state["max_active"], state["active"]
                    )
                time.sleep(0.01)
                with state_lock:
                    state["active"] -= 1
                return np.ones((len(texts), 2), dtype=np.float32)

        def make_encoder(_name):
            with state_lock:
                state["initializations"] += 1
            time.sleep(0.01)
            return FakeEncoder()

        fake_module = SimpleNamespace(SentenceTransformer=make_encoder)
        original_encoder = memory._ENCODER
        memory._ENCODER = None
        try:
            with patch.dict(
                sys.modules, {"sentence_transformers": fake_module}
            ):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    vectors = list(
                        pool.map(memory._embed, [["text"]] * 8)
                    )
        finally:
            memory._ENCODER = original_encoder

        self.assertEqual(state["initializations"], 1)
        self.assertEqual(state["max_active"], 1)
        self.assertTrue(all(vector.shape == (1, 2) for vector in vectors))


if __name__ == "__main__":
    unittest.main()
