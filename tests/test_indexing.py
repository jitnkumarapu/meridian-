from pathlib import Path

import numpy as np
import pytest

from mmrag.embedding import HashTextEncoder
from mmrag.indexing import BM25Store, DenseStore, financial_tokenize


def test_financial_tokenizer_preserves_important_tokens() -> None:
    tokens = financial_tokenize("FY2023 revenue was ₹1,234.5 and margin 12.4%; loss (45).")
    assert {"fy2023", "₹1,234.5", "12.4%", "(45)"} <= set(tokens)


def test_dense_store_search_and_roundtrip(tmp_path: Path) -> None:
    store = DenseStore(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), model_id="m")
    assert store.search(np.array([0.9, 0.1]), 1)[0].item_id == "a"
    store.save(tmp_path)
    loaded = DenseStore.load(tmp_path, expected_model_id="m")
    assert loaded.search(np.array([0.0, 1.0]), 1)[0].item_id == "b"
    with pytest.raises(ValueError, match="expected"):
        DenseStore.load(tmp_path, expected_model_id="other")


def test_bm25_store_roundtrip(tmp_path: Path) -> None:
    store = BM25Store(["revenue", "staff"], ["FY2023 revenue ₹1,234", "employee count 50"])
    assert store.search("FY2023 revenue", 1)[0].item_id == "revenue"
    path = tmp_path / "bm25.json"
    store.save(path)
    assert BM25Store.load(path).search("employee", 1)[0].item_id == "staff"


def test_hash_encoder_is_deterministic_and_normalized() -> None:
    encoder = HashTextEncoder(64)
    first = encoder.encode(["revenue growth", "table"])
    second = encoder.encode(["revenue growth", "table"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)
