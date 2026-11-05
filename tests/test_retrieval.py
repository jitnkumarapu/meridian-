from pathlib import Path

import pytest

from mmrag.embedding import HashTextEncoder
from mmrag.indexing import ScoredItem
from mmrag.retrieval import HybridRetriever, reciprocal_rank_fusion, weighted_rank_fusion
from mmrag.schemas import Chunk, Modality


def test_rrf_matches_hand_calculation() -> None:
    rankings = {
        "dense": [ScoredItem("a", 9), ScoredItem("b", 8)],
        "sparse": [ScoredItem("b", 100), ScoredItem("a", 1)],
    }
    result = reciprocal_rank_fusion(rankings, rrf_k=60)
    assert result[0].item_id == "a"  # deterministic id tie-break
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert result[1].score == pytest.approx(1 / 62 + 1 / 61)


def test_weighted_fusion_normalizes_each_source() -> None:
    rankings = {
        "dense": [ScoredItem("a", 0.9), ScoredItem("b", 0.1)],
        "sparse": [ScoredItem("b", 200), ScoredItem("a", 100)],
    }
    result = weighted_rank_fusion(rankings, weights={"dense": 0.75, "sparse": 0.25})
    assert result[0].item_id == "a"
    assert result[0].score == pytest.approx(0.75)
    assert result[1].score == pytest.approx(0.25)


def _chunks() -> list[Chunk]:
    source = Path("fixture.pdf")
    return [
        Chunk(
            chunk_id="text-1",
            doc_id="acme",
            page=1,
            modality=Modality.TEXT,
            content="The company was founded in 1999.",
            source_path=source,
        ),
        Chunk(
            chunk_id="table-1",
            doc_id="acme",
            page=2,
            modality=Modality.TABLE,
            content="Revenue | FY2023 = ₹1,234.5 million; FY2022 = ₹900 million",
            source_path=source,
        ),
        Chunk(
            chunk_id="figure-1",
            doc_id="acme",
            page=3,
            modality=Modality.FIGURE,
            content="Bar chart OCR: revenue rises from 2022 to 2023",
            caption="Figure 1: Revenue growth",
            source_path=source,
        ),
    ]


def test_hybrid_retrieval_finds_table_and_filters() -> None:
    retriever = HybridRetriever(_chunks(), HashTextEncoder(128), candidate_k=3)
    result = retriever.retrieve("What was FY2023 revenue ₹1,234.5?", 2)
    assert result.evidence[0].chunk.chunk_id == "table-1"
    filtered = retriever.retrieve("revenue", 5, modalities={Modality.FIGURE})
    assert [item.chunk.modality for item in filtered.evidence] == [Modality.FIGURE]


def test_retriever_persists_all_core_stores(tmp_path: Path) -> None:
    retriever = HybridRetriever(_chunks(), HashTextEncoder(64))
    retriever.save(tmp_path)
    assert (tmp_path / "text_dense" / "vectors.npy").exists()
    assert (tmp_path / "bm25.json").exists()
    assert (tmp_path / "chunks.json").exists()
    assert (tmp_path / "index_metadata.json").exists()


def test_retriever_loads_persisted_stores_without_reencoding(tmp_path: Path) -> None:
    encoder = HashTextEncoder(64)
    HybridRetriever(_chunks(), encoder, candidate_k=3).save(tmp_path)
    loaded = HybridRetriever.load(tmp_path, encoder, candidate_k=3)
    assert loaded.retrieve("FY2023 revenue", 1).evidence[0].chunk.chunk_id == "table-1"
    assert loaded.text_store.item_ids == [chunk.chunk_id for chunk in _chunks()]


def test_query_normalization_explanations_and_fiscal_filter() -> None:
    retriever = HybridRetriever(_chunks(), HashTextEncoder(64), candidate_k=3)
    result = retriever.retrieve("fiscal year 2023 sales", 3, fiscal_years={"FY2023"})
    assert result.query == "fiscal year 2023 sales"
    assert result.normalized_query and "FY2023" in result.normalized_query
    assert "revenue" in result.expansions
    assert all("FY2023" in item.chunk.content for item in result.evidence)
    assert result.evidence[0].explanation is not None
    assert result.evidence[0].explanation.final_rank == 1


def test_visual_query_detection_does_not_add_image_noise_to_plain_questions() -> None:
    assert HybridRetriever._is_visual_query("What values are shown in the revenue chart?")
    assert HybridRetriever._is_visual_query("Summarize the highlighted infographic")
    assert not HybridRetriever._is_visual_query("What revenue did the company report?")


def test_embedding_cache_reuses_content_and_validates_index(tmp_path: Path) -> None:
    first = HybridRetriever(_chunks(), HashTextEncoder(64), cache_dir=tmp_path / "cache")
    assert first.cache_stats["text_misses"] == 3
    second = HybridRetriever(_chunks(), HashTextEncoder(64), cache_dir=tmp_path / "cache")
    assert second.cache_stats["text_hits"] == 3
    second.save(tmp_path / "index")
    config = {
        "candidate_k": 20,
        "fusion": "rrf",
        "rrf_k": 60,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
    }
    metadata = HybridRetriever.validate_index(
        tmp_path / "index", text_encoder_id="test/hash-text-v1/64", figure_encoder_id=None, expected_config=config
    )
    assert metadata["cache"]["text_hits"] == 3
    with pytest.raises(ValueError, match="text encoder"):
        HybridRetriever.validate_index(tmp_path / "index", text_encoder_id="wrong", figure_encoder_id=None)


def test_optional_reranker_reorders_only_textual_evidence() -> None:
    class ReverseReranker:
        model_id = "test/reverse"

        def score(self, query: str, documents: list[str]) -> list[float]:
            del query
            return [float(index) for index in range(len(documents))]

    retriever = HybridRetriever(_chunks(), HashTextEncoder(64), candidate_k=3, reranker=ReverseReranker())
    result = retriever.retrieve("revenue", 3)
    textual = [item for item in result.evidence if item.chunk.modality is not Modality.FIGURE]
    assert textual[0].explanation is not None
    assert textual[0].explanation.reranker_score is not None
