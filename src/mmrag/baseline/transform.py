"""Convert multimodal chunks into the baseline corpus without changing provenance."""

from __future__ import annotations

from mmrag.schemas import Chunk, Modality


def _flatten_grid(grid: list[list[str]]) -> str:
    return " ".join(cell.strip() for row in grid for cell in row if cell.strip())


def to_text_only(chunks: list[Chunk]) -> list[Chunk]:
    """Drop figures and turn tables into unstructured text chunks.

    Chunk IDs and source locations remain stable, ensuring paired evaluation uses the same corpus.
    """
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.modality is Modality.FIGURE:
            continue
        if chunk.modality is Modality.TABLE:
            content = _flatten_grid(chunk.raw_grid) if chunk.raw_grid else chunk.content
            result.append(
                chunk.model_copy(
                    update={
                        "modality": Modality.TEXT,
                        "content": content,
                        "raw_grid": None,
                        "image_path": None,
                        "caption": "",
                        "metadata": {**chunk.metadata, "baseline_source_modality": "table"},
                    }
                )
            )
        else:
            result.append(chunk)
    return result
