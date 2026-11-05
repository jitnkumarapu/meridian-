"""OpenCLIP adapter for the independent figure embedding space."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .st_encoder import FloatArray, _normalise


class OpenCLIPEncoder:
    """Encode images and text queries with matching OpenCLIP towers."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        *,
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        try:
            open_clip = importlib.import_module("open_clip")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("OpenCLIP is not installed; install the 'models' project extra") from exc
        self._torch: Any = torch
        self._device = device
        self._model_id = f"open_clip/{model_name}/{pretrained}"
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        model.eval()
        self._model: Any = model
        self._preprocess: Any = preprocess
        self._tokenizer: Any = open_clip.get_tokenizer(model_name)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        projection = self._model.text_projection
        return int(projection.shape[-1])

    def encode_texts(self, texts: list[str]) -> FloatArray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokens = self._tokenizer(texts).to(self._device)
        with self._torch.inference_mode():
            values = self._model.encode_text(tokens).float().cpu().numpy()
        return _normalise(values)

    def encode_images(self, paths: list[Path]) -> FloatArray:
        if not paths:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise RuntimeError("Pillow is required for figure embeddings") from exc
        images = []
        for path in paths:
            with image_module.open(path) as image:
                images.append(self._preprocess(image.convert("RGB")))
        batch = self._torch.stack(images).to(self._device)
        with self._torch.inference_mode():
            values: NDArray[np.float32] = self._model.encode_image(batch).float().cpu().numpy()
        return _normalise(values)
