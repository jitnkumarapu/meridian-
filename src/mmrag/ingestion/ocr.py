"""OCR adapters; the native Tesseract dependency is checked at runtime."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class OCRBackend(Protocol):
    def extract(self, image_bytes: bytes, language: str) -> tuple[str, float | None]: ...


@dataclass(frozen=True)
class StaticOCRBackend:
    """Deterministic injectable backend used by offline integration tests."""

    text: str
    confidence: float = 100.0

    def extract(self, image_bytes: bytes, language: str) -> tuple[str, float | None]:
        del image_bytes, language
        return self.text, self.confidence


class TesseractOCRBackend:
    def __init__(self) -> None:
        detected = shutil.which("tesseract")
        if detected is None:
            windows_default = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
            detected = str(windows_default) if windows_default.is_file() else None
        if detected is None:
            raise RuntimeError("Tesseract executable not found; install it or provide another OCRBackend")
        self.executable = detected

    def extract(self, image_bytes: bytes, language: str) -> tuple[str, float | None]:
        import io

        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = self.executable
        image = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
        words: list[str] = []
        confidence: list[float] = []
        for word, score in zip(data["text"], data["conf"], strict=True):
            if word.strip():
                words.append(word.strip())
                numeric = float(score)
                if numeric >= 0:
                    confidence.append(numeric)
        return " ".join(words), sum(confidence) / len(confidence) if confidence else None
