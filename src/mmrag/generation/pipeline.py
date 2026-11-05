"""Answer pipeline with mandatory citation gate and one strict retry."""

from __future__ import annotations

import json
import logging
import time

from mmrag.generation.citations import validate_citations
from mmrag.generation.llm import LLMClient, LLMResponse
from mmrag.generation.prompts import build_prompt, prompt_hash
from mmrag.schemas import Answer, Evidence

LOGGER = logging.getLogger(__name__)


class AnswerPipeline:
    def __init__(self, llm: LLMClient, *, use_vision: bool = False, provider: str | None = None) -> None:
        self.llm = llm
        self.use_vision = use_vision
        self.provider = provider

    def answer(self, query: str, evidence: list[Evidence], *, style: str = "compact") -> Answer:
        start = time.perf_counter()
        if not evidence:
            return self._record(
                Answer(
                    text="Insufficient evidence",
                    citations=[],
                    evidence=[],
                    grounded=False,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    model_id=self.llm.model_id,
                    prompt_hash=prompt_hash(build_prompt(query, [], style=style)),
                    provider=self.provider,
                    style=style,
                )
            )

        response: LLMResponse | None = None
        final_prompt = ""
        for strict in (False, True):
            final_prompt = build_prompt(query, evidence, strict=strict, style=style)
            image_paths = (
                [
                    str(item.chunk.image_path)
                    for item in evidence
                    if item.chunk.modality.value == "figure" and item.chunk.image_path
                ]
                if self.use_vision
                else None
            )
            response = self.llm.complete(final_prompt, image_paths=image_paths)
            validation = validate_citations(response.text, evidence)
            if validation.valid:
                return self._record(
                    Answer(
                        text=response.text,
                        citations=validation.citations,
                        evidence=evidence,
                        grounded=True,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        token_usage=response.token_usage,
                        model_id=response.model_id,
                        prompt_hash=prompt_hash(final_prompt),
                        provider=self.provider,
                        retries=int(strict) + response.retries,
                        style=style,
                        metadata={"grounding": "valid", "evidence_count": len(evidence)},
                    )
                )

        assert response is not None
        return self._record(
            Answer(
                text="Insufficient evidence",
                citations=[],
                evidence=evidence,
                grounded=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                token_usage=response.token_usage,
                model_id=response.model_id,
                prompt_hash=prompt_hash(final_prompt),
                provider=self.provider,
                retries=1 + response.retries,
                style=style,
                metadata={"grounding": "failed", "evidence_count": len(evidence)},
            )
        )

    @staticmethod
    def _record(answer: Answer) -> Answer:
        LOGGER.info(
            json.dumps(
                {
                    "event": "answer_completed",
                    "provider": answer.provider,
                    "model": answer.model_id,
                    "latency_ms": round(answer.latency_ms, 3),
                    "token_usage": answer.token_usage,
                    "retries": answer.retries,
                    "grounded": answer.grounded,
                    "prompt_hash": answer.prompt_hash,
                },
                sort_keys=True,
            )
        )
        return answer
