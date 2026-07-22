"""Embedding provider contracts and implementations."""

from __future__ import annotations

from collections import deque
import hashlib
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Deque, Protocol, Sequence


DEFAULT_GEMINI_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSION = 768
DEFAULT_EMBEDDING_BATCH_SIZE = 10
DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE = 90


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    title: str | None = None


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = "gemini"
    model: str = DEFAULT_GEMINI_MODEL
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    requests_per_minute: int = DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        settings = cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "gemini").strip().casefold(),
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_GEMINI_MODEL).strip(),
            dimension=_positive_int_env(
                "EMBEDDING_DIMENSION", DEFAULT_EMBEDDING_DIMENSION
            ),
            batch_size=_positive_int_env(
                "EMBEDDING_BATCH_SIZE", DEFAULT_EMBEDDING_BATCH_SIZE
            ),
            requests_per_minute=_positive_int_env(
                "EMBEDDING_REQUESTS_PER_MINUTE",
                DEFAULT_EMBEDDING_REQUESTS_PER_MINUTE,
            ),
            api_key=os.getenv("GEMINI_API_KEY") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.provider != "gemini":
            raise ValueError(
                f"Unsupported embedding provider: {self.provider!r}; expected 'gemini'"
            )
        if not self.model:
            raise ValueError("Embedding model cannot be empty")
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        if self.requests_per_minute <= 0:
            raise ValueError("Embedding requests per minute must be positive")


class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        ...

    @property
    def dimensions(self) -> int:
        ...

    def embed_documents(
        self, inputs: Sequence[EmbeddingInput]
    ) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class GeminiEmbeddingProvider:
    """Gemini Embedding 2 adapter with asymmetric retrieval formatting."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        max_attempts: int = 3,
    ) -> None:
        settings.validate()
        if not settings.api_key and client is None:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        if client is None:
            from google import genai

            client = genai.Client(api_key=settings.api_key)

        self._settings = settings
        self._client = client
        self._sleep = sleep
        self._monotonic = monotonic
        self._max_attempts = max_attempts
        self._request_timestamps: Deque[float] = deque()

    @property
    def model_id(self) -> str:
        return self._settings.model

    @property
    def dimensions(self) -> int:
        return self._settings.dimension

    def embed_documents(
        self, inputs: Sequence[EmbeddingInput]
    ) -> list[list[float]]:
        normalized = [
            EmbeddingInput(
                text=_required_text(item.text, "document text"),
                title=(item.title or "none").strip() or "none",
            )
            for item in inputs
        ]
        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self._settings.batch_size):
            batch = normalized[start : start + self._settings.batch_size]
            texts = [
                f"title: {item.title} | text: {item.text}"
                for item in batch
            ]
            vectors.extend(self._embed_batch(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        query = _required_text(text, "query")
        return self._embed_batch(
            [f"task: search result | query: {query}"]
        )[0]

    def _embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        self._wait_for_capacity(len(texts))
        response = self._request_with_retry(texts)
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings is None:
            raise ValueError("Gemini response is missing embeddings")
        if len(embeddings) != len(texts):
            raise ValueError(
                "Gemini returned a different number of embeddings than inputs"
            )

        vectors = [_embedding_values(item) for item in embeddings]
        for vector in vectors:
            _validate_vector(vector, self.dimensions)
        return vectors

    def _request_with_retry(self, texts: Sequence[str]) -> Any:
        contents = [{"parts": [{"text": text}]} for text in texts]
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.models.embed_content(
                    model=self.model_id,
                    contents=contents,
                    config={"output_dimensionality": self.dimensions},
                )
            except Exception as exc:
                if attempt >= self._max_attempts or not _is_retryable(exc):
                    raise
                self._sleep(_retry_delay_seconds(exc, attempt))

        raise RuntimeError("Embedding request exhausted retries")

    def _wait_for_capacity(self, units: int) -> None:
        if units > self._settings.requests_per_minute:
            raise ValueError(
                "Embedding batch is larger than EMBEDDING_REQUESTS_PER_MINUTE"
            )

        while True:
            now = self._monotonic()
            cutoff = now - 60.0
            while self._request_timestamps and self._request_timestamps[0] <= cutoff:
                self._request_timestamps.popleft()

            if len(self._request_timestamps) + units <= self._settings.requests_per_minute:
                self._request_timestamps.extend([now] * units)
                return

            wait_seconds = max(
                self._request_timestamps[0] + 60.0 - now,
                0.1,
            )
            self._sleep(wait_seconds)


class FakeEmbeddingProvider:
    """Deterministic offline provider for unit tests."""

    def __init__(self, dimensions: int = 8, model_id: str = "fake") -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(
        self, inputs: Sequence[EmbeddingInput]
    ) -> list[list[float]]:
        return [_deterministic_vector(item.text, self.dimensions) for item in inputs]

    def embed_query(self, text: str) -> list[float]:
        return _deterministic_vector(_required_text(text, "query"), self.dimensions)


def build_embedding_provider(
    settings: EmbeddingSettings | None = None,
    *,
    client: Any | None = None,
) -> EmbeddingProvider:
    resolved = settings or EmbeddingSettings.from_env()
    if resolved.provider == "gemini":
        return GeminiEmbeddingProvider(resolved, client=client)
    raise ValueError(f"Unsupported embedding provider: {resolved.provider!r}")


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _required_text(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    return text


def _embedding_values(embedding: Any) -> list[float]:
    values = getattr(embedding, "values", None)
    if values is None and isinstance(embedding, dict):
        values = embedding.get("values")
    if values is None:
        raise ValueError("Gemini embedding is missing values")
    return [float(value) for value in values]


def _validate_vector(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise ValueError(
            f"Embedding has dimension {len(vector)}; expected {dimensions}"
        )
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains a non-finite value")


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "code", None)
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return False
    return status_code == 429 or status_code >= 500


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    match = re.search(r"retry in ([0-9.]+)s", str(exc), flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)), float(2 ** (attempt - 1)))
    return float(2 ** (attempt - 1))


def _deterministic_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(_required_text(text, "text").encode("utf-8")).digest()
    values = [
        (digest[index % len(digest)] / 127.5) - 1.0
        for index in range(dimensions)
    ]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]
