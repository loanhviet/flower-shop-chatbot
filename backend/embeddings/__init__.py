"""Provider-neutral text embedding interfaces."""

from .providers import (
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingSettings,
    FakeEmbeddingProvider,
    GeminiEmbeddingProvider,
    build_embedding_provider,
)

__all__ = [
    "EmbeddingInput",
    "EmbeddingProvider",
    "EmbeddingSettings",
    "FakeEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "build_embedding_provider",
]
