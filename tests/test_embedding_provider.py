import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.embeddings import (
    EmbeddingInput,
    EmbeddingSettings,
    FakeEmbeddingProvider,
    GeminiEmbeddingProvider,
)


class _FakeModels:
    def __init__(self, dimensions, failures=None):
        self.dimensions = dimensions
        self.failures = list(failures or [])
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.failures:
            raise self.failures.pop(0)
        embeddings = [
            SimpleNamespace(values=[float(index + 1)] * self.dimensions)
            for index, _ in enumerate(kwargs["contents"])
        ]
        return SimpleNamespace(embeddings=embeddings)


class _RetryableError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class EmbeddingProviderTest(unittest.TestCase):
    def _settings(self, **overrides):
        values = {
            "provider": "gemini",
            "model": "gemini-embedding-2",
            "dimension": 4,
            "batch_size": 2,
            "api_key": "test-key",
        }
        values.update(overrides)
        return EmbeddingSettings(**values)

    def test_settings_load_defaults_and_require_positive_numbers(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "secret"}, clear=True):
            settings = EmbeddingSettings.from_env()

        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.model, "gemini-embedding-2")
        self.assertEqual(settings.dimension, 768)
        self.assertEqual(settings.batch_size, 10)
        self.assertEqual(settings.requests_per_minute, 90)
        self.assertEqual(settings.api_key, "secret")

        with patch.dict(
            os.environ,
            {"EMBEDDING_DIMENSION": "0", "GEMINI_API_KEY": "secret"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                EmbeddingSettings.from_env()

    def test_gemini_formats_documents_and_preserves_batches(self):
        models = _FakeModels(dimensions=4)
        provider = GeminiEmbeddingProvider(
            self._settings(),
            client=SimpleNamespace(models=models),
        )

        vectors = provider.embed_documents(
            [
                EmbeddingInput("Nội dung một", "Hoa hồng"),
                EmbeddingInput("Nội dung hai"),
                EmbeddingInput("Nội dung ba", "Hoa lan"),
            ]
        )

        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(models.calls), 2)
        first_contents = models.calls[0]["contents"]
        self.assertEqual(
            first_contents[0]["parts"][0]["text"],
            "title: Hoa hồng | text: Nội dung một",
        )
        self.assertEqual(
            first_contents[1]["parts"][0]["text"],
            "title: none | text: Nội dung hai",
        )
        self.assertEqual(
            models.calls[0]["config"],
            {"output_dimensionality": 4},
        )

    def test_gemini_formats_query_and_retries_transient_error(self):
        models = _FakeModels(
            dimensions=4,
            failures=[_RetryableError(429)],
        )
        sleeps = []
        provider = GeminiEmbeddingProvider(
            self._settings(),
            client=SimpleNamespace(models=models),
            sleep=sleeps.append,
        )

        vector = provider.embed_query("hoa sinh nhật")

        self.assertEqual(len(vector), 4)
        self.assertEqual(sleeps, [1.0])
        self.assertEqual(len(models.calls), 2)
        self.assertEqual(
            models.calls[-1]["contents"][0]["parts"][0]["text"],
            "task: search result | query: hoa sinh nhật",
        )

    def test_gemini_rejects_wrong_dimension(self):
        provider = GeminiEmbeddingProvider(
            self._settings(),
            client=SimpleNamespace(models=_FakeModels(dimensions=3)),
        )

        with self.assertRaisesRegex(ValueError, "expected 4"):
            provider.embed_query("hoa hồng")

    def test_missing_api_key_is_rejected_without_injected_client(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            GeminiEmbeddingProvider(self._settings(api_key=None))

    def test_fake_provider_is_deterministic_and_normalized(self):
        provider = FakeEmbeddingProvider(dimensions=8)

        first = provider.embed_query("hoa hồng")
        second = provider.embed_query("hoa hồng")

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            math.sqrt(sum(value * value for value in first)),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
