import json
import tempfile
import unittest
from pathlib import Path

from qdrant_client import QdrantClient, models

from backend.embeddings import FakeEmbeddingProvider
from backend.rag.ingest import ingest_documents, load_corpus_documents
from backend.rag.qdrant_store import (
    QdrantDocumentStore,
    QdrantSettings,
    document_payload,
    stable_point_id,
)
from backend.rag.search import semantic_search


def _document(**overrides):
    document = {
        "id": "flower-01:overview",
        "document_type": "overview",
        "product_id": "flower-01",
        "chunk_index": None,
        "text": "Sản phẩm: bó hoa hồng đỏ tặng sinh nhật",
        "metadata": {
            "name": "Bó Hoa Hồng",
            "product_url": "https://example.com/hoa-hong",
            "image_url": "https://example.com/hoa-hong.jpg",
            "current_price": 600000,
            "display_price": "600.000 VND",
            "availability": "instock",
            "categories": ["Bó Hoa Sinh Nhật"],
            "product_type": "bó hoa",
            "occasion_tags": ["sinh nhật"],
            "flower_tags": ["hoa hồng"],
            "color_tags": ["đỏ"],
            "source": "example.com",
        },
    }
    document.update(overrides)
    return document


def _store(client=None, **overrides):
    values = {
        "embedding_provider": "fake",
        "embedding_model": "fake-model",
        "embedding_dimension": 8,
    }
    values.update(overrides)
    return QdrantDocumentStore(
        client or QdrantClient(":memory:"),
        QdrantSettings(url=":memory:", collection="test_flowers"),
        **values,
    )


class QdrantIngestionTest(unittest.TestCase):
    def test_point_id_and_payload_are_stable(self):
        self.assertEqual(
            stable_point_id("flower-01:overview"),
            stable_point_id("flower-01:overview"),
        )
        payload = document_payload(_document())

        self.assertEqual(payload["document_id"], "flower-01:overview")
        self.assertEqual(payload["current_price"], 600000)
        self.assertEqual(payload["flower_tags"], ["hoa hồng"])
        self.assertEqual(payload["product_url"], "https://example.com/hoa-hong")

    def test_ingestion_is_idempotent_and_search_returns_source(self):
        client = QdrantClient(":memory:")
        store = _store(client)
        provider = FakeEmbeddingProvider(dimensions=8, model_id="fake-model")
        documents = [_document()]

        first = ingest_documents(documents, provider, store)
        second = ingest_documents(documents, provider, store)
        results = semantic_search(
            "Sản phẩm: bó hoa hồng đỏ tặng sinh nhật",
            limit=1,
            provider=provider,
            store=store,
        )

        self.assertEqual(first.points_upserted, 1)
        self.assertEqual(second.points_upserted, 1)
        self.assertEqual(store.count(), 1)
        self.assertEqual(results[0]["payload"]["source"], "example.com")
        self.assertEqual(
            results[0]["payload"]["document_id"],
            "flower-01:overview",
        )

    def test_incompatible_collection_metadata_is_rejected(self):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="test_flowers",
            vectors_config={
                "dense": models.VectorParams(
                    size=8,
                    distance=models.Distance.COSINE,
                )
            },
            metadata={
                "managed_by": "someone-else",
                "embedding_provider": "fake",
                "embedding_model": "fake-model",
                "embedding_dimension": 8,
                "corpus_schema_version": 1,
            },
        )

        with self.assertRaisesRegex(ValueError, "metadata is incompatible"):
            _store(client).ensure_collection()

    def test_recreate_replaces_incompatible_collection(self):
        client = QdrantClient(":memory:")
        client.create_collection(
            collection_name="test_flowers",
            vectors_config=models.VectorParams(
                size=3,
                distance=models.Distance.DOT,
            ),
        )
        store = _store(client)

        store.ensure_collection(recreate=True)

        info = client.get_collection("test_flowers")
        self.assertEqual(info.config.params.vectors["dense"].size, 8)
        self.assertEqual(
            info.config.metadata["managed_by"],
            "flower-shop-chatbot",
        )

    def test_load_corpus_documents_validates_jsonl_and_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corpus.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(document, ensure_ascii=False)
                    for document in [_document(), _document(id="flower-02:overview")]
                ),
                encoding="utf-8",
            )

            documents = load_corpus_documents(path, limit=1)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["id"], "flower-01:overview")


if __name__ == "__main__":
    unittest.main()
