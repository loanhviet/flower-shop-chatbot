"""Qdrant collection schema and document persistence."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from qdrant_client import QdrantClient, models


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION_NAME = "flower_products_gemini_embedding_2_768"
DENSE_VECTOR_NAME = "dense"
CORPUS_SCHEMA_VERSION = 1
POINT_NAMESPACE = uuid.UUID("ea341862-bf75-56a7-b57c-9fd7a1bac063")

KEYWORD_PAYLOAD_FIELDS = (
    "document_type",
    "product_id",
    "availability",
    "source",
    "product_type",
    "categories",
    "occasion_tags",
    "flower_tags",
    "color_tags",
)


@dataclass(frozen=True)
class QdrantSettings:
    url: str = DEFAULT_QDRANT_URL
    collection: str = DEFAULT_COLLECTION_NAME
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "QdrantSettings":
        settings = cls(
            url=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).strip(),
            collection=os.getenv(
                "QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME
            ).strip(),
            api_key=os.getenv("QDRANT_API_KEY") or None,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.url:
            raise ValueError("QDRANT_URL cannot be empty")
        if not self.collection:
            raise ValueError("QDRANT_COLLECTION cannot be empty")


class QdrantDocumentStore:
    def __init__(
        self,
        client: QdrantClient,
        settings: QdrantSettings,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
        corpus_schema_version: int = CORPUS_SCHEMA_VERSION,
    ) -> None:
        settings.validate()
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        self.client = client
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.corpus_schema_version = corpus_schema_version

    @classmethod
    def from_settings(
        cls,
        settings: QdrantSettings,
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
    ) -> "QdrantDocumentStore":
        client = QdrantClient(
            url=settings.url,
            api_key=settings.api_key,
        )
        return cls(
            client,
            settings,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )

    @property
    def collection_name(self) -> str:
        return self.settings.collection

    @property
    def collection_metadata(self) -> dict[str, Any]:
        return {
            "managed_by": "flower-shop-chatbot",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "corpus_schema_version": self.corpus_schema_version,
        }

    def ensure_collection(self, *, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.embedding_dimension,
                        distance=models.Distance.COSINE,
                    )
                },
                metadata=self.collection_metadata,
            )
            self._create_payload_indexes()
            return

        self._validate_collection()

    def validate_existing_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} does not exist"
            )
        self._validate_collection()

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        vectors: Sequence[Sequence[float]],
    ) -> int:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")

        points = []
        for document, vector in zip(documents, vectors):
            _validate_vector(vector, self.embedding_dimension)
            points.append(
                models.PointStruct(
                    id=stable_point_id(str(document.get("id") or "")),
                    vector={DENSE_VECTOR_NAME: list(vector)},
                    payload=document_payload(document),
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        return len(points)

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> list[Any]:
        _validate_vector(vector, self.embedding_dimension)
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            using=DENSE_VECTOR_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return list(result.points)

    def count(self) -> int:
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def _validate_collection(self) -> None:
        info = self.client.get_collection(self.collection_name)
        vector_config = info.config.params.vectors
        dense_config = (
            vector_config.get(DENSE_VECTOR_NAME)
            if isinstance(vector_config, dict)
            else None
        )
        if dense_config is None:
            raise ValueError(
                f"Collection {self.collection_name!r} is missing named vector "
                f"{DENSE_VECTOR_NAME!r}"
            )
        if int(dense_config.size) != self.embedding_dimension:
            raise ValueError(
                f"Collection dimension is {dense_config.size}; "
                f"expected {self.embedding_dimension}"
            )
        if dense_config.distance != models.Distance.COSINE:
            raise ValueError("Collection distance must be cosine")

        actual_metadata = getattr(info.config, "metadata", None) or {}
        expected = self.collection_metadata
        mismatches = {
            key: (actual_metadata.get(key), value)
            for key, value in expected.items()
            if actual_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Collection metadata is incompatible: {mismatches}"
            )

    def _create_payload_indexes(self) -> None:
        for field_name in KEYWORD_PAYLOAD_FIELDS:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="current_price",
            field_schema=models.PayloadSchemaType.INTEGER,
            wait=True,
        )


def stable_point_id(document_id: str) -> str:
    if not document_id.strip():
        raise ValueError("document id cannot be empty")
    return str(uuid.uuid5(POINT_NAMESPACE, document_id))


def document_payload(document: Mapping[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("document metadata must be an object")

    payload = {
        "document_id": document.get("id"),
        "document_type": document.get("document_type"),
        "product_id": document.get("product_id"),
        "chunk_index": document.get("chunk_index"),
        "text": document.get("text"),
    }
    for field_name in (
        "name",
        "product_url",
        "image_url",
        "current_price",
        "display_price",
        "availability",
        "categories",
        "product_type",
        "occasion_tags",
        "flower_tags",
        "color_tags",
        "source",
    ):
        payload[field_name] = metadata.get(field_name)
    return payload


def _validate_vector(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise ValueError(
            f"Vector has dimension {len(vector)}; expected {dimensions}"
        )
    if not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("Vector contains a non-finite value")
