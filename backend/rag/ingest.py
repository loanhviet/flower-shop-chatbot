"""Embed the retrieval corpus and idempotently upsert it into Qdrant."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.embeddings import (
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingSettings,
    build_embedding_provider,
)
from backend.rag.corpus import DEFAULT_OUTPUT_PATH
from backend.rag.qdrant_store import (
    QdrantDocumentStore,
    QdrantSettings,
)


@dataclass(frozen=True)
class IngestionStats:
    documents_seen: int
    documents_embedded: int
    points_upserted: int
    batches: int
    collection: str
    embedding_model: str
    embedding_dimension: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_corpus_documents(
    path: Path | str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    documents = []
    with Path(path).open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc
            _validate_document(document, line_number, path)
            documents.append(document)
            if limit is not None and len(documents) >= limit:
                break
    return documents


def ingest_documents(
    documents: Sequence[Mapping[str, Any]],
    provider: EmbeddingProvider,
    store: QdrantDocumentStore,
    *,
    batch_size: int = 32,
    recreate: bool = False,
) -> IngestionStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    store.ensure_collection(recreate=recreate)
    embedded = 0
    upserted = 0
    batches = 0

    for batch in _batched(documents, batch_size):
        embedding_inputs = [
            EmbeddingInput(
                text=str(document["text"]),
                title=str((document.get("metadata") or {}).get("name") or "none"),
            )
            for document in batch
        ]
        vectors = provider.embed_documents(embedding_inputs)
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned an unexpected vector count")
        upserted += store.upsert_documents(batch, vectors)
        embedded += len(vectors)
        batches += 1

    return IngestionStats(
        documents_seen=len(documents),
        documents_embedded=embedded,
        points_upserted=upserted,
        batches=batches,
        collection=store.collection_name,
        embedding_model=provider.model_id,
        embedding_dimension=provider.dimensions,
    )


def _batched(
    values: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _validate_document(
    document: Any,
    line_number: int,
    path: Path | str,
) -> None:
    if not isinstance(document, dict):
        raise ValueError(f"Expected an object on line {line_number} of {path}")
    for field_name in ("id", "document_type", "product_id", "text", "metadata"):
        if document.get(field_name) is None:
            raise ValueError(
                f"Document on line {line_number} is missing {field_name}"
            )
    if not str(document["text"]).strip():
        raise ValueError(f"Document on line {line_number} has empty text")
    if not isinstance(document["metadata"], dict):
        raise ValueError(
            f"Document metadata on line {line_number} must be an object"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed retrieval documents and upsert them into Qdrant."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--collection")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    embedding_settings = EmbeddingSettings.from_env()
    if args.batch_size is not None:
        embedding_settings = EmbeddingSettings(
            provider=embedding_settings.provider,
            model=embedding_settings.model,
            dimension=embedding_settings.dimension,
            batch_size=args.batch_size,
            requests_per_minute=embedding_settings.requests_per_minute,
            api_key=embedding_settings.api_key,
        )
        embedding_settings.validate()

    qdrant_settings = QdrantSettings.from_env()
    if args.collection:
        qdrant_settings = QdrantSettings(
            url=qdrant_settings.url,
            collection=args.collection,
            api_key=qdrant_settings.api_key,
        )
        qdrant_settings.validate()

    provider = build_embedding_provider(embedding_settings)
    store = QdrantDocumentStore.from_settings(
        qdrant_settings,
        embedding_provider=embedding_settings.provider,
        embedding_model=provider.model_id,
        embedding_dimension=provider.dimensions,
    )
    documents = load_corpus_documents(args.input, limit=args.limit)
    stats = ingest_documents(
        documents,
        provider,
        store,
        batch_size=embedding_settings.batch_size,
        recreate=args.recreate,
    )
    print(json.dumps(stats.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
