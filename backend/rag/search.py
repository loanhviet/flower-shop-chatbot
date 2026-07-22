"""Minimal dense semantic-search smoke command for M2."""

from __future__ import annotations

import argparse
import json

from backend.embeddings import EmbeddingSettings, build_embedding_provider
from backend.rag.qdrant_store import QdrantDocumentStore, QdrantSettings


def semantic_search(
    query: str,
    *,
    limit: int,
    provider,
    store: QdrantDocumentStore,
) -> list[dict]:
    store.validate_existing_collection()
    vector = provider.embed_query(query)
    points = store.query(vector, limit=limit)
    return [
        {
            "id": str(point.id),
            "score": point.score,
            "payload": point.payload,
        }
        for point in points
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a dense semantic-search smoke query."
    )
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--collection")
    args = parser.parse_args()

    embedding_settings = EmbeddingSettings.from_env()
    qdrant_settings = QdrantSettings.from_env()
    if args.collection:
        qdrant_settings = QdrantSettings(
            url=qdrant_settings.url,
            collection=args.collection,
            api_key=qdrant_settings.api_key,
        )

    provider = build_embedding_provider(embedding_settings)
    store = QdrantDocumentStore.from_settings(
        qdrant_settings,
        embedding_provider=embedding_settings.provider,
        embedding_model=provider.model_id,
        embedding_dimension=provider.dimensions,
    )
    results = semantic_search(
        args.query,
        limit=args.limit,
        provider=provider,
        store=store,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
