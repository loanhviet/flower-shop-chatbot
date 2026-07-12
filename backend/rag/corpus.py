"""Build retrieval-ready documents from the normalized product catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_INPUT_PATH = Path("data/processed/hoatuoimymy_products_tagged.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/processed/retrieval_corpus.jsonl")

DETAIL_THRESHOLD_CHARS = 1_200
DETAIL_CHUNK_CHARS = 900
DETAIL_CHUNK_OVERLAP_CHARS = 150


@dataclass(frozen=True)
class CorpusBuildStats:
    overview_documents: int
    detail_documents: int
    duplicate_detail_products_skipped: int
    short_or_empty_products_skipped: int

    @property
    def total_documents(self) -> int:
        return self.overview_documents + self.detail_documents

    def to_dict(self) -> dict[str, int]:
        return asdict(self) | {"total_documents": self.total_documents}


def load_products(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL product catalog without silently accepting invalid rows."""
    products = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            product = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(product, dict):
            raise ValueError(f"Expected an object on line {line_number} of {path}")
        products.append(product)
    return products


def build_retrieval_corpus(
    products: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], CorpusBuildStats]:
    """Create one overview plus useful, non-duplicated detail chunks per product."""
    product_list = list(products)
    _validate_products(product_list)

    description_counts = Counter(
        _description_fingerprint(product.get("clean_description"))
        for product in product_list
        if product.get("clean_description")
    )

    documents: list[dict[str, Any]] = []
    detail_documents = 0
    duplicate_detail_products_skipped = 0
    short_or_empty_products_skipped = 0

    for product in product_list:
        documents.append(_build_overview_document(product))

        description = _normalized_description(product.get("clean_description"))
        if len(description) <= DETAIL_THRESHOLD_CHARS:
            short_or_empty_products_skipped += 1
            continue

        if description_counts[_description_fingerprint(description)] > 1:
            duplicate_detail_products_skipped += 1
            continue

        for chunk_index, chunk in enumerate(split_detail_text(description)):
            documents.append(_build_detail_document(product, chunk, chunk_index))
            detail_documents += 1

    stats = CorpusBuildStats(
        overview_documents=len(product_list),
        detail_documents=detail_documents,
        duplicate_detail_products_skipped=duplicate_detail_products_skipped,
        short_or_empty_products_skipped=short_or_empty_products_skipped,
    )
    return documents, stats


def split_detail_text(
    text: str,
    *,
    chunk_chars: int = DETAIL_CHUNK_CHARS,
    overlap_chars: int = DETAIL_CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split long Vietnamese text near natural boundaries, preserving overlap."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must be at least zero and smaller than chunk_chars")

    text = _normalized_description(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        target_end = min(start + chunk_chars, len(text))
        end = (
            target_end
            if target_end == len(text)
            else _find_chunk_end(text, start, target_end, chunk_chars)
        )
        chunks.append(text[start:end].strip())

        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def write_jsonl(documents: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    """Write a generated corpus as UTF-8 JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(document, ensure_ascii=False) + "\n")


def _find_chunk_end(text: str, start: int, target_end: int, chunk_chars: int) -> int:
    window = text[start:target_end]
    boundary_ends = []
    for separator in ("\n\n", "\n", ". ", "! ", "? ", ": ", " "):
        position = window.rfind(separator)
        if position >= 0:
            boundary_ends.append(position + len(separator))
    boundary_end = max(boundary_ends, default=0)
    if boundary_end > chunk_chars // 2:
        return start + boundary_end
    return target_end


def _build_overview_document(product: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{product['product_id']}:overview",
        "document_type": "overview",
        "product_id": product["product_id"],
        "chunk_index": None,
        "text": product.get("rag_text") or _compact_product_context(product),
        "metadata": _metadata(product),
    }


def _build_detail_document(
    product: Mapping[str, Any], chunk: str, chunk_index: int) -> dict[str, Any]:
    return {
        "id": f"{product['product_id']}:detail:{chunk_index}",
        "document_type": "detail",
        "product_id": product["product_id"],
        "chunk_index": chunk_index,
        "text": f"{_compact_product_context(product)}\nChi tiết: {chunk}",
        "metadata": _metadata(product),
    }


def _metadata(product: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
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
    )
    return {field: product.get(field) for field in fields}


def _compact_product_context(product: Mapping[str, Any]) -> str:
    lines = [
        f"Sản phẩm: {product.get('product_type') or product.get('name')}",
        f"Mã/tên gốc: {product.get('name')}",
    ]
    if product.get("occasion_tags"):
        lines.append(f"Dịp phù hợp: {', '.join(product['occasion_tags'])}")
    if product.get("flower_tags"):
        lines.append(f"Loại hoa: {', '.join(product['flower_tags'])}")
    if product.get("color_tags"):
        lines.append(f"Màu sắc: {', '.join(product['color_tags'])}")
    lines.append(f"Giá: {product.get('display_price') or 'liên hệ'}")
    return "\n".join(lines)


def _validate_products(products: Iterable[Mapping[str, Any]]) -> None:
    for index, product in enumerate(products):
        if not product.get("product_id"):
            raise ValueError(f"Product at index {index} is missing product_id")


def _normalized_description(value: Any) -> str:
    return " ".join(str(value or "").split())


def _description_fingerprint(value: Any) -> str:
    return _normalized_description(value).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the flower product retrieval corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    documents, stats = build_retrieval_corpus(load_products(args.input))
    write_jsonl(documents, args.output)
    print(json.dumps(stats.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
