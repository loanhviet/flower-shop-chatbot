# flower-shop-chatbot

Flower shop chatbot project. The current focus is preparing a clean product
catalog from flower shop pages so the chatbot can later search, recommend, and
answer with grounded product information.

## Structure

- `crawler/`
- `backend/`
- `data/`
- `docs/`
- `notebooks/`

## Crawler

The first data source is `hoatuoimymy.com`. The crawler exports product JSONL
with normalized prices, search tags, and compact `rag_text` for chatbot/RAG
ingestion.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Crawl Data

Run a small validation crawl first:

```bash
cd crawler
scrapy crawl hoatuoimymy \
  -s CLOSESPIDER_ITEMCOUNT=50 \
  -s FLOWER_QUALITY_REPORT_PATH=../data/processed/hoatuoimymy_quality_report_tags_sample.json \
  -L INFO \
  -O ../data/processed/sample_hoatuoimymy_tags_50.jsonl
```

Then crawl the full tagged catalog:

```bash
cd crawler
scrapy crawl hoatuoimymy \
  -s FLOWER_QUALITY_REPORT_PATH=../data/processed/hoatuoimymy_quality_report_full_tags.json \
  -L INFO \
  -O ../data/processed/hoatuoimymy_products_tagged.jsonl
```

Review the quality report before using the data. The key checks are missing
prices, missing images, missing search tags, duplicate URLs, and dropped items.

### Chatbot Data

Each JSONL product keeps the original product fields and adds chatbot-friendly
fields:

- `rag_text`: compact text to embed/search.
- `product_type`: inferred type such as `bó hoa`, `giỏ hoa`, `hoa khai trương`.
- `occasion_tags`, `flower_tags`, `color_tags`: simple search filters.
- `current_price`, `display_price`: normalized active price; missing-price products use `Giá: liên hệ` in `rag_text`.
- `image_url`, `product_url`: product references for chatbot answers.

For RAG, prefer `rag_text` plus the tag fields over raw `description`.

### Build Retrieval Corpus

Build the local corpus before adding embeddings or Qdrant:

```bash
python -m backend.rag.corpus
```

This generates `data/processed/retrieval_corpus.jsonl` locally. Each product
has one overview document from `rag_text`. Products with a unique
`clean_description` longer than 1,200 characters also receive detail chunks
of up to 900 characters with 150 characters of overlap. Repeated descriptions
do not create detail chunks, preventing duplicate retrieval results.
