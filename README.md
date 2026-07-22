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

## Gemini Embeddings and Qdrant

M2 uses `gemini-embedding-2` with 768-dimensional vectors. Unit tests use a
fake provider and do not require network access.

Copy the local configuration and add your Gemini key:

```bash
cp .env.example .env
# Edit GEMINI_API_KEY in .env, then export it for the current shell.
set -a
source .env
set +a
```

Never commit `.env` or print the API key.

### Start Qdrant

Run the pinned development server on localhost:

```bash
docker run --name flower-shop-qdrant -d \
  -p 127.0.0.1:6333:6333 \
  -v flower-shop-qdrant-data:/qdrant/storage \
  qdrant/qdrant:v1.18.2
```

The REST API and dashboard are available at
`http://127.0.0.1:6333` and `http://127.0.0.1:6333/dashboard`.

### Ingest Documents

Start with a small collection before spending quota on the full corpus:

```bash
python -m backend.rag.ingest \
  --limit 10 \
  --collection flower_products_gemini_embedding_2_768_smoke \
  --recreate
```

Then build the full collection:

```bash
python -m backend.rag.ingest --recreate
```

The Gemini free tier counts each input document toward its per-minute quota.
The default `EMBEDDING_REQUESTS_PER_MINUTE=90` keeps ingestion below the
100-input free-tier limit; a full 1,783-document run can therefore take about
20 minutes. Keep the process running so it can pause between quota windows.

Normal runs use stable point IDs and idempotent upserts. `--recreate` deletes
only the configured collection before rebuilding it, so verify
`QDRANT_URL` and `QDRANT_COLLECTION` before using the flag.

### Semantic Search Smoke Test

```bash
python -m backend.rag.search "bó hoa hồng đỏ tặng sinh nhật khoảng 600 nghìn"
python -m backend.rag.search "hoa khai trương sang trọng có hoa lan"
python -m backend.rag.search "lan hồ điệp trắng giá liên hệ"
```

Results are JSON objects containing similarity score, product metadata, price,
image, URL, and source. Metadata filtering, BM25, and hybrid fusion are added
in M3.

## Provider Boundary

Gemini is used only for embeddings. Answer generation is intentionally deferred
until M5, where `ChatProvider` will use Alibaba Cloud Model Studio and prioritize
a Qwen model selected against the available quality, latency, and free quota at
that time.
