# flower-shop-chatbot

Project scaffold for a flower shop chatbot.

## Structure

- `crawler/`
- `backend/`
- `data/`
- `docs/`
- `notebooks/`

## Getting started

Add the implementation for each component inside the matching folder.

## Crawler

The first crawler target is `hoatuoimymy.com`. It uses Scrapy sitemap crawling and writes a clean product catalog as JSONL for later chatbot/RAG ingestion.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

Run validation crawls before crawling the full catalog:

```bash
cd crawler
scrapy crawl hoatuoimymy -s CLOSESPIDER_ITEMCOUNT=50 -O ../data/processed/sample_hoatuoimymy_50.jsonl
scrapy crawl hoatuoimymy -s CLOSESPIDER_ITEMCOUNT=200 -O ../data/processed/sample_hoatuoimymy_200.jsonl
```

Then crawl the full catalog:

```bash
cd crawler
scrapy crawl hoatuoimymy -O ../data/processed/hoatuoimymy_products.jsonl
```

For a tiny smoke-test crawl:

```bash
cd crawler
scrapy crawl hoatuoimymy -s CLOSESPIDER_ITEMCOUNT=10 -O ../data/processed/sample_hoatuoimymy.jsonl
```

Each crawl writes a quality report to:

```text
data/processed/hoatuoimymy_quality_report.json
```

Override the report destination when needed:

```bash
cd crawler
scrapy crawl hoatuoimymy \
  -s FLOWER_QUALITY_REPORT_PATH=../data/processed/custom_quality_report.json \
  -s CLOSESPIDER_ITEMCOUNT=50 \
  -O ../data/processed/sample_hoatuoimymy_50.jsonl
```

Before running a full crawl, review the report and make sure missing price,
missing image, missing clean description, and duplicate URL counts are acceptable
for chatbot ingestion.

### Output schema

Each JSONL line is one product object:

- `product_id`: stable short hash from source and product URL.
- `name`: product name.
- `price_text`: original price text from the product page.
- `price_min`, `price_max`: parsed VND price bounds when available.
- `current_price`: price the chatbot should treat as the active price.
- `original_price`: listed/original price; equals `current_price` when there is no sale price.
- `sale_price`: discounted price when the page exposes multiple prices.
- `display_price`: formatted active price, for example `650.000 VND`.
- `currency`: `VND`.
- `availability`: stock/availability text when available.
- `short_description`, `description`: normalized product descriptions.
- `clean_description`: description cleaned for chatbot/RAG usage.
- `rag_text`: compact product text assembled for search/RAG ingestion.
- `categories`: list of product categories.
- `image_url`: absolute product image URL when available.
- `product_url`: canonical crawled product page URL.
- `source`: source domain.
- `crawled_at`: UTC ISO timestamp.
