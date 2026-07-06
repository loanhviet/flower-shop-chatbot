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

```bash
cd crawler
scrapy crawl hoatuoimymy -O ../data/processed/hoatuoimymy_products.jsonl
```

For a small validation crawl:

```bash
cd crawler
scrapy crawl hoatuoimymy -s CLOSESPIDER_ITEMCOUNT=10 -O ../data/processed/sample_hoatuoimymy.jsonl
```

### Output schema

Each JSONL line is one product object:

- `name`: product name.
- `price_text`: original price text from the product page.
- `price_min`, `price_max`: parsed VND price bounds when available.
- `currency`: `VND`.
- `availability`: stock/availability text when available.
- `short_description`, `description`: normalized product descriptions.
- `categories`: list of product categories.
- `image_url`: absolute product image URL when available.
- `product_url`: canonical crawled product page URL.
- `source`: source domain.
- `crawled_at`: UTC ISO timestamp.
