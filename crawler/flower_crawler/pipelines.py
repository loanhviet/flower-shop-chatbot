import logging
import re
from datetime import datetime, timezone

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


logger = logging.getLogger(__name__)


class FlowerProductPipeline:
    def __init__(self):
        self.seen_urls = set()

    def process_item(self, item):
        adapter = ItemAdapter(item)

        self._normalize_text_fields(adapter)
        self._normalize_categories(adapter)
        self._validate_required_fields(adapter)
        self._deduplicate(adapter)
        self._normalize_price(adapter)

        adapter["currency"] = adapter.get("currency") or "VND"
        adapter["crawled_at"] = datetime.now(timezone.utc).isoformat()

        if not adapter.get("price_text"):
            logger.warning("Product has no price: %s", adapter.get("product_url"))
        if not adapter.get("image_url"):
            logger.warning("Product has no image: %s", adapter.get("product_url"))

        return item

    @staticmethod
    def _normalize_text_fields(adapter):
        for field in (
            "name",
            "price_text",
            "description",
            "short_description",
            "availability",
            "image_url",
            "product_url",
            "source",
        ):
            value = adapter.get(field)
            if isinstance(value, str):
                adapter[field] = " ".join(value.split()) or None

    @staticmethod
    def _normalize_categories(adapter):
        categories = adapter.get("categories") or []
        normalized = []
        seen = set()

        for category in categories:
            clean_category = " ".join(str(category).split())
            key = clean_category.casefold()
            if clean_category and key not in seen:
                normalized.append(clean_category)
                seen.add(key)

        adapter["categories"] = normalized

    @staticmethod
    def _validate_required_fields(adapter):
        if not adapter.get("name"):
            raise DropItem("Missing product name")
        if not adapter.get("product_url"):
            raise DropItem("Missing product URL")

    def _deduplicate(self, adapter):
        product_url = adapter.get("product_url")
        if product_url in self.seen_urls:
            raise DropItem(f"Duplicate product URL: {product_url}")
        self.seen_urls.add(product_url)

    @staticmethod
    def _normalize_price(adapter):
        prices = _parse_vnd_prices(adapter.get("price_text"))
        adapter["price_min"] = min(prices) if prices else None
        adapter["price_max"] = max(prices) if prices else None


def _parse_vnd_prices(price_text):
    if not price_text:
        return []

    prices = []
    for match in re.findall(r"\d[\d.,]*", price_text):
        digits = re.sub(r"\D", "", match)
        if not digits:
            continue

        value = int(digits)
        if value > 0:
            prices.append(value)

    return prices
