import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from itemadapter import ItemAdapter
from scrapy.exceptions import DropItem


logger = logging.getLogger(__name__)


class FlowerProductPipeline:
    def __init__(self, report_path=None, spider_name=None):
        self.report_path = report_path
        self.spider_name = spider_name
        self.crawler = None
        self.seen_urls = set()
        self.stats = {
            "items_seen": 0,
            "items_processed": 0,
            "dropped_total": 0,
            "dropped_reasons": Counter(),
            "duplicate_urls": 0,
            "missing_price": 0,
            "missing_image": 0,
            "missing_clean_description": 0,
            "category_distribution": Counter(),
            "valid_prices": [],
        }

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls(report_path=crawler.settings.get("FLOWER_QUALITY_REPORT_PATH"))
        pipeline.crawler = crawler
        return pipeline

    def process_item(self, item, spider=None):
        self.stats["items_seen"] += 1
        adapter = ItemAdapter(item)

        try:
            self._normalize_text_fields(adapter)
            self._normalize_categories(adapter)
            self._validate_required_fields(adapter)
            self._deduplicate(adapter)
            self._normalize_price(adapter)
            self._set_product_id(adapter)
            self._set_clean_description(adapter)
            self._set_rag_text(adapter)
        except DropItem as exc:
            self._record_drop(exc)
            raise

        adapter["currency"] = adapter.get("currency") or "VND"
        adapter["crawled_at"] = datetime.now(timezone.utc).isoformat()

        if not adapter.get("price_text"):
            logger.warning("Product has no price: %s", adapter.get("product_url"))
        if not adapter.get("image_url"):
            logger.warning("Product has no image: %s", adapter.get("product_url"))

        self._record_success(adapter)
        return item

    def close_spider(self):
        self._write_quality_report(self._active_spider())

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
            self.stats["duplicate_urls"] += 1
            raise DropItem(f"Duplicate product URL: {product_url}")
        self.seen_urls.add(product_url)

    @staticmethod
    def _normalize_price(adapter):
        prices = _parse_vnd_prices(adapter.get("price_text"))
        adapter["price_min"] = min(prices) if prices else None
        adapter["price_max"] = max(prices) if prices else None

        if not prices:
            adapter["current_price"] = None
            adapter["original_price"] = None
            adapter["sale_price"] = None
            adapter["display_price"] = None
            return

        if len(prices) == 1:
            current_price = prices[0]
            original_price = prices[0]
            sale_price = None
        else:
            current_price = min(prices)
            original_price = max(prices)
            sale_price = current_price

        adapter["current_price"] = current_price
        adapter["original_price"] = original_price
        adapter["sale_price"] = sale_price
        adapter["display_price"] = _format_vnd(current_price)

    @staticmethod
    def _set_product_id(adapter):
        source = adapter.get("source") or ""
        product_url = adapter.get("product_url") or ""
        stable_key = f"{source}|{product_url}"
        adapter["product_id"] = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _set_clean_description(adapter):
        description_parts = [
            adapter.get("short_description"),
            adapter.get("description"),
        ]
        description = " ".join(
            part
            for part in description_parts
            if isinstance(part, str) and part.strip()
        )
        adapter["clean_description"] = _clean_product_description(description)

    @staticmethod
    def _set_rag_text(adapter):
        lines = [
            f"Tên sản phẩm: {adapter.get('name')}",
        ]

        categories = adapter.get("categories") or []
        if categories:
            lines.append(f"Danh mục: {', '.join(categories)}")

        if adapter.get("display_price"):
            lines.append(f"Giá hiện tại: {adapter.get('display_price')}")

        if adapter.get("availability"):
            lines.append(f"Tình trạng: {adapter.get('availability')}")

        if adapter.get("clean_description"):
            lines.append(f"Mô tả: {adapter.get('clean_description')}")

        lines.append(f"URL: {adapter.get('product_url')}")
        adapter["rag_text"] = "\n".join(lines)

    def _record_drop(self, exc):
        self.stats["dropped_total"] += 1
        self.stats["dropped_reasons"][_drop_reason(exc)] += 1

    def _record_success(self, adapter):
        self.stats["items_processed"] += 1

        if not adapter.get("price_text"):
            self.stats["missing_price"] += 1
        if not adapter.get("image_url"):
            self.stats["missing_image"] += 1
        if not adapter.get("clean_description"):
            self.stats["missing_clean_description"] += 1

        for category in adapter.get("categories") or []:
            self.stats["category_distribution"][category] += 1

        current_price = adapter.get("current_price")
        if isinstance(current_price, int) and current_price > 0:
            self.stats["valid_prices"].append(current_price)

    def _write_quality_report(self, spider=None):
        report_path = self._resolve_report_path(spider)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        category_distribution = dict(
            sorted(
                self.stats["category_distribution"].items(),
                key=lambda item: (-item[1], item[0].casefold()),
            )
        )
        valid_prices = self.stats["valid_prices"]
        spider_name = getattr(spider, "name", None) or self.spider_name
        report = {
            "spider": spider_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items_seen": self.stats["items_seen"],
            "items_processed": self.stats["items_processed"],
            "dropped_total": self.stats["dropped_total"],
            "dropped_reasons": dict(self.stats["dropped_reasons"]),
            "duplicate_urls": self.stats["duplicate_urls"],
            "missing_price": self.stats["missing_price"],
            "missing_image": self.stats["missing_image"],
            "missing_clean_description": self.stats["missing_clean_description"],
            "category_count": len(category_distribution),
            "category_distribution": category_distribution,
            "valid_price_min": min(valid_prices) if valid_prices else None,
            "valid_price_max": max(valid_prices) if valid_prices else None,
        }

        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote flower quality report to %s", report_path)

    def _resolve_report_path(self, spider):
        spider_name = getattr(spider, "name", None) or self.spider_name or "flower_products"
        if self.report_path:
            return Path(self.report_path.format(spider_name=spider_name))
        return Path(f"../data/processed/{spider_name}_quality_report.json")

    def _active_spider(self):
        if self.crawler:
            return getattr(self.crawler, "spider", None)
        return None


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


def _format_vnd(value):
    return f"{value:,}".replace(",", ".") + " VND"


def _clean_product_description(description):
    if not description:
        return None

    clean = " ".join(description.split())
    clean = re.sub(
        r"^Hoa Tươi My My luôn là lựa chọn tốt nhất.*?Bạn có thể đặt hoa nhanh ship 2-3h tại zalo shop\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"(Liên Hệ Đặt Hàng|Liên hệ đặt hàng|Đặt hàng:|Hotline\s*/\s*Zalo|Hotline|Zalo|☎️|📞).*",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" -–—:,.")
    return clean or None


def _drop_reason(exc):
    message = str(exc)
    if message.startswith("Duplicate product URL"):
        return "duplicate_product_url"
    if message == "Missing product name":
        return "missing_product_name"
    if message == "Missing product URL":
        return "missing_product_url"
    return re.sub(r"[^a-z0-9]+", "_", message.casefold()).strip("_") or "unknown"
