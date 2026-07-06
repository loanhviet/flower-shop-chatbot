from scrapy.spiders import SitemapSpider

from flower_crawler.items import FlowerProductItem


class HoatuoiMyMySpider(SitemapSpider):
    name = "hoatuoimymy"
    allowed_domains = ["hoatuoimymy.com"]

    sitemap_urls = [
        "https://hoatuoimymy.com/sitemap_index.xml",
    ]

    sitemap_follow = [
        r"product-sitemap\d+\.xml",
    ]

    sitemap_rules = [
        (r"https://hoatuoimymy\.com/.*", "parse_product"),
    ]

    def parse_product(self, response):
        name = self._join_text(
            response.css(
                "h1.product_title::text, "
                "h1.entry-title::text"
            ).getall()
        )

        price_text = self._join_text(
            response.css(
                ".entry-summary .price .woocommerce-Price-amount ::text, "
                ".summary .price .woocommerce-Price-amount ::text"
            ).getall()
        )

        short_description = self._join_text(
            response.css(
                ".woocommerce-product-details__short-description ::text"
            ).getall()
        )

        description = self._join_text(
            response.css(
                "#tab-description ::text, "
                ".woocommerce-Tabs-panel--description ::text"
            ).getall()
        )

        categories = [
            text.strip()
            for text in response.css(
                ".product_meta .posted_in a::text, "
                ".posted_in a::text"
            ).getall()
            if text.strip()
        ]

        availability = self._join_text(
            response.css(
                ".stock::text, "
                ".availability::text, "
                "meta[property='product:availability']::attr(content)"
            ).getall()
        )

        image_url = (
            response.css(".woocommerce-product-gallery__image img::attr(src)").get()
            or response.css("img.wp-post-image::attr(src)").get()
            or response.css("meta[property='og:image']::attr(content)").get()
        )

        if not name and not price_text:
            self.logger.debug("Skipping non-product page: %s", response.url)
            return

        item = FlowerProductItem()
        item["name"] = name
        item["price_text"] = price_text
        item["short_description"] = short_description
        item["description"] = description
        item["categories"] = categories
        item["image_url"] = response.urljoin(image_url) if image_url else None
        item["product_url"] = response.url
        item["source"] = "hoatuoimymy.com"
        item["availability"] = availability or None

        yield item

    @staticmethod
    def _join_text(texts):
        return " ".join(
            text.strip()
            for text in texts
            if text and text.strip()
        )
