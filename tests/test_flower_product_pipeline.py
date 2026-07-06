import unittest

from scrapy.exceptions import DropItem

from crawler.flower_crawler.items import FlowerProductItem
from crawler.flower_crawler.pipelines import FlowerProductPipeline, _parse_vnd_prices


class FlowerProductPipelineTest(unittest.TestCase):
    def test_parse_vnd_prices_with_single_and_range_values(self):
        self.assertEqual(_parse_vnd_prices("350.000 VND"), [350000])
        self.assertEqual(_parse_vnd_prices("250.000 VND - 450.000 VND"), [250000, 450000])

    def test_pipeline_normalizes_product_item(self):
        item = FlowerProductItem()
        item["name"] = "  Hoa hong   do  "
        item["price_text"] = "350.000 VND"
        item["categories"] = [" Hoa Sinh Nhat ", "hoa sinh nhat", "Hoa Hong"]
        item["image_url"] = " https://hoatuoimymy.com/image.jpg "
        item["product_url"] = " https://hoatuoimymy.com/san-pham/hoa-hong/ "
        item["source"] = " hoatuoimymy.com "

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(result["name"], "Hoa hong do")
        self.assertEqual(result["price_min"], 350000)
        self.assertEqual(result["price_max"], 350000)
        self.assertEqual(result["currency"], "VND")
        self.assertEqual(result["categories"], ["Hoa Sinh Nhat", "Hoa Hong"])
        self.assertEqual(
            result["product_url"],
            "https://hoatuoimymy.com/san-pham/hoa-hong/",
        )
        self.assertTrue(result["crawled_at"])

    def test_pipeline_drops_duplicate_product_url(self):
        pipeline = FlowerProductPipeline()
        first = FlowerProductItem(
            name="Hoa hong",
            price_text="350.000 VND",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/san-pham/hoa-hong/",
            categories=[],
        )
        duplicate = FlowerProductItem(
            name="Hoa hong",
            price_text="350.000 VND",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/san-pham/hoa-hong/",
            categories=[],
        )

        pipeline.process_item(first)

        with self.assertRaisesRegex(DropItem, "Duplicate product URL"):
            pipeline.process_item(duplicate)


if __name__ == "__main__":
    unittest.main()
