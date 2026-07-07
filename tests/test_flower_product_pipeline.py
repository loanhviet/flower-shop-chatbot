import json
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(result["current_price"], 350000)
        self.assertEqual(result["original_price"], 350000)
        self.assertIsNone(result["sale_price"])
        self.assertEqual(result["display_price"], "350.000 VND")
        self.assertEqual(result["currency"], "VND")
        self.assertEqual(result["categories"], ["Hoa Sinh Nhat", "Hoa Hong"])
        self.assertEqual(
            result["product_url"],
            "https://hoatuoimymy.com/san-pham/hoa-hong/",
        )
        self.assertTrue(result["product_id"])
        self.assertTrue(result["crawled_at"])

    def test_pipeline_sets_sale_price_fields(self):
        item = FlowerProductItem(
            name="Hoa cuoi",
            price_text="750.000 ₫ 650.000 ₫",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/hoa-cuoi/",
            source="hoatuoimymy.com",
            categories=[],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(result["price_min"], 650000)
        self.assertEqual(result["price_max"], 750000)
        self.assertEqual(result["current_price"], 650000)
        self.assertEqual(result["original_price"], 750000)
        self.assertEqual(result["sale_price"], 650000)
        self.assertEqual(result["display_price"], "650.000 VND")

    def test_pipeline_generates_stable_product_id(self):
        first = FlowerProductItem(
            name="Hoa hong",
            product_url="https://hoatuoimymy.com/san-pham/hoa-hong/",
            source="hoatuoimymy.com",
            categories=[],
        )
        second = FlowerProductItem(
            name="Hoa hong khac",
            product_url="https://hoatuoimymy.com/san-pham/hoa-hong/",
            source="hoatuoimymy.com",
            categories=[],
        )

        first_result = FlowerProductPipeline().process_item(first)
        second_result = FlowerProductPipeline().process_item(second)

        self.assertEqual(first_result["product_id"], second_result["product_id"])

    def test_pipeline_cleans_description_and_builds_rag_text(self):
        item = FlowerProductItem(
            name="Bó Hoa M250",
            price_text="730.000 ₫ 600.000 ₫",
            description=(
                "Hoa Tươi My My luôn là lựa chọn tốt nhất của những tín đồ yêu thích hoa. "
                "Với tiêu chí: Hoa tươi mới được nhập về trong ngày "
                "Bạn có thể đặt hoa nhanh ship 2-3h tại zalo shop "
                "Bó Hoa M250 là bó hoa hồng xanh và cúc họa mi đẹp. "
                "Liên Hệ Đặt Hàng 📞 Đặt hàng: Hotline / Zalo: 0979.424.145"
            ),
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/bo-hoa-m250/",
            source="hoatuoimymy.com",
            availability="instock",
            categories=["Bó Hoa Sinh Nhật"],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(
            result["clean_description"],
            "Bó Hoa M250 là bó hoa hồng xanh và cúc họa mi đẹp",
        )
        self.assertEqual(result["product_type"], "bó hoa")
        self.assertIn("sinh nhật", result["occasion_tags"])
        self.assertIn("hoa hồng", result["flower_tags"])
        self.assertIn("cúc họa mi", result["flower_tags"])
        self.assertIn("xanh", result["color_tags"])
        self.assertIn("Sản phẩm: bó hoa", result["rag_text"])
        self.assertIn("Mã/tên gốc: Bó Hoa M250", result["rag_text"])
        self.assertIn("Dịp phù hợp: sinh nhật", result["rag_text"])
        self.assertIn("Giá hiện tại: 600.000 VND", result["rag_text"])
        self.assertIn("Mô tả ngắn:", result["rag_text"])
        self.assertIn("URL: https://hoatuoimymy.com/bo-hoa-m250/", result["rag_text"])

    def test_pipeline_infers_khai_truong_tags(self):
        item = FlowerProductItem(
            name="Hoa Khai Trương M371",
            price_text="1.600.000 ₫ 1.400.000 ₫",
            description="Kệ hoa khai trương tông hồng sang trọng với hoa lan.",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/hoa-khai-truong-m371/",
            source="hoatuoimymy.com",
            categories=["Hoa Khai Trương", "Kệ Hoa Khai Trương"],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(result["product_type"], "hoa khai trương")
        self.assertIn("khai trương", result["occasion_tags"])
        self.assertIn("hoa lan", result["flower_tags"])
        self.assertIn("hồng", result["color_tags"])
        self.assertIn("Sản phẩm: hoa khai trương", result["rag_text"])

    def test_pipeline_prefers_name_for_product_type_over_categories(self):
        item = FlowerProductItem(
            name="Giỏ Hoa M166",
            price_text="900.000 ₫ 790.000 ₫",
            description="Giỏ hoa hồng đỏ và hoa cẩm tú cầu đẹp.",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/gio-hoa-m166/",
            source="hoatuoimymy.com",
            categories=["Giỏ Hoa", "Lẵng Hoa Khai Trương"],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(result["product_type"], "giỏ hoa")
        self.assertIn("khai trương", result["occasion_tags"])
        self.assertIn("Sản phẩm: giỏ hoa", result["rag_text"])

    def test_pipeline_does_not_treat_cam_ket_as_orange_color(self):
        item = FlowerProductItem(
            name="Hoa Chia Buồn M12",
            price_text="1.750.000 ₫ 1.649.000 ₫",
            description="Shop cam kết hoa lan trắng tươi mới và trang trọng.",
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/hoa-chia-buon-m12/",
            source="hoatuoimymy.com",
            categories=["Hoa Đám Tang"],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertIn("trắng", result["color_tags"])
        self.assertNotIn("cam", result["color_tags"])

    def test_pipeline_marks_missing_price_as_contact_price_and_shortens_rag_text(self):
        long_description = " ".join(["Lan hồ điệp trắng sang trọng"] * 80)
        item = FlowerProductItem(
            name="Hoa Lan Hồ Điệp M381",
            description=long_description,
            image_url="https://hoatuoimymy.com/image.jpg",
            product_url="https://hoatuoimymy.com/hoa-lan-ho-diep-m381/",
            source="hoatuoimymy.com",
            categories=["Lan Hồ Điệp"],
        )

        result = FlowerProductPipeline().process_item(item)

        self.assertEqual(result["product_type"], "lan hồ điệp")
        self.assertIn("lan hồ điệp", result["flower_tags"])
        self.assertIn("trắng", result["color_tags"])
        self.assertIn("Giá: liên hệ", result["rag_text"])
        self.assertIn("URL: https://hoatuoimymy.com/hoa-lan-ho-diep-m381/", result["rag_text"])
        self.assertLess(len(result["rag_text"]), len(long_description))

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

    def test_pipeline_writes_quality_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "quality.json"
            pipeline = FlowerProductPipeline(
                report_path=str(report_path),
                spider_name="testspider",
            )

            pipeline.process_item(
                FlowerProductItem(
                    name="Hoa hong",
                    price_text="350.000 VND",
                    description="Hoa hong do dep",
                    image_url="https://hoatuoimymy.com/image.jpg",
                    product_url="https://hoatuoimymy.com/hoa-hong/",
                    source="hoatuoimymy.com",
                    categories=["Hoa Hong"],
                )
            )
            pipeline.process_item(
                FlowerProductItem(
                    name="Hoa khong gia",
                    product_url="https://hoatuoimymy.com/hoa-khong-gia/",
                    source="hoatuoimymy.com",
                    categories=["Hoa Hong", "Hoa Sinh Nhat"],
                )
            )
            with self.assertRaisesRegex(DropItem, "Duplicate product URL"):
                pipeline.process_item(
                    FlowerProductItem(
                        name="Hoa trung",
                        product_url="https://hoatuoimymy.com/hoa-hong/",
                        source="hoatuoimymy.com",
                        categories=[],
                    )
                )

            pipeline.close_spider()

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(report["spider"], "testspider")
        self.assertEqual(report["items_seen"], 3)
        self.assertEqual(report["items_processed"], 2)
        self.assertEqual(report["dropped_total"], 1)
        self.assertEqual(report["dropped_reasons"], {"duplicate_product_url": 1})
        self.assertEqual(report["duplicate_urls"], 1)
        self.assertEqual(report["missing_price"], 1)
        self.assertEqual(report["missing_image"], 1)
        self.assertEqual(report["missing_clean_description"], 1)
        self.assertEqual(report["missing_product_type"], 2)
        self.assertEqual(report["missing_tags"], 2)
        self.assertEqual(report["contact_price"], 1)
        self.assertEqual(report["category_count"], 2)
        self.assertEqual(report["category_distribution"]["Hoa Hong"], 2)
        self.assertEqual(report["valid_price_min"], 350000)
        self.assertEqual(report["valid_price_max"], 350000)


if __name__ == "__main__":
    unittest.main()
