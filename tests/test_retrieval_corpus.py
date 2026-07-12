import unittest

from backend.rag.corpus import build_retrieval_corpus, split_detail_text


def _product(**overrides):
    product = {
        "product_id": "flower-01",
        "name": "Bó Hoa Hồng",
        "rag_text": "Sản phẩm: bó hoa\nMã/tên gốc: Bó Hoa Hồng",
        "clean_description": "",
        "product_url": "https://example.com/bo-hoa-hong",
        "image_url": "https://example.com/image.jpg",
        "current_price": 500000,
        "display_price": "500.000 VND",
        "availability": "instock",
        "categories": ["Bó Hoa Sinh Nhật"],
        "product_type": "bó hoa",
        "occasion_tags": ["sinh nhật"],
        "flower_tags": ["hoa hồng"],
        "color_tags": ["đỏ"],
        "source": "example.com",
    }
    product.update(overrides)
    return product


class RetrievalCorpusTest(unittest.TestCase):
    def test_short_or_empty_descriptions_only_create_overview(self):
        documents, stats = build_retrieval_corpus([_product()])

        self.assertEqual([document["id"] for document in documents], ["flower-01:overview"])
        self.assertEqual(documents[0]["document_type"], "overview")
        self.assertEqual(stats.overview_documents, 1)
        self.assertEqual(stats.detail_documents, 0)
        self.assertEqual(stats.short_or_empty_products_skipped, 1)

    def test_long_unique_description_creates_contextual_detail_chunks(self):
        description = " ".join(["Hoa hồng đỏ phù hợp sinh nhật."] * 80)
        documents, stats = build_retrieval_corpus([_product(clean_description=description)])

        detail_documents = [document for document in documents if document["document_type"] == "detail"]
        self.assertGreater(len(detail_documents), 1)
        self.assertEqual(stats.detail_documents, len(detail_documents))
        self.assertEqual(stats.total_documents, len(documents))
        self.assertEqual(
            [document["id"] for document in detail_documents],
            [f"flower-01:detail:{index}" for index in range(len(detail_documents))],
        )
        self.assertTrue(all("Mã/tên gốc: Bó Hoa Hồng" in document["text"] for document in detail_documents))
        self.assertTrue(all(document["metadata"]["current_price"] == 500000 for document in detail_documents))

    def test_split_detail_text_respects_limit_and_overlap(self):
        text = " ".join([f"Câu mô tả số {index}." for index in range(200)])

        chunks = split_detail_text(text, chunk_chars=200, overlap_chars=40)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 200 for chunk in chunks))
        self.assertTrue(chunks[0][-30:] in chunks[1])
        self.assertIn("Câu mô tả số 199.", chunks[-1])

    def test_duplicate_descriptions_keep_overviews_but_skip_details(self):
        description = " ".join(["Nội dung dùng chung cho hai sản phẩm."] * 80)
        first = _product(product_id="flower-01", clean_description=description)
        second = _product(product_id="flower-02", name="Giỏ Hoa", clean_description=description)

        documents, stats = build_retrieval_corpus([first, second])

        self.assertEqual([document["id"] for document in documents], ["flower-01:overview", "flower-02:overview"])
        self.assertEqual(stats.duplicate_detail_products_skipped, 2)
        self.assertEqual(stats.detail_documents, 0)

    def test_missing_product_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing product_id"):
            build_retrieval_corpus([_product(product_id=None)])
