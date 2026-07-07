import scrapy


class FlowerProductItem(scrapy.Item):
    product_id = scrapy.Field()
    name = scrapy.Field()
    price_text = scrapy.Field()
    price_min = scrapy.Field()
    price_max = scrapy.Field()
    current_price = scrapy.Field()
    original_price = scrapy.Field()
    sale_price = scrapy.Field()
    display_price = scrapy.Field()
    currency = scrapy.Field()
    availability = scrapy.Field()
    description = scrapy.Field()
    clean_description = scrapy.Field()
    short_description = scrapy.Field()
    rag_text = scrapy.Field()
    categories = scrapy.Field()
    image_url = scrapy.Field()
    product_url = scrapy.Field()
    source = scrapy.Field()
    crawled_at = scrapy.Field()
