import scrapy


class FlowerProductItem(scrapy.Item):
    name = scrapy.Field()
    price_text = scrapy.Field()
    price_min = scrapy.Field()
    price_max = scrapy.Field()
    currency = scrapy.Field()
    availability = scrapy.Field()
    description = scrapy.Field()
    short_description = scrapy.Field()
    categories = scrapy.Field()
    image_url = scrapy.Field()
    product_url = scrapy.Field()
    source = scrapy.Field()
    crawled_at = scrapy.Field()
