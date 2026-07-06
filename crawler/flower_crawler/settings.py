BOT_NAME = "flower_crawler"

SPIDER_MODULES = ["flower_crawler.spiders"]
NEWSPIDER_MODULE = "flower_crawler.spiders"

ROBOTSTXT_OBEY = True

CONCURRENT_REQUESTS = 4
DOWNLOAD_DELAY = 1

FEED_EXPORT_ENCODING = "utf-8"

TELNETCONSOLE_ENABLED = False

ITEM_PIPELINES = {
    "flower_crawler.pipelines.FlowerProductPipeline": 300,
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
