# Scrapy settings for XDA project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'XDA'
SPIDER_MODULES = ['XDA.spiders']
NEWSPIDER_MODULE = 'XDA.spiders'
ROOT = '/home/nanaeilish/projects'
ENVFILE = f'{ROOT}/Google-Opinion/.remote.env'
UTILSPATH = f'{ROOT}/Google-Opinion/utils'
# scrapy's import bahaves weird...can't get around with relative imports

# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = 'XDA (+http://www.yourdomain.com)'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False
# The integer values you assign to classes in this setting determine the order they run in- items go through pipelines from order number low to high. It’s customary to define these numbers in the 0-1000 range.
UTILSPATH = f'{ROOT}/Google-Opinion/crawler/'
DATABASE = 'crawler-raw'
COLLECTION = 'xda-202306'
KEYWORD = 'pixel'
PAGE_LIMIT = 100
# Configure maximum concurrent requests performed by Scrapy (default: 16)
CONCURRENT_REQUESTS = 8
LOG_LEVEL = 'ERROR'  # to only display errors
LOG_FORMAT = '%(levelname)s: %(message)s'
LOG_FILE = 'xda.log'
LOG_FILE_APPEND = False
LOG_STDOUT = True
PHONE2URL = {
    "pixel 5": "https://forum.xda-developers.com/c/google-pixel-5.11335/",
    'pixel 5a': "https://forum.xda-developers.com/f/google-pixel-5a.12359/",
    'pixel 6': 'https://forum.xda-developers.com/f/google-pixel-6.12311/',
    'pixel 6 pro':
    'https://forum.xda-developers.com/f/google-pixel-6-pro.12313/',
    'pixel 6a': 'https://forum.xda-developers.com/f/google-pixel-6a.12605/',
    'pixel 7': 'https://forum.xda-developers.com/f/google-pixel-7.12607/',
    'pixel 7 pro':
    'https://forum.xda-developers.com/f/google-pixel-7-pro.12609/',
    'pixel watch':
    'https://forum.xda-developers.com/f/google-pixel-watch.12613/'
}

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#   'Accept-Language': 'en',
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    'XDA.middlewares.XdaSpiderMiddleware': 543,
# }

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    'XDA.middlewares.XdaDownloaderMiddleware': 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    'XDA.pipelines.ThreadPipeline': 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = 'httpcache'
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
