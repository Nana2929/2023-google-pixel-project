# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from concurrent.futures import thread
import scrapy


class RedditPostItem(scrapy.Item):
    thread_id = scrapy.Field()
    forum = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    tag = scrapy.Field()
    author = scrapy.Field()
    post_text = scrapy.Field()
    datetime = scrapy.Field()
    upvote_ratio = scrapy.Field()
    comments = scrapy.Field()


class RedditCommentsItem(scrapy.Item):
    author = scrapy.Field()
    com_text = scrapy.Field()
    datetime = scrapy.Field()
