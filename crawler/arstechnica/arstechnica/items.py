# -*- coding: utf-8 -*-
# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import MapCompose, TakeFirst

from .utils import ArsTechnicaFieldProcessor


class ArstechnicaItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    thread_id = scrapy.Field()
    forum = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field(
        input_processor=MapCompose(ArsTechnicaFieldProcessor.format_title),
        output_processor=TakeFirst()
    )
    author = scrapy.Field()
    date_time = scrapy.Field(
        input_processor=MapCompose(ArsTechnicaFieldProcessor.format_time),
        output_processor=TakeFirst()
    )
    n_replies = scrapy.Field()
    n_views = scrapy.Field()
    posts = scrapy.Field(
        input_processor=MapCompose(ArsTechnicaFieldProcessor.format_replies),
    )
