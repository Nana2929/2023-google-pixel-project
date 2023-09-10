# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import pymongo
from pymongo import collection
import scrapy
from scrapy import exceptions
from reddit_crawler import items, settings


class RedditCrawlerPipeline:
    def __init__(self) -> None:
        host: str = settings.MONGODB_HOST
        port: int = settings.MONGODB_PORT
        dbname: str = settings.MONGODB_DBNAME
        collection_name: str = 'data'

        client = pymongo.MongoClient(host=host, port=port)
        db = client[dbname]

        self.collection: collection.Collection = db[collection_name]

    def process_item(self, item: items.RedditPostItem, spider: scrapy.Spider):
        self.to_mongodb(item, spider)
        # spider.log(f'save: {item["title"]}[{len(item["comments"])}]', 30)

        return item

    # insert item to mongodb
    def to_mongodb(self, item: items.RedditPostItem, spider: scrapy.Spider):
        self.collection.replace_one(
            {'thread_id': item['thread_id']}, dict(item), upsert=True)  # avoid duplicate data
        # self.collection.insert_one(dict(item))

        return item
