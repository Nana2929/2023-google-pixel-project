# -*- coding: utf-8 -*-
# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import pymongo
# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

from .settings import DATABASE, ENVFILE
from .utils import ArsTechnicaFieldProcessor, MongoDB


class ArstechnicaPipeline(object):

    def __init__(self) -> None:

        self.db_name: str = DATABASE
        MConn: pymongo.MongoClient = self.get_mongo_conn()
        self.db: pymongo.database.Database = MConn[self.db_name]
        self.collection = self.db['Ars_Technica_202306']

    def get_mongo_conn(self) -> pymongo.MongoClient:

        env_file = ENVFILE
        return MongoDB.by_env(env_file)

    def process_item(self, item, spider):

        item['title'] = ArsTechnicaFieldProcessor.format_title(item['title'])
        item['date_time'] = ArsTechnicaFieldProcessor.format_time(
            item['date_time'])
        item['posts'] = ArsTechnicaFieldProcessor.format_replies(item)
        item['n_replies'] = len(item['posts'])

        self.to_mongodb(item, spider)
        spider.log(f'save: {item["title"]}', 30)

        return item

    def to_mongodb(self, item, spider):

        self.collection.insert_one(dict(item))
        return item
