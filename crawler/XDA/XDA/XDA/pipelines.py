# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
# for Doc of type hints,
# see: https://pymongo.readthedocs.io/en/stable/examples/type_hints.html

from __future__ import absolute_import
from copy import deepcopy
import pymongo
import scrapy
from pymongo import collection, database
from scrapy import exceptions
import sys
from pathlib import Path
import pymongo
from XDA import items, settings


class ThreadPipeline:
    def __init__(self):

        self.settings = settings
        self.db_name:str = self.settings.DATABASE
        self.collection_name:str = self.settings.COLLECTION
        self.keyword:str = self.settings.KEYWORD
        self.utilspath:str = self.settings.UTILSPATH
        MConn:pymongo.MongoClient = self.getMongoConn()
        self.db:database.Database= MConn[self.db_name]


    def getMongoConn(self) -> pymongo.MongoClient:
        # print(self.utilspath) # /home/nanaeilish/projects/Google-Opinion/crawler/utils
        sys.path.append(self.utilspath)
        from utils import MongoDB
        env_file:Path = self.settings.ENVFILE
        return MongoDB.by_env(env_file)

    def process_item(self,
        item: items.ThreadItem,
        spider: scrapy.Spider)-> items.ThreadItem:
        # if self.keyword not in item['forum'].lower():
        #     raise exceptions.DropItem(f'This thread {item["thread_id"]} comes from wrong forum {item["forum"]}.')
        # if item['n_replies'] <= 1:
        #     raise exceptions.DropItem(f'This thread {item["thread_id"]} has no replies.')
        if item['title'].strip() == '':
            raise exceptions.DropItem(f'This thread {item["thread_id"]} has no title.')
        # check if the item with the same thread_id exists in db,
        # if yes, append the posts to the existing item and update the db
        tid = item['thread_id']
        if self.db[self.collection_name].count_documents({"thread_id": tid}) > 0:
            existing_doc = self.db[self.collection_name].find_one({"thread_id": tid})

            existing_posts = existing_doc['posts']
            existing_posts.extend(item['posts'])
            item['posts'] = sorted(existing_posts, key=lambda x: int(x['floor']))
            item['n_replies'] = len(item['posts'])
            # delete the existing doc
            self.db[self.collection_name].delete_many({"thread_id": tid})
            assert self.db[self.collection_name].count_documents({"thread_id": tid}) == 0
            print(f'Merging thread {item["thread_id"]} ({item["n_replies"]} replies).')
        self.to_mongodb(item, spider)

        return item

    def to_mongodb(self,
        item: items.ThreadItem,
        spider: scrapy.Spider) -> items.ThreadItem:
        # if self.collection.count_documents({"thread_id": item["thread_id"]}) <= 0:
        doc = dict(item)
        self.db[self.collection_name].replace_one(doc, doc, upsert = True)

        # print(f'Successfully saved into [{self.collection_name}]: \
        #       thread {item["thread_id"]} ({item["n_replies"]} replies).')
        return item














# %%
