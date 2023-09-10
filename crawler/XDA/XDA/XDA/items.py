import scrapy
from itemloaders import ItemLoader
from itemloaders.processors import TakeFirst, MapCompose, Compose, Join, Identity
import re
from typing import List
from datetime import datetime

# Define here the models for your scraped items
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

class FieldProcessor:
    @staticmethod
    def clean_text(text):
        text = re.sub('[\n|\t]', '', text)
        text = re.sub('<.*?>', '', text)
        text = text.strip()
        return text

    @staticmethod
    def format_time(rawstr:str):
        # '2021-12-01T18:23:04+0000'
        rawstrs = rawstr.split('T')
        date = datetime.strptime(rawstrs[0], '%Y-%m-%d')
        return date.strftime('%Y-%m-%d')

    @staticmethod
    def get_floor(floor:list):
        return re.sub('[ |\\n|\\t|#]', '', ''.join(floor))

    @staticmethod
    def get_thread_id(x):
        x = x.split('/')
        return re.search('\.(\d+)', x[-2]).group(1)

    @staticmethod
    def get_post_id(x):
        return re.search('(\d+)', x).group(0)


class ThreadItem(scrapy.Item):
    n_replies:int = scrapy.Field(
    )
    author:str = scrapy.Field()
    user_title:str = scrapy.Field()
    date_time:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.format_time),
        output_processor = Compose(lambda x:x[-1])
    )
    title:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.clean_text))
    url:str = scrapy.Field()   # link back to the crawled page
    thread_id:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.get_thread_id)
    )
    category:str =scrapy.Field(
        output_processor = TakeFirst()
    )
    posts:List[dict]=scrapy.Field(
        output_processor = Identity()
    )
    forum:str=scrapy.Field(
    )

class PostItem(scrapy.Item):
    post_id:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.get_post_id)
    )
    floor = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.get_floor)
    )
    author:str = scrapy.Field()
    user_title:str = scrapy.Field()
    date_time:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.format_time),
        output_processor = Compose(lambda x:x[-1])
    )   # last edit: 2022-02-01 12:50:00
    text:str = scrapy.Field(
        input_processor = MapCompose(FieldProcessor.clean_text),
        output_processor = Join()
    )
    thread_id:str=scrapy.Field()
    reaction:str = scrapy.Field()
    is_head:bool = scrapy.Field()   # True if floor is 1 else False
    url:str = scrapy.Field()   # link back to the crawled page


# #%% testing
# il = ItemLoader(item=ThreadItem())
# il.default_output_processor = TakeFirst()
# il.add_value('n_replies', 10)
# il.add_value('title', '\n\t <br> [ROM][G920/5-F-I-S-K-L-T-W8][9.0] UNOFFICIAL LineageOS 16.0')
# il.add_value('back_url', 'https://github.com/yiting-tom/GoogleOpinionScrapy/blob/master/crawler-system/pttCrawlerSystem/pipelines.py')
# il.add_value('author', ['lackalil'])
# il.add_value('user_title',  ['Member'])
# il.add_value('datetime', ['2021-12-01T18:23:04+0000', '2021-12-02T08:28:25+0000'])
# print(il.load_item())

# # %%
# il = ItemLoader(item=PostItem())
# il.default_output_processor = TakeFirst()
# il.add_value('back_url', 'https://github.com/yiting-tom/GoogleOpinionScrapy/blob/master/crawler-system/pttCrawlerSystem/pipelines.py')
# il.add_value('author', ['lackalil'])
# il.add_value('user_title',  ['Member'])
# il.add_value('datetime', ['2021-12-01T18:23:04+0000', '2021-12-02T08:28:25+0000'])
# il.add_value('text', ['This is beautiful', 'Very pixel'])
# # il.add_value('reaction', '(134)\n\n\n\n\n\n\n\n\n\t')
# il.add_value('floor', [['\n\n\n\n\n\n\n\n\n\t'], ['\n\n\n\n\n#20\n\n\n\n\t']])
# print('getting value:', il.get_output_value('author'))
# print(il.load_item())


