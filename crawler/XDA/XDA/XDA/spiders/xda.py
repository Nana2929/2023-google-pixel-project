from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional, Dict, Union, List
from itertools import groupby
import re
import logging

from easydict import EasyDict as edict
import scrapy
from inline_requests import inline_requests
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider
from itemloaders import ItemLoader
from itemloaders.processors import TakeFirst
from scrapy.utils.log import configure_logging


from ..items import ThreadItem, PostItem
from ..settings import PHONE2URL, PAGE_LIMIT
from ..models import ThreadAttr, PostAttr

cssD: Dict[str, Dict] = {
    'main': {
        'all_posts': 'article[id*="js-post"]',
    },
    'thread': {
        ThreadAttr.title: 'h1[class="p-title-value"]::text',
        ThreadAttr.date_time: 'time[class="u-dt"]::attr(datetime)',
    },
    'post': {
        PostAttr.date_time: 'time[class="u-dt"]::attr(datetime)',
        PostAttr.author: '::attr(data-author)',
        PostAttr.user_title: 'h5[class^="user"]::text',
        PostAttr.post_id: '::attr(id)',
        PostAttr.text: 'div[class="bbWrapper"]::text',
        PostAttr.reaction: 'a[class*="--reaction0"]::text',
    },
    'section': {
        'main': 'div.message-cell.message-cell--main',
        'user': 'div.message-cell.message-cell--user',
    }
}
XpathD: Dict[str, Dict] = {
    'forum': {
        'name':
        '//*[@id="top"]/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[1]/div/ul/li[4]/a/span',
        'page_num':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[2]/div[4]/div[1]/nav/div[1]/ul/li[5]/a/text()',
    },
    'thread': {
        'category':
        '//*[@id="top"]/div[3]/div[2]/div/div[2]/div[1]/div[1]/div/div/div[1]/h1/span[1]/text()',
        'curr_page':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[2]/div[4]/div[1]/nav/div[1]/ul/li[2]/a/text()',
        'page_num':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[3]/div[1]/div[1]/nav/div[1]/ul/li[6]/a/text()'
    },
    'post':{
        'floor': '//*[@id="js-post-{post_id}"]/div/div[2]/div/header/ul[2]/li[2]/a/text()',}
}
REACT_URL= 'https://forum.xda-developers.com/posts/{post_id}/reactions'
CssD = edict(cssD)
XpathD = edict(XpathD)

class XdaspiderSpider(CrawlSpider):

    configure_logging(install_root_handler=False)
    logging.basicConfig(filename='scrape_xda.log',
                        format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    name = 'xda'
    # logging.getLogger('scrapy').setLevel(logging.info)
    protocol = 'https://'
    allowed_domains = ['forum.xda-developers.com']
    keyword = 'pixel'
    rules = [
        Rule(LinkExtractor(allow=(r'^https:\/\/forum.xda-developers.com\/f')),
             callback='parse_forum',
             follow=False)
    ]

    start_urls = PHONE2URL.values()


    @staticmethod
    def get_integer(text: Union[str, List]) -> int:
        if isinstance(text, list):
            try:
                text = text[0]
                number = int(re.search(r'\d+', text).group())
            except:
                number = 1
        return number

    @staticmethod
    def get_next_url(url: str, page_num: int) -> Optional[str]:

        thread_root = re.sub(r'\/page-\d+', '', url.rstrip('/'))
        try:
            nextp = int(url.split('-')[-1]) + 1
            return f"{thread_root}/page-{nextp}"
        except:
            pass

    def parse_forum(self, forum_response):

        thread_extractor = LinkExtractor(
            allow=(r'^https:\/\/forum.xda-developers.com\/t'))
        thread_links = thread_extractor.extract_links(forum_response)
        for tlink in thread_links:
            yield scrapy.Request(tlink.url, callback=self.parse_thread)

        forum_page_limit = PAGE_LIMIT
        currp = forum_response.xpath(
            XpathD.thread.curr_page).extract()
        currp = self.get_integer(currp)
        #  ======================================
        self.logger.info(f'current forum url: {forum_response.url}')
        self.logger.info(f'flimit: {forum_page_limit}')
        self.logger.info(f'currp: {currp}')
        # =======================================

        if currp <= forum_page_limit:
            forum_root = re.sub(r'\/page-\d+', '',
                                forum_response.url.rstrip('/'))
            forum_next_page = f"{forum_root}/page-{currp+1}"

            yield scrapy.Request(forum_next_page, callback=self.parse_forum)


    def parse_reactions(self,
                        react_response: scrapy.http.Response) -> PostItem:

        reacts = react_response.css(CssD.post.reaction)
        react_string = reacts[-1].extract() if reacts else '(0)'
        pattern = '[^()]+(?=\))'
        reactions = re.search(pattern, react_string).group(0)
        return reactions

    @inline_requests
    def parse_thread(self, response: scrapy.http.Response) -> scrapy.Item:
        """parsing all webpages associated to a post
        Args:
            response (scrapy.http.Response): scrapy response
        """
        tl = ItemLoader(item=ThreadItem(), selector=response)
        tl.default_output_processor = TakeFirst()
        forum_name_xpath = f"{XpathD.forum.name}/text()"


        tl.add_value(ThreadAttr.thread_id, response.url)
        tl.add_value(ThreadAttr.url, response.url)
        tl.add_xpath(ThreadAttr.forum, forum_name_xpath)
        tl.add_xpath(ThreadAttr.category, XpathD.thread.category)
        raw_posts = response.css(CssD.main.all_posts)



        for key, search_css in CssD.thread.items():
            tl.add_css(key, search_css)
            self.logger.info(f'key: {key}, search_css: {search_css}, output: {tl.get_output_value(key)}')
        tl.add_value('n_replies', len(raw_posts))

        posts = []
        for raw_post in raw_posts:
            pl = ItemLoader(item=PostItem(), selector=raw_post)
            pl.default_output_processor = TakeFirst()
            for key, search_css in CssD.post.items():
                pl.add_css(key, search_css)
            id = pl.get_output_value(PostAttr.post_id)
            floor_path = XpathD.post.floor.format(post_id=id)
            react_url = REACT_URL.format(post_id=id)

            pl.add_xpath(PostAttr.floor, floor_path)
            react_response = yield scrapy.Request(react_url)
            pl.add_value(PostAttr.reaction,
                         self.parse_reactions(react_response))
            posts.append(dict(pl.load_item()))

        thread_page_num = response.xpath(
            XpathD.thread.page_num).extract()
        thread_page_num = self.get_integer(thread_page_num)
        next_page = self.get_next_url(response.url, thread_page_num)
        if next_page:
            yield scrapy.Request(next_page, callback=self.parse_thread)
        tl.add_value(ThreadAttr.posts, posts)



        yield tl.load_item()


