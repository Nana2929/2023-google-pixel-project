from collections import defaultdict
from typing import Final, Optional, Dict, Union, List
from itertools import groupby
import re
import logging

import scrapy
from inline_requests import inline_requests
from scrapy.selector import Selector
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider
from itemloaders import ItemLoader
from itemloaders.processors import TakeFirst
from scrapy.utils.log import configure_logging
from ..items import ThreadItem, PostItem
from ..settings import PHONE2URL
from dataclasses import dataclass


@dataclass
class ThreadAttr:
    thread_id: str = 'thread_id'
    forum: str = 'forum'
    title: str = 'title'
    n_replies: str = 'n_replies'
    date_time: str = 'date_time'
    category = 'category'
    posts = 'posts'
    url = 'url'


@dataclass
class PostAttr:
    post_id: str = 'post_id'
    date_time: str = 'date_time'
    text: str = 'text'
    author: str = 'author'
    date_time: str = 'date_time'
    user_title: str = 'user_title'
    reaction: str = 'reaction'
    floor: str = 'floor'


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
    CssD: Dict[str, Dict] = {
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
    }

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
        except:
            nextp = 2
        if nextp < page_num:
            return f"{thread_root}/page-{nextp}"

    def parse_forum(self, forum_response):

        thread_extractor = LinkExtractor(
            allow=(r'^https:\/\/forum.xda-developers.com\/t'))
        thread_links = thread_extractor.extract_links(forum_response)
        for tlink in thread_links:
            yield scrapy.Request(tlink.url, callback=self.parse_thread)

        forum_page_limit = forum_response.xpath(
            self.XpathD['forum']['page_num']).extract()

        forum_page_limit = int(forum_page_limit[0]) if forum_page_limit else 1
        currp = forum_response.xpath(
            self.XpathD['thread']['curr_page']).extract()
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

    @inline_requests
    def parse_thread(self, response: scrapy.http.Response) -> scrapy.Item:
        """parsing all webpages associated to a post
        Args:
            response (scrapy.http.Response): scrapy response
        """
        # print(f'Parsing thread {response.url}...')

        tl = ItemLoader(item=ThreadItem(), selector=response)
        tl.default_output_processor = TakeFirst()
        forum_name_xpath = f"{self.XpathD['forum']['name']}/text()"


        # tl.add_value(ThreadAttr.thread_id, )

        tl.add_value(ThreadAttr.thread_id, response.url) # items processor takes care 
        tl.add_value(ThreadAttr.url, response.url)
        tl.add_xpath(ThreadAttr.forum, forum_name_xpath)
        tl.add_xpath(ThreadAttr.category, self.XpathD['thread']['category'])
        postElems = response.css(self.CssD['main']['all_posts'])
        for key, search_css in self.CssD['thread'].items():
            tl.add_css(key, search_css)
            self.logger.info(f'key: {key}, search_css: {search_css}, output: {tl.get_output_value(key)}')
        tl.add_value('n_replies', len(postElems))

        posts = []
        for postElem in postElems:
            pl = ItemLoader(item=PostItem(), selector=postElem)
            pl.default_output_processor = TakeFirst()
            for key, search_css in self.CssD['post'].items():
                pl.add_css(key, search_css)
            post_id = pl.get_output_value(PostAttr.post_id)
            floor_xpath = f'//*[@id="js-post-{post_id}"]/div/div[2]/div/header/ul[2]/li[2]/a/text()'
            pl.add_xpath(PostAttr.floor, floor_xpath)
            reaction_url: str = f'https://forum.xda-developers.com/posts/{post_id}/reactions'
            react_response = yield scrapy.Request(reaction_url)
            pl.add_value(PostAttr.reaction,
                         self.parse_reactions(react_response))
            posts.append(dict(pl.load_item()))

        assert (len(posts) == tl.get_output_value('n_replies'),
                f"{len(posts) = } != {tl.get_output_value('n_replies')}")

        posts = [
            next(d) for _, d in groupby(posts, key=lambda _d: _d['floor'])
        ]
        tl.add_value(ThreadAttr.posts, posts)
        thread_page_num = response.xpath(
            self.XpathD['thread']['page_num']).extract()
        thread_page_num = self.get_integer(thread_page_num)
        next_page = self.get_next_url(response.url, thread_page_num)
        if next_page:
            yield scrapy.Request(next_page, callback=self.parse_thread)

        yield tl.load_item()

    def parse_reactions(self,
                        react_response: scrapy.http.Response) -> PostItem:

        reacts = react_response.css(self.CssD['post'][PostAttr.reaction])
        react_string = reacts[-1].extract() if reacts else '(0)'
        pattern = '[^()]+(?=\))'
        reactions = re.search(pattern, react_string).group(0)
        return reactions
