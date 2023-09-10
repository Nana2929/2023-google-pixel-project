# -*- coding: utf-8 -*-
import re
from collections import defaultdict
from typing import Any, Dict, List

import scrapy
from scrapy.loader import ItemLoader

from ..items import AndroidcentralItem
from ..model import AndroidcentralReplyInfo, AndroidcentralPostInfo


class AndroidSpider(scrapy.Spider):
    name = 'androidcentral'
    allowed_domains = ['androidcentral.com']
    start_urls = [
        'https://forums.androidcentral.com/google-pixel-7-pixel-7-pro/',
        'https://forums.androidcentral.com/google-pixel-6-pixel-6-pro/',
        'https://forums.androidcentral.com/google-pixel-6a/',
        'https://forums.androidcentral.com/google-pixel-5/',
        'https://forums.androidcentral.com/google-pixel-5a/',
        'https://forums.androidcentral.com/google-pixel-4-pixel-4-xl/',
        'https://forums.androidcentral.com/google-pixel-4a-pixel-4a-5g/',
        'https://forums.androidcentral.com/google-pixel-3-pixel-3-xl/',
        'https://forums.androidcentral.com/google-pixel-3a-pixel-3a-xl/',
        'https://forums.androidcentral.com/google-pixel-2-pixel-2-xl/',
        'https://forums.androidcentral.com/google-pixel-pixel-xl/',
    ]

    XPATH = {
        'topic_list': {
            'next_page': (
                '/html/body/div/div/div[@id="page-wrap"]'
                '/section/div/div[@class="main-content"]'
                '/div[@id="below_threadlist"]'
                '/div/form/span[@class="prev_next"][last()]'
                '/a/@href'
            ),
            'root': (
                '/html/body/div/div/div[@id="page-wrap"]'
                '/section/div/div[@class="main-content"]'
                '/div[@id="threadlist"]'
                '/form/div[1]'
                '/div[@id="threads"]'
                '//li'
            ),
            'url': (
                './div[@class="row"]'
                '/div[@class="col-xs-12 col-sm-7"]'
                '/div[@class="i"]'
                '/h1/a/@href'
            ),
            'n_replies': (
                './div[@class="row"]'
                '/div[@class="col-xs-12 col-sm-7"]'
                '/a'
            ),
        },
        'topic': {
            'root': (
                '/html/body/div[@id="site-global-wrap"]'
                '/div/div[@id="page-wrap"]'
                '/section/div/div[@class="main-content"]'
            ),
            'title': (
                './div[@id="breadcrumb"]'
                '/h1/text()'
            ),
            'author': (
                './div[@id="postlist"]'
                '/ol/li[1]'
                '/div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="desc"]'
                '/div[@class="byline"]'
                '/a/text()'
            ),
            'date_time': (
                './div[@id="postlist"]'
                '/ol/li[1]'
                '/div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="time"]'
                '/text()'
            ),
        },
        'posts': {
            'root': (
                '/html/body/div[@id="site-global-wrap"]'
                '/div/div[@id="page-wrap"]'
                '/section/div/div[@class="main-content"]'
                '/div[@id="postlist"]'
                '//ol/li'
            ),
            'url': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="ctrl_right"]'
                '/textarea/text()'
            ),
            'post_id': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="desc"]'
                '/a/@name'
            ),
            'author': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="desc"]'
                '/div[@class="byline"]'
                '/a/text()'
            ),
            'user_title': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="desc"]'
                '/div[@class="byline"]'
                '/div[@class="usertitle"]'
                '/text()'
            ),
            'text': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="desc"]'
                '/div[@class="message"]'
                '/text()'
            ),
            'date_time': (
                './div/div[@class="col-xs-10 p postbody"]'
                '/div[@class="time"]'
                '/text()'
            ),
        },
    }

    def parse(self, response):

        topic_infos = self.parse_topic_list(response=response)
        for topic in topic_infos[1:]:

            yield scrapy.Request(
                url=topic.url,
                callback=self.parse_topic,
                cb_kwargs={'topic_info': topic}
            )

        # if next_page exists, yield a new request
        next_page = response.xpath(self.XPATH['topic_list']['next_page']).get()
        if next_page:
            self.log(f"crawl next page url: {next_page}", 20)
            yield response.follow(
                url=response.urljoin(next_page),
                callback=self.parse
            )

    def parse_topic_list(self, response):

        topic_list = response.xpath(self.XPATH['topic_list']['root'])
        topic_infos = defaultdict(list)

        for topic in topic_list:

            target = 'url'
            xpath = self.XPATH['topic_list'][target]
            extracted = topic.xpath(xpath).get()
            topic_infos[target].append(response.urljoin(extracted))

            target = 'n_replies'
            xpath = self.XPATH['topic_list'][target]
            # TODO: don't know why, but the xpath is always empty while using text()
            extracted = topic.xpath(xpath).get().strip()
            extracted = re.search(r'\d+', extracted)
            topic_infos[target].append(int(extracted.group()))

        self.log(f"prase hub [{topic_infos}]: {response.url}", 30)

        return AndroidcentralPostInfo.from_dict(topic_infos)

    def parse_topic(self, response, topic_info):

        topic_root = response.xpath(self.XPATH['topic']['root'])
        topic_info.update(self.xpath_parse(topic_root, self.XPATH['topic']))

        replies = self.parse_replies(response=response)
        replies = [reply.to_dict() for reply in replies]
        topic_info.posts = replies

        self.log(f"prase post [{len(topic_info.posts)}]: {response.url}", 30)

        yield AndroidcentralItem(**topic_info.to_dict())

    def parse_replies(self, response):

        replies_root = response.xpath(self.XPATH['posts']['root'])
        replies_infos = defaultdict(list)

        for reply in replies_root:

            target = 'url'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(response.urljoin(extracted))

            target = 'post_id'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

            target = 'author'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

            target = 'user_title'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

            target = 'text'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

            target = 'date_time'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

        return AndroidcentralReplyInfo.from_dict(replies_infos)

    @staticmethod
    def xpath_parse(
        root: List[scrapy.Selector],
        xpath_dict: Dict[str, Any],
        *skip_targets
    ) -> Dict[str, List[str]]:

        skip_targets = *skip_targets, 'root'
        infos = {}

        for target, xpath in xpath_dict.items():
            # skip specific targets.
            if target in skip_targets:
                continue

            # assert the xpath is in string type.
            assert isinstance(xpath, str), \
                TypeError(f'xpath_parse: xpath type error: {xpath}')

            infos.update({target: root.xpath(xpath).get()})

        return infos
