# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Any, Dict, List

import scrapy

from ..items import ArstechnicaItem
from ..model import ArstechnicaReplyInfo, ArstechnicaTopicInfo


class ArstechnicaSpider(scrapy.Spider):
    name = 'arstechnica'
    allowed_domains = ['arstechnica.com']
    domain = 'https://arstechnica.com'
    start_urls = [
        'https://arstechnica.com/civis/forums/mobile-computing-outpost.9/'
    ]

    XPATH = {
        'topic_list': {
            'next_page': ('//div[@class="p-body-main  "]'
                          '//a[@class="pageNav-jump pageNav-jump--next"]'
                          '/@href'),
            'root':
                ('//div[re:test(@class,"structItem structItem--thread.+")]'),
            # url include thread_id, for example:
            # https://arstechnica.com/civis/viewtopic.php?f=9&t=1660000
            # thread_id = 1660000
            'url': ('./div[2]/div[@class="structItem-title"]'
                    '/a/@href'),
            'n_replies': ('./div[3]/dl[@class="pairs pairs--justified"]'
                          '/dd/a/text()'),
            'n_views':
                ('./div[3]/dl[@class="pairs pairs--justified structItem-minor"]'
                 '/dd/text()'),
        },
        'topic': {
            'root': ('//div[@class="uix_contentWrapper"]'),
            'title': ('.//div[@class="p-title "]/h1/text()'),
            'author': ('.//div[@class="block-body js-replyNewMessageContainer"]'
                       '//article[1]'
                       '//div[@class="message-cell message-cell--user"]'
                       '//div[@class="uix_messagePostBitWrapper"]'
                       '//div[@class="message-userDetails"]'
                       '/h4/a/text()'),
            'date_time': (
                './/div[@class="block-body js-replyNewMessageContainer"]'
                '//article[1]'
                '//div[@class="message-cell message-cell--main"]'
                '//header[@class="message-attribution message-attribution--split"]'
                '//ul[@class="message-attribution-main listInline "]'
                '/li/a/time/text()'),
        },
        'posts': {
            'root': ('//article[re:test(@class,"message    .+")]'),
            'url': (
                './/div[@class="message-cell message-cell--main"]'
                '//header[@class="message-attribution message-attribution--split"]'
                '//ul[@class="message-attribution-opposite message-attribution-opposite--list "]'
                '/li[3]/a/@href'),
            'post_id': (
                './/div[@class="message-cell message-cell--main"]'
                '//header[@class="message-attribution message-attribution--split"]'
                '//ul[@class="message-attribution-opposite message-attribution-opposite--list "]'
                '/li[1]/a/@href'),
            'floor': (
                './/div[@class="message-cell message-cell--main"]'
                '//header[@class="message-attribution message-attribution--split"]'
                '//ul[@class="message-attribution-opposite message-attribution-opposite--list "]'
                '/li[3]/a/text()'),
            'author': ('.//div[@class="message-cell message-cell--user"]'
                       '//div[@class="uix_messagePostBitWrapper"]'
                       '//div[@class="message-userDetails"]'
                       '/h4/a/text()'),
            'user_title': ('.//div[@class="message-cell message-cell--user"]'
                           '//div[@class="uix_messagePostBitWrapper"]'
                           '//div[@class="message-userDetails"]'
                           '/h5/text()'),
            'quote_link': ('.//div[@class="bbWrapper"]'
                           '//div[@class="bbCodeBlock-title"]'
                           '/a/@href'),
            'quote':
                ('.//div[@class="bbCodeBlock-content"]'
                 '//div[@class="bbCodeBlock-expandContent js-expandContent "]'
                 '/text()'),
            'text': ('.//div[@class="bbWrapper"]/text()'),
            'date_time': (
                './/div[@class="message-cell message-cell--main"]'
                '//header[@class="message-attribution message-attribution--split"]'
                '//ul[@class="message-attribution-main listInline "]'
                '/li/a/time/text()'),
        },
    }

    def parse(self, response):

        topic_infos = self.parse_topic_list(response=response)
        for topic in topic_infos:

            request = scrapy.Request(url=topic.url,
                                     callback=self.parse_topic,
                                     cb_kwargs={'topic_info': topic})
            yield request

        # if next_page exists, yield a new request
        next_page = response.xpath(self.XPATH['topic_list']['next_page']).get()
        if next_page:
            self.log(f"crawl next page url: {next_page}", 20)
            yield response.follow(url=self.domain + (next_page),
                                  callback=self.parse)

    def parse_topic_list(self, response):

        topic_list = response.xpath(self.XPATH['topic_list']['root'])
        topic_infos = defaultdict(list)

        for topic in topic_list:

            target = 'url'
            xpath = self.XPATH['topic_list'][target]
            extracted = topic.xpath(xpath).get()
            topic_infos[target].append(self.domain + extracted)

            target = 'n_replies'
            xpath = self.XPATH['topic_list'][target]
            extracted = topic.xpath(xpath).get()
            topic_infos[target].append(extracted)

            target = 'n_views'
            xpath = self.XPATH['topic_list'][target]
            extracted = topic.xpath(xpath).get()
            topic_infos[target].append(extracted)

        self.log(f"prase hub [{topic_infos}]: {response.url}", 30)

        return ArstechnicaTopicInfo.from_dict(topic_infos)

    def parse_topic(self, response, topic_info):

        topic_root = response.xpath(self.XPATH['topic']['root'])
        topic_info.update(self.xpath_parse(topic_root, self.XPATH['topic']))

        replies = self.parse_replies(response=response)
        replies = [reply.to_dict() for reply in replies]
        topic_info.posts = replies

        self.log(f"prase post [{len(topic_info.posts)}]: {response.url}", 30)

        yield ArstechnicaItem(**topic_info.to_dict())

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

            target = 'floor'
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

            target = 'quote_link'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            if extracted:
                extracted = response.urljoin(extracted)
            replies_infos[target].append(extracted)

            target = 'quote'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

            target = 'text'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get() or ''
            extracted = ''.join(extracted).strip()
            replies_infos[target].append(extracted)

            target = 'date_time'
            xpath = self.XPATH['posts'][target]
            extracted = reply.xpath(xpath).get()
            replies_infos[target].append(extracted)

        return ArstechnicaReplyInfo.from_dict(replies_infos)

    @staticmethod
    def xpath_parse(root: List[scrapy.Selector], xpath_dict: Dict[str, Any],
                    *skip_targets) -> Dict[str, List[str]]:

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
