import scrapy
import praw
import re
from datetime import datetime
from typing import List
from ..items import RedditPostItem, RedditCommentsItem


class RedditSpider(scrapy.Spider):
    name = 'reddit'
    allowed_domains = ['www.reddit.com']
    start_urls = ['https://reddit.com/r/Pixel6/']

    def parse(self, response):
        self.reddit = praw.Reddit(
            client_id='9XvxijsddKfkJdkcfVrDkQ',
            client_secret='tEYASVdkbXeSBZusB9O7ZtLpOrctPg',
            user_agent='my_bot/0.0.1'
        )

        subs = self.reddit.subreddit('Pixel6').hot(
            limit=20)  # get hot posts from subreddit Pixel6
        for sub in subs:
            yield scrapy.Request(f'https://www.reddit.com{sub.url}' if sub.url.startswith('/r') else sub.url,
                                 self._parse_post, meta={'title': sub.title, 'url': sub.url})

    def _parse_post(self, response: scrapy.http.Response):
        title = response.meta['title']
        url = response.meta['url']
        post_text = response.css('p._1qeIAgB0cPwnLhDF9XSiJM::text').getall()

        sub_id = re.split('/|comments', url)[-3]
        sub = self.reddit.submission(id=sub_id)
        sub.comments.replace_more(limit=1000)
        comments = self._parse_comments(sub.comments.list())
        tag = sub.link_flair_richtext[0]['t'] if len(
            sub.link_flair_richtext) != 0 else ''
        post_date = datetime.fromtimestamp(sub.created)
        post_date = post_date.strftime('%Y-%m-%d')

        post = RedditPostItem(thread_id=sub_id, forum=sub.subreddit.display_name, url=url,
                              title=title, tag=tag, author=sub.author.name,
                              post_text=''.join(post_text),
                              datetime=post_date, upvote_ratio=sub.upvote_ratio, comments=comments)

        yield post

    def _parse_comments(self, comments: List['praw.models.Comment']):
        if len(comments) == 0:
            return []

        comment_list = []
        for c in comments:
            if c.body in ('\'[removed]\'', '\'[deleted]\'', '[removed]', '[deleted]'):
                continue  # if the comment is deleted, we can't get its info

            com_date = datetime.fromtimestamp(c.created_utc)
            com_date = com_date.strftime('%Y-%m-%d')
            com = RedditCommentsItem(
                author=c.author.name, com_text=c.body, datetime=com_date)

            comment_list.append(com)

        return comment_list
