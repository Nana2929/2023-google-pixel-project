from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict


# process item完後進structs
# https://github.com/yiting-tom/GoogleOpinionScrapy/blob/master/crawler-system/pttCrawlerSystem/model.py
# https://github.com/lopentu/casa/blob/main/src/casa/opinion_types.py

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

@dataclass(frozen=True)
class User:
    name: str = ''
    title: str = ''
    id: str = ''
    link: str = ''

    def __repr__(self):
        return f'XDA User{{user_id: {self.user_id} name: {self.name}, title:{self.title}}}'

# @dataclass(frozen=True)
# class Reply:
#     author: Dict[str:str] = {}
#     post_id: str = ''
#     date: str = ''
#     time: str = ''
#     text: str  = ''
#     floor: int = 0
#     def __post_init__(self):
#         self.author = User(self.author)
#     def __repr__(self):
#         return f'XDA Reply{{post_id: {self.post_id}, author: {self.author.name}, text: {self.text[:20]}}}'

# @dataclass(frozen=True)
# class Post:
#     id: str
#     title: str
#     category: str
#     author: Dict[str:str]
#     text: str
#     date: str
#     time: str
#     floor: 0
#     replies: List[Reply]

#     def __post_init__(self):
#         self.author = User(self.author)

#     def __repr__(self):
#         return f'XDA Post{{id: {self.id}, author: {self.author.name}, title: {self.title}}}'

#     def add_reply(self, **kwargs):
#         self.replies.append(Reply(**kwargs))




