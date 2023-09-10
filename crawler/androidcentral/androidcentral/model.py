# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
# why frozen=True?
# see the following link for more details.
# https://stackoverflow.com/a/66195017
class AndroidcentralUserInfo(object):
    uid: int
    username: str = ''
    title: str = ''
    link: str = ''


@dataclass(frozen=True)
class AndroidcentralReplyInfo(object):
    # thread_id: int
    post_id: int
    url: str = ''
    author: str = ''
    user_title: str = ''
    text: str = ''
    date_time: str = ''

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]):
        return [
            cls(**dict(zip(d.keys(), tup)))
            for tup in zip(*d.values())
        ]

    def to_dict(self):
        return self.__dict__

    def update(self, update_dict: dict = None, **kwargs):
        if update_dict:
            self.__dict__.update(update_dict)
        self.__dict__.update(kwargs)


@dataclass(frozen=False)
# here we let frozen=False because we want to update the replies field
class AndroidcentralPostInfo(object):
    # thread_id: int
    forum: str = 'Android Central'
    url: str = ''
    title: str = ''
    author: str = ''
    date_time: str = ''
    n_replies: int = 0
    posts: List[Dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]):
        return [
            cls(**dict(zip(d.keys(), tup)))
            for tup in zip(*d.values())
        ]

    def to_dict(self):
        return self.__dict__

    def update(self, update_dict: dict = None, **kwargs):
        if update_dict:
            self.__dict__.update(update_dict)
        self.__dict__.update(kwargs)
