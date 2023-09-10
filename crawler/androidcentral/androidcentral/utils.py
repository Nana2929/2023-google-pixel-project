import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Final, List, Optional, Union

import dotenv
import pymongo
from ssh_pymongo import MongoSession


class AndroidcentralFieldProcessor(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def format_forum(url) -> str:

        pattern = re.compile(r'.*/google-(\w+)-(\w+)-?(\w+)?-?(\w+)?-?(\w+)?/.*')
        try:
            forum = pattern.findall(url)[0]
        except IndexError:
            return None

        forum = ' '.join(forum)
        return forum.strip()

    @staticmethod
    def format_title(title) -> str:

        return title.strip()

    @staticmethod
    def format_time(text) -> str:

        text = text.strip().split()
        try:
            date = datetime.strptime(text[0], '%m-%d-%Y')
        except ValueError:
            if text[0] == 'Today':
                date = datetime.now()
            elif text[0] == 'Yesterday':
                date = datetime.now() - timedelta(days=1)

        return date.strftime('%Y-%m-%d')

    @staticmethod
    def format_replies(item) -> List[Dict[str, str]]:

        for reply in item['posts']:

            try:
                reply['post_id'] = reply['post_id'].replace('post', '')
            except AttributeError:
                item['posts'].remove(reply)
                continue

            if reply['user_title'].strip() == '':
                reply['user_title'] = None

            try:
                reply['text'] = reply['text'].strip()
            except AttributeError:
                # some replies are empty or just paste image
                reply['text'] = None

            try:

                date = reply['date_time'].split()
                date = datetime.strptime(date[0], '%m-%d-%Y')
                reply['date_time'] = date.strftime('%Y-%m-%d')

            except ValueError:

                if reply['date_time'] == 'Today':
                    reply['date_time'] = datetime.now().strftime('%Y-%m-%d')
                elif reply['date_time'] == 'Yesterday':
                    reply['date_time'] = (
                        datetime.now() - timedelta(days=1)
                    ).strftime('%Y-%m-%d')
                else:
                    reply['date_time'] = None

        return item['posts']


class MongoDB:
    @classmethod
    def by_env(
        cls,
        env_file_path: Union[str, Path],
        database: Optional[str] = None
    ) -> pymongo.MongoClient:
        """Initialize `MongoClient` by specific env file

        Args:
            env_file_path (Union[str, Path]): The env file path to read.
            database (Optional[str], optional): Specify the database. Defaults to None.
        """
        connection = cls.__setup_ssh(env_file_path)
        return connection

    def __setup_ssh(
        env_file_path: Union[str, Path],
    ) -> pymongo.MongoClient:
        """initialization mongodb connection with ssh session

        Args:
            env_file_path (Union[str, Path]): The env file path to read.
        Returns:
            pymongo.MongoClient: The MongoClient object.
        """
        env: Dict[str, str] = dotenv.dotenv_values(env_file_path)

        HOST: Final[str] = env.get('HOST')
        PORT: Final[str] = env.get('PORT')
        USER: Final[str] = env.get('USER')
        PASSWORD: Final[Optional[str]] = env.get('PASSWORD', None)
        SSH_HOST: Final[str] = env.get('SSH_HOST')
        SSH_PORT: Final[int] = int(env.get('SSH_PORT'))

        session = MongoSession(
            host=SSH_HOST,
            port=SSH_PORT,
            user=USER,
            password=PASSWORD,
            uri=f'mongodb://{HOST}:{PORT}',
        )

        return session.connection
