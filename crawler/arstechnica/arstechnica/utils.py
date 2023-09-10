import re

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Final, List, Optional, Union

import dotenv
import pymongo
from ssh_pymongo import MongoSession


class ArsTechnicaFieldProcessor(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def format_title(title):

        return title.strip()

    @staticmethod
    def format_time(text):

        try:
            date = datetime.strptime(text, '%b %d, %Y')
            date = date.strftime('%Y-%m-%d')
        except ValueError:
            day = text.split()[0]
            if day == 'Today':
                date = datetime.now().strftime('%Y-%m-%d')
            elif day == 'Yesterday':
                date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                date = datetime.strptime(text, "%A at %I:%M %p")
                date = date.strftime('%Y-%m-%d')

        return date

    @staticmethod
    def format_replies(item):

        for reply in item['posts']:

            try:
                pattern = re.compile(r'.*posts/([\d]+)/bookmark')
                reply['post_id'] = re.match(pattern, reply['post_id'])
                reply['post_id'] = reply['post_id'].group(1)
            except Exception:
                item['posts'].remove(reply)
                continue

            reply['floor'] = reply['floor'].strip().strip('#')

            reply['user_title'] = reply['user_title'].strip()
            if reply['user_title'] == '':
                reply['user_title'] = None

            reply['quote'] = reply['quote'].strip() if reply['quote'] else None

            try:
                reply['text'] = reply['text'].strip()
            except AttributeError:
                # some replies are empty or just paste image
                reply['text'] = None

            try:
                date = datetime.strptime(reply['date_time'], '%b %d, %Y')
                reply['date_time'] = date.strftime('%Y-%m-%d')
            except ValueError:
                day = reply['date_time'].split()[0]
                if day == 'Today':
                    reply['date_time'] = datetime.now().strftime('%Y-%m-%d')
                elif day == 'Yesterday':
                    reply['date_time'] = (
                        datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    date = datetime.strptime(reply['date_time'],
                                             "%A at %I:%M %p")
                    reply['date_time'] = date.strftime('%Y-%m-%d')

        return item['posts']


class MongoDB:

    @classmethod
    def by_env(cls,
               env_file_path: Union[str, Path],
               database: Optional[str] = None) -> pymongo.MongoClient:
        """Initialize `MongoClient` by specific env file

        Args:
            env_file_path (Union[str, Path]): The env file path to read.
            database (Optional[str], optional): Specify the database. Defaults to None.
        """
        connection = cls.__setup_ssh(env_file_path)
        return connection

    def __setup_ssh(env_file_path: Union[str, Path],) -> pymongo.MongoClient:
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
