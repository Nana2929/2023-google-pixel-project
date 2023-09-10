#%%
import pymongo
from ssh_pymongo import MongoSession
from typing import Final, Optional, Dict, Union
from pathlib import Path
import dotenv

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


