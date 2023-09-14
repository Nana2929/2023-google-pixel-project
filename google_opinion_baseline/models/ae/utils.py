import json
from typing import List, Dict, Union
from os import PathLike


def save_to_json(path: Union[str, PathLike], data: List[Dict]):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)