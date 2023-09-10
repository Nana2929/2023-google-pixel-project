
import json 

def save_json(data: dict, filepath: str):
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data