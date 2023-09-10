
from pathlib import Path
import logging
import fire
from utils import load_json, save_json
from constants import (PROMPT_SWITCH,
                       ASPECT_CATEGORIES,
                       TARGET_CATEORIES,
                       SENTIMENT_POLARITIES)

logging.basicConfig(level=logging.INFO)
ROOT ="google_opinion_annotation/data/annotations_backup/20230730-16/"
DATA_DIR = ROOT + "processed"
OUTPUT_DIR = ROOT + "converted"
INF_FILE_PREFIX = "PATS_for_inference"
EVAL_FILE_PREFIX = "PATS_for_eval"

def convert_dialogs(dialogs: list):
    for dialog in dialogs:
        annotations = dialog["annotations"]
        ls_id = dialog["id"]
        mongo_id = dialog["data"]["id"]
        posts = dialog["data"]["posts"]; dialog_text = convert_dialog2text(posts)
        yield {"ls_id": ls_id, "mongo_id": mongo_id, "text": dialog_text, "annotations": annotations}


def convert_dialog2text(posts: list):
    """
    Convert a list of posts to a dialog string
    Convert the author names to A, B, C, ... following their order of appearance
    Parameters
    ----------
    posts : list
        _description_

    Returns
    -------
    string
    """
    author_set = set()
    for p in posts:
        author_set.add(p["author"])
    author_map = {a: chr(i+65) for i, a in enumerate(author_set)}
    dialog_text = []
    for p in posts:
        dialog_text.append(author_map[p["author"]] + ": " + p["text"])
    return "\n".join(dialog_text)

def main(data_dir:Path = DATA_DIR,
         output_dir:Path = OUTPUT_DIR,
         inf_file_prefix:str = INF_FILE_PREFIX,
         eval_file_prefix:str = EVAL_FILE_PREFIX,
         prompt_format:str = "AS"):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)


    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = [f for f in data_dir.iterdir() if f.suffix == ".json"]
    logging.info(input_files)
    input_jsons = [load_json(f) for f in input_files]
    prompt_template = PROMPT_SWITCH.get(prompt_format, None)
    if prompt_template is None:
        raise ValueError(f"Invalid prompt format: {prompt_format}, choose from {PROMPT_SWITCH.keys()}")

    eval_file = []
    inf_file = []
    for i, input_json in enumerate(input_jsons):
        for convert_d in convert_dialogs(input_json):
            convert_d["text"] = prompt_template.format(text=convert_d["text"],
                                                         target_categories=TARGET_CATEORIES,
                                                         aspect_categories=ASPECT_CATEGORIES,
                                                         sentiment_polarities=SENTIMENT_POLARITIES)
            eval_file.append(convert_d)
            inf_d = {
                'text': convert_d["text"],
            }
            inf_file.append(inf_d)
        logging.info(f"Converted {len(input_json)} dialogs from {input_files[i]}.")
    eval_filename = eval_file_prefix + "_" + prompt_format + ".json"
    inf_filename = inf_file_prefix + "_" + prompt_format + ".json"
    save_json(eval_file, output_dir / eval_filename)
    save_json(inf_file, output_dir / inf_filename)
    logging.info(f"Saved {len(eval_file)} dialogs to {output_dir / eval_filename}")


if __name__ == "__main__":
    fire.Fire(main)







