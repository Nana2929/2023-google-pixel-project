from pathlib import Path
from collections import defaultdict
import logging
import fire
from utils import load_json, save_json
from constants import (
    PROMPT_SWITCH,
    ASPECT_CATEGORIES,
    TARGET_CATEORIES,
    SENTIMENT_POLARITIES,
)

logging.basicConfig(level=logging.INFO)
ROOT = Path("pats")
DATA_DIR = ROOT / "processed"
OUTPUT_DIR = ROOT / "converted"
INF_FILE_PREFIX = "PATS_for_inference"
EVAL_FILE_PREFIX = "PATS_for_eval"


def get_relevant_posts(*elems):
    """
    Find the relevant posts' indices for completing the triplets
    """
    res = set()
    for elem in elems:
        if elem is None:
            continue
        res.add(int(elem["utterance_from"]))
        res.add(int(elem["utterance_to"]))
    res = list(res)
    res.sort()
    return res


def map_author(posts: list) -> tuple:
    """_summary_

    Parameters
    ----------
    posts : list
        _description_

    Returns
    -------
    tuple
        dict: author_map, {'Edwin': A, 'Lily': B, ...}
        list: author_list, list of author chars correspond to posts, e.g. [A, B, C, A, B, A, C, ...]
    """
    author_set = set()
    for p in posts:
        author_set.add(p["author"])
    author_map = {a: chr(i + 65) for i, a in enumerate(author_set)}
    author_list = [author_map[p["author"]] for p in posts]  # list of author chars
    return author_map, author_list


def get_enclosed_posts(posts: list, *elems):
    # check all sentences' where to put which bracket
    lparen_dict = defaultdict(list)
    rparen_dict = defaultdict(list)

    for elem in elems:
        if elem is None:
            continue
        elem_from = elem["from"]
        elem_to = elem["to"]
        elem_utt_from = int(elem["utterance_from"])
        elem_utt_to = int(elem["utterance_to"])
        lparen_dict[elem_utt_from].append(int(elem_from))
        rparen_dict[elem_utt_to].append(int(elem_to))

    # add brackets to texts
    all_paren_dict = defaultdict(list)
    for i, locs in lparen_dict.items():
        all_paren_dict[i].extend((loc, "<") for loc in locs)
    for i, locs in rparen_dict.items():
        all_paren_dict[i].extend((loc, ">") for loc in locs)
    # sort by first element
    for i, locs in all_paren_dict.items():
        all_paren_dict[i] = sorted(locs, key=lambda x: -x[0])
    # add paren
    for i, locs in all_paren_dict.items():
        for loc, paren in locs:
            posts[i]["text"] = posts[i]["text"][:loc] + paren + posts[i]["text"][loc:]
    return posts


def convert_alsc_text(posts: list, annotations: list, version: int = 1):
    """
    1. Map the speaker to A, B, C, D
    2. find the utterances (posts) indices for completing the triplets
    3. Version 1: Use bracket <> to enclose the target term and aspect term.
       Version 2: Do nothing
    """
    # use annotations
    annotation = annotations[0]  # assume all data is labeled by 1 person
    triplets = annotation["triplets"]
    _, author_list = map_author(posts)

    for t in triplets:
        # suppose all span is restricted to 1 utterance
        tgt = t["target"]
        tgt_sid = int(tgt["utterance_from"])
        opn = t["opinion"]
        opn_sid = int(opn["utterance_from"])
        if "aspect" in t:
            asp = t["aspect"]
            asp_sid = int(asp["utterance_from"])
        else:
            asp = None
            asp_sid = None
        relevant_post_ids = get_relevant_posts(tgt, asp, opn)

        # revise t triplet's target and aspect position
        # in full dialogue text
        if version == 1:
            new_posts = get_enclosed_posts(posts, tgt, asp)  # no opn!!!
            # note that new_posts here will have wrong indices. Fix it later or just leave them because enclosed version for Senti-llama does not need them
        else:
            new_posts = posts.copy()

        revised_t = t.copy()
        dialog_text = []
        # ======== make dict =========

        offset_map = {}
        offset = 0
        for s in relevant_post_ids:
            offset_map[s] = offset
            offset += len(new_posts[s]["text"]) + 1  # +1 for \n
        id2spans = defaultdict(list)
        for x, id in enumerate([tgt_sid, asp_sid, opn_sid]):
            if id is None:
                continue
            if x == 0:
                id2spans[id].append(("target", tgt["from"], tgt["to"]))
            elif x == 1:
                id2spans[id].append(("aspect", asp["from"], asp["to"]))
            elif x == 2:
                id2spans[id].append(("opinion", opn["from"], opn["to"]))

        new_spans = {}
        for id in relevant_post_ids:
            text = new_posts[id]["text"]
            author = author_list[id]
            prefixed_sent = f"{author}: {text}"
            utt_offset = offset_map[id]

            for name, start, end in id2spans[id]:
                final_start = start + utt_offset + len(prefixed_sent) - len(text)
                final_end = end + utt_offset + len(prefixed_sent) - len(text)
                new_spans[name] = (final_start, final_end)
            dialog_text.append(prefixed_sent)

            # accumulate text length before this sentence
            # and an extra "\n"

        if "aspect" in new_spans:
            revised_t["aspect"]["from"] = new_spans["aspect"][0]
            revised_t["aspect"]["to"] = new_spans["aspect"][1]
        revised_t["target"]["from"] = new_spans["target"][0]
        revised_t["target"]["to"] = new_spans["target"][1]
        revised_t["opinion"]["from"] = new_spans["opinion"][0]
        revised_t["opinion"]["to"] = new_spans["opinion"][1]
        dialog_text = "\n".join(dialog_text)

        # FIX: revise triplet index in dialogue text

        yield dialog_text, revised_t


def convert_dialogs(dialogs: list, version: int) -> dict:
    global mismatch_count
    mismatch_count = 0
    for dialog in dialogs:
        annotations = dialog["annotations"]
        ls_id = dialog["id"]
        mongo_id = dialog["data"]["id"]
        posts = dialog["data"]["posts"]
        for dialog_text, revised_t in convert_alsc_text(
            posts=posts, annotations=annotations, version=version
        ):
            # check span,
            if version == 2:
                target = revised_t["target"]
                tgt_s, tgt_t = target["from"], target["to"]
                tgt_text = dialog_text[tgt_s:tgt_t]
                logging.info(
                    f"after preprocessing: {tgt_text}, annotated: {target['text']}"
                )
                if tgt_text != target["text"]:
                    mismatch_count += 1
                    logging.info(
                        f"[Target text mismatch] current record: \"{tgt_text}\" vs  annotated: \"{target['text']}\""
                    )
            yield {
                "ls_id": ls_id,
                "mongo_id": mongo_id,
                "original_text": dialog_text,
                "annotation": revised_t,
            }


def main(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    inf_file_prefix: str = INF_FILE_PREFIX,
    eval_file_prefix: str = EVAL_FILE_PREFIX,
    version: int = 1,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    prompt_format = "ALSC"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = [f for f in data_dir.iterdir() if f.suffix == ".json"]
    input_jsons = [load_json(f) for f in input_files]
    prompt_template = PROMPT_SWITCH.get(prompt_format, None)
    logging.info(f"=== Conversion Script for {prompt_format} ===")
    logging.info(f"Prompt Template: {prompt_template}")
    logging.info(f"Input Dir: {data_dir}")
    logging.info(f"Input Files: {input_files}")
    logging.info(f"Use enclosed or not: {version == 1}")

    eval_file = []
    inf_file = []
    for i, input_json in enumerate(input_jsons):
        cnt = 0
        for convert_d in convert_dialogs(input_json, version=version):
            cnt += 1
            if version == 1:
                convert_d["prompted_text"] = prompt_template.format(
                    text=convert_d["original_text"],
                    target_categories=TARGET_CATEORIES,
                    aspect_categories=ASPECT_CATEGORIES,
                    sentiment_polarities=SENTIMENT_POLARITIES,
                )
            else:
                convert_d["text"] = convert_d["original_text"]
            eval_file.append(convert_d)

            if version == 1:
                inf_d = {
                    "text": convert_d["prompted_text"],
                }
                inf_file.append(inf_d)
        logging.info(
            f"Converted {len(input_json)} dialogs to {cnt} examples from {input_files[i]}."
        )
        logging.info(f"Total mismatch count: {mismatch_count}")
    version_mapping = {
        1: "SentiLLaMA",
        2: "RobertaABSA",
    }
    eval_filename = (
        eval_file_prefix
        + "_"
        + prompt_format
        + f"_for_{version_mapping[version]}"
        + ".json"
    )
    save_json(eval_file, output_dir / eval_filename)
    if version == 1:
        inf_filename = (
            inf_file_prefix
            + "_"
            + prompt_format
            + f"_for_{version_mapping[version]}"
            + ".json"
        )
        save_json(inf_file, output_dir / inf_filename)
    logging.info(f"Saved {len(eval_file)} dialogs to {output_dir / eval_filename}")


if __name__ == "__main__":
    """
    Version 1: for Senti-LLaMA
    Version 2: for RobertaABSA
    "prompted_text" does not align with the indices,
    "original_text" does.
    "text" in `for_SentiLLaMA` uses `prompted_text`
    "text" in `for_RobertaABSA` uses `original_text`
    """
    fire.Fire(main)
