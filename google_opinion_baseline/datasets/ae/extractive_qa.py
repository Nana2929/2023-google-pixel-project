# %%
from typing import Union, List, Dict, Any
from copy import deepcopy
import os
import json
from transformers.tokenization_utils_base import PaddingStrategy

import torch
from pathlib import Path
import logging
import torch.utils.data as data
from transformers import AutoTokenizer
import glob

# https://lightning-transformers.readthedocs.io/en/latest/tasks/nlp/question_answering.html

logging.basicConfig(level=logging.INFO, format='[%(module)s] %(message)s')


class ExtractiveQa(data.Dataset):
    """The dataset based on torch.utils.data.Dataset for turning sentiment aspect extraction dataset to extractive QA.

    The dataset instance, will later be instantialized as a data module by a data interface following
    https://github.com/miracleyoo/pytorch-lightning-template.


    Attributes
    ----------
    rpad : Bool
        Whether to pad the right side of the context, depending on the tokenizer.
    datas: List[Dict[str, Any]]
        The list of dictionaries that contains the data.
        Each data has a (text, aspect, sentiment) tuple or a (text, aspect) pair to be extracted.
        If the dataset is for testing, each data's `aspects` key needs to present, but the value can be an empty list.::
            {
                'aspects': [],...
            }
    process_datas: List[Dict[str, Any]]
        The list of dictionaries that contains the processed data.
        The output of `self.process_QA()` on `self.datas`.
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 base_name: str,
                 max_length: int = 256,
                 doc_stride: int = 128,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 expand_aspects: bool = False,
                 keep_keys: List[str] = ['id', 'context', 'answers'],
                 template: str = 'What is the aspect of this review?',
                 stage: str = 'train') -> None:
        """ Initialize the dataset with the given data_dir and tokenizer.

        Attributes
        ----------
        data_dir : Union[str, os.PathLike]
            The directory of the dataset that contains the json files,
            in the form of *_train.json and *_test.json, case insensitive.
        expand_aspects : bool, optional
            Whether to expand a data with multiple aspect terms to multiple data with single aspect term,
            by default False.
        base_name : str
            The tokenizer name sed to tokenize the text,
            can be the name of the pretrained tokenizer or the tokenizer itself.
        max_length : int, optional
            Tokenizer args. The maximum length of the input sequence (after concatenation), by default 256
        doc_stride : int, optional
            Tokenizer args. The stride (overlap) used when the context is too large and is split across several features, by default 128
        padding : Union[bool, str, PaddingStrategy], optional
            Tokenizer args. The padding strategy used, by default True. See https://huggingface.co/docs/transformers/pad_truncation for more.
        keep_keys : List[str], optional
            The keys (in the json dictionaries) to keep in the train batch (for reference after model output), by default ['id', 'context', 'answers'].
        template : str, optional
            The instruction template (question in QA) to used to elicit the desired aspect, by default 'What is the aspect of this review?'
        stage: str, optional
            The stage of the dataset ('train', 'validation', 'test'), by default 'train'.
            If 'train', the dataset will load the *_train.json files, otherwise, it will load the *_test.json files.

        """

        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_name)
        self.expand_aspects = expand_aspects
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.padding = padding
        # : Bool: True if the tokenizer pads on the right, False otherwise.
        self.rpad = self.tokenizer.padding_side == "right"
        self.keep_keys = keep_keys
        self.template = template
        self.stage = stage

        self.datas = self.read_file()
        # : List[Dict]: The list of dictionaries that contains the processed data. Each dictionary is a QA pair.
        self.processed_datas = []
        for d in self.datas:
            self.processed_datas += self.process_QA(d)
        logging.info(
            f'Loaded {len(self.processed_datas)} QA pairs from {len(self.datas)} {stage} data.'
        )

    def read_file(self) -> List[Dict]:
        """Read in json files from data directory and return a list of dictionaries (raw data).

        Returns
        -------
            List[Dict]:
                The list of dictionaries that contains the data.

        Examples
        --------
        >>> d = ExtractiveQa('data/ae', 'bert-base-uncased')
        >>> print(dataset.read_file()[0])
        {
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.'
            'aspects':
                [
                    {'term': ['cord'],
                    'polarity': 'neutral',
                    'from': 41,
                    'to': 45},
                    {'term': ['battery', 'life'],
                    'polarity': 'positive',
                    'from': 74,
                    'to': 86}
                ]
            ,...
        }
        """

        filepaths = glob.glob(os.path.join(self.data_dir, "*.json"))
        datas = []

        for filepath in filepaths:
            file_stem = Path(filepath).stem
            file_suffix = file_stem.split('_')[-1].lower()
            with open(filepath) as f:
                # reddit_train.json -> train
                if file_suffix == self.stage:
                    logging.info(f'loading {file_stem}.json')
                    datas += json.load(f)

        return datas

    def __len__(self) -> int:
        """Return dataset size.

        Returns
        -------
        int
            The number of data (the number of QA sequences) in the dataset.
        """
        return len(self.processed_datas)

    def process_QA(self, d: Dict) -> List[Dict]:
        """Process a raw dictionary (raw data) into QA dictionaries.

        Parameters
        ----------
        d : Dict
            The dictionary that contains the raw data.
            1 data can have more than 1 aspect.

        Returns
        -------
        List[Dict]
            The list of dictionaries that contains the processed data.
            Each QA pair is paired with 1 aspect,
            and therefore the length of the returned list is the number of aspects in the raw data.
            Each dictionary contains a QA pair. The processed data follows the squad dataset format, as can be seen in https://huggingface.co/docs/transformers/tasks/question_answering.
            Note that extra keys in the input dictionary will be preserved.
        Examples
        --------
        >>> d =  {
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.'
            'aspects':
                [
                    {'term': ['cord'],
                    'polarity': 'neutral',
                    'from': 41,
                    'to': 45},
                    {'term': ['battery', 'life'],
                    'polarity': 'positive',
                    'from': 74,
                    'to': 86}
                ]
        }
        >>> print(self.process_QA(d)) # expand_aspects = True; otherwise trim to the first aspect
        [
            {   'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
                'aspects': [
                    {'term': ['cord'], 'polarity': 'neutral', 'from': 41, 'to': 45},
                    {'term': ['battery', 'life'], 'polarity': 'positive', 'from': 74, 'to': 86}],
                'question': 'What is the aspect of this review?',
                'context': 'I charge it at night and skip taking the cord with me because of the good battery life.',
                'answers': {
                    'text': ['cord'],
                    'answer_start': [41],
                    'answer_end': [45]}},
            {   'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
                'aspects': [
                    {'term': ['cord'], 'polarity': 'neutral', 'from': 41, 'to': 45},
                    {'term': ['battery', 'life'], 'polarity': 'positive', 'from': 74, 'to': 86}],
                'question': 'What is the aspect of this review?',
                'context': 'I charge it at night and skip taking the cord with me because of the good battery life.',
                'answers': {
                    'text': ['battery life'],
                    'answer_start': [74],
                    'answer_end': [86]},}
        ]
        """
        qa_datas = []
        qa_data = deepcopy(d)
        qa_data['question'] = self.template
        qa_data['context'] = d['text']

        if self.stage == 'test':
            return [qa_data]  # no answer for test set
        if len(d['aspects']) == 0:
            # fix the issue that 422 validation examples have 11 examples without aspects
            # and 1692 training examples with 26 examples without aspects
            qa_data = deepcopy(qa_data)
            qa_data['answers'] = {
                'text': [''],
                'answer_start': [],
                'answer_end': []
            }
            return [qa_data]

        for aid in range(len(d['aspects'])):
            qa_data_ = deepcopy(qa_data)
            qa_data_['answers'] = {
                'text': [' '.join(d['aspects'][aid]['term'])],
                'answer_start': [d['aspects'][aid]['from']],
                'answer_end': [d['aspects'][aid]['to']]
            }
            qa_datas.append(qa_data_)
            if not self.expand_aspects:
                break  # only use the first aspect

        return qa_datas

    def __getitem__(self, index) -> Dict[str, Any]:
        """Return a processed data example.

        Parameters
        ----------
        index : int
            The index of the data example.
        Returns
        -------
        Dict[str,Any]
            the indexed example with `id` key if not already exists.
            This helps match back the example and the prediction answer.
        """
        data = self.processed_datas[index]
        if "id" not in data:  # a unique identifier (str, int, ...) for the example within the dataset
            data["id"] = index
        return data

    def train_collate_fn(self, inputs) -> Dict[str, Any]:
        """Collate function for training.

        This function should be passed into the dataloader as the collate_fn argument in lambda function.
        Revised based on
        https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

        Parameters
        ----------
        inputs : List[Dict]
            a list of input data

        Returns
        -------
        Dict[str,Any]
            a dictionary of batched tensors

        Examples
        --------
        >>> from pprint import pprint
        >>> import os
        >>> HOME = '/home/nanaeilish/projects/Google-Opinion' # change to your own dir
        >>> dataset = ExtractiveQa(
                data_dir=os.path.join(HOME, 'google_opinion/data/laptop14'),
                tokenizer="distilbert-base-uncased",
                keep_keys = ['id'],
                stage="train",
                template="What is the aspect of this review?")
        >>> inputs = [{'id': [0],
            'question': 'What is the aspect of this review?',
            'context': 'I charge it at night and skip taking the cord with me because of the good battery life.',
            'answers': {'text': ['cord'], 'answer_start': [41], 'answer_end': [45]}}]
        >>> print(dataset.train_collate_fn(inputs))
        {
            'id': [[0]],
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1]]),
            'input_ids': tensor([[  101,  2054,  2003,  1996,  7814,  1997,  2023,  3319,  1029,   102,
                    1045,  3715,  2009,  2012,  2305,  1998, 13558,  2635,  1996, 11601,
                    2007,  2033,  2138,  1997,  1996,  2204,  6046,  2166,  1012,   102]]),
            'start_positions': tensor([19]),
            'end_positions': tensor([19])
        }
        """

        examples = {k: [d[k] for d in inputs] for k in inputs[0].keys()}

        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.rpad else "context"],
            examples["context" if self.rpad else "question"],
            truncation="only_second" if self.rpad else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            padding=self.padding,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1
                                                          if self.rpad else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.rpad else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[
                            token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(
                        token_end_index + 1)

        batch = {
            "start_positions":
            torch.tensor(tokenized_examples["start_positions"]),
            "end_positions":
            torch.tensor(tokenized_examples["end_positions"]),
            "input_ids":
            torch.tensor([x for x in tokenized_examples["input_ids"]]),
            "attention_mask":
            torch.tensor([x for x in tokenized_examples["attention_mask"]]),
        }
        batch.update(
            {k: v
             for k, v in examples.items() if k in self.keep_keys})

        # logging.info(f"[return batch] keys: {batch.keys()}")
        return batch

    def val_collate_fn(self, inputs):
        """Collate function for validation and testing.

        This function should be passed into the dataloader as the collate_fn argument in lambda function.
        Revised based on
        https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb


        Parameters
        ----------
        inputs: List[Dict]
            a sub-list of self.processed_datas.

        Returns
        -------
        A batch of data for validation. The 10 key-value pairs are:
            id: List[int]
                The id of the example, used in `model_interface.postprocess_QA_predictions()`
            question: List[str]
                The question (`self.template`), used in `model_interface.postprocess_QA_predictions()`
            context: List[str]
                The context, same as `text` in a raw data dictionary. Used in `model_interface.postprocess_QA_predictions()`
            answers: List[Dict]
                The answer dictionary, same as `answers` in a raw data dictionary.
                Note that the `answer_start` and `answer_end` are character index.
                Used in `model_interface.postprocess_QA_predictions()`
            example_id: List[int]
                The example id, useful when a context generates multiple examples.
                Used in `model_interface.postprocess_QA_predictions()`
            offset_mapping: List[List[Union[float, Tuple]]]
                the offset mapping of the context,
                useful for computing the start/end position of the answer.
                Used in `model_interface.postprocess_QA_predictions()`
            start_positions: torch.tensor
                shape: torch.Size([1]); the start position of the answer (token index).
            end_positions: torch.tensor, shape: torch.Size([1])
                the end position of the answer (token index).
            input_ids: torch.tensor
                the input ids of the input sequence.
                shape: torch.Size([1, seq_len], where `seq_len` is the length of [CLS] context +[SEP] + question +[SEP] .
            attention_mask: torch.tensor
                the attention mask of the input sequence.
                shape: torch.Size([1, seq_len]), where `seq_len` is the length of [CLS] context +[SEP] + question +[SEP] .

        Examples
        --------
        >>> from pprint import pprint
        >>> import os
        >>> HOME = '/home/nanaeilish/projects/Google-Opinion' # change to your own dir
        >>> dataset = ExtractiveQa(
                data_dir=os.path.join(HOME, 'google_opinion/data/laptop14'),
                tokenizer="distilbert-base-uncased",
                keep_keys = ['id'],
                stage="train",
                template="What is the aspect of this review?")
        >>> inputs = [{'id': [0],
            'question': 'What is the aspect of this review?',
            'context': 'I charge it at night and skip taking the cord with me because of the good battery life.',
            'answers': {'text': ['cord'], 'answer_start': [41], 'answer_end': [45]}}]
        >>> print(dataset.val_collate_fn(inputs))
        {
            'answers': [{'answer_end': [45], 'answer_start': [41], 'text': ['cord']}],
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1]]),
            'context': ['I charge it at night and skip taking the cord with me because of '
                        'the good battery life.'],
            'end_positions': tensor([19]),
            'example_id': [[0]],
            'id': [[0]],
            'input_ids': tensor([[  101,  2054,  2003,  1996,  7814,  1997,  2023,  3319,  1029,   102,
                    1045,  3715,  2009,  2012,  2305,  1998, 13558,  2635,  1996, 11601,
                    2007,  2033,  2138,  1997,  1996,  2204,  6046,  2166,  1012,   102]]),
            'offset_mapping': [[None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                (0, 1),
                                (2, 8),
                                (9, 11),
                                (12, 14),
                                (15, 20),
                                (21, 24),
                                (25, 29),
                                (30, 36),
                                (37, 40),
                                (41, 45),
                                (46, 50),
                                (51, 53),
                                (54, 61),
                                (62, 64),
                                (65, 68),
                                (69, 73),
                                (74, 81),
                                (82, 86),
                                (86, 87),
                                None]],
            'question': ['What is the aspect of this review?'],
            'start_positions': tensor([19])
        }
        """

        examples = {k: [d.get(k, None) for d in inputs] for k in inputs[0].keys()} # reddit does not have post_id
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.rpad else "context"],
            examples["context" if self.rpad else "question"],
            truncation="only_second" if self.rpad else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            padding=self.padding,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples["offset_mapping"]

        # start, end positions for loss calc
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            # test set no answers
            if self.stage == 'test':
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                continue

            # validation set has answers
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1
                                                          if self.rpad else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.rpad else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[
                            token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(
                        token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(
                        token_end_index + 1)

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.rpad else 0

            # One example can give several spans (as answer), this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(
                examples["id"][sample_index])

            # Set to None in the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        batch = {

            # for matching back to answers in evaluation
            "id":
            examples["id"],
            "question":
            examples["question"],
            "context":
            examples["context"],
            "example_id":
            tokenized_examples["example_id"],
            "offset_mapping":
            tokenized_examples["offset_mapping"],

            # for loss calculation
            "start_positions":
            torch.tensor(tokenized_examples["start_positions"]),
            "end_positions":
            torch.tensor(tokenized_examples["end_positions"]),

            # for model inputs
            "input_ids":
            torch.tensor([x for x in tokenized_examples["input_ids"]]),
            "attention_mask":
            torch.tensor([x for x in tokenized_examples["attention_mask"]]),
        }
        if self.stage != 'test':
            batch["answers"] = examples["answers"]

        return batch


# %%
# import glob
# from common import *
# files = glob.glob(os.path.join(LAPTOP14_PATH, "*.json"))

# %%
# from pprint import pprint
# import os
# HOME = '/home/nanaeilish/projects/Google-Opinion'
# dataset = ExtractiveQa(
#     data_dir=os.path.join(HOME, 'google_opinion/data/laptop14'),
#     tokenizer="distilbert-base-uncased",
#     keep_keys = ['id'],
#     stage="train",
#     template="What is the aspect of this review?")
# pprint(dataset[0])

# from torch.utils.data import DataLoader
# pprint(dataset.val_collate_fn([ {'id': [0],
# 'question': 'What is the aspect of this review?',
# 'context': 'I charge it at night and skip taking the cord with me because of the good battery life.',
# 'answers': {'text': ['cord'], 'answer_start': [41], 'answer_end': [45]}}]))