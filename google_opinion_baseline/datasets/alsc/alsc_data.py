import os
import glob
import json
from pathlib import Path
import logging
from typing import Union, List
from copy import deepcopy

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy
import torch
import torch.utils.data as data

logging.basicConfig(level=logging.INFO, format='[%(module)s] %(message)s')


class AlscData(data.Dataset):
    """The dataset based on torch.utils.data.Dataset for turning sentiment aspect extraction dataset to extractive QA.

    The dataset instance, will later be instantialized as a data module by a data interface following
    https://github.com/miracleyoo/pytorch-lightning-template.

    Attributes
    ----------
    data: List[Dict[str, Any]]
        The list of dictionaries that contains the input data.
        Each data has a (text, aspect) tuple.
    process_data: List[Dict[str, Any]]
        The list of dictionaries that contains the processed data.
        The output of `self.process_data()` on `self.data`.
    """

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 tokenizer: Union[str, AutoTokenizer],
                 max_length: int = 512,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 stage: str = 'test'):
        """ Initialize the dataset with the given data_dir and tokenizer.

        Attributes
        ----------
        data_dir : Union[str, os.PathLike]
            The directory of the dataset that contains the json files.
        tokenizer : Union[str, AutoTokenizer]
            The tokenizer used to tokenize the text,
            can be the name of the pretrained tokenizer or the tokenizer itself.
        max_length : int, optional
            Tokenizer args. The maximum length of the input sequence (after concatenation), by default 512
        padding : Union[bool, str, PaddingStrategy], optional
            Tokenizer args. The padding strategy used, by default True. See https://huggingface.co/docs/transformers/pad_truncation for more.
        stage: str, optional
            The stage of the dataset ('train', 'valid', 'test'), by default 'test'.
        """

        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer) if isinstance(tokenizer, str) else tokenizer
        self.max_length = max_length
        self.padding = padding
        self.stage = stage

        self.data = self.read_file()
        self.processed_data = []

        for d in self.data:
            self.processed_data.append(self.process_data(d))

    def read_file(self):
        """Read in json files from data directory and return a list of dictionaries (raw data).

        Returns
        -------
            List[Dict]:
                The list of dictionaries that contains the data.

        Examples
        --------
        >>> print(dataset.read_file()[0])
        {
            'id': 0,
            'text': 'Boot time is super fast, around anywhere from 35 seconds to 1 minute.',
            'pred_aspect': 'time',
            'pred_score': 17.53968048095703
        }
        """

        filepaths = glob.glob(os.path.join(self.data_dir, '*.json'))
        data = []

        for filepath in filepaths:
            # file_stem = Path(filepath).stem
            # file_suffix = file_stem.split('_')[-1].lower()
            with open(filepath) as f:
                # if file_suffix == self.stage:
                logging.info(f'loading {filepath}')
                data = json.load(f)

        return data

    def __len__(self):
        """Return dataset size.

        Returns
        -------
        int
            The number of data in the dataset.
        """

        return len(self.processed_data)

    def process_data(self, data):
        """Process a raw text with adding [SEP] tokens.

        Parameters
        ----------
        data: Dict
            The dictionary that contains the raw data.
            1 text can only have 1 aspect.
            If a text has more than 1 aspects, the aspects should be separated into different dictionaries with same text
            before send into this function.

        Returns
        -------
        Dict
            The dictionary that contains the processed data.

        Examples
        --------
        >>> d = {
            'id': 0,
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.'
            'aspect': 'battery life'
        }

        >>> print(self.process_data(d))
        {
            'id': 0,
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
            'aspect': 'battery life'
            'processed_txt': 'I charge it at night and skip taking the cord with me because of the good battery life. [SEP] battery life'
        }
        """

        d = deepcopy(data)
        processed_txt = d['text'] + ' ' + '[SEP]' + ' ' + d['pred_aspect']
        processed_d = {
            'id': d['id'],
            'text': d['text'],
            'aspect': d['pred_aspect'],
            'processed_txt': processed_txt
        }

        return processed_d

    def __getitem__(self, idx):
        """Return a processed data example.

        Parameters
        ----------
        index : int
            The index of the data example.

        Returns
        -------
        Dict[str, Any]
            the indexed example with `id` key if not already exists.
            This helps match back the example and the prediction answer.
        """

        data = self.processed_data[idx]

        if 'id' not in data:  # a unique identifier (str, int, ...) for the example within the dataset
            data['id'] = idx

        return data

    def collate_fn(self, input):
        """Collate function for testing.

        This function should be passed into the dataloader as the collate_fn argument in lambda function.
        Revised based on
        https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb

        Parameters
        ----------
        inputs: List[Dict]
            a sub-list of self.processed_data.

        Returns
        -------
        A batch of data for validation. The 4 key-value pairs are:
            id: List[int]
                The id of the text.
            text: List[str]
                The context, same as `text` in a raw data dictionary.
            aspect: List[str]
                The corresponding aspect of the text.
            tokenized_txt: List[Dict]
                The output of the tokenizer. A dictionary includes 3 torch.tensors: input_ids, token_type_ids, and attention_mask.
                shape of 3 tensors: torch.Size([1, seq_len], where `seq_len` is the length of [CLS] text [SEP] aspect [SEP] .
            input_ids: torch.tensor
                the input ids of the input sequence.
                shape: torch.Size([1, seq_len], where `seq_len` is the length of [CLS] text [SEP] aspect [SEP] .

        Examples
        --------
        >>> dataset = AlscData(
                data_dir='/home/hedy881028/google-opinion/Google-Opinion/google_opinion/data/laptop14/',
                tokenizer='yangheng/deberta-v3-large-absa-v1.1')

        >>> inputs = [{'id': 0,
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
            'aspect': 'battery life',
            'processed_txt': 'I charge it at night and skip taking the cord with me because of the good battery life. [SEP] battery life'}]

        >>> print(dataset.collate_fn(inputs))
        {
            'id': 0,
            'text': 'I charge it at night and skip taking the cord with me because of the good battery life.',
            'aspect': 'battery life',
            'tokenized_txt': {'input_ids': tensor([[   1,  273, 1541,  278,  288,  661,  263, 7637,  787,  262, 7443,  275,  351,  401,  265,  262,  397,
                                                    2643,  432,  260,    2, 2643,  432,    2]]),
                              'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                              'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])},
            'input_ids': tensor([[   1,  273, 1541,  278,  288,  661,  263, 7637,  787,  262, 7443,  275,  351,  401,  265,  262,  397, 2643,  432,  260,
                                     2, 2643,  432,    2]])
        }
        """

        processed_txt = [i['processed_txt'] for i in input]
        tokenized_txt = self.tokenizer(processed_txt,
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_tensors='pt')
        id = [i['id'] for i in input]
        text = [i['text'] for i in input]
        aspect = [i['aspect'] for i in input]

        batch = {
            'id':
            id,
            'text':
            text,
            'aspect':
            aspect,
            'tokenized_txt':
            tokenized_txt,
            'input_ids':
            torch.tensor([x.tolist() for x in tokenized_txt["input_ids"]]),
        }

        return batch