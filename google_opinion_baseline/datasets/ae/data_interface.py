# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
# from torch.utils.data.sampler import WeightedRandomSampler


class DInterface(pl.LightningDataModule):
    """The interface of QA data module.
    Revised based on https://github.com/miracleyoo/pytorch-lightning-template/blob/master/classification/data/data_interface.py.

    Attributes
    ----------
    num_workers : int, optional
        How many subprocesses to use for data loading.
    dataset :str, optional
        Name of the dataset class to be instantialized. Defaults to ''.
    batch_size : int
        How many samples per batch to load for the instantialized dataloader.
    **kwargs: Dict[str,Any]
        Other hyprerparameters.
    """

    def __init__(self,
                 num_workers: int = 8,
                 dataset: str = '',
                 **kwargs) -> None:
        """ Inititalizes the interface of data module.

        Parameters
        ----------
        num_workers : int, optional
            How many subprocesses to use for data loading. Defaults to 8.
        dataset : str, optional
            Name of the dataset class to be instantiated, needs to be the same with the file stem name.
            For example, pass `extractive_qa` to instantiate `extractive_qa.py` in the `dataloaders/ae` folder.
            Defaults to ''.
        **kwargs: Dict[str,Any]
            Other hyperparameters.

        """

        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs

        self.load_data_module()

    def setup(self, stage: str = 'train') -> None:
        """Setup which dataloader to use.

        Parameters
        ----------
        stage : str
            The stage of the dataloader, can be `train`, `validation` or `test`.

        Raises
        ------
        ValueError
            If the stage name is not within the aforementioned valid names.
        """

        # Assign train/val datasets for use in dataloaders

        class_params = [
            'tokenizer', "max_length", "padding", "keep_keys", "template"
        ]
        class_args = {
            k: v
            for k, v in self.kwargs.items() if k in class_params
        }

        if stage == 'validation':
            self.valset = self.instancialize(**class_args, stage='validation')

        elif stage == 'train' or stage == 'fit':  # allow  self._call_lightning_datamodule_hook("setup", stage=`fit`) to work
            self.trainset = self.instancialize(**class_args, stage='train')
            # set data_counts=None to use full dataset
            self.valset = self.instancialize(**class_args, stage='validation')

        # Assign test dataset for use in dataloader(s)
        elif stage == 'test':
            # set data_counts=None to use full dataset
            print('setting up for stage: ', stage)
            self.testset = self.instancialize(**class_args, stage='test')
        else:
            raise ValueError(
                'Invalid stage name. Available choices include "train", "test" and "validation". '
            )

        # # If you need to balance your data using Pytorch Sampler,
        # # please uncomment the following lines.

        # with open('./data/ref/samples_weight.pkl', 'rb') as f:
        #     self.sample_weight = pkl.load(f)

    # def train_dataloader(self):
    #     sampler = WeightedRandomSampler(self.sample_weight, len(self.trainset)*20)
    #     return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, sampler = sampler)

    def train_dataloader(self) -> torch.utils.data.dataloader:
        """initalizes the train dataloader.
        Returns
        -------
        torch.utils.data.dataloader
        """
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=lambda x: self.trainset.train_collate_fn(x))

    def val_dataloader(self) -> torch.utils.data.dataloader:
        """initalizes the validation dataloader.
        Returns
        -------
        torch.utils.data.dataloader
        """
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: self.valset.val_collate_fn(x))

    def test_dataloader(self) -> torch.utils.data.dataloader:
        """initalizes the testing dataloader.
        Returns
        -------
        torch.utils.data.dataloader
        """
        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: self.testset.val_collate_fn(x))

    def load_data_module(self):
        """Load the corresponding data module from the `dataloaders` folder.

        Template function provided by Pytorch Lightning template.
        Please always name your dataset file as `snake_case.py` and
        class name corresponding `CamelCase`. For example, if you want
        to load dataset module `extractive_qa.py`, you need
        to have a class named `ExtractiveQa` in this file.

        Raises
        ------
        ValueError
            If the file name or class name is invalid.
        """
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        print(f'to_load: {camel_name}')
        try:
            self.data_module = getattr(
                importlib.import_module('.' + name, package=__package__),
                camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}'
            )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters from self.hparams dictionary.

        Template function provided by Pytorch Lightning template.
        You can also input any args to overwrite the corresponding value in `self.kwargs`.
        Parameters
        ----------
        **other_args: Dict[str,Any]
            Other hyperparameters to be included in initializing the `dataset` dataset module.
        """
        class_args = inspect.getfullargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args_ = {}

        for arg in class_args:
            if arg in inkeys:
                args_[arg] = self.kwargs[arg]
        args_.update(other_args)
        print(args_)

        return self.data_module(**args_)
