import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DInterface(pl.LightningDataModule):
    """The interface of ALSC data module.
    Revised based on https://github.com/miracleyoo/pytorch-lightning-template/blob/master/classification/data/data_interface.py.
    
    Attributes
    ----------
    num_workers: int, optional
        How many subprocesses to use for data loading.
    dataset: str, optional
        Name of the dataset class to be instantialized. Defaults to ''.
    batch_size: int
        How many samples per batch to load for the instantialized dataloader.
    **kwargs: Dict[str, Any]
        Other hyprerparameters.
    """

    def __init__(self, num_workers=4, dataset='', **kwargs):
        """ Inititalizes the interface of data module.
        
        Parameters
        ----------
        num_workers: int, optional
            How many subprocesses to use for data loading. Defaults to 8.
        dataset: str, optional
            Name of the dataset class to be instantiated, needs to be the same with the file stem name.
            For example, pass `alsc_data` to instantiate `alsc_data.py` in the `dataloaders/alsc` folder.
            Defaults to ''.
        **kwargs: Dict[str,Any]
            Other hyperparameters.
        """

        super().__init__()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.load_data_module()

    def setup(self, stage=None):
        """Setup which dataloader to use.
        
        Parameters
        ----------
        stage: str
            The stage of the dataloader, by default `test`.
        
        Raises
        ------
        ValueError
            If the stage name is not within the aforementioned valid names.
        """

        class_params = ['tokenizer', "max_length", "padding"]
        class_args = {
            k: v
            for k, v in self.kwargs.items() if k in class_params
        }

        if stage == 'test':
            self.testset = self.instancialize(**class_args, stage='test')

    def test_dataloader(self):
        """initalizes the testing dataloader.
        
        Returns
        -------
        torch.utils.data.dataloader
        """

        return DataLoader(self.testset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=lambda x: self.testset.collate_fn(x))

    def load_data_module(self):
        """Load the corresponding data module from the `dataloaders` folder.
        
        Template function provided by Pytorch Lightning template.
        Please always name your dataset file as `snake_case.py` and class name corresponding `CamelCase`.
        For example, if you want to load dataset module `alsc_data.py`, you need to have a class
        named `AlscData` in this file.
        """

        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        print(f'to_load: {camel_name}')

        self.data_module = getattr(
            importlib.import_module('.' + name, package=__package__),
            camel_name)

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
        args1 = {}

        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]

        args1.update(other_args)

        return self.data_module(**args1)