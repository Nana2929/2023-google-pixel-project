import inspect
import importlib
import json
import logging
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


class MInterface(pl.LightningModule):
    """The interface of Deberta model.
    Revised based on https://github.com/miracleyoo/pytorch-lightning-template/blob/master/classification/data/model/model_interface.py.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def test_step(self, batch, batch_idx):
        out = self(**batch)
        pred = F.softmax(out.logits, dim=-1)
        prediction = []

        for p in pred:
            prediction.append(self.model.config.id2label[p.argmax().item()])

        return batch, prediction

    def f1(self, pred, ref):
        """Validate testing result by f1-score.

        Parameters
        ----------
        pred:
            The prediction results of the model.
        ref:
            Golden answers.
        """

        pred_sent = [p['sentiment'].lower() for p in pred]
        ref_sent = []
        for r in ref:
            if r['aspects'] != []:
                ref_sent.append(r['aspects'][0]['polarity'])
            else:
                ref_sent.append('none')

        mlb = MultiLabelBinarizer(
            classes=['positive', 'negative', 'neutral', 'none'])
        mlb.fit_transform(pred_sent)
        mlb.fit_transform(ref_sent)
        f1 = f1_score(ref_sent, pred_sent, average='macro')

        return f1

    def acc(self, pred, ref):
        """Validate testing result by accuracy.

        Parameters
        ----------
        pred:
            The prediction results of the model.
        ref:
            Golden answers.
        """

        pred_sent = [p['sentiment'].lower() for p in pred]
        ref_sent = []
        for r in ref:
            if r['aspects'] != []:
                ref_sent.append(r['aspects'][0]['polarity'])
            else:
                ref_sent.append('none')

        acc = sum([
            1 if pred_sent[i] == ref_sent[i] else 0
            for i in range(len(pred_sent))
        ]) / len(pred_sent)

        return acc

    def test_epoch_end(self, output):
        """Show testing score and write output json file.

        The function conbines the  batch information and corresponding predictions into a dictionary after
        a test epochs end.

        Parameters
        ----------
        output:
            The returns from self.test_step(). Including batch information and predictions.
        """

        pred_file_path = self.hparams.pred_file_path
        ref_file_path = self.hparams.ref_file_path
        pred_data = []

        for batch, prediction in output:
            for i in range(len(prediction)):
                prediction_dict = {
                    'id': batch['id'][i],
                    'text': batch['text'][i],
                    'aspect': batch['aspect'][i],
                    'sentiment': prediction[i]
                }

                pred_data.append(prediction_dict)
        with open(pred_file_path, 'w') as f:
            json.dump(pred_data, f, indent=4, ensure_ascii=False)
        if ref_file_path:
            # do comparison
            with open(ref_file_path, 'r') as f:
                ref = json.load(f)
                f1 = self.f1(pred_data, ref)
                acc = self.acc(pred_data, ref)

            logging.info(f'F1-score of predictions on {self.hparams.data_dir} testset: {f1}.')
            logging.info(
                f'Accuracy of predictions on {self.hparams.data_dir} testset: {acc}.'
            )
        logging.info(
            f'Predictions on {self.hparams.data_dir} testset ({len(pred_data)} examples) saved to {pred_file_path}.'
        )

    def configure_optimizers(self):
        pass
        # if hasattr(self.hparams, 'weight_decay'):
        #     weight_decay = self.hparams.weight_decay
        # else:
        #     weight_decay = 0
        # optimizer = torch.optim.Adam(self.parameters(),
        #                              lr=self.hparams.lr,
        #                              weight_decay=weight_decay)

        # if self.hparams.lr_scheduler is None:
        #     return optimizer
        # else:
        #     if self.hparams.lr_scheduler == 'step':
        #         scheduler = lrs.StepLR(optimizer,
        #                                step_size=self.hparams.lr_decay_steps,
        #                                gamma=self.hparams.lr_decay_rate)
        #     elif self.hparams.lr_scheduler == 'cosine':
        #         scheduler = lrs.CosineAnnealingLR(
        #             optimizer,
        #             T_max=self.hparams.lr_decay_steps,
        #             eta_min=self.hparams.lr_decay_min_lr)
        #     else:
        #         raise ValueError('Invalid lr_scheduler type!')
        #     return [optimizer], [scheduler]

    def configure_loss(self):
        pass
        # loss = self.hparams.loss.lower()
        # if loss == 'mse':
        #     self.loss_function = F.mse_loss
        # elif loss == 'l1':
        #     self.loss_function = F.l1_loss
        # elif loss == 'bce':
        #     self.loss_function = F.binary_cross_entropy
        # else:
        #     raise ValueError("Invalid Loss Type!")

    def load_model(self):
        """Load the corresponding model from the `models` folder.

        Template function provided by Pytorch Lightning template.
        Please always name your dataset file as `snake_case.py` and class name corresponding `CamelCase`.
        For example, if you want to load dataset module `deberta.py`, you need to have a class
        named `Deberta` in this file.
        """

        name = self.hparams.model_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        print(f'to_load: {camel_name}')

        Model = getattr(
            importlib.import_module('.' + name, package=__package__),
            camel_name)
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters from self.hparams dictionary.

        Template function provided by Pytorch Lightning template.
        You can also input any args to overwrite the corresponding value in `self.kwargs`.

        Parameters
        ----------
        Model: torch.nn.Module
            The model to be instancialized.
        **other_args: Dict[str,Any]
            Other hyperparameters to be included in initializing the `dataset` dataset module.
        """

        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}

        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)

        args1.update(other_args)

        return Model(**args1)