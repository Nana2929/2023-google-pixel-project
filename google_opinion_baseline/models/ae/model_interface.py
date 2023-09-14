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

import importlib
import collections
import inspect
import os
from typing import List, Dict, Tuple, Union, Optional, Any
import logging

import numpy as np
import torch
from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import wandb

from .utils import save_to_json

logging.basicConfig(level=logging.INFO)


class MInterface(pl.LightningModule):
    """Model interface for QA models based on transformers.
    Child class of LightningModule for the pytorch lightning trainer to work.
    """

    # __doc__ += pl.LightningModule.__doc__ automatically for Sphinx
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Initializing the model interface.

        Parameters
        ----------
        kargs : Dict[str, Any]
            Keyword arguments for the model interface, including `model_name` which is the model to be loaded,
            the hparams for the loaded model and the training arguments.

        """
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        self.cls_token_id = kwargs[
            'cls_token_id'] if 'cls_token_id' in kwargs else 101
        self.n_best_size = 20

    def forward(self, **kwargs):
        """Forward pass of the model."""
        return self.model(**kwargs)

    def postprocess_qa_predictions(
            self,
            batch: Dict[str, Union[List, torch.Tensor]],
            raw_predictions: Tuple[torch.Tensor, torch.Tensor],
            is_test: bool = False,
            n_best_size: int = 20,
            max_answer_length: int = 30) -> List[Dict[str, Any]]:
        """Postprocessing the raw predictions of the model, used for obtaining `n_best_size` QA predictions in validation and testing stages.
        Revised based on https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb.

        Parameters
        ----------
        batch : Dict[str, Union[List, torch.Tensor]]
            The whole batch of data.

        raw_predictions : Tuple[torch.Tensor, torch.Tensor]
            raw predictions of the model, including start logits and end logits
            of shape (batch_size, seq_len) where `seq_len` is the length of the maximum sequence in the batch.
        is_test : bool, optional
            the stage of dataloader to be used, by default False
            If `is_test` is True, the dataloader is the testing dataloader, and the `answers` field is not included in the batch.
        n_best_size : int, optional
            how many top predictions to be preserved and choosen from, by default 20
        max_answer_length : int, optional
            length of the max answer span, by default 30

        Note
        ____
        Use `is_test` to control the stage.


        Returns
        -------
        List[Dict[str, Any]]
            the prediction results of the model::
            [
                {
                    'score': float, 'text': str ,
                }, ...
            ]
        """

        all_start_logits, all_end_logits = raw_predictions

        # Build a map example to its corresponding features.
        batch_size = len(batch['id'])
        if is_test:
            dataset_keys = ['id', 'context', 'question']
        else:
            dataset_keys = ['id', 'context', 'question', 'answers']

        feature_keys = set(batch.keys()) - set(dataset_keys)
        # Split a batch into `examples` and `features`.
        examples = [{k: batch[k][i]
                     for k in dataset_keys} for i in range(batch_size)]

        features = [{k: batch[k][i]
                     for k in feature_keys} for i in range(batch_size)]
        for i, feature in enumerate(features):
            for k in feature_keys:
                if k not in ['example_id', 'offset_mapping']:
                    feature[k] = feature[k].tolist()

        example_id_to_index = {
            examples[i]['id']: i
            for i in range(len(examples))
        }
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[
                feature["example_id"]]].append(i)

        predictions = []
        # Logging.
        # logging.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!

        for example_index, example in enumerate(examples):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                # update: list indexing to tensor idexing
                cls_index = features[feature_index]["input_ids"].index(
                    self.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[
                    cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1:-n_best_size -
                                                         1:-1].tolist()
                end_indexes = np.argsort(end_logits)[-1:-n_best_size -
                                                     1:-1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append({
                            "score":
                            float(start_logits[start_index] +
                                  end_logits[end_index]),
                            "text":
                            context[start_char:end_char]
                        })

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers,
                                     key=lambda x: x["score"],
                                     reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}
            predictions.append(best_answer)
        assert len(predictions) == len(examples)
        return predictions

    def training_step(
            self, batch: Dict[str, Union[List,
                                         torch.Tensor]]) -> Dict[str, float]:
        """step for training

        Parameters
        ----------
        batch : Dict[str, Union[List, torch.Tensor]]
            The batch of data to train on.

        Returns
        -------
        Dict[str,float]
            The loss of the model. This key is required for the training loop.
        """
        out = self(**batch)
        loss = out.loss
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        wandb.log({"train/loss": loss})
        return {'loss': loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, Any]:
        """step for validation

        Parameters
        ----------
        batch : Dict[str, Union[List, torch.Tensor]]
            The batch of data to validate on.

        Returns:
            Dict[str,Any]: The loss, output and answer of the model. The last 2 keys are for evaluation.
            example::
            {
                'loss':[...], # List[torch.tensor of shape ()]
                'predictions':
                [{
                    'text':  'battery life',
                    'score': 15.0 # logits sum
                }, ...] # the output for self.postprocess_qa_predictions
                'references': List[Dict[str, Any]]
                [{
                    'text': ['battery life'],
                    'answer_start': [74],
                    'answer_end': [86]
                },
                ...]
            }
        """
        out = self(**batch)
        loss = out.loss
        loss = loss.cpu().numpy() if isinstance(loss,
                                                torch.Tensor) else loss.cpu()

        all_start_logits, all_end_logits = out.start_logits, out.end_logits
        all_start_logits, all_end_logits = out.start_logits.cpu().numpy(
        ), out.end_logits.cpu().numpy()
        raw_predictions = all_start_logits, all_end_logits
        predictions = self.postprocess_qa_predictions(
            batch=batch, raw_predictions=raw_predictions)
        references = batch['answers']
        wandb.log({"val/loss": loss})
        return {
            'loss': loss,
            'predictions': predictions,
            'references': references
        }

    def validation_epoch_end(self, output: List[Dict[str, Any]]) -> None:
        """Calculating and logging the validation set eval statistics at the end of the epoch.
        Parameters
        ----------
        output : List[Dict[str, Any]]
            The total outputs accumulated from batch validation step.
        """
        # Make the Progress Bar leave there
        batch_size = len(output[0]['predictions'])
        total_loss = 0
        all_preds, all_refs = [], []
        for batch_output in output:
            b_preds = batch_output['predictions']
            b_refs = batch_output['references']
            total_loss += batch_output['loss']
            all_preds.extend(b_preds)
            all_refs.extend(b_refs)

        em, f1 = self.compute_metrics(all_preds, all_refs)
        avg_loss = total_loss / len(output)

        self.log('val_loss',
                 avg_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=batch_size)
        self.log('val_em', em, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({"val/em": em, "val/f1": f1})

    def compute_metrics(self, predictions: Dict[str, Any],
                        references: Dict[str, Any]) -> Tuple[float, float]:
        """Computing the exact match and f1 score over a list of predictions and references.
        Returns
        -------
        Tuple[float, float]
            The exact match and f1 score.
        """
        from .metrics import exact_match, f1_score
        exact_match = exact_match(preds=predictions, references=references)
        f1_score = f1_score(preds=predictions, references=references)
        return exact_match, f1_score

    def test_step(self, batch: Dict[str, Any], batch_idx):
        """

        """

        out = self(**batch)
        loss = out.loss
        loss = loss.cpu().numpy() if isinstance(loss,
                                                torch.Tensor) else loss.cpu()

        all_start_logits, all_end_logits = out.start_logits, out.end_logits
        all_start_logits, all_end_logits = out.start_logits.cpu().numpy(
        ), out.end_logits.cpu().numpy()
        raw_predictions = all_start_logits, all_end_logits
        predictions = self.postprocess_qa_predictions(
            batch=batch, raw_predictions=raw_predictions, is_test=True)

        return batch, predictions

    def test_epoch_end(self, output):

        pred_file_path = self.hparams.pred_file_path
        os.makedirs(os.path.dirname(pred_file_path), exist_ok=True)
        d = []

        for batch, batch_prediction in output:
            batch_ids = batch['id']
            batch_texts = batch['context']
            if 'answers' in batch:
                batch_answers = batch['answers']

            for idx in range(len(batch_ids)):
                p = batch_prediction[idx]
                bid = batch_ids[idx]
                data_id = bid
                pred_text = p['text']
                pred_score = p['score']
                prediction_dict = {
                    'id': data_id,
                    'text': batch_texts[idx],
                    'pred_aspect': pred_text,
                    'pred_score': pred_score
                }
                if 'answers' in batch:
                    answer = batch_answers[idx]['text']
                    ref_text = answer if isinstance(answer, str) else answer[0]
                    prediction_dict['gold_aspect'] = ref_text
                d.append(prediction_dict)
        save_to_json(path=pred_file_path, data=d)
        logging.info(
            f'Predictions on {self.hparams.data_dir} testset ({len(d)} examples) saved to {pred_file_path}.'
        )

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.lr_decay_steps,
                    eta_min=self.hparams.lr_decay_min_lr)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def configure_loss(self):
        loss = self.hparams.loss.lower()
        if loss == 'mse':
            self.loss_function = F.mse_loss
        elif loss == 'l1':
            self.loss_function = F.l1_loss
        elif loss == 'bce':
            self.loss_function = F.binary_cross_entropy
        else:
            raise ValueError("Invalid Loss Type!")

    def load_model(self):

        name = self.hparams.model_name
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        # outer test
        try:
            Model = getattr(
                importlib.import_module('.' + name, package=__package__),
                camel_name)
        # inner test
        # except:
        # Model = getattr(importlib.import_module(name, package=__package__),
        #                 camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {name}.{camel_name}!'
            )
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


# %%
# import pickle

# with open(
#         '/home/nanaeilish/projects/Google-Opinion/valid_batch_0_for_test.pkl',
#         'rb') as f:
#     batch = pickle.load(f)
# print(batch.keys())
# from easydict import EasyDict as edict

# base_name = "distilbert-base-uncased-distilled-squad"
# args = edict({
#     'model_name': "distil_bert",
#     'data_dir':
#     "/home/nanaeilish/projects/Google-Opinion/google_opinion/data/laptop14",
#     'template': "What is the aspect of this review?",
#     'dataset': "extractive_qa",
#     'loss': 'mse',
#     'max_length': 256,
#     'padding': True,
#     'lr': 2e-5,
#     'lr_scheduler': 'cosine',
#     'lr_decay_steps': 20,
#     'lr_decay_rate': 0.5,
#     'lr_decay_min_lr': 1e-5,
#     'num_epochs': 1,
#     'seed': 123,
#     'config': base_name,
#     'model_base': base_name,
#     'tokenizer': base_name,
#     'batch_size': 16
# })
# model = MInterface(**vars(args))

# outs = model(**batch)
# # %%
# all_start_logits, all_end_logits = outs.start_logits, outs.end_logits
# all_start_logits, all_end_logits = outs.start_logits.detach().numpy(
# ), outs.end_logits.detach().numpy()
# raw_preds = all_start_logits, all_end_logits

# predictions = model.postprocess_qa_predictions(batch=batch,
#                                                raw_predictions=raw_preds)
# from metrics import *

# em = exact_match(predictions, references=batch['answers'])
# f1 = f1_score(predictions, references=batch['answers'])
# print(em, f1)
# # # %%
# # %%

# %%
