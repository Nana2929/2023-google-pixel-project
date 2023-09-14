# %%
import os
import logging
from argparse import ArgumentParser
from easydict import EasyDict as edict
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning.callbacks as plc

from datasets.ae.data_interface import DInterface
from models.ae.model_interface import MInterface

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# https://wandb.ai/manan-goel/MNIST/reports/How-to-Integrate-PyTorch-Lightning-with-Weights-Biases--VmlldzoxNjg1ODQ1


def load_callbacks(args, exp_name):
    callbacks = []
    callbacks.append(
        plc.EarlyStopping(
            monitor='val_f1',
            mode='max',
            patience=args.early_stop_epoch,
            min_delta=0.01,
        ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval='step'))

    callbacks.append(
        plc.ModelCheckpoint(monitor='val_f1',
                            dirpath=f'./saved_models/{exp_name}',
                            filename='{epoch:02d}-[val_f1:{val_f1:.4f}]',
                            save_top_k=1,
                            mode='max',
                            save_last=False,
                            auto_insert_metric_name=False,
                            save_on_train_epoch_end=True))

    return callbacks


def main(args):

    logging.basicConfig(level=logging.INFO)
    do_log = False
    if args.checkpoint_path is None:
        logging.info("Training from scratch.")
        data_module = DInterface(**vars(args))
        model = MInterface(**vars(args))
        do_log = True
        wandb_logger = WandbLogger(project=args.project_name,
                               name=args.exp_name if args.exp_name else
                               None)  # created a random exp name

    else:
        logging.info("Resuming from checkpoint or used for testing.")
        model = MInterface.load_from_checkpoint(**vars(args))
        data_module = DInterface(**vars(args))
    exp_name = wandb_logger.experiment.name if do_log else None
    trainer = Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        enable_progress_bar=True,
        callbacks=load_callbacks(args, exp_name=exp_name),
        logger=wandb_logger if do_log else None,
        devices='0,',
    )
    if args.mode == 'train':
        # data_module.setup(stage='train')
        trainer.fit(model, datamodule=data_module)
    elif args.mode == 'test':
        # produce prediction file
        model.freeze()
        data_module.setup(stage='test')
        trainer.test(model, dataloaders=data_module.test_dataloader())
    elif args.mode == 'validation':
        # calculate score against validation set
        model.freeze()
        data_module.setup(stage='validation')
        trainer.validate(model, dataloaders=data_module.val_dataloader())
    else:
        raise ValueError(
            "Invalid mode. Mode should be `train`, `test` or `validation`.")


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--early_stop_epoch', default=4, type=int)

    # LR Scheduler
    parser.add_argument('--lr_scheduler',
                        default='cosine',
                        choices=['step', 'cosine'],
                        type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    # Restart Control
    parser.add_argument('--checkpoint_path',
                        type=str,
                        help="If validation or testing or resuming training,\
                        pleas pass the checkpoint model path. eg. ./saved_models/silver-lining/08-[val_f1:0.8376].ckpt"
                        )

    # Training Info
    parser.add_argument('--dataset', default='extractive_qa', type=str)
    parser.add_argument('--data_dir',
                        type=str,
                        help='path to data directory, eg. "./data/laptop14"')
    parser.add_argument('--model_name', default='distil_bert', type=str)
    parser.add_argument('--loss', default='mse', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    # Model & dataset Hyperparameters
    parser.add_argument('--expand_aspects', action='store_true')
    parser.add_argument('--template',
                        default='What is the aspect of this review?',
                        type=str)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--doc_stride', default=128, type=int)
    parser.add_argument('--base_name',
                        default='distilbert-base-cased-distilled-squad',
                        type=str)

    # Other
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--project_name',
                        default='Google-Opinion',
                        type=str,
                        help='name of the project logged in logger')
    parser.add_argument('--exp_name',
                        type=str,
                        help='name of the experiment logged in logger')
    parser.add_argument('--pred_file_path',
                        default='./predictions/ae/Laptop14_pred_test.json',
                        type=str,
                        help='path to save prediction file')

    # Add pytorch lightning's args to parser as a group.

    args = parser.parse_args()
    # Reset Some Default Trainer Arguments' Default Values

    pl.seed_everything(args.seed, workers=True)

    main(args)
