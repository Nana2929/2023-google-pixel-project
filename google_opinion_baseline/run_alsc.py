import logging
from argparse import ArgumentParser
from pytorch_lightning import Trainer

from datasets.alsc.data_interface import DInterface
from models.alsc.model_interface import MInterface

base_name = 'yangheng/deberta-v3-large-absa-v1.1'


def main(args):
    logging.basicConfig(level=logging.INFO)
    data_module = DInterface(**vars(args))
    model = MInterface(**vars(args))

    trainer = Trainer.from_argparse_args(
        args,
        accelerator='gpu',
        enable_progress_bar=True,
        devices='1'
    )

    # produce prediction file
    model.freeze()
    data_module.setup(stage='test')
    trainer.test(model, dataloaders=data_module)
    logging.info(f'Prediction file saved to {args.pred_file_path}')


if __name__ == '__main__':
    parser = ArgumentParser()

    # Basic Training Control
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Training Info
    parser.add_argument('--dataset', default='alsc_data', type=str)
    parser.add_argument('--data_dir',
                        default='./data/laptop14',
                        type=str,
                        help='path to data directory')
    parser.add_argument('--model_name', default='deberta', type=str)

    # Model & dataset hyperparameters
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--config', default=base_name, type=str)
    parser.add_argument('--model_base', default=base_name, type=str)
    parser.add_argument('--tokenizer', default=base_name, type=str)

    # Other
    parser.add_argument('--pred_file_path',
                        default='./predictions/alsc/pred.json',
                        type=str,
                        help='path to save prediction file')
    parser.add_argument('--ref_file_path',
                        # default='./data/laptop14/Laptop_ours_Validation.json',
                        type=str,
                        help='path to reference file')

    # Add pytorch lightning's args to parser as a group.
    args = parser.parse_args()

    main(args)