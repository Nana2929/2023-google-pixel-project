# BERT Baseline
- Date: 2023 Winter
- Tool for simple aspect extraction and sentiment classification based on BERT
- Framework
    - [PyTorch Lightning](https://lightning.ai/)
    - [miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template)
- Folder structure
    ```shell
    google_opinion_baseline
    ├── datasets
    │   ├── ae
    │   │   ├── common.py
    │   │   ├── data_interface.py
    │   │   └── extractive_qa.py
    │   └── alsc
    │       ├── alsc_data.py
    │       └── data_interface.py
    ├── models
    │   ├── ae
    │   │   ├── distil_bert.py
    │   │   ├── metrics.py
    │   │   ├── model_interface.py
    │   │   └── utils.py
    │   └── alsc
    │       ├── deberta.py
    │       └── model_interface.py
    ├── requirements.txt
    ├── inference.sh
    ├── run_ae.py
    └── run_alsc.py
    ```

## Before Start
- Install the required packages
    - Note that `torch` or `cublas` related packages should be installed manually after you survey your CUDA version and locate the correct distribution. Therefore they are *commented* in `requirements.txt`.
    ```shell
    pip install -r requirements.txt
    ```
- Prepare the data.
    - We provide 1 dataset `laptop14` (copied from [ROGERDJQ/RoBERTaABSA](https://github.com/ROGERDJQ/RoBERTaABSA/tree/main/Dataset/Laptop)), with pre-split `train` and `validation` files. Note that the `*_Test.json` file is actually copied from `*_Validation.json`, but without the ground truth.
## Aspect Extraction (`ae`) Sub-module
- Input: one short review sentence eg. "The screen is high-quality."
- Output: a contiguous aspect span in the sentence. eg. "screen".
### I. Modes
- `mode = train`: train the model to do aspect-term extraction.
    - Data: train set comes from `--data_dir` with stem suffix `train` (case-insensitive), eg. `laptop_train.json`; validation split (if exists) comes from the files in `data_dir` with stem suffix `validation` (case-insensitive), eg. `laptop_validation.json`.
    - Specify the model class to use with `--model_name`.
        - Find the models available in `models/ae`, eg. `distil_bert`.
    - Specify the dataset class to use with `--dataset`.
        - Find the data classes available in `datasets/ae`, eg. `extractive_qa`.
- `mode = test`: test with a specified checkpoint (`checkpoint_path`).
    - Save the prediction files to `pred_file_path`.
- `mode = validation`: evaluate the model with the validation split.
- Demo of score table
    ```
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃      Validate metric      ┃       DataLoader 0        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │          val_em           │    0.6516608595848083     │
    │          val_f1           │     0.72415691614151      │
    │         val_loss          │    0.8118242885057743     │
    └───────────────────────────┴───────────────────────────┘
    ```

### II. Commands

`cd google_opinion_baseline`.

- training
- using `laptop14`
    ```shell
    python3 run_ae.py \
    --model_name="distil_bert" \
    --dataset="extractive_qa" \
    --data_dir="./data/laptop14" \
    --mode=train
    ```
- testing
    - using `laptop14`
    ```shell
    python3 run_ae.py \
    --data_dir="./data/laptop14" \
    --mode=test \
    --model_name="distil_bert" \
    --dataset="extractive_qa" \
    --checkpoint_path="./saved_models/silver-lining/08-[val_f1:0.8376].ckpt" \
    --pred_file_path="./predictions/ae/laptop14_prediction.json"
    ```
- validating
    - using `laptop14`
    ```shell
    python3 run_ae.py \
    --data_dir=./data/laptop14 \
    --mode=validation \
    --model_name="distil_bert" \
    --dataset="extractive_qa" \
    --checkpoint_path="./saved_models/silver-lining/08-[val_f1:0.8376].ckpt"
    ```

### Aspect-based Sentiment Classification (`alsc`) Sub-module
- Input: a review sentence and an aspect term span. eg. "The screen is high-quality." and "screen".
- Output: the sentiment of the aspect term. eg. "positive".
#### I. Modes
- Note that we only provide `test` mode for `alsc` sub-module.
- `mode = test`
    - Save the prediction files to `--pred_file_path`: a `.json` that includes the review sentences, the aspect spans, and the predicted sentiment.
    - Test dataset comes from the files in `--data_dir` with stem suffix 'test' (case-insenstive), eg. `laptop_test.json`.
#### II. Commands
- testing
    - using `laptop14`
    ```shell
    python3 run_alsc.py \
    --data_dir="./data/laptop14" \
    --model_name="deberta" \
    --dataset="alsc_data" \
    --pred_file_path="./predictions/alsc/laptop14_prediction.json"
    ```
