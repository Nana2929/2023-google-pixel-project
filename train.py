import argparse
import wandb

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from torch.optim import AdamW

from utils import compute_metrics, group_by_length, tokenize_and_align_labels

def main(args):
    dataset = load_dataset("conll2003")
    ner_feature = dataset["train"].features["ner_tags"]
    label_names = ner_feature.feature.names
    selected_dataset = dataset["train"].select(range(args.data_size))

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    train_set = selected_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=selected_dataset.column_names,
    )
    valid_set = train_set.map(
        lambda examples: group_by_length(examples, args.valid_set_length),
        batched=True,
        batch_size=None,
        num_proc=4,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    id2label = {i: label for i, label in enumerate(label_names)}
    label2id = {v: k for k, v in id2label.items()}
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.d_model,
        max_position_embeddings=128,
        num_attention_heads=12,
        num_hidden_layers=6,
        intermediate_size=args.d_model * 4,
        classifier_dropout=0.1,
        id2label=id2label,
        label2id=label2id,
        num_labels = 9,
    )

    model = BertForTokenClassification(config=config)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    training_args = TrainingArguments(
        evaluation_strategy="steps",
        eval_steps=50,
        max_steps=args.max_steps,
        metric_for_best_model="f1",
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        output_dir="./models/"+f"{args.d_model}_{args.data_size}",
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        warmup_ratio=0.03,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.evaluate()
    wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./data/conll2003_train.txt")
    parser.add_argument("--tokenizer-path", type=str, default="./models/")
    parser.add_argument("--max-steps", type=int, default=4500)
    parser.add_argument("--data-size", nargs="+", type=int, default=[1400])
    parser.add_argument("--d-model", nargs="+", type=int, default=[768])
    parser.add_argument("--valid-set-length", nargs="+", type=int, default=[128])

    args = parser.parse_args()
    # range(1400, 15400, 1400)
    data_size = args.data_size
    # range(192, 804, 36)
    d_model = args.d_model
    # range(128, 288, 32)
    valid_set_length = args.valid_set_length

    for ds in data_size:
        for dm in d_model:
            for dl in valid_set_length:
                wandb.init(
                    project=f"conll2003_length_test",
                    name=f"d_{dm}_l_{dl}",
                )

                args.data_size = ds
                args.d_model = dm
                args.valid_set_length = dl

                main(args)