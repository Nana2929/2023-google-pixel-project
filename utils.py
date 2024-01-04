import evaluate
import numpy as np

from datasets import load_dataset
from transformers import BertTokenizerFast

DATASET = load_dataset("conll2003")
LABEL_NAMES = DATASET["train"].features["ner_tags"].feature.names

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[LABEL_NAMES[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABEL_NAMES[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


def group_by_length(examples, max_len=128):

    curr_idx = 0
    # examples is a dictionary with keys being the column names
    column_names = list(examples.data.keys())

    while curr_idx < len(examples[column_names[0]]):
        next_idx = curr_idx + 1
        curr_len = len(examples[column_names[0]][curr_idx])

        while curr_len < max_len and next_idx < len(examples[column_names[0]]):
            remaining_len = max_len - curr_len - 1
            next_example_len = len(examples[column_names[0]][next_idx])
            # If the next example is too long, truncate it
            if next_example_len <= remaining_len:
                for column_name in column_names:
                    examples[column_name][curr_idx] += examples[column_name][next_idx][1:]
                    # Remove next example and we won't increment next_idx
                    examples[column_name].pop(next_idx)

                curr_len += next_example_len

            else:
                for column_name in column_names:
                    examples[column_name][curr_idx] += examples[column_name][next_idx][1:remaining_len]
                    # Truncate the next example by remaining_len
                    examples[column_name][next_idx] = \
                        [101] + examples[column_name][next_idx][remaining_len:]

                curr_len += remaining_len

        curr_idx = next_idx

    return examples


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs