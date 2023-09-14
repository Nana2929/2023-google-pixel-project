#%%
from typing import List, Dict
from collections import Counter


def exact_match(preds: List[Dict], references: List[Dict]):
    """
    Parameters
    ----------
    preds : List[Dict]
        the processed batch predictions
    references : List[Dict]
        the batch answers
    Returns
    -------
    float
        the number of exact matches/total number of predictions
    Examples
    --------
    >>> preds = [{'score': 1.859975814819336, 'text': 'battery life'},
    ...          {'score': 1.3222908973693848, 'text': 'sales" team'}]
    >>> references = [{'text': ['battery life'], 'answer_start': [74], 'answer_end': [86]},
    ...               {'text': ['" sales " team'], 'answer_start': [109], 'answer_end': [121]}]
    >>> exact_match(preds, references)
    0.5
    """
    score = 0
    for pred, reference in zip(preds, references):
        pred_text = pred['text']
        reference_text = reference['text'][0] if isinstance(
            reference['text'], list) else reference['text']
        if pred_text == reference_text:
            score += 1
    return score / len(preds)


def f1_score(preds: List[Dict],
            references: List):
    """
    Parameters
    ----------
    preds : List[Dict]
        the processed batch predictions
    references : List[Dict]
        the batch answers
    Returns
    -------
    float
        the average f1 score over the same-domain examples
        # needs to be calculated over the full testset
    """
    S, G = set(), set()
    for pred, reference in zip(preds, references):
        pred_text = pred['text']
        reference_text = reference['text'][0] if isinstance(
            reference['text'], list) else reference['text']
        reference_text = reference_text.strip() # cleaning up spaces 
        G.add(reference_text)
        S.add(pred_text)
    e = 1e-10
    p = len(S & G) / (len(S) + e)
    r = len(S & G) / (len(G) + e)
    f1 = 2 * p * r / (p + r + e)
    return f1

# def f1_score(preds: List[Dict], references: List[Dict]):
    # score = 0
    # for pred, reference in zip(preds, references):
    #     pred_text = pred['text']
    #     reference_text = reference['text'][0] if isinstance(
    #         reference['text'], list) else reference['text']

    #     pred_tokens = pred_text.split()
    #     reference_tokens = reference_text.split()
    #     common = Counter(pred_tokens) & Counter(reference_tokens)
    #     num_same_tokens = sum(common.values())
    #     if len(pred_tokens) == 0 or len(reference_tokens) == 0:
    #         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    #         score += int(pred_tokens == reference_tokens)
    #     elif num_same_tokens == 0:
    #         # If they have no words in common, then F1 is 0
    #         score += 0
    #     else:
    #         precision = 1.0 * num_same_tokens / len(pred_tokens)
    #         recall = 1.0 * num_same_tokens / len(reference_tokens)
    #         f1 = (2 * precision * recall) / (precision + recall)
    #         score += f1
    # return score / len(preds)
