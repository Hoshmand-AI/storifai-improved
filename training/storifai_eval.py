"""Evaluation metrics: BLEU-1..4 and METEOR for VIST stories."""
import nltk
from nltk.translate.bleu_score import (
    SmoothingFunction,
    corpus_bleu,
)
from nltk.translate.meteor_score import meteor_score


def decode_tokens(tokens, idx2word, eos_idx, pad_idx):
    out = []
    for tok in tokens:
        t = int(tok)
        if t == eos_idx or t == pad_idx:
            break
        if t in (0, 1, 2):  # PAD, UNK, SOS
            continue
        if t < len(idx2word):
            out.append(idx2word[t])
    return out


def compute_corpus_metrics(predicted_stories, reference_stories):
    """Compute BLEU-1..4 and METEOR at the full-story level.

    predicted_stories: list of list[str]  (one list of words per story,
        formed by concatenating the 5 generated sentences)
    reference_stories: list of list[list[str]]  (each story has a list of
        reference word lists; for VIST single-reference, this is length 1)
    """
    smooth = SmoothingFunction().method1

    bleu1 = corpus_bleu(reference_stories, predicted_stories,
                        weights=(1.0, 0, 0, 0),
                        smoothing_function=smooth)
    bleu2 = corpus_bleu(reference_stories, predicted_stories,
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smooth)
    bleu3 = corpus_bleu(reference_stories, predicted_stories,
                        weights=(1 / 3, 1 / 3, 1 / 3, 0),
                        smoothing_function=smooth)
    bleu4 = corpus_bleu(reference_stories, predicted_stories,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smooth)

    meteor_scores = []
    for refs, hyp in zip(reference_stories, predicted_stories):
        try:
            m = meteor_score(refs, hyp)
        except Exception:
            m = 0.0
        meteor_scores.append(m)
    meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    return {
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
    }
