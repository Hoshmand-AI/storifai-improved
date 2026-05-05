"""
Storifai evaluation — BLEU-1/2/3/4 and METEOR scoring.

For VIST, the convention is full-story scoring: concatenate the 5 generated
sentences into one story, and the 5 ground-truth sentences into one reference,
then compute BLEU/METEOR over those.

We also report per-sentence scores for debugging.
"""
import torch
from collections import defaultdict


def ids_to_tokens(ids, vocab, sos_idx, eos_idx, pad_idx):
    """Convert a 1D tensor/list of IDs to a list of word tokens, stopping at EOS."""
    tokens = []
    for tid in ids:
        if isinstance(tid, torch.Tensor):
            tid = tid.item()
        if tid == eos_idx or tid == pad_idx:
            break
        if tid == sos_idx:
            continue
        tokens.append(vocab[tid])
    return tokens


def evaluate_model(model, dataloader, vocab, word2idx, device, max_len=30,
                   max_batches=None, n_sample_stories=5):
    """
    Run model on dataloader, decode greedily, compute BLEU & METEOR.

    Returns:
        metrics: dict with 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor'
        samples: list of (gt_story, generated_story) pairs (for paper qualitative analysis)
    """
    # Lazy imports because nltk may need data downloads
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score

    model.eval()
    sos_idx = word2idx['<SOS>']
    eos_idx = word2idx['<EOS>']
    pad_idx = word2idx['<PAD>']

    # Per-story (full 5-sentence) refs and hyps for BLEU corpus-level
    story_refs = []  # list of [list of tokens] (one ref per story)
    story_hyps = []  # list of list of tokens

    samples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = batch['images'].to(device)
            captions = batch['captions'].to(device)
            B = images.shape[0]

            # Generate
            generated = model.generate(images, max_len=max_len,
                                       sos_idx=sos_idx, eos_idx=eos_idx, pad_idx=pad_idx)
            # generated: (B, 5, T-1)

            for b in range(B):
                # Build ground-truth full story (concatenated 5 sentences)
                gt_tokens_per_sent = []
                for s in range(5):
                    gt_ids = captions[b, s].tolist()
                    # Strip SOS at start, stop at EOS
                    gt_tokens_per_sent.append(ids_to_tokens(gt_ids, vocab, sos_idx, eos_idx, pad_idx))

                gt_story = []
                for ts in gt_tokens_per_sent:
                    gt_story.extend(ts)

                # Build generated full story
                gen_tokens_per_sent = []
                for s in range(5):
                    gen_ids = generated[b, s].tolist()
                    gen_tokens_per_sent.append(ids_to_tokens(gen_ids, vocab, sos_idx, eos_idx, pad_idx))

                gen_story = []
                for ts in gen_tokens_per_sent:
                    gen_story.extend(ts)

                story_refs.append([gt_story])  # corpus_bleu wants list of refs per hyp
                story_hyps.append(gen_story)

                if len(samples) < n_sample_stories:
                    samples.append({
                        'gt_sentences': [' '.join(s) for s in gt_tokens_per_sent],
                        'gen_sentences': [' '.join(s) for s in gen_tokens_per_sent],
                    })

    # BLEU computation (corpus-level)
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(story_refs, story_hyps, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(story_refs, story_hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(story_refs, story_hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(story_refs, story_hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # METEOR — compute per-story and average
    meteor_scores = []
    for refs, hyp in zip(story_refs, story_hyps):
        if len(hyp) == 0:
            meteor_scores.append(0.0)
            continue
        try:
            score = meteor_score(refs, hyp)
            meteor_scores.append(score)
        except Exception:
            meteor_scores.append(0.0)
    avg_meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    metrics = {
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4,
        'meteor': avg_meteor,
        'num_stories': len(story_hyps),
    }
    return metrics, samples


def format_metrics(metrics):
    return (f"BLEU-1: {metrics['bleu1']*100:.2f}  "
            f"BLEU-2: {metrics['bleu2']*100:.2f}  "
            f"BLEU-3: {metrics['bleu3']*100:.2f}  "
            f"BLEU-4: {metrics['bleu4']*100:.2f}  "
            f"METEOR: {metrics['meteor']*100:.2f}  "
            f"(n={metrics['num_stories']})")
