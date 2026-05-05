"""Training and evaluation for Storifai baseline and improved models."""
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from storifai_dataset import (
    VISTStoryDataset,
    collect_required_photo_ids,
    download_images_from_gcs,
    load_processed_dataset,
)
from storifai_eval import compute_corpus_metrics, decode_tokens
from storifai_models import BaselineModel, ImprovedModel, count_params


def build_transforms(model_type):
    if model_type == "improved":
        # CLIP normalization
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        # ImageNet normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def make_dataloader(stories, word2idx, image_dir, model_type,
                     batch_size, max_caption_len, shuffle, num_workers=2):
    tfm = build_transforms(model_type)
    ds = VISTStoryDataset(stories, word2idx, image_dir, tfm,
                           max_caption_len=max_caption_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, pin_memory=True,
                       drop_last=False)


def build_model(model_type, vocab_size, max_len, pad_idx):
    if model_type == "improved":
        return ImprovedModel(vocab_size, embed_dim=512, num_heads=8,
                              decoder_layers=4, attn_layers=2,
                              max_len=max_len, pad_idx=pad_idx)
    elif model_type == "baseline":
        return BaselineModel(vocab_size, embed_dim=512,
                              max_len=max_len, pad_idx=pad_idx)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps,
                  base_lr, min_lr=1e-6):
        self.opt = optimizer
        self.warmup_steps = max(warmup_steps, 1)
        self.total_steps = max(total_steps, warmup_steps + 1)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.base_lr * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * min(progress, 1.0))
            )
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def compute_loss(logits, captions, pad_idx):
    # logits: (B*P, T-1, V), captions: (B, P, T)
    B, P, T = captions.shape
    targets = captions[:, :, 1:].reshape(-1)
    V = logits.size(-1)
    logits_flat = logits.reshape(-1, V)
    return F.cross_entropy(logits_flat, targets, ignore_index=pad_idx)


@torch.no_grad()
def evaluate(model, loader, vocab, word2idx, device, max_len, pad_idx,
              sos_idx, eos_idx):
    model.eval()
    idx2word = {i: w for w, i in word2idx.items()}

    total_loss = 0.0
    total_steps = 0
    pred_stories = []
    ref_stories = []

    for batch_idx, (images, captions) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        logits = model(images, captions)
        loss = compute_loss(logits, captions, pad_idx)
        total_loss += loss.item()
        total_steps += 1

        gen = model.generate(images, sos_idx=sos_idx, eos_idx=eos_idx,
                              max_len=max_len, temperature=1.0)
        # gen: (B, P, T)
        B, P, T = gen.shape
        for b in range(B):
            pred_words = []
            ref_words = []
            for p in range(P):
                pred_words.extend(decode_tokens(gen[b, p].tolist(),
                                                  idx2word, eos_idx, pad_idx))
                ref_words.extend(decode_tokens(captions[b, p].tolist(),
                                                  idx2word, eos_idx, pad_idx))
            pred_stories.append(pred_words)
            ref_stories.append([ref_words])

    metrics = compute_corpus_metrics(pred_stories, ref_stories)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics, pred_stories, ref_stories


def upload_to_gcs(local_path, gcs_path):
    subprocess.run(["gsutil", "-q", "cp", local_path, gcs_path], check=False)


def train(
    model_type,
    dataset_json_path,
    image_local_dir,
    output_dir,
    smoke_test=False,
    epochs=15,
    batch_size=8,
    lr=1e-4,
    weight_decay=0.01,
    warmup_frac=0.1,
    grad_clip=1.0,
    max_caption_len=30,
    early_stop_patience=3,
    eval_every=1,
    gcs_checkpoint_prefix=None,
    log_every=50,
    **_extra,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    data = load_processed_dataset(dataset_json_path)
    splits = data["splits"]
    vocab = data["vocab"]
    word2idx = data["word2idx"]

    pad_idx = word2idx["<PAD>"]
    sos_idx = word2idx["<SOS>"]
    eos_idx = word2idx["<EOS>"]
    vocab_size = len(vocab)

    train_loader = make_dataloader(
        splits["train"], word2idx, image_local_dir, model_type,
        batch_size=batch_size, max_caption_len=max_caption_len,
        shuffle=True,
    )
    val_loader = make_dataloader(
        splits["val"], word2idx, image_local_dir, model_type,
        batch_size=batch_size, max_caption_len=max_caption_len,
        shuffle=False,
    )
    test_loader = make_dataloader(
        splits["test"], word2idx, image_local_dir, model_type,
        batch_size=batch_size, max_caption_len=max_caption_len,
        shuffle=False,
    )
    print(f"  Vocab: {vocab_size}, train batches: {len(train_loader)}, "
          f"val batches: {len(val_loader)}")

    model = build_model(model_type, vocab_size, max_caption_len, pad_idx)
    model = model.to(device)
    total, trainable = count_params(model)
    print(f"  Total params: {total / 1e6:.2f}M  "
          f"Trainable: {trainable / 1e6:.2f}M")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(warmup_frac * total_steps))
    scheduler = WarmupCosine(optimizer, warmup_steps, total_steps,
                              base_lr=lr)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    if smoke_test:
        model.train()
        images, captions = next(iter(train_loader))
        images = images.to(device)
        captions = captions.to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images, captions)
            loss = compute_loss(logits, captions, pad_idx)
        return {
            "smoke_loss": float(loss.item()),
            "logits_shape": tuple(logits.shape),
            "captions_shape": tuple(captions.shape),
            "trainable_params": trainable,
        }

    best_bleu4 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    # Optional unfreeze for baseline at epoch 5
    unfreeze_at = 5 if model_type == "baseline" else None

    for epoch in range(1, epochs + 1):
        if unfreeze_at is not None and epoch == unfreeze_at + 1:
            print(f"  Unfreezing CNN backbone at epoch {epoch}")
            model.encoder.unfreeze()
            optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr * 0.1, weight_decay=weight_decay,
            )

        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",
                                     enabled=device.type == "cuda"):
                logits = model(images, captions)
                loss = compute_loss(logits, captions, pad_idx)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            cur_lr = scheduler.step()

            running_loss += loss.item()
            n_batches += 1

            if batch_idx % log_every == 0:
                print(f"  Epoch {epoch}/{epochs} batch {batch_idx}/"
                      f"{len(train_loader)} loss={loss.item():.4f}  "
                      f"lr={cur_lr:.2e}")

        train_loss = running_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        if epoch % eval_every == 0:
            print(f"  Validating epoch {epoch}...")
            val_metrics, _, _ = evaluate(
                model, val_loader, vocab, word2idx, device,
                max_caption_len, pad_idx, sos_idx, eos_idx,
            )
            print(
                f"  Epoch {epoch}: train_loss={train_loss:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  "
                f"BLEU-4={val_metrics['bleu4']*100:.2f}  "
                f"METEOR={val_metrics['meteor']*100:.2f}  "
                f"({epoch_time:.0f}s)"
            )
            history.append({"epoch": epoch, "train_loss": train_loss,
                            **val_metrics})

            if val_metrics["bleu4"] > best_bleu4:
                best_bleu4 = val_metrics["bleu4"]
                best_epoch = epoch
                no_improve = 0
                ckpt_path = os.path.join(output_dir,
                                          f"{model_type}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "vocab": vocab,
                    "word2idx": word2idx,
                    "model_type": model_type,
                    "metrics": val_metrics,
                }, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")
                if gcs_checkpoint_prefix:
                    upload_to_gcs(
                        ckpt_path,
                        f"{gcs_checkpoint_prefix.rstrip('/')}/"
                        f"{model_type}_best.pth",
                    )
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch {best_epoch})")
                    break

    # Test evaluation with best checkpoint
    ckpt_path = os.path.join(output_dir, f"{model_type}_best.pth")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
    print(f"  Running test evaluation...")
    test_metrics, pred_stories, ref_stories = evaluate(
        model, test_loader, vocab, word2idx, device, max_caption_len,
        pad_idx, sos_idx, eos_idx,
    )
    print(
        f"  Test: loss={test_metrics['loss']:.4f}  "
        f"BLEU-1={test_metrics['bleu1']*100:.2f}  "
        f"BLEU-2={test_metrics['bleu2']*100:.2f}  "
        f"BLEU-3={test_metrics['bleu3']*100:.2f}  "
        f"BLEU-4={test_metrics['bleu4']*100:.2f}  "
        f"METEOR={test_metrics['meteor']*100:.2f}"
    )

    samples = []
    for i in range(min(5, len(pred_stories))):
        samples.append({
            "predicted": " ".join(pred_stories[i]),
            "reference": " ".join(ref_stories[i][0]),
        })

    results = {
        "model_type": model_type,
        "best_val_bleu4": best_bleu4,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "history": history,
        "samples": samples,
    }
    results_path = os.path.join(output_dir, f"{model_type}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")

    if gcs_checkpoint_prefix:
        upload_to_gcs(
            results_path,
            f"{gcs_checkpoint_prefix.rstrip('/')}/{model_type}_results.json",
        )

    return results
