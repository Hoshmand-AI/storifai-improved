"""VIST dataset loader for Storifai.

Reads the processed dataset JSON (train/val/test splits + vocab) and
loads images from a local directory. Images should be pre-downloaded
from GCS before training begins.
"""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def tokenize(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def encode_sentence(words, word2idx, max_len=30):
    sos = word2idx["<SOS>"]
    eos = word2idx["<EOS>"]
    pad = word2idx["<PAD>"]
    unk = word2idx["<UNK>"]
    ids = [sos] + [word2idx.get(w, unk) for w in words[: max_len - 2]] + [eos]
    ids = ids[:max_len]
    ids = ids + [pad] * (max_len - len(ids))
    return ids


def collect_required_photo_ids(splits):
    ids = set()
    for split_name in ("train", "val", "test"):
        for story in splits[split_name]:
            for pid in story["photo_ids"]:
                ids.add(str(pid))
    return ids


def download_images_from_gcs(photo_ids, gcs_prefix, local_dir,
                              max_workers=64, log_every=1000):
    """Download a set of photo IDs from GCS to local directory in parallel.

    Skips photos that already exist locally (so this is resumable).
    Returns the set of photo IDs successfully present locally after download.
    """
    from google.cloud import storage as gcs_storage

    os.makedirs(local_dir, exist_ok=True)

    bucket_name = gcs_prefix.replace("gs://", "").split("/", 1)[0]
    prefix_path = gcs_prefix.replace(f"gs://{bucket_name}/", "").rstrip("/")

    client = gcs_storage.Client()
    bucket = client.bucket(bucket_name)

    existing = set()
    for f in os.listdir(local_dir):
        if f.endswith(".jpg"):
            existing.add(f.replace(".jpg", ""))

    needed = [pid for pid in photo_ids if pid not in existing]
    print(f"  Total photos required: {len(photo_ids)}")
    print(f"  Already cached locally: {len(existing)}")
    print(f"  To download: {len(needed)}")

    if not needed:
        return set(existing)

    def fetch(pid):
        try:
            blob = bucket.blob(f"{prefix_path}/{pid}.jpg")
            blob.download_to_filename(os.path.join(local_dir, f"{pid}.jpg"))
            return (pid, True)
        except Exception:
            return (pid, False)

    import time
    start = time.time()
    done = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch, pid) for pid in needed]
        for fut in as_completed(futures):
            pid, ok = fut.result()
            done += 1
            if not ok:
                failed.append(pid)
            if done % log_every == 0 or done == len(needed):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(needed) - done) / rate if rate > 0 else 0
                print(f"    [{done}/{len(needed)}] {rate:.0f} files/sec, ETA {eta:.0f}s")

    if failed:
        print(f"  Failed to download {len(failed)} photos")

    final = set()
    for f in os.listdir(local_dir):
        if f.endswith(".jpg"):
            final.add(f.replace(".jpg", ""))
    return final


class VISTStoryDataset(Dataset):
    """One example = one VIST story (5 images + 5 sentences)."""

    def __init__(self, stories, word2idx, image_dir, image_transform,
                 max_caption_len=30):
        self.stories = stories
        self.word2idx = word2idx
        self.image_dir = image_dir
        self.transform = image_transform
        self.max_len = max_caption_len

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        story = self.stories[idx]
        photo_ids = story["photo_ids"]
        sentences = story["sentences"]

        images = []
        for pid in photo_ids:
            path = os.path.join(self.image_dir, f"{pid}.jpg")
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            images.append(self.transform(img))
        images = torch.stack(images, dim=0)

        captions = []
        for s in sentences:
            tokens = tokenize(s)
            ids = encode_sentence(tokens, self.word2idx, self.max_len)
            captions.append(ids)
        captions = torch.tensor(captions, dtype=torch.long)

        return images, captions


def load_processed_dataset(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data
