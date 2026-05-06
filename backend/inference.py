"""
Story generation inference for the improved Storifai model.

Loads a trained ImprovedModel checkpoint and exposes a generate_story method
that takes 5 image bytes and returns 5 sentences. Files are loaded from local
disk if present (baked into the image at build time), otherwise downloaded
from GCS at startup.
"""

import io
import json
import os
import time
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torchvision import transforms

from model_code.storifai_models import ImprovedModel


GCS_PROJECT = "hoshmand-ai-c51f6"
GCS_BUCKET = "hoshmand-ai-datasets"
GCS_CHECKPOINT_BLOB = "storifai/checkpoints/improved_best.pth"
GCS_DATASET_BLOB = "storifai/dataset/storifai_dataset.json"

ASSETS_DIR = Path("/app/assets")
LOCAL_CHECKPOINT = ASSETS_DIR / "improved_best.pth"
LOCAL_DATASET = ASSETS_DIR / "storifai_dataset.json"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 224

NUM_PHOTOS = 5
MAX_LEN = 30


class StoryGenerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.sos_idx = None
        self.eos_idx = None
        self.pad_idx = None
        self.transform = self._build_transform()

    @staticmethod
    def _build_transform():
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    @staticmethod
    def _ensure_file(local_path: Path, blob_name: str) -> None:
        if local_path.exists() and local_path.stat().st_size > 0:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"[startup] using local {local_path} ({size_mb:.1f} MB)")
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        from google.cloud import storage
        print(f"[startup] downloading gs://{GCS_BUCKET}/{blob_name} -> {local_path}")
        client = storage.Client(project=GCS_PROJECT)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"[startup] downloaded {size_mb:.1f} MB")

    def load(self) -> None:
        t0 = time.time()
        self._ensure_file(LOCAL_DATASET, GCS_DATASET_BLOB)
        self._ensure_file(LOCAL_CHECKPOINT, GCS_CHECKPOINT_BLOB)

        with open(LOCAL_DATASET, "r") as f:
            dataset_meta = json.load(f)
        self.vocab = dataset_meta["vocab"]
        self.word2idx = dataset_meta["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.pad_idx = self.word2idx["<PAD>"]
        self.sos_idx = self.word2idx["<SOS>"]
        self.eos_idx = self.word2idx["<EOS>"]
        vocab_size = len(self.vocab)
        print(f"[startup] vocab_size={vocab_size}")

        ckpt = torch.load(str(LOCAL_CHECKPOINT), map_location=self.device, weights_only=False)
        ckpt_epoch = ckpt.get("epoch", "?")
        ckpt_metrics = ckpt.get("metrics", {})
        print(f"[startup] checkpoint epoch={ckpt_epoch} metrics={ckpt_metrics}")

        self.model = ImprovedModel(
            vocab_size=vocab_size,
            embed_dim=512,
            num_heads=8,
            decoder_layers=4,
            attn_layers=2,
            max_len=MAX_LEN,
            pad_idx=self.pad_idx,
            dropout=0.1,
        )
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        print(f"[startup] model loaded in {time.time() - t0:.1f}s")

    def _prepare_images(self, image_bytes_list: List[bytes]) -> torch.Tensor:
        if len(image_bytes_list) != NUM_PHOTOS:
            raise ValueError(f"expected {NUM_PHOTOS} images, got {len(image_bytes_list)}")
        tensors = []
        for raw in image_bytes_list:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            tensors.append(self.transform(img))
        return torch.stack(tensors, dim=0).unsqueeze(0).to(self.device)

    def _decode_tokens(self, token_ids: List[int]) -> str:
        words = []
        for idx in token_ids:
            if idx == self.eos_idx or idx == self.pad_idx:
                break
            if idx == self.sos_idx:
                continue
            word = self.idx2word.get(int(idx), "<UNK>")
            if word in ("<PAD>", "<SOS>", "<EOS>"):
                continue
            words.append(word)
        if not words:
            return ""
        sentence = " ".join(words).strip()
        sentence = sentence[0].upper() + sentence[1:]
        if sentence and sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    @torch.no_grad()
    def generate_story(self, image_bytes_list: List[bytes]) -> List[str]:
        if self.model is None:
            raise RuntimeError("model not loaded; call load() first")
        images = self._prepare_images(image_bytes_list)
        tokens = self.model.generate(
            images,
            sos_idx=self.sos_idx,
            eos_idx=self.eos_idx,
            max_len=MAX_LEN,
            temperature=1.0,
        )
        tokens = tokens[0].cpu().tolist()
        sentences = [self._decode_tokens(seq) for seq in tokens]
        return sentences
