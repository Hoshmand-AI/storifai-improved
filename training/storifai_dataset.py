"""
Storifai dataset — shared by both baseline and improved models.

Reads processed dataset JSON from GCS, streams images from GCS on demand
with in-memory caching so epoch 2+ are fast.
"""
import io
import re
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from google.cloud import storage
from collections import OrderedDict


# CLIP normalization (used by both — we'll use same preprocessing for fair comparison)
# This matches torchvision's standard ImageNet stats which works for ResNet-50,
# and CLIP itself uses slightly different stats but the difference is small for
# fine-tuning purposes. For strict fairness we use ImageNet stats throughout.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def tokenize(text):
    """Match the tokenizer used in Phase 4 vocab building."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class LRUImageCache:
    """Simple LRU cache for decoded image tensors. Avoids re-downloading from GCS."""
    def __init__(self, max_items=20000):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


class StorifaiDataset(Dataset):
    """
    Each item is one story: 5 images + 5 tokenized captions.

    Returns:
        images: (5, 3, 224, 224) tensor
        captions: (5, max_len) tensor of token IDs (left-shifted, with <SOS> and <EOS>)
        masks: (5, max_len) tensor — 1 where token is real, 0 where padding
        story_id: int (for debugging)
    """
    def __init__(self, split_data, vocab, word2idx, gcs_bucket, gcs_image_prefix,
                 max_len=30, training=False, cache=None):
        self.stories = split_data
        self.vocab = vocab
        self.word2idx = word2idx
        self.max_len = max_len
        self.training = training
        self.transform = TRAIN_TRANSFORM if training else DEFAULT_TRANSFORM
        self.cache = cache if cache is not None else LRUImageCache(max_items=25000)

        # GCS client (shared across workers — pickle-safe init)
        self._gcs_bucket_name = gcs_bucket
        self._gcs_prefix = gcs_image_prefix.rstrip('/')
        self._client = None
        self._bucket = None

        # Special tokens
        self.PAD = word2idx['<PAD>']
        self.UNK = word2idx['<UNK>']
        self.SOS = word2idx['<SOS>']
        self.EOS = word2idx['<EOS>']

    def _ensure_client(self):
        """Lazy-init GCS client per worker to avoid pickle issues."""
        if self._client is None:
            self._client = storage.Client()
            self._bucket = self._client.bucket(self._gcs_bucket_name)

    def __len__(self):
        return len(self.stories)

    def _load_image(self, photo_id):
        """Load one image from cache or GCS."""
        # Cache key only depends on photo_id (transform is per-instance, not cached)
        cached = self.cache.get(photo_id)
        if cached is not None:
            # Re-apply transform (in case it's stochastic for training)
            if self.training:
                # Decode original PIL is needed for stochastic transforms — store PIL bytes
                pil_image = cached
                return self.transform(pil_image)
            else:
                # For eval, the cached tensor is fine
                if isinstance(cached, torch.Tensor):
                    return cached
                return self.transform(cached)

        # Cache miss: download from GCS
        self._ensure_client()
        blob_path = f"{self._gcs_prefix}/{photo_id}.jpg"
        blob = self._bucket.blob(blob_path)
        image_bytes = blob.download_as_bytes()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Cache the PIL image (for training) or processed tensor (for eval)
        if self.training:
            self.cache.put(photo_id, pil_image)
            return self.transform(pil_image)
        else:
            tensor = self.transform(pil_image)
            self.cache.put(photo_id, tensor)
            return tensor

    def _encode_caption(self, sentence):
        """Tokenize + convert to IDs + add SOS/EOS + pad to max_len."""
        tokens = tokenize(sentence)
        ids = [self.SOS]
        for tok in tokens:
            ids.append(self.word2idx.get(tok, self.UNK))
            if len(ids) >= self.max_len - 1:
                break
        ids.append(self.EOS)

        # Pad to max_len
        mask = [1] * len(ids) + [0] * (self.max_len - len(ids))
        ids = ids + [self.PAD] * (self.max_len - len(ids))
        return ids[:self.max_len], mask[:self.max_len]

    def __getitem__(self, idx):
        story = self.stories[idx]
        photo_ids = story['photo_ids']
        sentences = story['sentences']

        images = torch.stack([self._load_image(pid) for pid in photo_ids])  # (5, 3, 224, 224)

        caption_ids = []
        masks = []
        for sent in sentences:
            ids, mask = self._encode_caption(sent)
            caption_ids.append(ids)
            masks.append(mask)

        captions = torch.tensor(caption_ids, dtype=torch.long)  # (5, max_len)
        masks = torch.tensor(masks, dtype=torch.long)  # (5, max_len)

        return {
            'images': images,
            'captions': captions,
            'masks': masks,
            'story_id': story['story_id'],
        }


def load_dataset_json(local_path):
    """Load the storifai_dataset.json built in Phase 4."""
    with open(local_path) as f:
        data = json.load(f)
    return data


def build_dataloaders(dataset_json_path, gcs_bucket, batch_size=8, num_workers=2,
                     max_len=30, shared_cache=True):
    """
    Returns (train_loader, val_loader, test_loader, vocab, word2idx).
    """
    from torch.utils.data import DataLoader

    data = load_dataset_json(dataset_json_path)
    vocab = data['vocab']
    word2idx = data['word2idx']
    splits = data['splits']
    gcs_prefix = data['gcs_image_prefix'].replace(f'gs://{gcs_bucket}/', '')

    # Shared cache across all 3 datasets — saves memory and re-downloads
    cache = LRUImageCache(max_items=25000) if shared_cache else None

    train_ds = StorifaiDataset(splits['train'], vocab, word2idx, gcs_bucket,
                                gcs_prefix, max_len=max_len, training=True, cache=cache)
    val_ds = StorifaiDataset(splits['val'], vocab, word2idx, gcs_bucket,
                              gcs_prefix, max_len=max_len, training=False, cache=cache)
    test_ds = StorifaiDataset(splits['test'], vocab, word2idx, gcs_bucket,
                               gcs_prefix, max_len=max_len, training=False, cache=cache)

    # num_workers=0 because LRU cache must be shared in main process. With workers>0
    # each worker has own cache (still works, just less efficient on epoch 2+).
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, vocab, word2idx
