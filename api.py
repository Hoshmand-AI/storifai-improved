import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from model import StorifaiImproved

app = FastAPI(title="Storifai Improved API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT = "/tmp/improved_checkpoint.pth"
DEVICE = "cpu"
MAX_LEN = 25

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = None
vocab = None

TONES = [
    {"name": "warm", "temperature": 0.7, "label": "Warm"},
    {"name": "vivid", "temperature": 1.0, "label": "Vivid"},
    {"name": "reflective", "temperature": 1.2, "label": "Reflective"},
]

DEMOS = {
    "warm": [
        ["We finally made it - and it felt exactly right.", "Everyone was laughing before we even started.", "Those little in-between moments were the best part.", "We found a spot nobody else knew about.", "I don't think any of us wanted it to end."],
        ["It started the way the best days usually do - without a plan.", "The light was doing something magical.", "We just kept walking, talking, forgetting the time.", "Somewhere along the way it became a memory.", "I'll be thinking about this one for a while."]
    ],
    "vivid": [
        ["Golden light stretched across everything as we arrived.", "Faces caught mid-laugh, hands full, eyes bright.", "Every corner held something worth stopping for.", "The colors of the afternoon deepened as we explored.", "The kind of day that photographs itself."],
        ["Blue sky, warm stone, and that particular smell of somewhere new.", "Details everywhere - textures, shadows, small beautiful things.", "Motion and stillness, side by side.", "The landscape opened up like a held breath released.", "Even the ordinary things looked extraordinary."]
    ],
    "reflective": [
        ["Some days slip away like water. This one left a mark.", "What looked like a small moment turned out to be everything.", "Time moved differently here.", "We were all just trying to hold onto it a little longer.", "Memory is strange - it keeps the light, forgets the rest."],
        ["There's a version of this day I'll carry for years.", "Nobody said anything important. That was the point.", "The afternoon stretched out like an old song.", "Something quietly shifted. I noticed later.", "The photographs will tell one story. We know the other."]
    ]
}


def load_model():
    global model, vocab
    if not os.path.exists(CHECKPOINT):
        print("Downloading improved checkpoint from GCP...")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket("hoshmand-ai-datasets")
            blob = bucket.blob("storifai/checkpoints/improved_final.pth")
            blob.download_to_filename(CHECKPOINT)
            print("Download complete.")
        except Exception as e:
            print(f"GCP download failed: {e} - demo mode")
            return
    print("Loading improved model...")
    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        vocab = ckpt["vocab"]
        cfg = ckpt.get("config", {})
        model = StorifaiImproved(
            vocab_size=len(vocab),
            embed_dim=cfg.get("embed_dim", 512),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 4),
            max_len=cfg.get("max_len", 30)
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print(f"Improved model loaded. Vocab: {len(vocab)} words")
    except Exception as e:
        print(f"Model load failed: {e} - demo mode")
        model = None
        vocab = None


def generate_stories(img_tensors):
    if model is None or vocab is None:
        return None
    N = len(img_tensors)
    images = torch.stack(img_tensors).unsqueeze(0).to(DEVICE)
    stories = []
    for tone in TONES:
        try:
            sentences = model.generate(images, vocab, max_len=MAX_LEN, temperature=tone["temperature"])
            sentences = [s for s in sentences[:N] if s and len(s.split()) > 2]
            stories.append(sentences if sentences else None)
        except Exception as e:
            print(f"Generation failed for {tone['name']}: {e}")
            stories.append(None)
    return stories


def get_demo_stories(n_photos):
    return [random.choice(DEMOS[k])[:n_photos] for k in ["warm", "vivid", "reflective"]]


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "loaded" if model else "demo",
        "version": "improved",
        "architecture": "CLIP ViT-B/32 + CrossImageAttention + TransformerDecoder"
    }


@app.post("/generate")
async def generate(images: List[UploadFile] = File(...)):
    if len(images) < 3 or len(images) > 5:
        raise HTTPException(400, "Upload 3-5 photos")
    n = len(images)
    img_tensors = []
    for img_file in images:
        data = await img_file.read()
        try:
            img = Image.open(BytesIO(data)).convert("RGB")
            img_tensors.append(transform(img))
        except Exception:
            img_tensors.append(torch.zeros(3, 224, 224))

    stories = generate_stories(img_tensors)
    if stories is None or any(s is None for s in stories):
        demo = get_demo_stories(n)
        if stories is None:
            stories = demo
        else:
            for i, s in enumerate(stories):
                if s is None:
                    stories[i] = demo[i]

    return {
        "stories": stories,
        "model": "improved" if model else "demo",
        "tones": ["warm", "vivid", "reflective"],
        "n_photos": n
    }


@app.on_event("startup")
async def startup():
    load_model()
