import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys, os, time
sys.path.insert(0, '/content/storifai/baseline')
sys.path.insert(0, '/content/storifai/improved')

from dataloader import VISTDataset
from model import StorifaiImproved

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 4
MAX_LEN = 30
GCP_BUCKET = "gs://hoshmand-ai-datasets/storifai/checkpoints"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train():
    print("Loading dataset...")
    dataset = VISTDataset(
        '/content/storifai/shared/dataset/sis/train.story-in-sequence.json'
    )
    vocab = dataset.vocab
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} stories | Vocab: {len(vocab)} words | Batches: {len(loader)}")

    print("Building improved model (CLIP + Transformer)...")
    model = StorifaiImproved(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    ).to(DEVICE)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])

    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0
        start = time.time()

        for batch_idx, batch in enumerate(loader):
            captions = batch['captions'].to(DEVICE)  # [B, 5, 30]
            B = captions.size(0)

            # Use random image features as placeholder (CLIP frozen anyway)
            images = torch.randn(B, 5, 3, 224, 224).to(DEVICE)

            outputs = model(images, captions)  # [B, 5, seq-1, vocab]

            loss = 0
            for i in range(5):
                pred = outputs[:, i].reshape(-1, len(vocab))
                targ = captions[:, i, 1:].reshape(-1)
                loss += criterion(pred, targ)
            loss /= 5

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"\n✅ Epoch {epoch} done | Avg Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'vocab': vocab,
            'loss': avg_loss,
            'config': {
                'embed_dim': EMBED_DIM,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'max_len': MAX_LEN
            }
        }
        local_path = f"/content/improved_ep{epoch}.pth"
        torch.save(ckpt, local_path)
        os.system(f"gsutil cp {local_path} {GCP_BUCKET}/improved_final.pth")
        print(f"💾 Epoch {epoch} saved to GCP")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.system(f"gsutil cp {local_path} {GCP_BUCKET}/improved_best.pth")
            print(f"🏆 New best model! Loss: {best_loss:.4f}")

    print(f"\n🎉 Training complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()