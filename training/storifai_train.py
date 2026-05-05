"""
Storifai unified training script.

Usage in Colab:
    %run storifai_train.py --model improved
    %run storifai_train.py --model baseline

Or as a function call:
    from storifai_train import train
    train(model_type='improved', num_epochs=15)
"""
import os
import json
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from storifai_dataset import build_dataloaders
from storifai_models import build_model
from storifai_eval import evaluate_model, format_metrics


def train(model_type='improved',
          dataset_json_path='/content/storifai_dataset.json',
          gcs_bucket='hoshmand-ai-datasets',
          num_epochs=15,
          batch_size=8,
          learning_rate=1e-4,
          max_len=30,
          unfreeze_baseline_after=5,
          patience=3,
          checkpoint_dir='/content/checkpoints',
          gcs_checkpoint_prefix='storifai/checkpoints',
          num_workers=2,
          eval_max_batches=None,
          warmup_epochs=1):
    """
    Train baseline or improved Storifai model on the prepared dataset.

    Returns: dict with final test metrics + path to best checkpoint in GCS
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Data ----
    print(f"\nLoading dataset from {dataset_json_path}...")
    train_loader, val_loader, test_loader, vocab, word2idx = build_dataloaders(
        dataset_json_path=dataset_json_path,
        gcs_bucket=gcs_bucket,
        batch_size=batch_size,
        num_workers=num_workers,
        max_len=max_len,
    )
    vocab_size = len(vocab)
    pad_idx = word2idx['<PAD>']
    print(f"  Vocab: {vocab_size}, train batches: {len(train_loader)}, val batches: {len(val_loader)}")

    # ---- Model ----
    print(f"\nBuilding {model_type} model...")
    model = build_model(model_type, vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {n_params/1e6:.2f}M  Trainable: {n_trainable/1e6:.2f}M")

    # ---- Optimizer ----
    # AdamW with linear warmup + cosine decay
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=0.01,
    )
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    scaler = GradScaler() if device.type == 'cuda' else None

    # ---- Training loop ----
    history = []
    best_val_bleu4 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    best_ckpt_local = os.path.join(checkpoint_dir, f"{model_type}_best.pth")

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Unfreeze baseline encoder after N epochs
        if model_type == 'baseline' and epoch == unfreeze_baseline_after:
            print(f"\n  >>> Unfreezing baseline encoder at epoch {epoch}")
            model.unfreeze_encoder()
            # Reinit optimizer to include the new params
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate * 0.1, weight_decay=0.01,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=(num_epochs - epoch) * len(train_loader))

        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['images'].to(device, non_blocking=True)
            captions = batch['captions'].to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                logits = model(images, captions)  # (B, 5, T-1, V)
                # Target: captions[:, :, 1:] (everything except SOS)
                targets = captions[:, :, 1:]  # (B, 5, T-1)
                # Flatten for cross-entropy
                loss = criterion(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            if batch_idx % 50 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{num_epochs} batch {batch_idx}/{len(train_loader)} "
                      f"loss={loss.item():.4f}  lr={cur_lr:.2e}")

        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- Validate ----
        print(f"\n  Validating epoch {epoch+1}...")
        val_metrics, val_samples = evaluate_model(
            model, val_loader, vocab, word2idx, device,
            max_len=max_len, max_batches=eval_max_batches, n_sample_stories=2)
        val_str = format_metrics(val_metrics)
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  {val_str}  ({elapsed/60:.1f} min)")

        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_metrics': val_metrics,
            'elapsed_sec': elapsed,
        })

        # ---- Save best model ----
        if val_metrics['bleu4'] > best_val_bleu4:
            best_val_bleu4 = val_metrics['bleu4']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'vocab': vocab,
                'word2idx': word2idx,
                'model_type': model_type,
            }, best_ckpt_local)
            print(f"  ✅ New best — saved to {best_ckpt_local}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")

        # ---- Early stopping ----
        if epochs_without_improvement >= patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # ---- Final test eval (load best checkpoint) ----
    print(f"\n{'='*60}")
    print(f"Loading best checkpoint (epoch {best_epoch}) for test eval...")
    print(f"{'='*60}")
    ckpt = torch.load(best_ckpt_local, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_metrics, test_samples = evaluate_model(
        model, test_loader, vocab, word2idx, device,
        max_len=max_len, n_sample_stories=5)
    print(f"\n📊 TEST RESULTS ({model_type}):")
    print(f"   {format_metrics(test_metrics)}")

    # ---- Show samples ----
    print(f"\n📝 SAMPLE GENERATIONS ({model_type}):\n")
    for i, sample in enumerate(test_samples):
        print(f"--- Sample {i+1} ---")
        print("  Ground truth:")
        for s in sample['gt_sentences']:
            print(f"    • {s}")
        print("  Generated:")
        for s in sample['gen_sentences']:
            print(f"    • {s}")
        print()

    # ---- Save final results bundle to GCS ----
    print(f"Uploading checkpoint and results to GCS...")
    import subprocess
    gcs_ckpt = f"gs://{gcs_bucket}/{gcs_checkpoint_prefix}/{model_type}_v2_best.pth"
    subprocess.run(['gsutil', 'cp', best_ckpt_local, gcs_ckpt], check=True, capture_output=True)
    print(f"  ✅ Checkpoint: {gcs_ckpt}")

    # Also save metrics + samples as JSON for paper writing
    results_bundle = {
        'model_type': model_type,
        'best_epoch': best_epoch,
        'best_val_bleu4': best_val_bleu4,
        'test_metrics': test_metrics,
        'test_samples': test_samples,
        'history': history,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'max_len': max_len,
            'patience': patience,
        },
    }
    results_local = os.path.join(checkpoint_dir, f"{model_type}_results.json")
    with open(results_local, 'w') as f:
        json.dump(results_bundle, f, indent=2, default=str)
    gcs_results = f"gs://{gcs_bucket}/{gcs_checkpoint_prefix}/{model_type}_v2_results.json"
    subprocess.run(['gsutil', 'cp', results_local, gcs_results], check=True, capture_output=True)
    print(f"  ✅ Results: {gcs_results}")

    print(f"\n🎉 Done training {model_type}.")
    return results_bundle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='improved',
                        choices=['baseline', 'improved'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()

    train(model_type=args.model,
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          patience=args.patience,
          num_workers=args.num_workers)
