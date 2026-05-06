"""Model architectures for Storifai: baseline (ResNet-50 + LSTM) and
improved (CLIP ViT-B/32 + Cross-Image Attention + Transformer Decoder)."""
import math



def _sample_next(logits, do_sample=False, top_p=0.9):
    """Pick next token: argmax (greedy) or top-p nucleus sampling."""
    import torch
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = sorted_probs.cumsum(dim=-1)
    mask = cumprobs > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    idx_in_sorted = torch.multinomial(probs, num_samples=1)
    return sorted_idx.gather(-1, idx_in_sorted)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# --------------------------- baseline ---------------------------

class BaselineEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(backbone.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.proj = nn.Linear(2048, embed_dim)
        self.frozen = True
        self._set_frozen(True)

    def _set_frozen(self, frozen):
        for p in self.cnn.parameters():
            p.requires_grad = not frozen
        self.frozen = frozen

    def unfreeze(self):
        self._set_frozen(False)

    def forward(self, images):
        # images: (B, P, 3, H, W)
        B, P, C, H, W = images.shape
        x = images.view(B * P, C, H, W)
        with torch.set_grad_enabled(not self.frozen):
            feat = self.cnn(x).squeeze(-1).squeeze(-1)
        feat = self.proj(feat)
        return feat.view(B, P, -1)


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512,
                 max_len=30, pad_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.image_to_h = nn.Linear(embed_dim, hidden_dim)
        self.image_to_c = nn.Linear(embed_dim, hidden_dim)
        self.max_len = max_len
        self.pad_idx = pad_idx

    def forward(self, image_feat, captions):
        # image_feat: (B*P, embed_dim), captions: (B*P, T)
        h0 = self.image_to_h(image_feat).unsqueeze(0)
        c0 = self.image_to_c(image_feat).unsqueeze(0)

        embedded = self.embed(captions[:, :-1])
        out, _ = self.lstm(embedded, (h0, c0))
        logits = self.out(out)
        return logits

    def generate(self, image_feat, sos_idx, eos_idx, max_len=None,
                 temperature=1.0, do_sample=False, top_p=0.9):
        if max_len is None:
            max_len = self.max_len
        device = image_feat.device
        N = image_feat.size(0)

        h = self.image_to_h(image_feat).unsqueeze(0)
        c = self.image_to_c(image_feat).unsqueeze(0)
        tokens = torch.full((N, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            emb = self.embed(tokens[:, -1:])
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.out(out[:, -1]) / max(temperature, 1e-6)
            nxt = _sample_next(logits, do_sample=do_sample, top_p=top_p)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens


class BaselineModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, max_len=30, pad_idx=0):
        super().__init__()
        self.encoder = BaselineEncoder(embed_dim=embed_dim)
        self.decoder = LSTMDecoder(vocab_size, embed_dim=embed_dim,
                                    max_len=max_len, pad_idx=pad_idx)
        self.pad_idx = pad_idx
        self.max_len = max_len

    def forward(self, images, captions):
        # images: (B, P, 3, H, W), captions: (B, P, T)
        B, P, T = captions.shape
        feat = self.encoder(images)              # (B, P, D)
        feat_flat = feat.view(B * P, -1)
        cap_flat = captions.view(B * P, T)
        logits = self.decoder(feat_flat, cap_flat)  # (B*P, T-1, V)
        return logits

    @torch.no_grad()
    def generate(self, images, sos_idx, eos_idx, max_len=None,
                  temperature=1.0, do_sample=False, top_p=0.9):
        B, P = images.shape[:2]
        feat = self.encoder(images).view(B * P, -1)
        tokens = self.decoder.generate(feat, sos_idx, eos_idx,
                                        max_len=max_len,
                                        temperature=temperature,
                                        do_sample=do_sample,
                                        top_p=top_p)
        return tokens.view(B, P, -1)


# --------------------------- improved ---------------------------

class CLIPVisualEncoder(nn.Module):
    """Wraps OpenAI CLIP ViT-B/32 visual encoder (frozen)."""

    def __init__(self, embed_dim=512):
        super().__init__()
        import clip
        model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()
        # CLIP ViT-B/32 outputs 512-d features
        self.proj = nn.Linear(512, embed_dim) if embed_dim != 512 else nn.Identity()

    def forward(self, images):
        # images: (B, P, 3, 224, 224) — assumed already CLIP-normalized
        B, P, C, H, W = images.shape
        x = images.view(B * P, C, H, W)
        with torch.no_grad():
            feat = self.visual(x.type(self.visual.conv1.weight.dtype))
        feat = feat.float()
        feat = self.proj(feat)
        return feat.view(B, P, -1)


class CrossImageAttention(nn.Module):
    """Self-attention across the 5 images in a story."""

    def __init__(self, embed_dim=512, num_heads=8, num_layers=2,
                 dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                              num_layers=num_layers)
        self.pos = nn.Parameter(torch.randn(1, 5, embed_dim) * 0.02)

    def forward(self, image_feats):
        # image_feats: (B, P, D)
        x = image_feats + self.pos[:, : image_feats.size(1)]
        return self.encoder(x)


class TransformerStoryDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8,
                 num_layers=4, max_len=30, pad_idx=0, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim,
                                   padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,
                                              num_layers=num_layers)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len
        self.pad_idx = pad_idx

    @staticmethod
    def causal_mask(T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device),
                           diagonal=1)

    def forward(self, image_context, captions):
        # image_context: (B*P, S, D) — per-image attention context
        # captions: (B*P, T)
        T = captions.size(1) - 1
        tgt = self.embed(captions[:, :-1]) + self.pos[:, :T]
        tgt_mask = self.causal_mask(T, captions.device)
        out = self.decoder(tgt=tgt, memory=image_context, tgt_mask=tgt_mask)
        return self.out(out)

    @torch.no_grad()
    def generate(self, image_context, sos_idx, eos_idx, max_len=None,
                 temperature=1.0, do_sample=False, top_p=0.9):
        if max_len is None:
            max_len = self.max_len
        device = image_context.device
        N = image_context.size(0)
        tokens = torch.full((N, 1), sos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            T = tokens.size(1)
            tgt = self.embed(tokens) + self.pos[:, :T]
            tgt_mask = self.causal_mask(T, device)
            out = self.decoder(tgt=tgt, memory=image_context,
                                tgt_mask=tgt_mask)
            logits = self.out(out[:, -1]) / max(temperature, 1e-6)
            nxt = _sample_next(logits, do_sample=do_sample, top_p=top_p)
            tokens = torch.cat([tokens, nxt], dim=1)
        return tokens


class ImprovedModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8,
                  decoder_layers=4, attn_layers=2, max_len=30,
                  pad_idx=0, dropout=0.1):
        super().__init__()
        self.encoder = CLIPVisualEncoder(embed_dim=embed_dim)
        self.cross_image = CrossImageAttention(embed_dim=embed_dim,
                                                 num_heads=num_heads,
                                                 num_layers=attn_layers,
                                                 dropout=dropout)
        self.decoder = TransformerStoryDecoder(
            vocab_size, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=decoder_layers, max_len=max_len, pad_idx=pad_idx,
            dropout=dropout,
        )
        self.pad_idx = pad_idx
        self.max_len = max_len

    def _build_context(self, images):
        B, P = images.shape[:2]
        feat = self.encoder(images)              # (B, P, D)
        feat = self.cross_image(feat)            # (B, P, D)
        # Each per-image generation conditions on its own attended feature
        ctx = feat.view(B * P, 1, -1)            # (B*P, 1, D)
        return ctx

    def forward(self, images, captions):
        B, P, T = captions.shape
        ctx = self._build_context(images)        # (B*P, 1, D)
        cap_flat = captions.view(B * P, T)
        logits = self.decoder(ctx, cap_flat)
        return logits

    @torch.no_grad()
    def generate(self, images, sos_idx, eos_idx, max_len=None,
                  temperature=1.0, do_sample=False, top_p=0.9):
        B, P = images.shape[:2]
        ctx = self._build_context(images)
        tokens = self.decoder.generate(ctx, sos_idx, eos_idx,
                                        max_len=max_len,
                                        temperature=temperature,
                                        do_sample=do_sample,
                                        top_p=top_p)
        return tokens.view(B, P, -1)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
