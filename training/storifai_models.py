"""
Storifai models — baseline (ResNet-50 + LSTM) and improved (CLIP + Cross-Attn + Transformer).

Both have identical interfaces:
    forward(images, captions) -> logits over vocab
    generate(images, max_len) -> generated token IDs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import math


# ======================================================================
#  BASELINE: ResNet-50 + LSTM
# ======================================================================

class BaselineModel(nn.Module):
    """
    ResNet-50 image encoder + LSTM decoder.

    Each of the 5 images is encoded independently. For each image, we generate
    its caption using an LSTM whose initial hidden state is the projected image
    feature.

    Forward:
        images:   (B, 5, 3, 224, 224)
        captions: (B, 5, T)  — token IDs (input shifted right with SOS)
    Returns:
        logits: (B, 5, T, V)
    """
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, num_layers=1,
                 dropout=0.3, freeze_encoder=True, pad_idx=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Image encoder — ResNet-50 (ImageNet pretrained)
        resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        # Remove final FC; output is (B, 2048, 1, 1) -> flatten to (B, 2048)
        self.image_feature_dim = resnet.fc.in_features  # 2048
        resnet.fc = nn.Identity()
        self.encoder = resnet
        self.encoder_frozen = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Project image feature → LSTM init hidden state
        self.feat_to_h = nn.Linear(self.image_feature_dim, hidden_dim * num_layers)
        self.feat_to_c = nn.Linear(self.image_feature_dim, hidden_dim * num_layers)

        # Word embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # LSTM decoder
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)

        # Output projection
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder_frozen = False

    def encode_image(self, images):
        """images: (N, 3, 224, 224) -> (N, 2048)"""
        if self.encoder_frozen:
            with torch.no_grad():
                feats = self.encoder(images)
        else:
            feats = self.encoder(images)
        return feats

    def forward(self, images, captions):
        """
        images:   (B, 5, 3, 224, 224)
        captions: (B, 5, T)  — for teacher forcing, captions[:, :, :-1] is input,
                               captions[:, :, 1:] is target
        Returns:
            logits: (B, 5, T-1, V)
        """
        B, P, C, H, W = images.shape  # P=5
        T = captions.shape[2]

        # Encode all 5 images at once
        flat_images = images.view(B * P, C, H, W)
        feats = self.encode_image(flat_images)  # (B*5, 2048)

        # Project to LSTM init state
        h0 = self.feat_to_h(feats).view(B * P, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.feat_to_c(feats).view(B * P, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # Reshape captions to (B*5, T)
        flat_captions = captions.view(B * P, T)
        # Input: all but last token (teacher forcing)
        decoder_input = flat_captions[:, :-1]  # (B*5, T-1)
        embeds = self.embedding(decoder_input)  # (B*5, T-1, E)

        outputs, _ = self.lstm(embeds, (h0, c0))  # (B*5, T-1, H)
        outputs = self.dropout(outputs)
        logits = self.out(outputs)  # (B*5, T-1, V)
        return logits.view(B, P, T - 1, self.vocab_size)

    @torch.no_grad()
    def generate(self, images, max_len, sos_idx, eos_idx, pad_idx):
        """
        images: (B, 5, 3, 224, 224)
        Returns: (B, 5, max_len) token IDs (greedy decoding)
        """
        B, P, C, H, W = images.shape
        flat_images = images.view(B * P, C, H, W)
        feats = self.encode_image(flat_images)

        h = self.feat_to_h(feats).view(B * P, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c = self.feat_to_c(feats).view(B * P, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

        # Start with SOS
        N = B * P
        current = torch.full((N, 1), sos_idx, dtype=torch.long, device=images.device)
        outputs = []
        finished = torch.zeros(N, dtype=torch.bool, device=images.device)

        for _ in range(max_len - 1):
            embeds = self.embedding(current)  # (N, 1, E)
            out, (h, c) = self.lstm(embeds, (h, c))
            logits = self.out(out[:, -1, :])  # (N, V)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (N, 1)
            # Once finished, force pad
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_idx), next_token)
            outputs.append(next_token)
            finished = finished | (next_token.squeeze(1) == eos_idx)
            current = next_token
            if finished.all():
                break

        result = torch.cat(outputs, dim=1)  # (N, generated_len)
        # Pad to max_len-1 if early-stopped
        if result.shape[1] < max_len - 1:
            pad_amount = max_len - 1 - result.shape[1]
            result = F.pad(result, (0, pad_amount), value=pad_idx)
        return result.view(B, P, max_len - 1)


# ======================================================================
#  IMPROVED: CLIP ViT-B/32 + Cross-Image Attention + Transformer Decoder
# ======================================================================

class CrossImageAttention(nn.Module):
    """
    Self-attention across the 5 image features so each one is aware of the others.
    This lets the model build a coherent narrative across the photo sequence.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_feats):
        """image_feats: (B, 5, dim) — features of 5 images per story"""
        # Self-attn across the 5 images
        attn_out, _ = self.attn(image_feats, image_feats, image_feats)
        x = self.norm1(image_feats + self.dropout(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x  # (B, 5, dim)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class ImprovedModel(nn.Module):
    """
    CLIP ViT-B/32 (frozen) + Cross-Image Attention + Transformer Decoder.

    Architecture:
        1. Each image -> CLIP image features (B*5, 512)
        2. Reshape to (B, 5, 512), apply Cross-Image Attention
        3. For each of the 5 images, decode caption with Transformer Decoder
           where memory is the cross-attended image feature for that image
    """
    def __init__(self, vocab_size, d_model=512, num_decoder_layers=4, num_heads=8,
                 dim_feedforward=2048, dropout=0.1, pad_idx=0, max_len=30):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.max_len = max_len

        # CLIP encoder (frozen) — loaded lazily to allow swap if needed
        try:
            import clip  # type: ignore
            self._has_openai_clip = True
            self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_dim = 512
        except ImportError:
            # Fallback: use HuggingFace transformers CLIP if openai clip is not installed
            self._has_openai_clip = False
            from transformers import CLIPVisionModel, CLIPImageProcessor
            self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_dim = 768  # vision_model output dim

        # Project CLIP feature to d_model
        self.image_proj = nn.Linear(self.clip_dim, d_model)

        # Cross-image attention (2 layers for richer cross-image reasoning)
        self.cross_image_layers = nn.ModuleList([
            CrossImageAttention(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(2)
        ])

        # Token embedding + positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len + 2)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.out = nn.Linear(d_model, vocab_size)

        # Init
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)

    def encode_clip(self, images):
        """images: (N, 3, 224, 224) -> (N, clip_dim)"""
        with torch.no_grad():
            if self._has_openai_clip:
                # OpenAI CLIP expects float and 224x224 input
                feats = self.clip_model.encode_image(images)
                feats = feats.float()
            else:
                # HuggingFace returns BaseModelOutputWithPooling; use pooler_output
                outputs = self.clip_model(pixel_values=images)
                feats = outputs.pooler_output
        return feats

    def encode_images(self, images):
        """images: (B, 5, 3, 224, 224) -> (B, 5, d_model)"""
        B, P, C, H, W = images.shape
        flat = images.view(B * P, C, H, W)
        clip_feats = self.encode_clip(flat)  # (B*P, clip_dim)
        feats = self.image_proj(clip_feats)  # (B*P, d_model)
        feats = feats.view(B, P, self.d_model)

        # Cross-image attention across the 5 images
        for layer in self.cross_image_layers:
            feats = layer(feats)
        return feats  # (B, 5, d_model)

    def _causal_mask(self, sz, device):
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, images, captions):
        """
        images:   (B, 5, 3, 224, 224)
        captions: (B, 5, T)
        Returns: (B, 5, T-1, V)
        """
        B, P, C, H, W = images.shape
        T = captions.shape[2]

        image_feats = self.encode_images(images)  # (B, 5, d_model)

        # Decode each image's caption independently, using its own (cross-attended)
        # image feature as memory.
        # Reshape to process all (B*5) sequences in parallel
        flat_captions = captions.view(B * P, T)
        decoder_input = flat_captions[:, :-1]  # (B*P, T-1)
        tgt_emb = self.embedding(decoder_input) * math.sqrt(self.d_model)
        tgt_emb = self.pos_enc(tgt_emb)

        # Memory: each story-image is one "token" of memory length 1
        # (B, 5, d_model) -> (B*5, 1, d_model)
        memory = image_feats.view(B * P, 1, self.d_model)

        # Causal mask for self-attn
        tgt_len = decoder_input.shape[1]
        causal = self._causal_mask(tgt_len, decoder_input.device)

        # Padding mask for tgt
        tgt_key_padding_mask = (decoder_input == self.pad_idx)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (B*P, T-1, d_model)

        logits = self.out(out)  # (B*P, T-1, V)
        return logits.view(B, P, T - 1, self.vocab_size)

    @torch.no_grad()
    def generate(self, images, max_len, sos_idx, eos_idx, pad_idx):
        """images: (B, 5, 3, 224, 224) -> (B, 5, max_len-1)"""
        B, P, C, H, W = images.shape
        image_feats = self.encode_images(images)  # (B, 5, d_model)
        memory = image_feats.view(B * P, 1, self.d_model)

        N = B * P
        current = torch.full((N, 1), sos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(N, dtype=torch.bool, device=images.device)
        generated = []

        for step in range(max_len - 1):
            tgt_emb = self.embedding(current) * math.sqrt(self.d_model)
            tgt_emb = self.pos_enc(tgt_emb)
            causal = self._causal_mask(current.shape[1], current.device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal)
            logits = self.out(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_idx), next_token)
            generated.append(next_token)
            finished = finished | (next_token.squeeze(1) == eos_idx)
            current = torch.cat([current, next_token], dim=1)
            if finished.all():
                break

        result = torch.cat(generated, dim=1)
        if result.shape[1] < max_len - 1:
            pad_amount = max_len - 1 - result.shape[1]
            result = F.pad(result, (0, pad_amount), value=pad_idx)
        return result.view(B, P, max_len - 1)


def build_model(model_type, vocab_size, pad_idx=0, max_len=30):
    if model_type == 'baseline':
        return BaselineModel(vocab_size=vocab_size, pad_idx=pad_idx)
    elif model_type == 'improved':
        return ImprovedModel(vocab_size=vocab_size, pad_idx=pad_idx, max_len=max_len)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
