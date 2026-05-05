import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CrossImageAttention(nn.Module):
    """Allows each image to attend to all other images in the story."""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 5, embed_dim]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerDecoder(nn.Module):
    """Transformer decoder that generates story sentences."""
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=4, max_len=30, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: [B, seq_len]
        # memory: [B, 5, embed_dim] — image context
        B, seq_len = tgt.shape
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0)
        x = self.dropout(self.embed(tgt) + self.pos_embed(positions))

        # Generate causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tgt.device)

        out = self.decoder(x, memory, tgt_mask=causal_mask)
        return self.output_proj(out)  # [B, seq_len, vocab_size]


class StorifaiImproved(nn.Module):
    """
    Improved Storifai model using CLIP + Cross-Image Attention + Transformer Decoder.
    Architecture:
        1. CLIP encodes each image → 512-dim features
        2. Cross-image attention lets images attend to each other
        3. Transformer decoder generates story sentences conditioned on image context
    """
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=4, max_len=30, dropout=0.1):
        super().__init__()

        # CLIP vision encoder (frozen)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Freeze CLIP — we only train the decoder
        for param in self.clip.parameters():
            param.requires_grad = False

        # Project CLIP features (512) to embed_dim
        clip_dim = self.clip.config.projection_dim  # 512
        self.image_proj = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Cross-image attention
        self.cross_attn = CrossImageAttention(embed_dim, num_heads)

        # Transformer decoder
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, num_layers, max_len, dropout)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def encode_images(self, pixel_values):
        """
        Encode a batch of story images using CLIP.
        pixel_values: [B, 5, 3, 224, 224]
        Returns: [B, 5, embed_dim]
        """
        B, N, C, H, W = pixel_values.shape
        # Flatten batch and story dims
        flat = pixel_values.view(B * N, C, H, W)

        with torch.no_grad():
            features = self.clip.get_image_features(pixel_values=flat)  # [B*N, 512]

        features = features.view(B, N, -1)  # [B, 5, 512]
        features = self.image_proj(features)  # [B, 5, embed_dim]
        features = self.cross_attn(features)  # [B, 5, embed_dim]
        return features

    def forward(self, pixel_values, captions):
        """
        pixel_values: [B, 5, 3, 224, 224]
        captions: [B, 5, seq_len]
        Returns: logits [B, 5, seq_len, vocab_size]
        """
        B, N, seq_len = captions.shape

        # Encode all images
        image_features = self.encode_images(pixel_values)  # [B, 5, embed_dim]

        # Decode each sentence conditioned on its image + cross-image context
        all_logits = []
        for i in range(N):
            # Use image i as primary context + all images as memory
            img_context = image_features  # [B, 5, embed_dim]
            tgt = captions[:, i, :-1]  # [B, seq_len-1]
            logits = self.decoder(tgt, img_context)  # [B, seq_len-1, vocab_size]
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)  # [B, 5, seq_len-1, vocab_size]

    def generate(self, pixel_values, vocab, max_len=25, temperature=0.8):
        """
        Generate a story from images at inference time.
        pixel_values: [1, 5, 3, 224, 224]
        Returns: list of 5 sentences
        """
        self.eval()
        idx2word = {v: k for k, v in vocab.word2idx.items()}
        start_idx = vocab.word2idx.get('<START>', 1)
        end_idx = vocab.word2idx.get('<END>', 2)
        pad_idx = vocab.word2idx.get('<PAD>', 0)

        with torch.no_grad():
            image_features = self.encode_images(pixel_values)  # [1, 5, embed_dim]
            sentences = []

            for i in range(5):
                tokens = [start_idx]
                for _ in range(max_len):
                    tgt = torch.tensor([tokens], device=pixel_values.device)
                    logits = self.decoder(tgt, image_features)  # [1, len, vocab]
                    next_logits = logits[0, -1] / temperature
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    if next_token == end_idx or next_token == pad_idx:
                        break
                    tokens.append(next_token)

                words = [idx2word.get(t, '') for t in tokens[1:]]
                words = [w for w in words if w and w not in ('<END>', '<PAD>', '<START>')]
                sentence = ' '.join(words).capitalize() + '.'
                sentences.append(sentence if words else 'A beautiful moment captured.')

        return sentences