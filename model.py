"""
model.py

Defines a minimal transformer model for chess move prediction using positional encoding,
component-based biasing, and standard transformer encoder layers.
"""
import torch.nn as nn
from positional_encoding import get_or_generate_encoding
from move_vocab_builder import load_or_build_vocab

class MinimalChessTransformer(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=3, num_heads=8, num_classes=1968, device='cuda'):
        super().__init__()
        self.device = device

        # === Core Encoder ===
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoding = get_or_generate_encoding(d_model=hidden_dim, board_size=8)
        self.register_buffer("positional_encoding", encoding)

        ff_dim = hidden_dim * 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward= ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # === Load move component IDs (non-trainable) ===
        _, _, from_ids, to_ids, promo_ids = load_or_build_vocab()
        self.register_buffer("from_ids", from_ids)   # shape: (num_classes,)
        self.register_buffer("to_ids", to_ids)
        self.register_buffer("promo_ids", promo_ids)

        # === Component Bias Embeddings (learnable) ===
        self.from_bias = nn.Embedding(64, 1)
        self.to_bias = nn.Embedding(64, 1)
        self.promo_bias = nn.Embedding(5, 1)

    def forward(self, board_tensor):  # board_tensor: (batch_size, 64, input_dim)
        x = self.embedding(board_tensor)                           # (B, 64, H)
        x = x + self.positional_encoding.unsqueeze(0)              # (B, 64, H)
        x = self.transformer(x)                                    # (B, 64, H)
        x = x.mean(dim=1)                                          # (B, H)
        logits = self.classifier(x)                                # (B, num_classes)

        # === Add learned bias correction per move ===
        bias = (
            self.from_bias(self.from_ids) +       # (num_classes, 1)
            self.to_bias(self.to_ids) +           # (num_classes, 1)
            self.promo_bias(self.promo_ids)       # (num_classes, 1)
        ).squeeze(-1)                            # (num_classes,)

        logits = logits + bias  # Broadcasted over batch

        return logits
