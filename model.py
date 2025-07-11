import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import get_or_generate_encoding

class MinimalChessTransformer(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=3, num_heads=8, num_classes=1968, device= 'cuda'):
        super().__init__()
        self.device= device

        # Linear projection from input features to hidden_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Non-trainable positional encoding stored as buffer
        # This tensor is part of the model's state (non-trainable), automatically moves with .to(device),
        # and is included in state_dict() for saving/loading.
        encoding = get_or_generate_encoding(d_model=hidden_dim, board_size=8)
        self.register_buffer("positional_encoding", encoding)

        ff_dim= hidden_dim * 4


        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward= ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, board_tensor):  # board_tensor: (batch_size, 64, input_dim)
        x = self.embedding(board_tensor)  # (batch_size, 64, hidden_dim)
        x = x + self.positional_encoding.unsqueeze(0)  # add positional encoding
        x = self.transformer(x)  # (batch_size, 64, hidden_dim)

        x = x.mean(dim=1)  # simple mean pooling over squares
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits