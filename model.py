import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalChessTransformer(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=3, num_heads=8, num_classes=4672, device= 'cuda'):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.file_embed = nn.Embedding(8, hidden_dim)
        self.rank_embed = nn.Embedding(8, hidden_dim)
        self.device= device

        files = torch.arange(8).repeat(8).to(self.device)     # 0–7, repeated by rank
        ranks = torch.arange(8).repeat_interleave(8).to(self.device)  # 0–7, each 8 times

        file_emb = self.file_embed(files)  # (64, hidden_dim)
        rank_emb = self.rank_embed(ranks)  # (64, hidden_dim)

        self.positional_encoding = file_emb + rank_emb  # (64, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
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

# Example loss:

def compute_loss(probs, target_index):
    """
    Cross entropy loss given masked probabilities and target index.
    """
    target_prob = probs[target_index]
    loss = -torch.log(target_prob + 1e-9)  # avoid log(0)
    return loss
