import torch
import torch.nn as nn
import torch.nn.functional as F

class MinimalChessTransformer(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (learnable or fixed)
        self.pos_embed = nn.Parameter(torch.randn(133, hidden_dim))

        # Simple Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, board_tensor, legal_mask):
 
        x = self.embedding(board_tensor) 
    #    print(f"x shape is: {x.shape}")  
        x = x + self.pos_embed.unsqueeze(0)
    #    print(f"self.pos_embed.unsqueeze(0) shape is: {self.pos_embed.unsqueeze(0).shape}")
        x = self.transformer(x)   
    #    print(f"self.transformer(x)  shape is: {self.transformer(x).shape}")

        square_tokens = x[:, :64, :]   
        logits = self.classifier(square_tokens).squeeze(-1)  
    #    print(f"logits  shape is: {logits.shape}")

        masked_logits = logits.masked_fill(legal_mask == 0, float('-inf'))  
        probs = F.softmax(masked_logits, dim=1)  # softmax over 64 squares

        return logits, probs


# Example loss:

def compute_loss(probs, target_index):
    """
    Cross entropy loss given masked probabilities and target index.
    """
    target_prob = probs[target_index]
    loss = -torch.log(target_prob + 1e-9)  # avoid log(0)
    return loss
