# model.py
import torch
import torch.nn as nn
from core.positional_encoding import get_or_generate_encoding
from core.move_vocab_builder import load_or_build_vocab



class MinimalChessTransformer(nn.Module):
    """
A Transformer-based chess move predictor using a single shared affine‑bilinear interaction.

0. CLS Token & Logit Bias
   - Learn a dedicated [CLS] token (no positional encoding).
   - After the Transformer, use the CLS token output to compute
     an additional bias term on the final move logits via a learned
     linear layer.

1. Embedding & Transformer
   - Embed each of the 64 squares (input_dim → hidden_dim).
   - Add fixed positional encodings to the squares.
   - Prepend the learned CLS token to the sequence.
   - Process through TransformerEncoder layers for global context.

2. Square Feature Projection
   - Linearly project each square’s hidden_dim vector to proj_dim.
   - Append a constant 1 to each projected vector, yielding
     square_feats of shape (B, 64, proj_dim+1).

3. Shared Affine‑Bilinear Decoder
   - Learn one weight matrix W' of shape (proj_dim+1, proj_dim+1).
   - For move i:
       u' = [ square_feats[:, from_ids[i], :] ; 1 ]   # (B, proj_dim+1)
       v' = [ square_feats[:, to_ids[i],   :] ; 1 ]   # (B, proj_dim+1)
     Compute:
       logit_i = u'ᵀ · W' · v'
   - This single W' encodes:
       • Pure bilinear interactions: uᵀ W v  
       • Linear “from” term: uᵀ w_u  
       • Linear “to” term: w_vᵀ v  
       • Global bias: b

4. Precomputed Move Indices
   - Load `from_ids` and `to_ids` (shape `(num_classes,)`) to know which square
     each move departs from and lands on.

Args:
    input_dim (int):   Features per square (e.g. 13).
    hidden_dim (int):  Transformer embedding size.
    num_layers (int):  Number of TransformerEncoder layers.
    num_heads (int):   Number of attention heads.
    proj_dim (int):    Dimension of per-square features before appending 1.
    num_classes (int): Total moves in the vocabulary (e.g. 1968).
    """
    def __init__(
        self,
        input_dim=13,
        hidden_dim=128,
        num_layers=3,
        num_heads=8,
        proj_dim=89,
        num_classes=1968,
        device='cuda',
        git_version= None,
        model_version = f"v1.0.1" # CLS token introduced
    ):
        super().__init__()
        self.device = device
        self.git_version= git_version
        self.model_version= model_version

        # CLS token (learned, no positional encoding)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # shape: (1, 1, H)

        # 1. Transformer encoder
        self.embedding = nn.Linear(input_dim, hidden_dim)
        pe = get_or_generate_encoding(d_model=hidden_dim, board_size=8)
        self.register_buffer("positional_encoding", pe)  # (64, hidden_dim)

        ff = hidden_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # 2. Square projection
        self.square_proj = nn.Linear(hidden_dim, proj_dim)  # (hidden_dim → proj_dim)

        # 3. Move → square mappings
        _, _, from_ids, to_ids, _ = load_or_build_vocab()
        self.register_buffer("from_ids", from_ids)  # (num_classes,)
        self.register_buffer("to_ids",   to_ids)    # (num_classes,)

        # 4. Affine‑bilinear weight matrix of size (proj_dim+1)×(proj_dim+1)
        self.W_aug = nn.Parameter(torch.randn(proj_dim+1, proj_dim+1))

        self.cls_to_logit_bias = nn.Linear(hidden_dim, num_classes)


    def forward(self, board_tensor):  # (B, 64, input_dim)
        B = board_tensor.size(0)

        x = self.embedding(board_tensor)  # (B, 64, H)

        # Add positional encoding to 64 squares only
        x = x + self.positional_encoding.unsqueeze(0)  # (B, 64, H)

        # Prepend [CLS] token (broadcast to batch)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([cls, x], dim=1)  # (B, 65, H)

        # Transformer
        x = self.transformer(x)  # (B, 65, H)
        cls_out = x[:, 0, :]     # (B, H) — the [CLS] token output
        sq = x[:, 1:, :]         # (B, 64, H)

        # Project squares
        sq_proj = self.square_proj(sq)  # (B, 64, M)

        # Gather from/to features
        u = sq_proj[:, self.from_ids, :]  # (B, C, M)
        v = sq_proj[:, self.to_ids, :]    # (B, C, M)

        # Append 1 to each vector
        ones = u.new_ones(B, u.size(1), 1)  # (B, C, 1)
        u_aug = torch.cat([u, ones], dim=-1)
        v_aug = torch.cat([v, ones], dim=-1)

        # Bilinear interaction
        uW = torch.einsum("bcm,mn->bcn", u_aug, self.W_aug)
        logits = (uW * v_aug).sum(dim=-1)  # (B, C)

        # Optional: add [CLS]-based logit bias
        logits = logits + self.cls_to_logit_bias(cls_out)  # (B, C)

        return logits