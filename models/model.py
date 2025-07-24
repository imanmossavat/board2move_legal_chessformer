# model.py
import torch
import torch.nn as nn
from core.positional_encoding import get_or_generate_encoding
from core.move_vocab_builder import load_or_build_vocab

class MinimalChessTransformer(nn.Module):
    """
A Transformer-based chess move predictor using a single shared affine‑bilinear interaction.

1. Embedding & Transformer
   - Embed each of the 64 squares (input_dim → hidden_dim).
   - Add fixed positional encodings.
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
        device='cuda'
    ):
        super().__init__()
        self.device = device

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

    def forward(self, board_tensor):  # board_tensor: (B, 64, input_dim)
        B = board_tensor.size(0)

        # 1. Encode
        x = self.embedding(board_tensor)                   # (B, 64, H)
        x = x + self.positional_encoding.unsqueeze(0)      # (B, 64, H)
        x = self.transformer(x)                            # (B, 64, H)

        # 2. Project to proj_dim
        sq = self.square_proj(x)                           # (B, 64, M)

        # 3. Gather u, v for each move
        u = sq[:, self.from_ids, :]                        # (B, C, M)
        v = sq[:, self.to_ids,   :]                        # (B, C, M)

        # 4. Augment with constant 1
        ones = u.new_ones(B, u.size(1), 1)                 # (B, C, 1), C= 1968
        u_aug = torch.cat([u, ones], dim=-1)               # (B, C, M+1)
        v_aug = torch.cat([v, ones], dim=-1)               # (B, C, M+1)

        # 5. Affine‑bilinear: u'ᵀ W' v'
        #   first: (B,C,M+1) @ (M+1,M+1) → (B,C,M+1)
        uW = torch.einsum("bcm,mn->bcn", u_aug, self.W_aug)
        #   then dot with v_aug: elementwise * sum over last dim
        logits = (uW * v_aug).sum(dim=-1)                  # (B, C)

        return logits
