"""
Positional Encoding for Chessboard Squares (2D Sinusoidal)

This module implements a 2D sinusoidal positional encoding designed
for an 8x8 chessboard.

Where each square is uniquely represented by 
its file (column) and rank (row).

Encoding Logic:
- The positional encoding vector has dimension `d_model`.
- `d_model` must be divisible by 4, because:
  - Half of the dimensions (`d_model / 2`) encode the file position,
  - The other half (`d_model / 2`) encode the rank position.
- Each half is further split equally between sine and cosine components:
  - `d_model / 4` dimensions use sine encoding of the position scaled by
    exponentially decreasing frequencies.
  - `d_model / 4` dimensions use cosine encoding similarly.

Mathematically, for dimension index i (0 ≤ i < d_model/4), file f, and rank r:

    PE(2i)     = sin( f × 10000^(-2i / d_model) )
    PE(2i + 1) = cos( f × 10000^(-2i / d_model) )
    PE(d_model/2 + 2i)     = sin( r × 10000^(-2i / d_model) )
    PE(d_model/2 + 2i + 1) = cos( r × 10000^(-2i / d_model) )


Functions:
- `sinusoidal_2d_encoding(d_model)` generates the full positional encoding tensor.
- `save_positional_encoding_to_json(tensor, path)` saves the tensor for reuse.
- `load_positional_encoding_from_json(path)` loads the tensor back.
- `plot_positional_encoding(tensor, unified_cmap=True)` visualizes the encoding for inspection.
"""


import json
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def get_file_rank_grid(board_size=8):
    files = torch.arange(board_size).repeat(board_size)  # x (columns)
    ranks = torch.arange(board_size).repeat_interleave(board_size)  # y (rows)
    return files, ranks


def generate_2d_sinusoidal_encoding(d_model=32, board_size=8):
    assert d_model % 4 == 0, "d_model must be divisible by 4"
    files, ranks = get_file_rank_grid(board_size)

    frequencies = torch.pow(10000, -2 * torch.arange(0, d_model // 4) / d_model)

    file_embed = torch.zeros(board_size * board_size, d_model // 2)
    rank_embed = torch.zeros(board_size * board_size, d_model // 2)

    for i in range(d_model // 4):
        file_embed[:, 2 * i]     = torch.sin(files * frequencies[i])
        file_embed[:, 2 * i + 1] = torch.cos(files * frequencies[i])
        rank_embed[:, 2 * i]     = torch.sin(ranks * frequencies[i])
        rank_embed[:, 2 * i + 1] = torch.cos(ranks * frequencies[i])

    combined = torch.cat([file_embed, rank_embed], dim=1)  # shape: (64, d_model)
    return combined


def save_positional_encoding_to_json(
        tensor, 
        path="data/positional_encoding/pos_enc.json"):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)  # 👈 create the directory if needed
    array = tensor.detach().cpu().numpy().tolist()
    with open(path, "w") as f:
        json.dump(array, f)
    print(f'Positional Encoding JSON saved to {path}')


def load_positional_encoding_from_json(
        path="data/positional_encoding/pos_enc.json"):
    
    with open(path, "r") as f:
        data = json.load(f)
    return torch.tensor(data, dtype=torch.float32)


def get_or_generate_encoding(
        d_model=32, 
        board_size=8, 
        path="data/positional_encoding/pos_enc.json"):
    
    if path and os.path.exists(path):
        return load_positional_encoding_from_json(path)
    encoding = generate_2d_sinusoidal_encoding(d_model, board_size)
    if path:
        save_positional_encoding_to_json(encoding, path)
    return encoding


def plot_positional_encoding(tensor, unified_color_scale=True, d_model=32):
    n_cols = 8
    n_rows = (d_model + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten()

    if unified_color_scale:
        vmin = tensor.min().item()
        vmax = tensor.max().item()

    for dim in range(d_model):
        board_matrix = tensor[:, dim].view(8, 8).numpy()

        sns.heatmap(
            board_matrix,
            cmap="coolwarm",
            linewidths=0.5,
            square=True,
            cbar=False,
            vmin=(vmin if unified_color_scale else None),
            vmax=(vmax if unified_color_scale else None),
            ax=axes[dim],
        )
        axes[dim].set_title(f"Dim {dim}")
        axes[dim].invert_yaxis()
        axes[dim].set_xticks([])
        axes[dim].set_yticks([])

    for ax in axes[d_model:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
