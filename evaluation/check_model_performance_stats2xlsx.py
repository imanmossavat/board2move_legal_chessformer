import chess
import torch
import torch.nn.functional as F
from models.model import MinimalChessTransformer
from core.dataset import ChessMoveDataset
from core.move_vocab_builder import load_or_build_vocab
from core.dataset import ChessMoveDataset, BufferedShuffleDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
uci_to_index, index_to_uci, _, _, _ = load_or_build_vocab()

import pandas as pd
failure_cases = []
checkpoint_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\data\checkpoints\model_prior_tojul24\model_epoch1_batch230000.pth"

pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
num_samples= 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {checkpoint_path}")
# Load dataset (streaming)
base_dataset = ChessMoveDataset(pgn_path, epsilon=0.001, include_board= True)  
move_vocab_size = len(base_dataset.uci_to_index)

model = MinimalChessTransformer(num_classes=move_vocab_size, device=device).to(device)

ckpt = torch.load(checkpoint_path, map_location=device)
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model version: {ckpt['model_version']}, Git commit: {ckpt['git_version']}")
else:
    model.load_state_dict(ckpt)
    print("legacy model with no versioning info")


model.eval()

dataset = iter(base_dataset)  # raw iterator without shuffling


with torch.no_grad():
    for i in range(num_samples):
        board_tensor, target_distribution, meta = next(dataset)
        board_tensor = board_tensor.unsqueeze(0).to(device)
        target_distribution = target_distribution.squeeze().cpu()

        logits = model(board_tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu()

        actual_index = target_distribution.argmax().item()
        actual_prob = probs[actual_index].item()
        sorted_indices = torch.argsort(probs, descending=True)
        rank = (sorted_indices == actual_index).nonzero(as_tuple=True)[0].item() + 1
        model_top_index = probs.argmax().item()
        model_predicted_uci = index_to_uci.get(model_top_index, "UNKNOWN")
        legal_mask = target_distribution > 0
        legal_sum = probs[legal_mask].sum().item()
        illegal_sum = probs[~legal_mask].sum().item()

        board = meta["board"]
        fen = board.fen()
        actual_uci = meta["actual_uci"]

        move = chess.Move.from_uci(actual_uci)
        is_check = board.is_check()
        is_promotion = move.promotion is not None
        is_castle = board.is_castling(move)
        is_capture = board.is_capture(move)
        ply = board.ply()

        # Estimate phase based on ply
        if ply < 20:
            phase = "opening"
        elif ply < 40:
            phase = "middlegame"
        else:
            phase = "endgame"

        # Determine if it's a "failure"
        is_failure = (
            rank > 10
            or actual_prob < 0.1
            or (actual_prob > 0.8 and rank > 1)
        )

        if is_failure:
            failure_cases.append({
                "fen": fen,
                "uci": actual_uci,
                "model_predicted_move": model_predicted_uci,
                "rank": rank,
                "actual_prob": actual_prob,
                "legal_mass": legal_sum,
                "illegal_mass": illegal_sum,
                "is_check": is_check,
                "is_promotion": is_promotion,
                "is_castle": is_castle,
                "is_capture": is_capture,
                "phase": phase,
                "ply": ply,
            })

        # Optionally log progress
        print(f"Sample {i+1:04d}: rank={rank}, prob={actual_prob:.4f}, fail={is_failure}")


df = pd.DataFrame(failure_cases)
df.to_csv("chess_model_failures.csv", index=False)
df.to_excel("chess_model_failures.xlsx", index=False)
print(f"Saved {len(failure_cases)} failure cases to Excel.")
