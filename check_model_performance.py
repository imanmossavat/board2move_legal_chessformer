import torch
import torch.nn.functional as F
from model import MinimalChessTransformer
from dataset import ChessMoveDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import statistics
import time

# === Settings ===
pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
model_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\data\minimal_transformer_final.pth"
limit_dataset = None  # e.g., 5000 for fast testing
batch_size = 64
topk_values = [1, 3, 5, 10]

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# === Load Model ===
model = MinimalChessTransformer(device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Load Dataset ===
dataset = ChessMoveDataset(pgn_path, epsilon=0.0)
if limit_dataset:
    dataset = Subset(dataset, range(limit_dataset))
uci_to_index = dataset.dataset.uci_to_index if isinstance(dataset, Subset) else dataset.uci_to_index
index_to_uci = {v: k for k, v in uci_to_index.items()}

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

# === Metric Storage ===
correct_probs = []
legal_sums = []
ranks = []
topk_correct = {k: 0 for k in topk_values}
total = 0

# === Evaluation Loop ===
start_time = time.time()
with torch.no_grad():
    for board_tensor, target_distribution, boards in tqdm(dataloader, total=len(dataloader)):
        board_tensor = board_tensor.to(device, non_blocking=True)
        target_distribution = target_distribution.to(device, non_blocking=True)

        logits = model(board_tensor)
        probs = F.softmax(logits, dim=1)  # (batch, 1968)
        batch_size_actual = probs.size(0)

        # True move indices
        true_indices = target_distribution.argmax(dim=1)

        # p(correct move)
        correct_p_batch = probs.gather(1, true_indices.unsqueeze(1)).squeeze(1)
        correct_probs.extend(correct_p_batch.tolist())

        # p(legal moves)
        for i in range(batch_size_actual):
            legal_uci = [m.uci() for m in boards[i].legal_moves]
            legal_idx = [uci_to_index[m] for m in legal_uci if m in uci_to_index]
            legal_sum = probs[i, legal_idx].sum()
            legal_sums.append(legal_sum.item())

        # Rank of correct move
        sorted_indices = torch.argsort(probs, dim=1, descending=True)
        match_positions = (sorted_indices == true_indices.unsqueeze(1)).nonzero(as_tuple=False)
        batch_ranks = match_positions[:, 1] + 1  # 1-based rank
        ranks.extend(batch_ranks.tolist())

        # Top-k accuracy
        topk_preds = torch.topk(probs, max(topk_values), dim=1).indices  # (batch, k)
        for k in topk_values:
            correct_in_topk = (topk_preds[:, :k] == true_indices.unsqueeze(1)).any(dim=1).sum().item()
            topk_correct[k] += correct_in_topk

        total += batch_size_actual

# === Results ===
duration = time.time() - start_time
print("\n=== Evaluation Results ===")
print(f"Total samples: {total}")
print(f"Runtime: {duration:.2f} sec")
print(f"Average p(correct move): {statistics.mean(correct_probs):.4f}")
print(f"Average sum p(legal moves): {statistics.mean(legal_sums):.4f}")
print(f"Median rank of correct move: {statistics.median(ranks)}")
for k in topk_values:
    print(f"Top-{k} accuracy: {topk_correct[k] / total:.2%}")
