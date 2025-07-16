import torch
import torch.nn.functional as F
from model import MinimalChessTransformer
from dataset import ChessMoveDataset, BufferedShuffleDataset
from tokenizer import fen2board
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np

def plot_move_info(legal_moves_mask, target_distribution, probs):
    """
    Plot legal moves mask, target distribution, and predicted probabilities side by side.
    
    Args:
        legal_moves_mask: 1D array or tensor, binary mask where legal moves=1, illegal=0
        target_distribution: 1D array or tensor of move probabilities (target labels)
        probs: 1D array or tensor of predicted move probabilities from the model
    """
    # Convert to numpy arrays if tensors
    if not isinstance(legal_moves_mask, np.ndarray):
        legal_moves_mask = legal_moves_mask.cpu().numpy()
    if not isinstance(target_distribution, np.ndarray):
        target_distribution = target_distribution.cpu().numpy()
    if not isinstance(probs, np.ndarray):
        probs = probs.cpu().numpy()
        
    vocab_size = len(probs)
    x = np.arange(vocab_size)
    
    fig, axs = plt.subplots(3, 1, figsize=(18, 5))
    
    # Legal moves mask barplot
    axs[0].bar(x, legal_moves_mask, color='green')
    axs[0].set_title("Legal Moves Mask")
    axs[0].set_xlabel("Move Index")
    axs[0].set_ylabel("Legal (1) or Illegal (0)")
    axs[0].set_ylim([-0.1, 1.1])
    
    # Target distribution barplot
    axs[1].bar(x, target_distribution, color='blue')
    axs[1].set_title("Target Move Distribution")
    axs[1].set_xlabel("Move Index")
    axs[1].set_ylabel("Probability")
    
    # Predicted probabilities barplot
    axs[2].bar(x, probs, color='orange')
    axs[2].set_title("Model Predicted Probabilities")
    axs[2].set_xlabel("Move Index")
    axs[2].set_ylabel("Probability")
    
    plt.tight_layout()
    plt.show()

def evaluate_model(checkpoint_path, pgn_path, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}")
    # Load dataset (streaming)
    base_dataset = ChessMoveDataset(pgn_path, epsilon=0.001)  
    move_vocab_size = len(base_dataset.uci_to_index)

    model = MinimalChessTransformer(num_classes=move_vocab_size, device=device).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = iter(base_dataset)  # raw iterator without shuffling

    correct_ranks = []
    actual_probs = []
    legal_mass = []
    illegal_mass = []
    
    if False:
        with torch.no_grad():
            for i in range(num_samples):
                board_tensor, target_distribution, *_ = next(dataset)
                board_tensor = board_tensor.unsqueeze(0).to(device)  # add batch dim
                target_distribution = target_distribution.squeeze().cpu()

                logits = model(board_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu()  # shape: [num_moves]

                # Determine actual move index (argmax of target distribution)
                actual_index = target_distribution.argmax().item()
                actual_prob = probs[actual_index].item()

                # Compute rank of actual move
                sorted_indices = torch.argsort(probs, descending=True)
                rank = (sorted_indices == actual_index).nonzero(as_tuple=True)[0].item() + 1

                # Mass on legal vs illegal
                legal_moves_mask = target_distribution > 0

                # Print summary stats
                print(f"Target distribution non-zero entries: {np.count_nonzero(target_distribution)}")
                print(f"Legal moves mask sum (num legal moves): {legal_moves_mask.sum().item()}")
                print(f"Top-5 probs: {probs.topk(5).values.tolist()}")
                print(f"Target distribution max value: {target_distribution.max().item():.4f}")
                print(f"Target move index: {actual_index}")

                plot_move_info(legal_moves_mask, target_distribution, probs.cpu())

    
    with torch.no_grad():
        for i in range(num_samples):
            board_tensor, target_distribution, *_ = next(dataset)
            board_tensor = board_tensor.unsqueeze(0).to(device)  # add batch dim
            target_distribution = target_distribution.squeeze().cpu()

            logits = model(board_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu()  # shape: [num_moves]

            # Determine actual move index (argmax of target distribution)
            actual_index = target_distribution.argmax().item()
            actual_prob = probs[actual_index].item()

            # Compute rank of actual move
            sorted_indices = torch.argsort(probs, descending=True)
            rank = (sorted_indices == actual_index).nonzero(as_tuple=True)[0].item() + 1

            # Mass on legal vs illegal
            legal_moves_mask = target_distribution > 0
            legal_sum = probs[legal_moves_mask].sum().item()
            illegal_sum = probs[~legal_moves_mask].sum().item()

            correct_ranks.append(rank)
            actual_probs.append(actual_prob)
            legal_mass.append(legal_sum)
            illegal_mass.append(illegal_sum)

            print(f"Sample {i+1}:")
            print(f"  Actual move index: {actual_index}")
            print(f"  Rank: {rank}")
            print(f"  Prob of actual move: {actual_prob:.4f}")
            print(f"  Legal move mass: {legal_sum:.4f}")
            print(f"  Illegal move mass: {illegal_sum:.4f}")
            print("-" * 40)


    # Summary stats
    print("\n=== Summary ===")
    print(f"Average rank of actual move: {sum(correct_ranks)/len(correct_ranks):.2f}")
    print(f"Average probability on actual move: {sum(actual_probs)/len(actual_probs):.4f}")
    print(f"Average legal move mass: {sum(legal_mass)/len(legal_mass):.4f}")
    print(f"Average illegal move mass: {sum(illegal_mass)/len(illegal_mass):.4f}")

    # Histogram of actual move ranks
    plt.figure(figsize=(10, 5))
    plt.hist(correct_ranks, bins=range(1, max(correct_ranks) + 2), edgecolor='black', align='left')
    plt.title("Histogram of Actual Move Ranks")
    plt.xlabel("Rank of Ground Truth Move")
    plt.ylabel("Frequency")
    plt.xticks(range(1, max(correct_ranks) + 1))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Histogram of actual probabilities
    plt.figure(figsize=(10, 5))
    plt.hist(actual_probs, bins=100, edgecolor='black', align='left')
    plt.title("Histogram of Probabilities of Actual Moves")
    plt.xlabel("Probabilities of Actual Moves")
    plt.ylabel("Frequency")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    checkpoint = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\data\checkpoints\model_epoch1_batch230000.pth"
    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    evaluate_model(checkpoint, pgn_path, num_samples=1000)
