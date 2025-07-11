import torch
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset, BufferedShuffleDataset
from model import MinimalChessTransformer
import torch.nn.functional as F
import os
import csv
from datetime import datetime

def train_loop(model, dataloader, optimizer, device, csv_writer, epoch, global_step, save_every=5000, data_dir=None):
    """
    Train the model for one epoch using soft target distributions.

    Each training target is a probability distribution over legal moves,
    combining (1 - epsilon) weight on the actual move played and 
    epsilon weight distributed among all legal alternatives.

    Args:
        model: The neural network model.
        dataloader: A DataLoader yielding (board_tensor, soft_target_distribution) batches.
        optimizer: Optimizer for model parameters.
        device: Torch device (e.g. "cuda" or "cpu").
        csv_writer: CSV writer to log training metrics.
        epoch: Current epoch number.
        global_step: Global training step counter.
        save_every: Save model checkpoint every N batches.
        data_dir: Root directory to store checkpoints.
    """
    model.train()
    total_loss = 0

    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    for batch_idx, (board_tensor, target_distribution, _) in enumerate(dataloader):
        board_tensor = board_tensor.to(device)
        target_distribution = target_distribution.to(device)

        optimizer.zero_grad()
        logits = model(board_tensor)
        log_probs = F.log_softmax(logits, dim=1)
        loss = criterion(log_probs, target_distribution)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        lr = optimizer.param_groups[0]['lr']
        csv_writer.writerow([global_step, epoch, batch_idx, loss.item(), lr])

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx > 0 and batch_idx % save_every == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"model_epoch{epoch}_batch{batch_idx}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")
    
    return global_step



def main():
    NEPOCHS= 1
    epsilon= 0.1
    print(f'Model learns to follow the game with probability 1-epsilon= {1-epsilon}')
    print('epsilon=1 means fully random legal move')
    print('epsilon=0 means deterministic user response')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    data_dir = os.path.join(os.path.dirname(pgn_path), "data")

    print(f'data is saved to: {data_dir}')

    dataset = BufferedShuffleDataset(ChessMoveDataset(pgn_path, epsilon= epsilon), buffer_size=10000)
    num_classes = len(dataset.dataset.uci_to_index)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = MinimalChessTransformer(num_classes=num_classes, device=device)
    model.to(model.device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Logs folder inside data folder next to PGN
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(data_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "training_loss.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["global_step", "epoch", "batch_idx", "loss", "lr"])
        global_step= 0
        for epoch in range(1, NEPOCHS+1):
            print(f"Epoch {epoch}")
            global_step= train_loop(model=model, 
                       dataloader=dataloader, 
                       optimizer=optimizer, 
                       device=device, 
                       csv_writer=csv_writer, 
                       epoch=epoch, 
                       global_step= global_step, 
                       data_dir=data_dir)

    final_path = os.path.join(data_dir, "minimal_transformer_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to: {final_path}")

if __name__ == "__main__":
    main()
