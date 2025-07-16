"""
train.py

This script handles training for a minimal chess move prediction model using a transformer-based architecture.
It includes logging, checkpointing, and data loading.

Modules:
- train_loop: Runs a single training loop.
- main: Loads data, initializes model, and handles the training process.
"""
import torch
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset, BufferedShuffleDataset
from model import MinimalChessTransformer
import torch.nn.functional as F
import os
import csv
from datetime import datetime

def train_loop(model, dataloader, optimizer, device, csv_writer, epoch, global_step, save_every=5000, data_dir=None):
    model.train()
    total_loss = 0
    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_buffer = []

    for batch_idx, (board_tensor, target_distribution, _) in enumerate(dataloader):
        board_tensor = board_tensor.to(device, non_blocking=True)
        target_distribution = target_distribution.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(board_tensor)
        log_probs = F.log_softmax(logits, dim=1)
        loss = criterion(log_probs, target_distribution)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        lr = optimizer.param_groups[0]['lr']
        log_buffer.append([global_step, epoch, batch_idx, loss.item(), lr])

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Flush CSV logs every 100 batches
        if len(log_buffer) >= 100:
            csv_writer.writerows(log_buffer)
            log_buffer.clear()

        if batch_idx > 0 and batch_idx % save_every == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"model_epoch{epoch}_batch{batch_idx}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

    # Flush remaining logs after epoch
    if len(log_buffer) > 0:
        csv_writer.writerows(log_buffer)
        log_buffer.clear()

    return global_step


def main():
    NEPOCHS = 1
    epsilon = 0.01
    print(f'Model learns to follow the game with probability 1-epsilon= {1 - epsilon}')
    print('epsilon=1 means fully random legal move')
    print('epsilon=0 means deterministic user response')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    data_dir = os.path.join(os.path.dirname(pgn_path), "data")

    print(f'data is saved to: {data_dir}')

    dataset = BufferedShuffleDataset(ChessMoveDataset(pgn_path, epsilon=epsilon), buffer_size=1000)
    num_classes = len(dataset.dataset.uci_to_index)

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = MinimalChessTransformer(num_classes=num_classes, device=device)
    model.to(device)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(data_dir, f"logs_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "training_loss.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["global_step", "epoch", "batch_idx", "loss", "lr"])
        global_step = 0
        for epoch in range(1, NEPOCHS + 1):
            print(f"Epoch {epoch}")
            global_step = train_loop(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                csv_writer=csv_writer,
                epoch=epoch,
                global_step=global_step,
                data_dir=data_dir,
                save_every=10000
            )

    final_path = os.path.join(data_dir, "minimal_transformer_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Timestamp: {timestamp}")
    print(f"Training log saved to: {csv_path}")
    print(f"Saved final model to: {final_path}")

if __name__ == "__main__":
    main()
