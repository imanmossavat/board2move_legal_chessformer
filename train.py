import torch
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset, BufferedShuffleDataset
from model import MinimalChessTransformer
import torch.nn.functional as F
import os
import csv
from datetime import datetime


def train_loop(model, dataloader, optimizer, device, csv_writer, epoch,global_step, save_every=5000, data_dir=None):
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    checkpoints_dir = os.path.join(data_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    for batch_idx, (board_tensor, target_index) in enumerate(dataloader):
        board_tensor = board_tensor.to(device)
        target_index = target_index.to(device)

        optimizer.zero_grad()
        logits = model(board_tensor)
        loss = criterion(logits, target_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1  # increment by 1 per batch

        csv_writer.writerow([global_step, epoch, batch_idx, loss.item()])

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if batch_idx > 0 and batch_idx % save_every == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"model_epoch{epoch}_batch{batch_idx}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")


def main():
    NEPOCHS= 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    data_dir = os.path.join(os.path.dirname(pgn_path), "data")

    print(f'data is saved to: {data_dir}')

    dataset = BufferedShuffleDataset(ChessMoveDataset(pgn_path), buffer_size=10000)
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
        csv_writer.writerow(["global_step", "epoch", "batch_idx", "loss"])
        global_step= 0
        for epoch in range(1, NEPOCHS+1):
            print(f"Epoch {epoch}")
            train_loop(model=model, 
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
