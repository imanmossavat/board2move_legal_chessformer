import torch
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset, BufferedShuffleDataset
from model import MinimalChessTransformer
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
def train_loop(model, dataloader, optimizer, device, writer, epoch, save_every=5000):
    model.train()
    total_loss = 0
    for batch_idx, (board_tensor, legal_mask, move_from_index) in enumerate(dataloader):
        board_tensor = board_tensor.to(device)
        legal_mask = legal_mask.to(device)
        move_from_index = move_from_index.to(device)

        optimizer.zero_grad()
        logits, _ = model(board_tensor, legal_mask)
        loss = F.cross_entropy(logits, move_from_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging
        global_step = epoch * 100000 + batch_idx
        writer.add_scalar("Loss/train", loss.item(), global_step)

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Save model checkpoint every N batches
        if batch_idx > 0 and batch_idx % save_every == 0:
            ckpt_path = f"checkpoints/model_epoch{epoch}_batch{batch_idx}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to: {ckpt_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    dataset = BufferedShuffleDataset(ChessMoveDataset(pgn_path), buffer_size=10000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = MinimalChessTransformer()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    writer = SummaryWriter(log_dir="runs/chess_transformer")

    for epoch in range(1, 6):
        print(f"Epoch {epoch}")
        train_loop(model, dataloader, optimizer, device, writer, epoch)

    torch.save(model.state_dict(), "minimal_transformer_final.pth")
    writer.close()

if __name__ == "__main__":
    main()
