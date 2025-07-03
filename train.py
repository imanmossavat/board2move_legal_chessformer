import torch
from torch.utils.data import DataLoader
from dataset import ChessMoveDataset
from model import MinimalChessTransformer
import torch.nn.functional as F

def train_loop(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (board_tensor, legal_mask, move_from_index) in enumerate(dataloader):
        # Move to device (CUDA or CPU)
        board_tensor = board_tensor.to(device)         
#        print(f"board tensor shape is: {board_tensor.shape}")

        legal_mask = legal_mask.to(device)            
#        print(f"legal mask shape is: {legal_mask.shape}")
        move_from_index = move_from_index.to(device)  
#        print(f"move_from_index shape is: {move_from_index.shape}")

        optimizer.zero_grad()
        logits,_ = model(board_tensor, legal_mask)     

        loss = F.cross_entropy(logits, move_from_index)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Avg loss: {avg_loss:.4f}")

def main():
    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset and DataLoader
    pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
    dataset = ChessMoveDataset(pgn_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Model
    model = MinimalChessTransformer()
    model.to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 5. Train loop
    for epoch in range(1, 6):  # Train 5 epochs as example
        print(f"Epoch {epoch}")
        train_loop(model, dataloader, optimizer, device)

    # 6. Save model checkpoint
    torch.save(model.state_dict(), "minimal_transformer.pth")

if __name__ == "__main__":
    main()
