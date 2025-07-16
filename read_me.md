## ♟️ Chess Move Prediction Project

This project implements a neural network-based system to predict the next move in a chess game. The model learns from PGN (Portable Game Notation) files and outputs probability distributions over a large vocabulary of chess moves. It's designed with flexibility, modularity, and practicality in mind for experimenting with transformer architectures on board game data.

---

## 📁 Project Structure & File Descriptions

### `train.py`

**Purpose**:
Main training script that:

* Loads a PGN file.
* Prepares a custom dataset of board states and target move distributions.
* Initializes the transformer model.
* Trains the model using KL divergence between predicted and target move distributions.
* Logs training progress to CSV and saves model checkpoints.

---

### `model.py`

**Purpose**:
Defines `MinimalChessTransformer`, a transformer encoder architecture for chess board state classification. It:

* Embeds board features.
* Uses positional encodings.
* Adds learned biases for move components (from-square, to-square, promotion).
* Outputs logits over the entire move vocabulary.

---

### `dataset.py`

**Purpose**:
Contains custom `IterableDataset` subclasses:

* `ChessMoveDataset`: Parses PGN files into `(board_tensor, target_distribution)` samples, optionally with epsilon noise.
* `BufferedShuffleDataset`: Buffers and randomly yields samples for better shuffling when using streaming datasets.

---

### `tokenizer.py`

**Purpose**:
Tokenizes a `chess.Board` into a tensor representation used as model input. Includes:

* Piece and color encodings.
* Castling rights.
* En passant square.
* Legal move indicators.

Provides utility functions such as:

* `fen2board()` – convert FEN strings to boards.
* `square_name_to_index()` – map square notation to index.

---

### `move_vocab_builder.py`

**Purpose**:
Builds and loads a vocabulary of all possible pseudo-legal chess moves. Also:

* Handles special moves like castling and promotions.
* Saves and loads vocab from disk.
* Computes per-move component indices (from/to/promo) used for bias embeddings.

---

### `positional_encoding.py` *(Referenced but not shown)*

**Purpose**:
Presumably generates positional encodings for transformer input, e.g., using sinusoidal or learned embeddings for square positions on the board.

---

## 📊 Key Features

* **PGN Parsing**: Converts real games into model-ready training samples.
* **Transformer Architecture**: Adapted to board games, with positional and piece-aware inputs.
* **Custom Move Vocabulary**: Comprehensive set of chess moves mapped to integer indices.
* **Streaming Dataset with Buffered Shuffle**: Efficient for large PGN files.
* **Epsilon-Greedy Target Distributions**: Blends correct move with legal alternatives for robustness.

---

## 🚀 How to Run

1. Place a `.pgn` file in your directory.
2. Edit the `pgn_path` in `train.py` to match your PGN file.
3. Run training:

   ```bash
   python train.py
   ```
4. Check `logs_<timestamp>/training_loss.csv` and `data/checkpoints/` for logs and saved models.



 list of **all functions** 
Sure! Here's the same list in a plain, dev-friendly format — useful for referencing or pasting into comments or docs:

---

**File: `train.py`**

* `train_loop(model, dataloader, optimizer, device, csv_writer, epoch, global_step, save_every=10000, data_dir=None)`
* `main()`

---

**File: `model.py`**

* `MinimalChessTransformer.__init__(self, input_dim=13, hidden_dim=128, num_layers=3, num_heads=8, num_classes=1968, device='cuda')`
* `MinimalChessTransformer.forward(self, board_tensor)`

---

**File: `dataset.py`**

* `BufferedShuffleDataset.__init__(self, dataset, buffer_size=5000)`
* `BufferedShuffleDataset.__iter__(self)`
* `ChessMoveDataset.__init__(self, pgn_file_path: str, epsilon: float = 0.1)`
* `ChessMoveDataset.parse_game(self, game: chess.pgn.Game)`
* `ChessMoveDataset.game_generator(self)`
* `ChessMoveDataset.__iter__(self)`

---

**File: `tokenizer.py`**

* `board_to_tensor(board: chess.Board)`
* `encode_castling_rights(board: chess.Board)`
* `encode_en_passant(board: chess.Board)`
* `encode_legal_moves_side(board: chess.Board)`
* `tokenize_board_uniform(board: chess.Board)`
* `square_name_to_index(square: str)`
* `square_name_to_file_rank(square: str)`
* `fen2board(fen: str)`

---

**File: `move_vocab_builder.py`**

* `build_move_vocab()`
* `build_move_component_indices(index_to_uci)`
* `load_or_build_vocab()`

