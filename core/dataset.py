"""
dataset.py

Provides iterable datasets for training chess models on PGN game data or preprocessed (board, move) pairs.

Includes:

1. ChessMoveDataset:
   - Parses PGN files into training samples of (board_tensor, target_distribution).
   - Supports epsilon-smoothing across legal moves for robustness.
   - Can optionally return board metadata (actual move, legal moves, and board object) for curriculum filtering or debugging.
   - Can filter moves to include only positions from the winning side of the game.

2. BoardMovePairDataset:
   - Loads preprocessed (board_tensor, move_index) pairs from disk.
   - Designed for use in curriculum learning setups with curated or filtered data.

3. BufferedShuffleDataset:
   - Buffers and randomly shuffles samples from a wrapped iterable dataset.
   - Enables better stochasticity for stream-based datasets (e.g., large PGN files).

Usage:
    Use `ChessMoveDataset` for raw PGN training.
    Use `BoardMovePairDataset` when training on extracted (board, move) pairs.
    Wrap either in `BufferedShuffleDataset` for randomized sampling.

"""


import torch
import chess.pgn
from torch.utils.data import IterableDataset
from typing import Tuple, Generator
from move_vocab_builder import load_or_build_vocab
from tokenizer import tokenize_board_uniform  # make sure this is imported
from random import shuffle
from collections import deque
import pickle



class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=5000):
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = deque()
        iterator = iter(self.dataset)

        # Initial fill
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            print("[BufferedShuffleDataset] Reached end of dataset while filling buffer")

        # Continuous yield + replace
        while buffer:
            idx = torch.randint(0, len(buffer), (1,)).item()
            sample = buffer[idx]
            yield sample

            try:
                new_sample = next(iterator)
                buffer[idx] = new_sample  # replace the one we yielded
            except StopIteration:
                # No more data, remove the one we yielded
                del buffer[idx]

class ChessMoveDataset(IterableDataset):
    def __init__(
        self,
        pgn_file_path: str,
        epsilon: float = 0.1,
        include_meta: bool = False,
        only_winning_side: bool = False
    ):
        self.pgn_file_path = pgn_file_path
        self.epsilon = epsilon
        self.include_meta = include_meta
        self.only_winning_side = only_winning_side
        self.uci_to_index, _, _, _, _ = load_or_build_vocab()

    def parse_game(self, game: chess.pgn.Game) -> Generator:
        board = game.board()
        winning_side = None
        valid_results = {"1-0": "white", "0-1": "black", "1/2-1/2": "draw"}

        result = game.headers.get("Result", "*")
        winning_side = valid_results.get(result, None)  # None if invalid or "*"

        game_id = game.headers.get("Site", "Unknown Site") + " | " + game.headers.get("White", "Unknown White") + " vs " + game.headers.get("Black", "Unknown Black")

        for move_number, move in enumerate(game.mainline_moves(), 1):
            board_tensor = tokenize_board_uniform(board)
            legal_uci = [m.uci() for m in board.legal_moves if m.uci() in self.uci_to_index]

            if not legal_uci:
                print(f"[WARNING] Skipping position with no legal moves at move {move_number} in game: {game_id}")
                board.push(move)
                continue

            legal_indices = [self.uci_to_index[uci] for uci in legal_uci]
            target_distribution = torch.zeros(len(self.uci_to_index), dtype=torch.float32)
            legal_tensor = torch.tensor(legal_indices, dtype=torch.long)
            prob_per_legal = self.epsilon / len(legal_indices)
            target_distribution[legal_tensor] = prob_per_legal

            actual_uci = move.uci()
            assert actual_uci in self.uci_to_index, f"Actual move {actual_uci} not in move vocabulary!"

            actual_idx = self.uci_to_index[actual_uci]
            target_distribution[actual_idx] += 1.0 - self.epsilon

            # If filtering by only winning side
            if self.only_winning_side and winning_side is not None:
                if (winning_side == "white" and not board.turn) or (winning_side == "black" and board.turn):
                    board.push(move)
                    continue

            if self.include_meta:
                yield board_tensor, target_distribution, {
                    "actual_uci": actual_uci,
                    "legal_indices": legal_indices,
                    "board": board.copy(),  # defensive copy
                    "winning_side": winning_side,
                    "move_number": move_number
                }
            else:
                yield board_tensor, target_distribution

            board.push(move)

    def game_generator(self):
        with open(self.pgn_file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                yield from self.parse_game(game)

    def __iter__(self):
        return self.game_generator()


class BoardMovePairDataset(IterableDataset):
    """
    Dataset that loads saved (board_tensor, move_index) pairs from disk.
    """
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.samples = pickle.load(f)  # List of (board_tensor, move_idx)

    def __iter__(self):
        for board_tensor, move_index in self.samples:
            yield board_tensor, move_index
