import torch
import chess.pgn
from torch.utils.data import IterableDataset
from typing import List, Tuple, Generator
from move_vocab_builder import build_move_vocab
from tokenizer import tokenize_board_uniform  # make sure this is imported
from random import shuffle
from collections import deque



class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=1000):
        self.dataset = dataset
        self.buffer_size = buffer_size


    def __iter__(self):
        buffer = deque()
        iterator = iter(self.dataset)

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass

        while buffer:
            shuffle_buffer = list(buffer)
            shuffle(shuffle_buffer)
            for sample in shuffle_buffer:
                yield sample

            try:
                for _ in range(self.buffer_size):
                    buffer.append(next(iterator))
                    buffer.popleft()
            except StopIteration:
                break

class ChessMoveDataset(IterableDataset):
    def __init__(self, pgn_file_path: str, epsilon: float = 0.1):
        self.pgn_file_path = pgn_file_path
        self.epsilon= epsilon
        self.uci_to_index, _ = build_move_vocab()


    def parse_game(self, game: chess.pgn.Game) -> Generator[Tuple[torch.Tensor, torch.Tensor, str], None, None]:
        board = game.board()
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

            yield board_tensor, target_distribution, actual_uci
            board.push(move)

    def game_generator(self) -> Generator[Tuple[torch.Tensor, int], None, None]:
        with open(self.pgn_file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                yield from self.parse_game(game)

    def __iter__(self):
        return self.game_generator()
