import torch
import chess.pgn
from torch.utils.data import IterableDataset
from typing import List, Tuple, Generator

from tokenizer import tokenize_board_uniform, square_name_to_index  # make sure this is imported

class ChessMoveDataset(IterableDataset):
    def __init__(self, pgn_file_path: str):
        self.pgn_file_path = pgn_file_path

    def parse_game(self, game: chess.pgn.Game) -> Generator[Tuple[torch.Tensor, torch.Tensor, int], None, None]:
        board = game.board()
        for move in game.mainline_moves():
            board_tensor = tokenize_board_uniform(board)  # shape [X, 8]
            
            legal_mask = torch.zeros(64, dtype=torch.float32)
            for move_ in board.legal_moves:
                legal_mask[move_.from_square] = 1.0

            move_from_idx = move.from_square  # already an int [0–63]

            yield board_tensor, legal_mask, move_from_idx
            board.push(move)

    def game_generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor, int], None, None]:
        with open(self.pgn_file_path, 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                yield from self.parse_game(game)

    def __iter__(self):
        return self.game_generator()
