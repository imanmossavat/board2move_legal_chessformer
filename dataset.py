import torch
import chess.pgn
from torch.utils.data import IterableDataset
from typing import List, Tuple, Generator

from tokenizer import tokenize_board_uniform, build_move_vocab  # make sure this is imported
from random import shuffle
from collections import deque



class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=5000):
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
    def __init__(self, pgn_file_path: str):
        self.pgn_file_path = pgn_file_path
        self.uci_to_index, _ = build_move_vocab()



    def parse_game(self, game: chess.pgn.Game) -> Generator[Tuple[torch.Tensor, int], None, None]:

        board = game.board()
        for move in game.mainline_moves():
            board_tensor = tokenize_board_uniform(board)  # shape [X, 8]
            
            target_index = self.uci_to_index[move.uci()]


            yield board_tensor, target_index
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
