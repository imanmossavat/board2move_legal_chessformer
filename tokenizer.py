import torch
import chess
from typing import Tuple

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((64, 8), dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = piece.piece_type - 1
            color_idx = 0 if piece.color == chess.WHITE else 1

            # Set piece type (first 6 dims)
            tensor[square, piece_idx] = 1.0

            # Set color (last 2 dims)
            tensor[square, 6] = 1.0 if color_idx == 0 else 0.0
            tensor[square, 7] = 1.0 if color_idx == 1 else 0.0

    return tensor


def encode_castling_rights(board: chess.Board) -> torch.Tensor:
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]
    tensor = torch.zeros((4, 8), dtype=torch.float32)
    for i, flag in enumerate(rights):
        if flag:
            tensor[i] = 1.0
    return tensor


def encode_en_passant(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((64, 8), dtype=torch.float32)
    if board.ep_square is not None:
        tensor[board.ep_square] = 1.0
    return tensor


def encode_side_to_move(board: chess.Board) -> torch.Tensor:
    value = 1.0 if board.turn == chess.WHITE else 0.0
    return torch.full((1, 8), value, dtype=torch.float32)


def tokenize_board_uniform(board: chess.Board) -> torch.Tensor:
    return torch.cat([
        board_to_tensor(board),
        encode_castling_rights(board),
        encode_en_passant(board),
        encode_side_to_move(board),
    ], dim=0)

def square_name_to_index(square: str) -> int:
    return chess.SQUARE_NAMES.index(square)


def square_name_to_file_rank(square: str) -> Tuple[int, int]:
    file = ord(square[0]) - ord('a')
    rank = int(square[1]) - 1
    return file, rank

def fen2board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    return board
