import torch
import chess
from typing import Tuple

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((64, 8), dtype=torch.float32)
    # Channels 0-5: piece types (pawn=1,...,king=6)
    # Channel 6: white piece presence
    # Channel 7: black piece presence
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
    tensor = torch.zeros((64, 2), dtype=torch.float32)
    # For white castling rights: mark squares of king and rooks if rights exist
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[chess.E1, 0] = 1.0  # white king square
        tensor[chess.H1, 0] = 1.0  # white kingside rook square
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[chess.E1, 0] = 1.0
        tensor[chess.A1, 0] = 1.0
    # For black castling rights:
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[chess.E8, 1] = 1.0  # black king square
        tensor[chess.H8, 1] = 1.0  # black kingside rook
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[chess.E8, 1] = 1.0
        tensor[chess.A8, 1] = 1.0
    return tensor



def encode_en_passant(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((64, 1), dtype=torch.float32)
    if board.ep_square is not None:
        tensor[board.ep_square, 0] = 1.0
    return tensor

def encode_legal_moves_side(board: chess.Board) -> torch.Tensor:
    tensor = torch.zeros((64,2), dtype=torch.float32)
    
    for move_ in board.legal_moves:
        tensor[move_.from_square, board.turn == chess.WHITE] = 1.0
    return tensor


def tokenize_board_uniform(board: chess.Board) -> torch.Tensor:
    return torch.cat([
        board_to_tensor(board),             # (64, 8)
        encode_castling_rights(board),      # (64, 2), one for each color
        encode_en_passant(board),           # (64, 1)
        encode_legal_moves_side(board),     # (64, 2), one for each color
    ], dim=1)

def square_name_to_index(square: str) -> int:
    return chess.SQUARE_NAMES.index(square)


def square_name_to_file_rank(square: str) -> Tuple[int, int]:
    file = ord(square[0]) - ord('a')
    rank = int(square[1]) - 1
    return file, rank

def fen2board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    return board


# def build_move_vocab():
#     uci_to_index = {}
#     index_to_uci = {}
#     index = 0

#     for from_square in chess.SQUARES:
#         for to_square in chess.SQUARES:
#             move = chess.Move(from_square, to_square)
#             uci = move.uci()
#             uci_to_index[uci] = index
#             index_to_uci[index] = uci
#             index += 1

#             # Add promotions
#             if chess.square_rank(to_square) in [0, 7]:
#                 for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
#                     promo_move = chess.Move(from_square, to_square, promotion=promo)
#                     uci = promo_move.uci()
#                     uci_to_index[uci] = index
#                     index_to_uci[index] = uci
#                     index += 1

#     return uci_to_index, index_to_uci
