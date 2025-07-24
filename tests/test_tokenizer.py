import unittest
import torch
import chess
from core.tokenizer import (
    board_to_tensor,
    encode_castling_rights,
    encode_en_passant,
    encode_legal_moves_side,
    square_name_to_index,
    square_name_to_file_rank,
    tokenize_board_uniform,
    fen2board
)


class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.start_fen = chess.STARTING_FEN
        self.ep_fen = "rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 3"
        self.no_castling_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        self.white_kingside_only_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w K - 0 1"
        self.black_queenside_only_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b q - 0 1"

    def test_board_to_tensor_shape(self):
        board = fen2board(self.start_fen)
        tensor = board_to_tensor(board)
        self.assertEqual(tensor.shape, (64, 8))

    def test_tokenize_board_uniform_shape(self):
        board = fen2board(self.start_fen)
        tokenized = tokenize_board_uniform(board)
        self.assertEqual(tokenized.shape, (64, 13))  # 8 (piece) + 2 (castling) + 1 (en passant) + 2 (legal move)

    def test_castling_rights_none(self):
        board = fen2board(self.no_castling_fen)
        castling_tensor = encode_castling_rights(board)
        self.assertTrue(torch.all(castling_tensor == 0.0))

    def test_castling_rights_white_kingside_only(self):
        board = fen2board(self.white_kingside_only_fen)
        tensor = encode_castling_rights(board)
        self.assertEqual(tensor[7, 0].item(), 1.0)   # White kingside rook
        self.assertEqual(tensor[4, 0].item(), 1.0)   # White king
        self.assertEqual(tensor[0, 0].item(), 0.0)   # Queenside rook should be 0

    def test_castling_rights_black_queenside_only(self):
        board = fen2board(self.black_queenside_only_fen)
        tensor = encode_castling_rights(board)
        self.assertEqual(tensor[60, 1].item(), 1.0)  # Black king
        self.assertEqual(tensor[56, 1].item(), 1.0)  # Black queenside rook
        self.assertEqual(tensor[63, 1].item(), 0.0)  # Kingside rook should be 0

    def test_en_passant_encoding(self):
        board = fen2board(self.ep_fen)
        tensor = encode_en_passant(board)
        ep_square = chess.parse_square('e6')
        self.assertTrue(torch.all(tensor[ep_square] == 1.0))
        self.assertTrue(torch.all(tensor[chess.parse_square('a1')] == 0))

    def test_legal_moves_encoding(self):
        board = fen2board(self.start_fen)
        tensor = encode_legal_moves_side(board)
        self.assertEqual(tensor.shape, (64, 2))

        for sq in range(64):
            if tensor[sq, 0] == 1.0:  # White legal move from here
                piece = board.piece_at(sq)
                self.assertIsNotNone(piece)
                self.assertTrue(piece.color == chess.WHITE)

    def test_legal_moves_black_to_move(self):
        board = fen2board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        tensor = encode_legal_moves_side(board)

        for sq in range(64):
            if tensor[sq, 1] == 1.0:
                piece = board.piece_at(sq)
                self.assertIsNotNone(piece)
                self.assertTrue(piece.color == chess.BLACK)

    def test_square_name_conversion(self):
        idx = square_name_to_index('e4')
        self.assertEqual(idx, chess.parse_square('e4'))

        file, rank = square_name_to_file_rank('e4')
        self.assertEqual((file, rank), (4, 3))


if __name__ == "__main__":
    unittest.main()
