import unittest
import torch
import chess
from tokenizer import (
    board_to_tensor,
    encode_castling_rights,
    encode_en_passant,
    encode_side_to_move,
    square_name_to_index,
    square_name_to_file_rank,
    tokenize_board_uniform,
    fen2board
)

class TestTokenizer(unittest.TestCase):

    def setUp(self):
        self.start_fen = chess.STARTING_FEN
        self.ep_fen = "rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq e6 0 3"
        self.castling_fen = "rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQK1NR w KQ - 1 2"
        self.black_to_move_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"

    def test_board_to_tensor_shape(self):
        board = fen2board(self.start_fen)
        tensor = board_to_tensor(board)
        self.assertEqual(tensor.shape, (64, 8))

    def test_board_to_tensor_empty_squares(self):
        board = fen2board(self.start_fen)
        tensor = board_to_tensor(board)
        self.assertTrue(torch.all(tensor[27] == 0))  # d4

    def test_board_to_tensor_piece_encoding(self):
        board = fen2board(self.start_fen)
        tensor = board_to_tensor(board)
        square = chess.parse_square('e2')
        self.assertEqual(tensor[square, 0].item(), 1.0)  # pawn dim
        self.assertEqual(tensor[square, 6].item(), 1.0)  # white
        self.assertEqual(tensor[square, 7].item(), 0.0)  # not black

    def test_encode_castling_rights(self):
        board = fen2board(self.start_fen)
        tensor = encode_castling_rights(board)
        self.assertTrue(torch.all(tensor == 1.0))

        board2 = fen2board(self.castling_fen)
        tensor2 = encode_castling_rights(board2)
        self.assertEqual(tensor2[0, 0].item(), 1.0)  # white kingside = 1
        self.assertEqual(tensor2[2, 0].item(), 0.0)  # black kingside = 0

    def test_encode_en_passant(self):
        board = fen2board(self.ep_fen)
        tensor = encode_en_passant(board)
        ep_square = chess.parse_square('e6')
        self.assertTrue(torch.all(tensor[ep_square] == 1.0))
        self.assertTrue(torch.all(tensor[chess.parse_square('a1')] == 0))

    def test_encode_side_to_move(self):
        board = fen2board(self.start_fen)
        tensor = encode_side_to_move(board)
        self.assertTrue(torch.all(tensor == 1.0))  # white

        board2 = fen2board(self.black_to_move_fen)
        tensor2 = encode_side_to_move(board2)
        self.assertTrue(torch.all(tensor2 == 0.0))  # black

    def test_square_name_to_index_and_file_rank(self):
        idx = square_name_to_index('e4')
        self.assertEqual(idx, chess.parse_square('e4'))
        file, rank = square_name_to_file_rank('e4')
        self.assertEqual(file, 4)
        self.assertEqual(rank, 3)

    def test_tokenize_board_uniform_shape(self):
        board = fen2board(self.start_fen)
        tensor = tokenize_board_uniform(board)
        self.assertEqual(tensor.shape, (133, 8))


if __name__ == "__main__":
    unittest.main()
