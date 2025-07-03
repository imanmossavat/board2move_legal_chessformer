import unittest
import tempfile
import os
import chess
import chess.pgn
from dataset import ChessMoveDataset
from torch.utils.data import DataLoader
import torch

class TestChessMoveDataset(unittest.TestCase):

    def setUp(self):
        self.test_pgn = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn")
        self.test_pgn.write(b"""
[Event "Test"]
[Site "Local"]
[Date "2025.06.29"]
[Round "1"]
[White "White"]
[Black "Black"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
""")
        self.test_pgn.close()

    def tearDown(self):
        try:
            self.test_pgn.close()
        except Exception:
            pass
        os.unlink(self.test_pgn.name)

    def test_dataset_output_structure(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        iterator = iter(dataset)
        sample = next(iterator)

        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 3)

        board_tensor, legal_mask, move_from = sample

        self.assertIsInstance(board_tensor, torch.Tensor)
        self.assertEqual(board_tensor.shape[1], 8)  # Shape: [X, 8]
        self.assertIsInstance(legal_mask, torch.Tensor)
        self.assertEqual(legal_mask.shape, (64,))
        self.assertIsInstance(move_from, int)
        self.assertTrue(0 <= move_from < 64)
        self.assertEqual(legal_mask[move_from].item(), 1.0)  # move_from must be legal

    def test_dataset_yields_all_moves(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        samples = list(dataset)
        self.assertEqual(len(samples), 6)  # 6 half-moves (plies)

    def test_correct_next_move_square(self):
        from tokenizer import tokenize_board_uniform  # Ensure consistent encoding

        dataset = ChessMoveDataset(self.test_pgn.name)
        with open(self.test_pgn.name, 'r') as f:
            game = chess.pgn.read_game(f)

        board = game.board()
        for (board_tensor, _, move_from), move in zip(dataset, game.mainline_moves()):
            expected_tensor = tokenize_board_uniform(board)
            self.assertTrue(torch.equal(board_tensor, expected_tensor))
            self.assertEqual(move_from, move.from_square)
            board.push(move)

    def test_multiple_games(self):
        multi_pgn = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn")
        multi_pgn.write(b"""
[Event "Game1"]
[Site "Test"]
[Date "2025.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. d4 d5 2. c4 c6 1-0

[Event "Game2"]
[Site "Test"]
[Date "2025.01.02"]
[Round "1"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 0-1
""")
        multi_pgn.close()
        dataset = ChessMoveDataset(multi_pgn.name)
        samples = list(dataset)
        self.assertEqual(len(samples), 8)
        os.unlink(multi_pgn.name)

    def test_dataloader_integration(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        loader = DataLoader(dataset, batch_size=2)

        board_batch, mask_batch, from_batch = next(iter(loader))

        self.assertEqual(board_batch.shape[0], 2)
        self.assertEqual(mask_batch.shape, (2, 64))
        self.assertEqual(from_batch.shape, (2,))
        self.assertTrue(isinstance(from_batch[0].item(), int))

    def test_empty_pgn_file(self):
        empty_pgn = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn")
        empty_pgn.write(b"")
        empty_pgn.close()
        dataset = ChessMoveDataset(empty_pgn.name)
        samples = list(dataset)
        self.assertEqual(samples, [])
        os.unlink(empty_pgn.name)

    def test_deterministic_output(self):
        dataset1 = list(ChessMoveDataset(self.test_pgn.name))
        dataset2 = list(ChessMoveDataset(self.test_pgn.name))

        for (b1, l1, m1), (b2, l2, m2) in zip(dataset1, dataset2):
            self.assertTrue(torch.equal(b1, b2))
            self.assertTrue(torch.equal(l1, l2))
            self.assertEqual(m1, m2)

if __name__ == '__main__':
    unittest.main()
