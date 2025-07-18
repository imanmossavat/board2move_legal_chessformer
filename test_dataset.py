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
        self.assertEqual(len(sample), 4)

        board_tensor, target_distribution, actual_uci,_ = sample

        self.assertIsInstance(board_tensor, torch.Tensor)
        self.assertEqual(board_tensor.shape[0], 64)  # 64 squares
        self.assertEqual(board_tensor.shape[1], 13)  # updated channel count

        self.assertIsInstance(target_distribution, torch.Tensor)
        self.assertEqual(target_distribution.shape[0], len(dataset.uci_to_index))
        self.assertAlmostEqual(target_distribution.sum().item(), 1.0, places=5)
        self.assertTrue((target_distribution >= 0).all().item())  # probs non-negative

        self.assertIsInstance(actual_uci, str)
        self.assertIn(actual_uci, dataset.uci_to_index)

        # Check that the probability for actual move is highest or close to (1 - epsilon)
        actual_idx = dataset.uci_to_index[actual_uci]
        self.assertGreater(target_distribution[actual_idx].item(), 1.0 - dataset.epsilon)

    def test_dataset_yields_all_moves(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        samples = list(dataset)
        self.assertEqual(len(samples), 6)  # 6 plies

    def test_correct_next_move_square(self):
        from tokenizer import tokenize_board_uniform

        dataset = ChessMoveDataset(self.test_pgn.name)
        with open(self.test_pgn.name, 'r') as f:
            game = chess.pgn.read_game(f)

        board = game.board()
        for (board_tensor, target_distribution, actual_uci), move in zip(dataset, game.mainline_moves()):
            expected_tensor = tokenize_board_uniform(board)
            self.assertTrue(torch.equal(board_tensor, expected_tensor))

            # actual_uci string should match move.uci()
            self.assertEqual(actual_uci, move.uci())

            # Target distribution should put most weight on actual move index
            actual_idx = dataset.uci_to_index[actual_uci]
            self.assertGreater(target_distribution[actual_idx].item(), 1.0 - dataset.epsilon)

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
        self.assertEqual(len(samples), 8)  # total moves from both games
        os.unlink(multi_pgn.name)

    def test_dataloader_integration(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        loader = DataLoader(dataset, batch_size=2)

        board_batch, target_batch, actual_uci_batch,_ = next(iter(loader))

        self.assertEqual(board_batch.shape[0], 2)
        self.assertEqual(board_batch.shape[1], 64)
        self.assertEqual(board_batch.shape[2], 13)  # batch x squares x channels

        self.assertEqual(target_batch.shape[0], 2)
        self.assertEqual(target_batch.shape[1], len(dataset.uci_to_index))

        self.assertEqual(len(actual_uci_batch), 2)
        self.assertIsInstance(actual_uci_batch[0], str)

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

        for (b1, t1, u1,_), (b2, t2, u2,_) in zip(dataset1, dataset2):
            self.assertTrue(torch.equal(b1, b2))
            self.assertTrue(torch.equal(t1, t2))
            self.assertEqual(u1, u2)

if __name__ == '__main__':
    unittest.main()
