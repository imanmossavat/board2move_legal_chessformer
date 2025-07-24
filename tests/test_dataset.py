import unittest
import tempfile
import os
import chess
import chess.pgn
import torch
import pickle
from core.dataset import ChessMoveDataset, BoardMovePairDataset
from torch.utils.data import DataLoader
from core.tokenizer import tokenize_board_uniform


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
            os.unlink(self.test_pgn.name)
        except Exception:
            pass

    def test_dataset_output_structure(self):
        dataset = ChessMoveDataset(self.test_pgn.name, include_meta=True)
        sample = next(iter(dataset))

        self.assertIsInstance(sample, tuple)
        self.assertEqual(len(sample), 3)

        board_tensor, target_distribution, meta = sample

        self.assertIsInstance(board_tensor, torch.Tensor)
        self.assertEqual(board_tensor.shape, (64, 13))

        self.assertIsInstance(target_distribution, torch.Tensor)
        self.assertEqual(target_distribution.shape[0], len(dataset.uci_to_index))
        self.assertAlmostEqual(target_distribution.sum().item(), 1.0, places=4)

        actual_uci = meta["actual_uci"]
        self.assertIsInstance(actual_uci, str)
        self.assertIn(actual_uci, dataset.uci_to_index)

    def test_dataset_yields_all_moves(self):
        dataset = ChessMoveDataset(self.test_pgn.name)
        samples = list(dataset)
        self.assertEqual(len(samples), 6)  # 6 plies (half-moves)

    def test_correct_next_move_tensor(self):
        dataset = ChessMoveDataset(self.test_pgn.name, include_meta=True)
        with open(self.test_pgn.name, 'r') as f:
            game = chess.pgn.read_game(f)

        board = game.board()
        for (board_tensor, target_distribution, meta), move in zip(dataset, game.mainline_moves()):
            expected_tensor = tokenize_board_uniform(board)
            self.assertTrue(torch.equal(board_tensor, expected_tensor))
            self.assertEqual(meta["actual_uci"], move.uci())

            actual_idx = dataset.uci_to_index[meta["actual_uci"]]
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
        self.assertEqual(len(samples), 8)  # 8 plies
        os.unlink(multi_pgn.name)
    
    def test_dataloader_training_mode(self):
        # Meta is OFF → batching is OK
        dataset = ChessMoveDataset(self.test_pgn.name, include_meta=False)
        loader = DataLoader(dataset, batch_size=2)

        board_batch, target_batch = next(iter(loader))

        self.assertEqual(board_batch.shape[0], 2)
        self.assertEqual(board_batch.shape[1:], (64, 13))
        self.assertEqual(target_batch.shape[0], 2)
        self.assertEqual(target_batch.shape[1], len(dataset.uci_to_index))


    def test_evaluation_mode_no_batching(self):
        # Meta is ON → no batching
        dataset = ChessMoveDataset(self.test_pgn.name, include_meta=True)
        sample = next(iter(dataset))

        board_tensor, target_distribution, meta = sample

        self.assertIsInstance(board_tensor, torch.Tensor)
        self.assertIsInstance(target_distribution, torch.Tensor)
        self.assertIsInstance(meta, dict)
        self.assertIn("board", meta)

    def test_empty_pgn_file(self):
        empty_pgn = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn")
        empty_pgn.write(b"")
        empty_pgn.close()
        dataset = ChessMoveDataset(empty_pgn.name)
        samples = list(dataset)
        self.assertEqual(samples, [])
        os.unlink(empty_pgn.name)

    def test_deterministic_output(self):
        dataset1 = list(ChessMoveDataset(self.test_pgn.name, include_meta=True))
        dataset2 = list(ChessMoveDataset(self.test_pgn.name, include_meta=True))

        for (b1, t1, m1), (b2, t2, m2) in zip(dataset1, dataset2):
            self.assertTrue(torch.equal(b1, b2))
            self.assertTrue(torch.equal(t1, t2))
            self.assertEqual(m1["actual_uci"], m2["actual_uci"])

    def test_board_move_pair_dataset(self):
        # Create dummy data and save it to a temp file
        dummy_data = [(torch.randn(64, 13), 12), (torch.randn(64, 13), 44)]
        temp_dir = tempfile.mkdtemp()
        data_path = os.path.join(temp_dir, "temp_board_moves.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(dummy_data, f)

        dataset = BoardMovePairDataset(data_path)
        samples = list(dataset)
        self.assertEqual(len(samples), 2)

        for board_tensor, move_idx in samples:
            self.assertIsInstance(board_tensor, torch.Tensor)
            self.assertEqual(board_tensor.shape, (64, 13))
            self.assertIsInstance(move_idx, int)

        os.unlink(data_path)
        os.rmdir(temp_dir)


if __name__ == '__main__':
    unittest.main()
