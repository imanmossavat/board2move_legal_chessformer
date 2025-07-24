import unittest
from core.move_vocab_builder import  load_or_build_vocab
from core.dataset import ChessMoveDataset

class TestMoveVocabContents(unittest.TestCase):
    def setUp(self):
        self.uci_to_index, self.index_to_uci, self.from_ids, self.to_ids, self.promo_ids = load_or_build_vocab()


    def test_castling_moves_present(self):
        castling_moves = ['e1g1', 'e1c1', 'e8g8', 'e8c8']
        for move in castling_moves:
            self.assertIn(move, self.uci_to_index, f"Castling move {move} missing from vocab")

    def test_pawn_promotion_moves_present(self):
        promotions = [
            'a7a8q', 'h7h8r', 'c7c8b', 'e7e8n',  # white promotions
            'a2a1q', 'h2h1r', 'c2c1b', 'e2e1n'   # black promotions
        ]
        for promo in promotions:
            self.assertIn(promo, self.uci_to_index, f"Promotion move {promo} missing from vocab")

    def test_vocab_size_reasonable(self):
        self.assertGreater(len(self.uci_to_index), 1000, "Vocab size is suspiciously small")


class TestMoveVocabulary(unittest.TestCase):
    def setUp(self):
        self.uci_to_index, self.index_to_uci, self.from_ids, self.to_ids, self.promo_ids = load_or_build_vocab()
        self.pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
        self.dataset = ChessMoveDataset(self.pgn_path, epsilon=0.1)

    def test_moves_are_in_vocab(self):
        unknown_moves = set()
        num_tests = 10000

        for i, (_, _, uci,_) in enumerate(self.dataset):
            if i >= num_tests:
                break
            if uci not in self.uci_to_index:
                unknown_moves.add(uci)

        if unknown_moves:
            print(f"Unknown moves: {sorted(unknown_moves)}")
        self.assertEqual(len(unknown_moves), 0, f"Found unknown moves: {unknown_moves}")

    def test_component_shapes(self):
        self.assertEqual(self.from_ids.shape[0], len(self.index_to_uci))
        self.assertEqual(self.to_ids.shape[0], len(self.index_to_uci))
        self.assertEqual(self.promo_ids.shape[0], len(self.index_to_uci))


    def test_component_value_ranges(self):
        self.assertTrue(all(0 <= x < 64 for x in self.from_ids.tolist()))
        self.assertTrue(all(0 <= x < 64 for x in self.to_ids.tolist()))
        self.assertTrue(all(0 <= x <= 4 for x in self.promo_ids.tolist()))

if __name__ == "__main__":
    unittest.main()
