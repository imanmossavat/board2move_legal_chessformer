# test_stockfish.py
import unittest
import chess
import chess.engine

class TestStockfishEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Define this once for all tests
        cls.stockfish_path = r"C:\Users\imanm\Downloads\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"
        cls.depth = 12

    def test_engine_analysis_returns_score(self):
        board = chess.Board()
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].white().score()
            
            # Assert score is an int (centipawn)
            self.assertIsInstance(score, int)
        finally:
            engine.quit()

if __name__ == "__main__":
    unittest.main()
