import time
import chess.pgn
import torch
from tokenizer import tokenize_board_uniform  # must match your dataset logic

def profile_dataset(pgn_path=r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn",
                    max_games=1000):
    total_games = 0
    total_moves = 0

    disk_time = 0.0
    chess_ops_time = 0.0

    with open(pgn_path, 'r') as pgn_file:
        while total_games < max_games:
            start_disk = time.time()
            game = chess.pgn.read_game(pgn_file)
            disk_time += time.time() - start_disk

            if game is None:
                break

            total_games += 1
            board = game.board()
            moves_in_game = 0

            for move in game.mainline_moves():
                start_chess = time.time()

                # Simulate dataset internals
                board_tensor = tokenize_board_uniform(board)  # 13 x 8 x 8
                legal_mask = torch.zeros(64)
                for legal_move in board.legal_moves:
                    legal_mask[legal_move.from_square] = 1.0
                move_from = move.from_square

                chess_ops_time += time.time() - start_chess

                board.push(move)
                moves_in_game += 1

            total_moves += moves_in_game

    print(f"\n--- Profiling Summary ---")
    print(f"Total games processed: {total_games}")
    print(f"Total moves processed: {total_moves}")
    print(f"Total disk reading time: {disk_time:.3f} seconds")
    print(f"Total chess operations time (encoding + legal mask): {chess_ops_time:.3f} seconds")
    print(f"Average moves per game: {total_moves / total_games if total_games else 0:.2f}")
    print(f"Average disk time per game: {disk_time / total_games if total_games else 0:.4f} seconds")
    print(f"Average chess ops time per move: {chess_ops_time / total_moves if total_moves else 0:.5f} seconds")

if __name__ == "__main__":
    profile_dataset()
