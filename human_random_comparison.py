import chess.pgn
import chess.engine
import random
import pandas as pd
import matplotlib.pyplot as plt

# === Config ===

PGN_PATH = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
STOCKFISH_PATH = r"C:\Users\imanm\Downloads\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

MAX_POSITIONS = 100
STOCKFISH_DEPTH = 12

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def get_eval(board):
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
        return info["score"].white().score(mate_score=10000)
    except Exception:
        return None

def process_game(game):
    board = game.board()
    results = []

    for move in game.mainline_moves():
        if board.is_game_over():
            break

        # Save position before move
        board_before = board.copy()
        turn = "white" if board.turn else "black"

        # Human move
        human_move = move
        board.push(human_move)
        human_eval = get_eval(board)

        # Random move
        board_random = board_before.copy()
        random_move = random.choice(list(board_random.legal_moves))
        board_random.push(random_move)
        random_eval = get_eval(board_random)

        # Pre-move eval
        eval_before = get_eval(board_before)

        if None not in (eval_before, human_eval, random_eval):
            delta_human = (human_eval - eval_before) if turn == "white" else (eval_before - human_eval)
            delta_random = (random_eval - eval_before) if turn == "white" else (eval_before - random_eval)
            results.append({
                "fen": board_before.fen(),
                "delta_human": delta_human,
                "delta_random": delta_random
            })

        if len(results) >= MAX_POSITIONS:
            break

    return results

def run_experiment():
    all_results = []
    with open(PGN_PATH, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            all_results.extend(process_game(game))
            if len(all_results) >= MAX_POSITIONS:
                break

    df = pd.DataFrame(all_results)
    df.to_csv("human_vs_random.csv", index=False)

    print("Avg Δ (Human):", df["delta_human"].mean())
    print("Avg Δ (Random):", df["delta_random"].mean())

    # Plot
    plt.hist(df["delta_human"], bins=50, alpha=0.6, label="Human", color="blue")
    plt.hist(df["delta_random"], bins=50, alpha=0.6, label="Random", color="red")
    plt.title("Stockfish Evaluation Delta")
    plt.xlabel("Δ (centipawns)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig("histogram_human_vs_random.png")
    plt.show()

    engine.quit()

if __name__ == "__main__":

    run_experiment()
