import chess
import chess.engine


def best_move_and_value(fen: str, engine_path: str, depth: int = 12):
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))

        # Best move suggested by engine
        best_move = info.get("pv")[0]  # principal variation first move
        if best_move is None:
            best_move = info.get("bestmove")  # fallback, sometimes None

        # Evaluation score from white's perspective
        score = info["score"].white()
        centipawn_score = score.score(mate_score=10000)  # converts mate to large int

    finally:
        engine.quit()

    return best_move, centipawn_score





def evaluate_fen(fen: str, engine_path: str, depth: int = 12) -> int:
    import chess
    import chess.engine

    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white().score(mate_score=10000)  # handle mate scores
    finally:
        engine.quit()
    
    return score

def test_stockfish():
    # Load Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    # Example position (starting position)
    board = chess.Board()

    # Analyze position at fixed depth (e.g., 12)
    info = engine.analyse(board, chess.engine.Limit(depth=12))

    # Print centipawn score
    score = info["score"].white()
    print(f"Evaluation: {score.score()} centipawns")

    engine.quit()

if __name__ ==  "__main__":
    stockfish_path = r"C:\Users\imanm\Downloads\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

    test_stockfish()


    fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3"  # Italian Game

    # The score is Stockfish’s evaluation of the position from White’s perspective, measured in centipawns (1/100th of a pawn).
    # Positive score → White is better
    # Negative score → Black is better
    score = evaluate_fen(fen, stockfish_path)
    print(f"Score: {score} cp")


    move, score = best_move_and_value(fen, stockfish_path)
    print(f"Best move: {move}")
    print(f"Evaluation (cp): {score}")