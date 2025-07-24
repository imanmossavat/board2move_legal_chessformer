import os
import torch
import chess
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from model import MinimalChessTransformer
from dataset import ChessMoveDataset
from move_vocab_builder import load_or_build_vocab
import chess.engine
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Config ===
VERBOSE = True  # Set False to silence prints
FAILURE_RANK_THRESHOLD = 7
FAILURE_PROBABILITY_THRESHOLD = 0.01
FAILURE_HIGH_PROBABILITY_RANK = 3
ESTIMATED_ELO_K = 10

num_samples = 100
checkpoint_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\data\checkpoints\model_prior_tojul24\model_epoch1_batch230000.pth"
pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
stockfish_path = r"C:\Users\imanm\Downloads\stockfish\stockfish\stockfish-windows-x86-64-avx2.exe"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(os.path.dirname(pgn_path), "data", f"analysis_{timestamp}")
os.makedirs(log_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def load_model_and_dataset(checkpoint_path, pgn_path, device):
    vprint("Loading vocab and dataset...")
    uci_to_index, index_to_uci, _, _, _ = load_or_build_vocab()
    base_dataset = ChessMoveDataset(pgn_path, epsilon=0.001, include_meta=True)
    move_vocab_size = len(base_dataset.uci_to_index)
    vprint(f"Vocabulary size: {move_vocab_size}")
    vprint("Loading model...")
    model = MinimalChessTransformer(num_classes=move_vocab_size, device=device).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model, base_dataset, uci_to_index, index_to_uci


def evaluate_move_sample(board_tensor, target_distribution, meta, model, index_to_uci, engine, move_vocab_size, device):
    board_tensor = board_tensor.unsqueeze(0).to(device)
    target_distribution = target_distribution.squeeze().cpu()

    logits = model(board_tensor)
    if logits.shape[1] != move_vocab_size:
        raise ValueError(f"Model output size mismatch: got {logits.shape[1]}, expected {move_vocab_size}.")

    probs = F.softmax(logits, dim=1).squeeze().cpu()

    actual_index = target_distribution.argmax().item()
    actual_prob = probs[actual_index].item()

    sorted_indices = torch.argsort(probs, descending=True)
    rank = (sorted_indices == actual_index).nonzero(as_tuple=True)[0].item() + 1

    model_top_index = probs.argmax().item()
    model_predicted_uci = index_to_uci.get(model_top_index, "UNKNOWN")

    legal_mask = target_distribution > 0
    legal_sum = probs[legal_mask].sum().item()
    illegal_sum = probs[~legal_mask].sum().item()

    board = meta["board"].copy()
    actual_uci = meta["actual_uci"]
    if not actual_uci:
        vprint("No actual UCI move found in metadata. Skipping sample.")
        return None

    turn = "white" if board.turn else "black"

    # Stockfish eval before move
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=12))
        eval_before = info["score"].white().score(mate_score=10000)
    except Exception as e:
        vprint(f"Stockfish eval failed before move: {e}")
        eval_before = None

    # Human move delta
    delta_human = raw_delta_human = None
    try:
        board_human = board.copy()
        board_human.push_uci(actual_uci)
        info_human = engine.analyse(board_human, chess.engine.Limit(depth=12))
        eval_human = info_human["score"].white().score(mate_score=10000)
        if eval_before is not None and eval_human is not None:
            raw_delta_human = eval_human - eval_before
            delta_human = raw_delta_human if turn == "white" else -raw_delta_human
    except Exception as e:
        vprint(f"Error analyzing human move: {e}")

    # AI move delta
    delta_ai = raw_delta_ai = None
    illegal_move_failure = False
    try:
        board_ai = board.copy()
        if model_predicted_uci == "UNKNOWN":
            vprint("Skipping AI move analysis — predicted UCI is UNKNOWN.")
        else:
            move_obj = chess.Move.from_uci(model_predicted_uci)
            if move_obj in board_ai.legal_moves:
                board_ai.push_uci(model_predicted_uci)
                info_ai = engine.analyse(board_ai, chess.engine.Limit(depth=12))
                eval_ai = info_ai["score"].white().score(mate_score=10000)
                if eval_before is not None and eval_ai is not None:
                    raw_delta_ai = eval_ai - eval_before
                    delta_ai = raw_delta_ai if turn == "white" else -raw_delta_ai
            else:
                vprint(f"AI move {model_predicted_uci} is not legal in this position.")
                illegal_move_failure = True
    except Exception as e:
        vprint(f"Error analyzing AI move '{model_predicted_uci}': {e}")

    ply = board.ply()
    phase = "opening" if ply < 20 else "middlegame" if ply < 40 else "endgame"

    is_failure = (
        illegal_move_failure or
        rank > FAILURE_RANK_THRESHOLD or
        actual_prob < FAILURE_PROBABILITY_THRESHOLD or
        (actual_prob > 0.8 and rank > FAILURE_HIGH_PROBABILITY_RANK)
    )

    failure_data = {
        "fen": board.fen(),
        "uci": actual_uci,
        "model_predicted_move": model_predicted_uci,
        "rank": rank,
        "actual_prob": actual_prob,
        "legal_mass": legal_sum,
        "illegal_mass": illegal_sum,
        "stockfish_eval_before": eval_before,
        "raw_delta_human": raw_delta_human,
        "delta_human": delta_human,
        "raw_delta_ai": raw_delta_ai,
        "delta_ai": delta_ai,
        "turn": turn,
        "phase": phase,
        "ply": ply
    }

    return {"is_failure": is_failure, "failure_data": failure_data, "delta_ai": delta_ai}


def save_report_and_plots(failure_cases, delta_ai_list, board_tensor_shape, log_path, meta_data):
    df = pd.DataFrame(failure_cases)

    # Estimate model ELO from delta_ai
    estimated_model_elo = None
    if delta_ai_list:
        delta_ai_series = pd.Series([abs(d) for d in delta_ai_list if d is not None])
        if not delta_ai_series.empty:
            avg_delta_ai = delta_ai_series.mean()
            median_delta_ai = delta_ai_series.median()
            estimated_model_elo = 3000 - ESTIMATED_ELO_K * avg_delta_ai
            vprint(f"Estimated model ELO (based on avg |delta_ai|): {estimated_model_elo:.0f}")
        else:
            avg_delta_ai = median_delta_ai = None
            vprint("No valid delta_ai values for ELO estimation.")
    else:
        avg_delta_ai = median_delta_ai = None
        vprint("No delta_ai values available for ELO estimation.")

    # Update metadata with ELO estimates and counts
    meta_data.update({
        "input_tensor_shape": str(board_tensor_shape),
        "failures": len(df),
        "avg_delta_ai": avg_delta_ai,
        "median_delta_ai": median_delta_ai,
        "estimated_model_elo": estimated_model_elo
    })

    # Write Excel report
    writer = pd.ExcelWriter(os.path.join(log_path, "training_report.xlsx"), engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="failures")
    pd.DataFrame([meta_data]).to_excel(writer, index=False, sheet_name="metadata")
    writer.close()

    vprint("\n=== Run Summary ===")
    for key, value in meta_data.items():
        vprint(f"{key}: {value}")
    vprint("===================\n")

    # Plot helpers
    def plot_and_save(data, title, xlabel, filename, eval_delta=False):
        plt.figure()
        if eval_delta:
            plt.hist(data, bins=300, range=(-300, 300), color='skyblue', edgecolor='black')
        else:
            plt.hist(data, bins=100, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(log_path, filename))
        plt.close()


    def plot_human_vs_ai_delta_scatter(scatter_df, log_path=None):
        # Filter only numeric rows (optional safety check)
        df = scatter_df[["delta_human", "delta_ai"]].dropna()
        
        # Create scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=df,
            x="delta_human",
            y="delta_ai",
            s=60,
            edgecolor="w",
            alpha=0.6
        )
        
        # Add diagonal reference line
        min_val = min(df.min().min(), 0)
        max_val = max(df.max().max(), 0)
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Match")

        # Labels and title
        plt.xlabel("Stockfish Δ (Human Move)")
        plt.ylabel("Stockfish Δ (AI Move)")
        plt.title("Human vs AI Stockfish Evaluation Δ")
        plt.legend()
        plt.grid(True)
        plt.xlim(-300, 300)
        plt.ylim(-300, 300)


        if log_path:
            plt.savefig(f"{log_path}/scatter_human_vs_ai_delta.png", bbox_inches="tight")
        else:
            plt.show()

        plt.close()



    # Generate plots if data is available
    if not df.empty:
        if "rank" in df.columns:
            vprint("Plotting: Rank")
            plot_and_save(df["rank"], "Model Rank of Actual Move", "Rank", "rank_histogram.png")
        if "actual_prob" in df.columns:
            vprint("Plotting: Actual Probability")
            plot_and_save(df["actual_prob"], "Model Probability of Actual Move", "Probability", "actual_prob_histogram.png")
        if "legal_mass" in df.columns:
            vprint("Plotting: Legal Move Mass")
            plot_and_save(df["legal_mass"], "Model Probability Mass on Legal Moves", "Probability Mass", "legal_mass_histogram.png")
        if "illegal_mass" in df.columns:
            vprint("Plotting: Illegal Move Mass")
            plot_and_save(df["illegal_mass"], "Model Probability Mass on Illegal Moves", "Probability Mass", "illegal_mass_histogram.png")
        if "delta_ai" in df.columns:
            delta_ai_vals = df["delta_ai"].dropna()
            if not delta_ai_vals.empty:
                vprint("Plotting: AI Move Stockfish Evaluation Delta")
                plot_and_save(delta_ai_vals, "AI Move Stockfish Evaluation Delta", "Evaluation Delta (Centipawns)", "delta_ai_histogram.png", eval_delta=True)
        # Scatter plot of human vs AI delta (only where both are present)
        if "delta_human" in df.columns and "delta_ai" in df.columns:
            scatter_df = df.dropna(subset=["delta_human", "delta_ai"])
            if not scatter_df.empty:
                vprint("Plotting: Scatter Plot of Human vs AI Deltas")
                plot_human_vs_ai_delta_scatter(scatter_df, log_path)


def main():
    vprint(f"Device: {device}")
    vprint(f"Checkpoint path: {checkpoint_path}")
    vprint(f"PGN path: {pgn_path}")
    vprint(f"Stockfish engine path: {stockfish_path}")
    vprint(f"Log directory: {log_path}")

    model, base_dataset, uci_to_index, index_to_uci = load_model_and_dataset(checkpoint_path, pgn_path, device)
    failure_cases = []
    delta_ai_list = []

    dataset_iter = iter(base_dataset)

    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Analyzing moves"):
            try:
                board_tensor, target_distribution, meta = next(dataset_iter)
            except StopIteration:
                vprint("Reached end of dataset early.")
                break

            result = evaluate_move_sample(
                board_tensor,
                target_distribution,
                meta,
                model,
                index_to_uci,
                engine,
                len(uci_to_index),
                device
            )
            if result is None:
                continue

            if result["is_failure"]:
                failure_cases.append(result["failure_data"])
            if result["delta_ai"] is not None:
                delta_ai_list.append(abs(result["delta_ai"]))

    meta_data = {
        "checkpoint_path": checkpoint_path,
        "pgn_path": pgn_path,
        "stockfish_path": stockfish_path,
        "num_samples_analyzed": len(delta_ai_list),
        "timestamp": timestamp,
    }

    if failure_cases:
        save_report_and_plots(failure_cases, delta_ai_list, board_tensor.shape, log_path, meta_data)
    else:
        vprint("No failure cases detected.")
    print(f"Analysis complete. Logs and results saved in: {log_path}")

    engine.quit()


if __name__ == "__main__":
    main()
