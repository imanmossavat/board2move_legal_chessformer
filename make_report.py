import os
import pandas as pd
import chess
import chess.svg
from datetime import datetime
from pathlib import Path
from cairosvg import svg2png

# === Config ===
pgn_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\lichess_elite_2025-02.pgn"
data_dir = os.path.join(os.path.dirname(pgn_path), "data")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_dir = os.path.join(data_dir, f"logs_{timestamp}")
boards_dir = os.path.join(report_dir, "boards")
os.makedirs(boards_dir, exist_ok=True)

excel_path = r'C:\Users\imanm\OneDrive - Office 365 Fontys\Documenten\myCode\Chess\MyTransformer_v1\chess_model_failures.xlsx'
df = pd.read_excel(excel_path)

# === Markdown Report ===
md_lines = [
    f"# Chess Model Failure Report ({timestamp})\n",
    f"Total cases: {len(df)}\n",
    "---\n"
]

# === Board Renderer ===
def save_board_svg_png(board, filename, size=400):
    svg = chess.svg.board(board, size=size)
    svg_path = filename.replace(".png", ".svg")
    with open(svg_path, "w") as f:
        f.write(svg)
    svg2png(url=svg_path, write_to=filename)
    os.remove(svg_path)

# === Generate images and report lines ===
for i, row in df.iterrows():
    fen = row["fen"]
    human_uci = row["uci"]
    model_uci = row["model_predicted_move"]

    board = chess.Board(fen)
    board_before = board.copy()

    # Human move
    board_human = board.copy()
    try:
        board_human.push_uci(human_uci)
    except Exception as e:
        print(f"Row {i}: bad human move: {human_uci} ({e})")

    # AI move
    board_ai = board.copy()
    try:
        board_ai.push_uci(model_uci)
    except Exception as e:
        print(f"Row {i}: bad model move: {model_uci} ({e})")

    # Save PNGs
    before_path = os.path.join(boards_dir, f"{i}_before.png")
    human_path = os.path.join(boards_dir, f"{i}_human.png")
    ai_path = os.path.join(boards_dir, f"{i}_ai.png")

    save_board_svg_png(board_before, before_path)
    save_board_svg_png(board_human, human_path)
    save_board_svg_png(board_ai, ai_path)

    # Write to markdown
    md_lines.append(f"## Position {i+1}")
    md_lines.append(f"**FEN**: `{fen}`  \n**Human move**: `{human_uci}`  \n**Model move**: `{model_uci}`  \n")
    md_lines.append(
        f'<table><tr>'
        f'<td><strong>Before</strong><br><img src="boards/{i}_before.png" width="300"/></td>'
        f'<td><strong>Human</strong><br><img src="boards/{i}_human.png" width="300"/></td>'
        f'<td><strong>Model</strong><br><img src="boards/{i}_ai.png" width="300"/></td>'
        f'</tr></table>\n'
    )
    md_lines.append("---\n")

# === Write Markdown file ===
md_path = os.path.join(report_dir, "report.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\n✅ Report saved to: {md_path}")
