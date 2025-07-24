import os
import shutil

# Define the new folder structure and files to move
structure = {
    "utils": ["git_info.py"],  # Add this once you create it
    "models": ["model.py"],
    "core": [
        "dataset.py", "dataset_profile.py", "move_vocab_builder.py",
        "positional_encoding.py", "tokenizer.py"
    ],
    "training": ["train_standard.py"],
    "evaluation": [
        "check_model_performance.py",
        "check_model_performance_stat2xlsx_advanced.py",
        "check_model_performance_stats2xlsx.py",
        "human_random_comparison.py",
        "read_csv_logs.py"
    ],
    "tests": [
        "test_model.py", "test_dataset.py", "test_tokenizer.py",
        "test_stockfish.py", "test_move_vocab.py"
        # Add "test_git_status.py" later if you implement it
    ],
}

# Create folders and move files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        if os.path.exists(file):
            print(f"Moving {file} → {folder}/")
            shutil.move(file, os.path.join(folder, file))
        else:
            print(f"⚠️ File not found: {file}")

# Optionally move temp or other misc files
# shutil.move("temp.py", "sandbox/temp.py")  # If you create a sandbox dir