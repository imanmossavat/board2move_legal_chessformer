import pandas as pd
import matplotlib.pyplot as plt

log_path = r"C:\Users\imanm\Downloads\lichess_elite_2025-02\data\logs_20250711_225930\training_loss.csv"
df = pd.read_csv(log_path)

# Compute moving average (window=200 steps is typical; adjust as needed)
window_size = 200
df["loss_smooth"] = df["loss"].rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12, 6))
plt.plot(df["global_step"], df["loss"], alpha=0.3, label='Raw Loss')
plt.plot(df["global_step"], df["loss_smooth"], color='red', label=f'Smoothed Loss (window={window_size})')

plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.title("Training Loss with Smoothed Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()