"""
Plot training reward curve from SB3 Monitor CSV logs.

Reads the Monitor log file written during training and plots
episode rewards over time with a smoothed rolling average.
Saves the plot as training_curve.png.

Note: We use Monitor CSV logs instead of parsing TensorBoard event files.
This is simpler, more portable, and doesn't require TensorBoard as a
dependency for plotting. You can still view live TensorBoard logs during
training with: tensorboard --logdir ./logs/tensorboard/

Limitation: Monitor logs capture per-episode reward and length, but not
internal PPO metrics (policy loss, value loss, entropy). For those,
use TensorBoard directly.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_monitor_csv(log_dir):
    """Load all Monitor CSV files from a directory.

    SB3 Monitor writes CSV files with a 2-line header:
    Line 1: JSON metadata (starts with #)
    Line 2: column names (r, l, t = reward, length, time)
    """
    csv_files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No monitor CSV files found in {log_dir}. "
            "Run train.py first to generate training logs."
        )

    all_rewards = []
    all_lengths = []
    all_times = []

    for f in csv_files:
        with open(f, "r") as fp:
            lines = fp.readlines()

        # Skip the JSON header line (starts with #) and column header
        data_lines = [l.strip() for l in lines[2:] if l.strip()]
        for line in data_lines:
            parts = line.split(",")
            if len(parts) >= 3:
                all_rewards.append(float(parts[0]))
                all_lengths.append(int(float(parts[1])))
                all_times.append(float(parts[2]))

    return np.array(all_rewards), np.array(all_lengths), np.array(all_times)


def smooth(values, window=20):
    """Simple rolling average for smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    log_dir = os.path.join("logs", "monitor")
    rewards, lengths, times = load_monitor_csv(log_dir)

    episodes = np.arange(len(rewards))
    smoothed = smooth(rewards, window=20)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Plot 1: Episode rewards
    ax1.plot(episodes, rewards, alpha=0.3, color="blue", label="Raw")
    ax1.plot(
        np.arange(len(smoothed)) + 10,  # offset for rolling window center
        smoothed,
        color="blue",
        linewidth=2,
        label="Smoothed (window=20)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward Curve — Kuka Grasping PPO")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode lengths
    smoothed_lengths = smooth(lengths.astype(float), window=20)
    ax2.plot(episodes, lengths, alpha=0.3, color="green", label="Raw")
    ax2.plot(
        np.arange(len(smoothed_lengths)) + 10,
        smoothed_lengths,
        color="green",
        linewidth=2,
        label="Smoothed (window=20)",
    )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length (steps)")
    ax2.set_title("Episode Length Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("Saved training_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
