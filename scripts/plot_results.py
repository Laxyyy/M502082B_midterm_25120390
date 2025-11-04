
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results(log_path="results/training_log.csv", save_dir="results"):
    """
    Reads the training log CSV, plots the training loss, validation loss,
    and a combined plot, then saves them to the specified directory.
    """
    assert os.path.exists(log_path), f"Log file not found at {log_path}"
    
    # Read the data
    df = pd.read_csv(log_path)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # --- Plot 1: Training Loss ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["train_loss"], label="Training Loss", color="blue")
    plt.title("Training Loss over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    train_loss_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(train_loss_path)
    plt.close()
    print(f"Saved training loss plot to {train_loss_path}")

    # --- Plot 2: Validation Loss ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["val_loss"], label="Validation Loss", color="orange")
    plt.title("Validation Loss over Steps")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    val_loss_path = os.path.join(save_dir, "validation_loss.png")
    plt.savefig(val_loss_path)
    plt.close()
    print(f"Saved validation loss plot to {val_loss_path}")

    # --- Plot 3: Training vs. Validation Loss ---
    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["train_loss"], label="Training Loss", color="blue", alpha=0.8)
    plt.plot(df["step"], df["val_loss"], label="Validation Loss", color="orange", alpha=0.8)
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    comparison_path = os.path.join(save_dir, "comparison_loss.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"Saved comparison plot to {comparison_path}")

if __name__ == "__main__":
    # Need to add pandas and matplotlib to requirements
    try:
        plot_training_results()
    except ImportError:
        print("Pandas and Matplotlib are required. Please install them using:")
        print("pip install pandas matplotlib")

