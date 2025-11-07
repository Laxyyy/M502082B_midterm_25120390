import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_ablation_results(results_root="results", output_filename="ablation_comparison.png"):
    """
    Scans subdirectories in the results_root for training_log.csv files,
    plots their validation loss on a single graph, and saves the plot.
    """
    plt.figure(figsize=(14, 8))
    
    # Define a clear naming scheme for the legend
    experiment_names = {
        "baseline": "Baseline",
        "ablation_no_pe": "No Positional Encoding",
        "ablation_single_head": "Single-Head Attention",
        "ablation_no_residual": "No Residual Connections",
        "ablation_no_layernorm": "No Layer Normalization"
    }

    # Define colors for consistency
    colors = {
        "baseline": "blue",
        "ablation_no_pe": "red",
        "ablation_single_head": "green",
        "ablation_no_residual": "purple",
        "ablation_no_layernorm": "orange"
    }

    found_any_logs = False
    
    # Iterate through predefined experiment order for a clean plot legend
    for dir_name in experiment_names.keys():
        log_path = os.path.join(results_root, dir_name, "training_log.csv")
        
        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                if "step" in df.columns and "val_loss" in df.columns:
                    found_any_logs = True
                    label = experiment_names.get(dir_name, dir_name)
                    color = colors.get(dir_name, None)
                    plt.plot(df["step"], df["val_loss"], label=label, color=color, alpha=0.9)
                    print(f"Plotting data from: {log_path}")
                else:
                    print(f"Warning: 'step' or 'val_loss' column not found in {log_path}")
            except Exception as e:
                print(f"Error reading or plotting {log_path}: {e}")
        else:
            print(f"Warning: Log file not found at {log_path}")

    if not found_any_logs:
        print("Error: No valid training log files were found to plot.")
        plt.close()
        return

    plt.title("Ablation Study: Validation Loss Comparison", fontsize=16)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.yscale('log') # Use a log scale for better visibility if losses vary widely
    
    output_path = os.path.join(results_root, output_filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nSuccessfully saved ablation comparison plot to: {output_path}")

if __name__ == "__main__":
    try:
        plot_ablation_results()
    except ImportError:
        print("Pandas and Matplotlib are required. Please install them using:")
        print("pip install pandas matplotlib")
