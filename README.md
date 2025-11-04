# Tiny Shakespeare Transformer

This project is an implementation of a Transformer model from scratch using PyTorch, designed for character-level language modeling on the Tiny Shakespeare dataset. It serves as a hands-on assignment to understand and build the core components of the Transformer architecture as described in the paper "Attention Is All You Need".

## Project Structure

```
.
├───data/
│   └───input.txt         # Tiny Shakespeare dataset
├───results/
│   ├───training_log.csv
│   └───*.png             # Output plots
├───scripts/
│   ├───run.sh            # Main execution script
│   └───plot_results.py   # Script to plot training curves
├───src/
│   ├───attention.py      # Multi-head self-attention module
│   ├───model.py          # Transformer model definition
│   ├───training.py       # Training loop and utilities
│   └───...               # Other modules (FFN, positional encoding, etc.)
├───run_train.py          # Main script to start training
└───requirements.txt      # Project dependencies
```

## Features Implemented

- **Multi-Head Self-Attention**: Core attention mechanism.
- **Positional Encoding**: Sinusoidal positional encodings to inject sequence order.
- **Position-wise Feed-Forward Networks**: Applied independently at each position.
- **Residual Connections & Layer Normalization**: For stabilizing deep networks.
- **Encoder-Only Transformer**: A complete model assembled from the above components.
- **Detailed Training Loop**: Includes validation, checkpointing, and logging.
- **Advanced Training Techniques**:
  - **AdamW Optimizer**: A robust optimization algorithm.
  - **Gradient Clipping**: Prevents exploding gradients.
  - **Cosine Annealing Learning Rate Scheduler**: For smoother convergence.
- **Result Visualization**: Scripts to plot training and validation loss curves.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd tiny-shakespeare-transformer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Reproduce Training

To run the training and reproduce the results, simply execute the provided shell script. This script will train the model with the default configuration specified in `run_train.py` and save the training log and loss curves to the `results/` directory.

```bash
bash scripts/run.sh
```

The script uses a fixed random seed (`1337`) to ensure reproducibility.

### Plotting Results

If you have a `training_log.csv` file (either from running the training or from your own experiments), you can generate the loss curve plots by running:

```bash
python scripts/plot_results.py
```
