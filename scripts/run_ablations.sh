#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Baseline Experiment ---
echo "--- Running Baseline Experiment ---"
python run_train.py --results_dir results/baseline --num_heads 4
echo "--- Baseline Experiment Finished ---"

# --- Ablation 1: No Positional Encoding ---
echo "--- Running Ablation: No Positional Encoding ---"
python run_train.py --results_dir results/ablation_no_pe --ablation_no_pe
echo "--- Ablation No PE Finished ---"

# --- Ablation 2: Single-Head Attention ---
echo "--- Running Ablation: Single-Head Attention ---"
python run_train.py --results_dir results/ablation_single_head --num_heads 1
echo "--- Ablation Single-Head Finished ---"

# --- Ablation 3: No Residual Connections ---
echo "--- Running Ablation: No Residual Connections ---"
python run_train.py --results_dir results/ablation_no_residual --ablation_no_residual
echo "--- Ablation No Residual Finished ---"

# --- Ablation 4: No Layer Normalization ---
echo "--- Running Ablation: No Layer Normalization ---"
python run_train.py --results_dir results/ablation_no_layernorm --ablation_no_layernorm
echo "--- Ablation No LayerNorm Finished ---"

echo "--- All Ablation Experiments Completed ---"
