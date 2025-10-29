import time
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def estimate_loss(model: nn.Module, data_ids, eval_iters: int, batch_size: int, block_size: int, device: torch.device):
	model.eval()
	losses = {}
	loss_f = nn.CrossEntropyLoss()
	with torch.no_grad():
		iters = max(1, eval_iters)
		total = 0.0
		for _ in range(iters):
			ix = torch.randint(0, len(data_ids) - block_size, (batch_size,), device=device)
			x = torch.stack([torch.tensor(data_ids[i:i+block_size], device=device) for i in ix])
			y = torch.stack([torch.tensor(data_ids[i+1:i+block_size+1], device=device) for i in ix])
			logits = model(x)
			B, T, C = logits.shape
			loss = loss_f(logits.view(B*T, C), y.view(B*T))
			total += loss.item()
		avg = total / iters
	model.train()
	return avg


def train(
    model: nn.Module, 
    data_ids, 
    device: torch.device, 
    *, 
    epochs: int = 1, 
    batch_size: int = 32, 
    block_size: int = 128, 
    lr: float = 1e-3, 
    eval_interval: int = 200, 
    eval_iters: int = 20, 
    log_path: str = None, 
    max_steps: Optional[int] = None,
    dropout_adj_threshold: Optional[float] = None,
    dynamic_dropout_patience: int = 5, 
    dynamic_dropout_factor: float = 0.5,
    early_stop_threshold: Optional[float] = None,
    early_stop_stability_steps: int = 0
):
	"""Minimal train loop.

	model: nn.Module
	data_ids: list or array of token ids
	"""
	model.to(device)
	optimizer = optim.AdamW(model.parameters(), lr=lr)
	loss_f = nn.CrossEntropyLoss()
	steps = 0
	
	# early stopping state
	early_stop_monitoring_start_step = None

	# dynamic dropout state
	val_loss_lower_counter = 0
	dropout_adj_activated = False

	# prepare CSV logging
	if log_path is not None:
		import os, csv
		os.makedirs(os.path.dirname(log_path), exist_ok=True)
		write_header = not os.path.exists(log_path)
		csv_file = open(log_path, "a", buffering=1)
		csv_writer = csv.writer(csv_file)
		if write_header:
			csv_writer.writerow(["step", "train_loss", "val_loss", "timestamp"])
	
	training_stopped = False
	for epoch in range(epochs):
		t0 = time.time()
		# iterate with random batches
		num_iters = max(1, len(data_ids) // batch_size)
		for it in range(num_iters):
			ix = torch.randint(0, len(data_ids) - block_size, (batch_size,), device=device)
			x = torch.stack([torch.tensor(data_ids[i:i+block_size], device=device) for i in ix])
			y = torch.stack([torch.tensor(data_ids[i+1:i+block_size+1], device=device) for i in ix])
			optimizer.zero_grad()
			logits = model(x)
			B, T, C = logits.shape
			loss = loss_f(logits.view(B*T, C), y.view(B*T))
			loss.backward()
			optimizer.step()
			steps += 1

			if max_steps is not None and steps >= max_steps:
				print(f"Maximum steps ({max_steps}) reached. Stopping training.")
				training_stopped = True
				break

			if steps % eval_interval == 0:
				val_loss = estimate_loss(model, data_ids, eval_iters, batch_size, block_size, device)
				print(f"step {steps}: train loss {loss.item():.4f}, val loss {val_loss:.4f}")
				if log_path is not None:
					csv_writer.writerow([steps, f"{loss.item():.6f}", f"{val_loss:.6f}", time.time()])
				
				# Conditional activation for dynamic dropout
				if dropout_adj_threshold is not None and not dropout_adj_activated:
					if loss.item() < dropout_adj_threshold and val_loss < dropout_adj_threshold:
						print(f"Losses below {dropout_adj_threshold}. Activating dynamic dropout adjustment.")
						dropout_adj_activated = True

				if dropout_adj_activated:
					if val_loss < loss.item():
						val_loss_lower_counter += 1
					else:
						val_loss_lower_counter = 0
					
					if val_loss_lower_counter >= dynamic_dropout_patience:
						for module in model.modules():
							if isinstance(module, nn.Dropout):
								new_dropout = module.p * dynamic_dropout_factor
								print(f"Adjusting dropout from {module.p:.4f} to {new_dropout:.4f}")
								module.p = new_dropout
						val_loss_lower_counter = 0

				# Simplified early stopping logic
				if early_stop_threshold is not None:
					if loss.item() < early_stop_threshold and val_loss < early_stop_threshold:
						if early_stop_monitoring_start_step is None:
							early_stop_monitoring_start_step = steps
						elif (steps - early_stop_monitoring_start_step) >= early_stop_stability_steps:
							print(f"Losses stable below {early_stop_threshold} for {early_stop_stability_steps} steps. Triggering early stop.")
							training_stopped = True
							break
					else:
						# reset if the loss goes back up
						early_stop_monitoring_start_step = None

		if training_stopped:
			break
		t1 = time.time()
		print(f"epoch {epoch+1}/{epochs} completed in {t1-t0:.2f}s")
	if log_path is not None:
		csv_file.close()


def save_checkpoint(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
	payload = {"model_state": model.state_dict()}
	if optimizer is not None:
		payload["optim_state"] = optimizer.state_dict()
	torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, device: Optional[torch.device] = None):
	data = torch.load(path, map_location=device)
	model.load_state_dict(data["model_state"])
	if optimizer is not None and "optim_state" in data:
		optimizer.load_state_dict(data["optim_state"])
	return model, optimizer


__all__ = ["estimate_loss", "train", "save_checkpoint", "load_checkpoint"]