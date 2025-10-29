import math
import random
from typing import Tuple, Dict, List

import numpy as np
import torch


def set_seed(seed: int):
	"""Set random seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def build_vocab(text: str) -> Tuple[Dict[str, int], List[str]]:
	"""Build character-level vocab from text.

	Returns (stoi, itos)
	"""
	chars = sorted(list(set(text)))
	itos = chars
	stoi = {ch: i for i, ch in enumerate(itos)}
	return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
	return [stoi[ch] for ch in text]


def decode(indices: List[int], itos: List[str]) -> str:
	return "".join([itos[i] for i in indices])


def get_batch(data: List[int], batch_size: int, block_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Create a random batch for language modeling.

	Returns x,y where x: (B, T) and y: (B, T)
	"""
	# data is a list or 1D array of token ids
	data_len = len(data)
	ix = np.random.randint(0, data_len - block_size, size=(batch_size,))
	x = np.stack([data[i:i+block_size] for i in ix])
	y = np.stack([data[i+1:i+block_size+1] for i in ix])
	x = torch.tensor(x, dtype=torch.long, device=device)
	y = torch.tensor(y, dtype=torch.long, device=device)
	return x, y


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
	return t.to(device)


__all__ = [
	"set_seed",
	"build_vocab",
	"encode",
	"decode",
	"get_batch",
	"to_device",
]