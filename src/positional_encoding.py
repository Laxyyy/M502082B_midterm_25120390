import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
	"""Sinusoidal positional encoding as in 'Attention is All You Need'."""

	def __init__(self, embed_dim: int, max_len: int = 5000):
		super().__init__()
		pe = torch.zeros(max_len, embed_dim)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		if embed_dim % 2 == 1:
			# odd dim: last column remains zero for cos
			pe[:, 1::2] = torch.cos(position * div_term[:-1])
		else:
			pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
		self.register_buffer("pe", pe)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Add positional encoding to input tensor.

		x: (B, T, C)
		returns: (B, T, C)
		"""
		B, T, C = x.shape
		return x + self.pe[:, :T, :C]


__all__ = ["SinusoidalPositionalEncoding"]
