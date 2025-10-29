import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFFN(nn.Module):
	"""Position-wise feed-forward network: two linear layers with activation.

	Applies the same FFN to each position independently.
	"""

	def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0, activation: str = "relu"):
		super().__init__()
		self.fc1 = nn.Linear(embed_dim, ff_dim)
		self.fc2 = nn.Linear(ff_dim, embed_dim)
		self.dropout = nn.Dropout(dropout)
		if activation == "relu":
			self.act = F.relu
		elif activation == "gelu":
			self.act = F.gelu
		else:
			raise ValueError("Unsupported activation")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, T, C)
		x = self.fc1(x)
		x = self.act(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x


__all__ = ["PositionwiseFFN"]
