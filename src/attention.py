import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadSelfAttention(nn.Module):
	"""Multi-head self-attention with optional causal mask.

	Args:
		embed_dim: input embedding dimension
		num_heads: number of attention heads
		dropout: attention dropout
		causal: if True, apply causal mask (for autoregressive models)
	"""

	def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
		super().__init__()
		assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.causal = causal

		# combined projection for q,k,v
		self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		self.attn_dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Forward pass.

		x: (B, T, C)
		mask: optional attention mask broadcastable to (B, num_heads, T, T)
		returns: (B, T, C)
		"""
		B, T, C = x.shape
		qkv = self.qkv_proj(x)  # (B, T, 3*C)
		qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
		qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
		q, k, v = qkv[0], qkv[1], qkv[2]

		# scaled dot-product attention
		# q,k: (B, heads, T, head_dim)
		attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, heads, T, T)
		attn_scores = attn_scores / (self.head_dim ** 0.5)

		# causal mask
		if self.causal:
			causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
			attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

		if mask is not None:
			# mask should contain 0 for keep, 1 for masked positions OR boolean mask
			attn_scores = attn_scores.masked_fill(mask.bool(), float('-inf'))

		attn = torch.softmax(attn_scores, dim=-1)
		attn = self.attn_dropout(attn)

		out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
		out = out.transpose(1, 2).contiguous().reshape(B, T, C)
		out = self.out_proj(out)
		return out


__all__ = ["MultiHeadSelfAttention"]