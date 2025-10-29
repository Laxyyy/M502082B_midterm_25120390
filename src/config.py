from dataclasses import dataclass
import torch
import math

@dataclass
class TransformerConfig:
	vocab_size: int = 65
	embed_dim: int = 128
	num_heads: int = 8
	ff_dim: int = 512
	num_layers: int = 4
	max_len: int = 1024
	dropout: float = 0.1
	# default to encoder-style (non-causal) for encoder-only experiments
	causal: bool = False

__all__ = ["TransformerConfig"]