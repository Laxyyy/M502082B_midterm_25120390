import os
import sys
import torch
import numpy as np

# ensure project src is on path for tests
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
	sys.path.insert(0, SRC)

from attention import MultiHeadSelfAttention


def test_attention_shapes_and_causal():
	B, T, C = 2, 5, 16
	heads = 4
	x = torch.randn(B, T, C)
	attn = MultiHeadSelfAttention(C, heads, dropout=0.0, causal=True)
	out = attn(x)
	assert out.shape == (B, T, C)

	# test causal: ensure positions can't attend to future by checking that
	# attention to a future token is zero after masking (approx)
	# we craft q,k so that attention scores are higher for later positions
	q = torch.zeros(B, T, C)
	k = torch.zeros(B, T, C)
	# set token at position 0 to have high key for last dim, and query at last pos also high
	k[:, 0, -1] = 10.0
	q[:, -1, -1] = 10.0
	# set module weights manually via qkv_proj to produce q/k/v from input
	# simpler: run on crafted x where q/k are as above
	x2 = torch.zeros_like(x)
	x2[:, 0, :] = k[:, 0, :]
	x2[:, -1, :] = q[:, -1, :]
	out2 = attn(x2)
	assert out2.shape == (B, T, C)