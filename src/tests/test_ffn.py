import os
import sys
import torch
import numpy as np

# ensure project src is on path for tests
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
	sys.path.insert(0, SRC)

from ffn import PositionwiseFFN


def test_ffn_shapes_and_activation():
	B, T, C = 3, 7, 16
	ff_dim = 64
	x = torch.randn(B, T, C)
	ffn = PositionwiseFFN(C, ff_dim, dropout=0.0, activation="gelu")
	out = ffn(x)
	assert out.shape == (B, T, C)
