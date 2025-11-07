# 好的，我很高兴能帮助你完成这个任务。我们将分步骤实现多头自注意力（multi-head self-attention）、位置前馈网络（position-wise FFN）、残差连接和层归一化（LayerNorm）、以及位置编码（positional encoding）。以下是我们将要完成的步骤：

# ### 步骤 1: 导入必要的库
# 我们需要导入一些基本的库来处理数据和构建模型。通常我们会使用 PyTorch 或 TensorFlow。你更倾向于使用哪个框架？

# ### 步骤 2: 实现位置编码（Positional Encoding）
# 位置编码用于为输入序列中的每个位置提供位置信息。我们将实现一个函数来生成位置编码。

# ### 步骤 3: 实现多头自注意力（Multi-Head Self-Attention）
# 我们将实现一个类来计算多头自注意力机制。这个类将包括查询（Q）、键（K）、值（V）的线性变换，以及注意力权重的计算。

# ### 步骤 4: 实现位置前馈网络（Position-wise Feed-Forward Network）
# 我们将实现一个简单的前馈神经网络，它将应用于每个位置的输出。

# ### 步骤 5: 实现残差连接和层归一化（Residual Connection + LayerNorm）
# 我们将实现一个函数来处理残差连接和层归一化，以便在每个子层之后应用。

# ### 步骤 6: 整合所有组件
# 最后，我们将把所有组件整合到一个模型中，以便进行训练和评估。

# ### 具体实现
# 我们可以从步骤 1 开始。请告诉我你选择的框架（PyTorch 或 TensorFlow），然后我将提供相应的代码示例。
import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .ffn import PositionwiseFFN
from .positional_encoding import SinusoidalPositionalEncoding
from typing import Optional


class TransformerBlock(nn.Module):
	"""Single Transformer block: attention -> add&norm -> ffn -> add&norm"""

	def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1, causal: bool = False,
				 ablation_no_residual: bool = False, ablation_no_layernorm: bool = False):
		super().__init__()
		self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout, causal=causal)
		self.ln1 = nn.LayerNorm(embed_dim) if not ablation_no_layernorm else nn.Identity()
		self.ffn = PositionwiseFFN(embed_dim, ff_dim, dropout=dropout, activation="gelu")
		self.ln2 = nn.LayerNorm(embed_dim) if not ablation_no_layernorm else nn.Identity()
		self.dropout = nn.Dropout(dropout)
		self.ablation_no_residual = ablation_no_residual

	def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# attention sublayer
		attn_out = self.attn(x, mask=mask)
		if self.ablation_no_residual:
			x = self.dropout(attn_out)
		else:
			x = x + self.dropout(attn_out)
		x = self.ln1(x)

		# ffn sublayer
		ffn_out = self.ffn(x)
		if self.ablation_no_residual:
			x = self.dropout(ffn_out)
		else:
			x = x + self.dropout(ffn_out)
		x = self.ln2(x)
		return x


class TransformerModel(nn.Module):
	"""Minimal transformer for language modeling over characters/tokens."""

	def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, max_len: int = 1024, dropout: float = 0.1, causal: bool = True,
				 ablation_no_pe: bool = False, ablation_no_residual: bool = False, ablation_no_layernorm: bool = False):
		super().__init__()
		self.token_emb = nn.Embedding(vocab_size, embed_dim)
		self.pos_enc = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
		self.layers = nn.ModuleList([
			TransformerBlock(embed_dim, num_heads, ff_dim, dropout=dropout, causal=causal,
						   ablation_no_residual=ablation_no_residual,
						   ablation_no_layernorm=ablation_no_layernorm)
			for _ in range(num_layers)
		])
		self.ln_f = nn.LayerNorm(embed_dim)
		self.head = nn.Linear(embed_dim, vocab_size, bias=False)
		self.ablation_no_pe = ablation_no_pe

	def forward(self, idx: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
		# idx: (B, T)
		x = self.token_emb(idx) * (self.token_emb.embedding_dim ** 0.5)
		if not self.ablation_no_pe:
			x = self.pos_enc(x)
		for layer in self.layers:
			x = layer(x, mask=mask)
		x = self.ln_f(x)
		logits = self.head(x)
		return logits


__all__ = ["TransformerBlock", "TransformerModel"]