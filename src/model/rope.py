"""
Everyone uses rotary positional embeddings nowdays, so why not do it here 
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"Expected: ({x.shape[1]}, {x.shape[-1]}). Found: {freqs_cis.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    return torch.view_as_real(x_ * freqs_cis).view(x.shape)


class MultiHeadSelfAttention(nn.Module):
    """
    Multihead self-attention with Rotary Position Embedding
    Prerry much inspired by Llama3 implementation
    https://github.com/meta-llama/llama3/blob/main/llama/model.py#L49-L190
    """
    def __init__(self, n_features: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.head_features = n_features // n_heads
        self.n_heads = n_heads
        self.n_features = n_features
        assert self.head_features * n_heads == n_features, "n_features must be divisible by n_heads"

        self.wq = nn.Linear(
            in_features=n_features,
            out_features=n_features,
            bias=False,
        )
        self.wk = nn.Linear(
            in_features=n_features,
            out_features=n_features,
            bias=False,
        )
        self.wv = nn.Linear(
            in_features=n_features,
            out_features=n_features,
            bias=False,
        )
        self.wo = nn.Linear(
            in_features=n_features,
            out_features=n_features,
            bias=False,
        )

    def _transform_mask(self, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        mask_expanded = mask.view(batch_size, 1, 1, seq_len).expand(-1, self.n_heads, -1, -1)
        mask_expanded = mask_expanded.float().masked_fill(mask_expanded == 0, float('-inf')).masked_fill(mask_expanded == 1, float(0.0))
        return mask_expanded

    def forward(self, x: torch.Tensor, freq: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: tensor with shape [batch_size, seq_len, n_features]
            freq: tensor of torch.complex64 with shape [n_features // n_heads, seq_len]
            padding_mask: tensor with shape [batch_size, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq: torch.Tensor = xq.view(batch_size, seq_len, self.n_heads, self.head_features)
        xk: torch.Tensor = xk.view(batch_size, seq_len, self.n_heads, self.head_features)
        xv: torch.Tensor = xv.view(batch_size, seq_len, self.n_heads, self.head_features)

        xq, xk = (apply_rotary_emb(x, freq) for x in (xq, xk))

        # Make [batch_size, n_heads, seq_len, head_features]
        xq, xk, xv, = (x.transpose(1, 2) for x in (xq, xk, xv))
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.n_features)
        if padding_mask is not None:
            scores = scores + self._transform_mask(padding_mask)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # [batch_size, n_heads, seq_len, head_features]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)
