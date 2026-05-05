from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ast_vec: torch.Tensor, llm_vec: torch.Tensor) -> torch.Tensor:
        if ast_vec.dim() == 1:
            ast_vec = ast_vec.unsqueeze(0)
        if llm_vec.dim() == 1:
            llm_vec = llm_vec.unsqueeze(0)
        seq = torch.stack([ast_vec, llm_vec], dim=1)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        fused = attn_out.mean(dim=1)
        return self.norm(fused)
