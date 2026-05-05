from __future__ import annotations

import torch
import torch.nn as nn

from iac_guard.ast_module import TreeMessagePassingEncoder
from iac_guard.ast_module.extract import build_tree_features
from iac_guard.classifier import RiskClassifier
from iac_guard.collector.diff_collector import IaCDiff
from iac_guard.fusion import AttentionFusion
from iac_guard.llm_encoder import DiffTextEncoder


class IaCMaliciousChangeModel(nn.Module):
    def __init__(
        self,
        llm_model_name: str = "distilbert-base-uncased",
        ast_hidden_dim: int = 256,
        ast_layers: int = 3,
        fusion_heads: int = 8,
        num_classes: int = 3,
        dropout: float = 0.3,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.ast_encoder = TreeMessagePassingEncoder(
            hidden_dim=ast_hidden_dim,
            num_layers=ast_layers,
        )
        self.llm = DiffTextEncoder(model_name=llm_model_name, max_length=max_length)
        llm_dim = self.llm.out_dim
        self.ast_proj = nn.Linear(ast_hidden_dim, llm_dim)
        self.fusion = AttentionFusion(embed_dim=llm_dim, num_heads=fusion_heads)
        self.head = RiskClassifier(in_dim=llm_dim, num_classes=num_classes, dropout=dropout)

    def forward_from_diff(self, diff: IaCDiff) -> torch.Tensor:
        feats = build_tree_features(diff)
        ast_h = self.ast_encoder(feats)
        text = diff.raw if diff.raw.strip() else "\n".join(diff.added_lines + diff.removed_lines)
        llm_h = self.llm(text)
        ast_p = self.ast_proj(ast_h)
        fused = self.fusion(ast_p, llm_h)
        logits = self.head(fused)
        return logits
