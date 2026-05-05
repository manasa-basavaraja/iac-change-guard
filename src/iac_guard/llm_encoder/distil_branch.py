from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class DiffTextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.out_dim = self.encoder.config.hidden_size

    def forward(self, text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.encoder(**enc).last_hidden_state[:, 0, :]
        return out
