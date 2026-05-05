from __future__ import annotations

import hashlib
import torch
import torch.nn as nn

from iac_guard.ast_module.extract import TreeFeatures, TreeNode


def _label_id(label: str, vocab: int = 4096) -> int:
    h = hashlib.md5(label.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % vocab


def flatten_tree(root: TreeNode) -> tuple[list[int], list[tuple[int, int]]]:
    ids: list[int] = []
    edges: list[tuple[int, int]] = []

    def visit(node: TreeNode, parent_idx: int | None) -> int:
        idx = len(ids)
        ids.append(_label_id(node.label))
        if parent_idx is not None:
            edges.append((parent_idx, idx))
            edges.append((idx, parent_idx))
        for ch in node.children:
            visit(ch, idx)
        return idx

    visit(root, None)
    return ids, edges


class TreeMessagePassingEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_dim: int = 256,
        num_layers: int = 3,
        heuristic_dim: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.act = nn.ReLU()
        self.heuristic_mlp = nn.Sequential(
            nn.Linear(heuristic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._heuristic_dim = heuristic_dim

    def forward(self, features: TreeFeatures) -> torch.Tensor:
        device = next(self.parameters()).device
        h_heur = torch.tensor(features.heuristic, dtype=torch.float32, device=device).view(1, -1)
        if h_heur.shape[1] < self._heuristic_dim:
            pad = self._heuristic_dim - h_heur.shape[1]
            h_heur = torch.nn.functional.pad(h_heur, (0, pad))
        elif h_heur.shape[1] > self._heuristic_dim:
            h_heur = h_heur[:, : self._heuristic_dim]

        base = self.heuristic_mlp(h_heur)

        if features.root is None or not features.root.children and not features.root.label:
            return base

        ids, edges = flatten_tree(features.root)
        if not ids:
            return base

        x = self.embed(torch.tensor(ids, dtype=torch.long, device=device))
        for layer in self.layers:
            agg = torch.zeros_like(x)
            deg = torch.zeros(x.shape[0], 1, device=device)
            for a, b in edges:
                agg[a] += x[b]
                deg[a] += 1
            deg = torch.clamp(deg, min=1.0)
            x = self.act(layer(agg / deg + x))
        pooled = x.mean(dim=0, keepdim=True)
        return pooled + base
