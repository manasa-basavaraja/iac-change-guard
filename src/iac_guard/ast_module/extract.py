from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml

from iac_guard.collector.diff_collector import IaCDiff


@dataclass
class TreeNode:
    label: str
    children: list["TreeNode"] = field(default_factory=list)
    value: str | None = None


def _walk_dict(obj: Any, prefix: str = "root") -> TreeNode:
    if isinstance(obj, dict):
        node = TreeNode(label=prefix)
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                node.children.append(_walk_dict(v, child_prefix))
            else:
                node.children.append(TreeNode(label=f"{child_prefix}", value=str(v)[:120]))
        return node
    if isinstance(obj, list):
        node = TreeNode(label=prefix)
        for i, v in enumerate(obj[:50]):
            node.children.append(_walk_dict(v, f"{prefix}[{i}]"))
        return node
    return TreeNode(label=prefix, value=str(obj)[:120])


def _try_hcl_to_tree(source: str) -> TreeNode | None:
    try:
        import hcl2  # type: ignore

        parsed = hcl2.loads(source)
        return _walk_dict(parsed, "hcl")
    except Exception:
        return None


def _try_yaml_to_tree(source: str) -> TreeNode | None:
    try:
        data = yaml.safe_load(source)
        if data is None:
            return None
        return _walk_dict(data, "yaml")
    except Exception:
        return None


def build_tree_from_iac_text(path: str, text: str) -> TreeNode | None:
    p = path.replace("\\", "/").lower()
    if p.endswith(".tf") or p.endswith(".tfvars"):
        tree = _try_hcl_to_tree(text)
        if tree:
            return tree
    if p.endswith((".yml", ".yaml")):
        return _try_yaml_to_tree(text)
    return _try_yaml_to_tree(text)


def heuristic_diff_features(diff: IaCDiff) -> list[float]:
    blob = "\n".join(diff.added_lines).lower()
    feats = [
        float("0.0.0.0/0" in blob or "::/0" in blob),
        float("privileged" in blob and "true" in blob),
        float(blob.count("secret") + blob.count("password")),
        float("admin" in blob or "iam" in blob),
        float(diff.path.lower().endswith(".tf")),
        float(".github/workflows" in diff.path.replace("\\", "/").lower()),
        float(len(diff.added_lines)) / 100.0,
        float(len(diff.removed_lines)) / 100.0,
    ]
    return feats


@dataclass
class TreeFeatures:
    root: TreeNode | None
    heuristic: list[float]


def build_tree_features(diff: IaCDiff) -> TreeFeatures:
    text = diff.after_blob
    root = build_tree_from_iac_text(diff.path, text) if text.strip() else None
    return TreeFeatures(root=root, heuristic=heuristic_diff_features(diff))
