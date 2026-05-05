from __future__ import annotations

import re
from typing import List


_TF_RESOURCE = re.compile(
    r'^\s*(resource|data)\s+"([^"]+)"\s+"([^"]+)"\s*\{',
    re.IGNORECASE,
)
_K8S_KIND = re.compile(r"^\s*kind:\s*(\S+)", re.IGNORECASE)
_GHA_NAME = re.compile(r"^\s*name:\s*(.+)$", re.IGNORECASE)


def tokenize_diff_text(lines: list[str]) -> list[str]:
    tokens: List[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            tokens.append("<blank>")
            continue

        m = _TF_RESOURCE.match(line)
        if m:
            tokens.append(f"<tf:{m.group(1)}:{m.group(2)}:{m.group(3)}>")
            continue

        m = _K8S_KIND.match(line)
        if m:
            tokens.append(f"<k8s:kind:{m.group(1)}>")
            continue

        if "runs-on:" in line or "uses:" in line:
            tokens.append("<gha:step_or_runner>")

        clipped = line.strip()[:200]
        tokens.append(clipped)
    return tokens
