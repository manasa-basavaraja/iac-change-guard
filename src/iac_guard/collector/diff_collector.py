from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class IaCDiff:
    path: str
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def after_blob(self) -> str:
        return "\n".join(self.added_lines + self.context_lines)


_IAC_SUFFIXES = (".tf", ".tfvars", ".yml", ".yaml")
_WORKFLOW_MARKER = ".github/workflows/"


def _is_iac_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    if _WORKFLOW_MARKER in p and p.endswith((".yml", ".yaml")):
        return True
    return p.endswith(_IAC_SUFFIXES)


def collect_from_unified_diff(
    diff_text: str,
    *,
    preferred_path: str | None = None,
) -> IaCDiff | None:
    if not diff_text.strip():
        return None

    current_file: str | None = preferred_path
    added: list[str] = []
    removed: list[str] = []
    context: list[str] = []

    path_from_header = re.compile(r"^\+\+\+ b/(.*)$")

    for line in diff_text.splitlines():
        m = path_from_header.match(line)
        if m:
            current_file = m.group(1).strip()
            continue

        if line.startswith("+++") or line.startswith("---"):
            continue

        if not line:
            continue

        if line.startswith("@@"):
            continue

        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith(" "):
            context.append(line[1:])

    path = current_file or "unknown"
    if preferred_path:
        path = preferred_path

    if not _is_iac_path(path):
        return None

    return IaCDiff(
        path=path,
        added_lines=added,
        removed_lines=removed,
        context_lines=context,
        raw=diff_text,
    )


def load_diff_file(path: str | Path) -> IaCDiff | None:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return collect_from_unified_diff(text)


def iter_iac_diffs(diff_text: str) -> Iterable[IaCDiff]:
    parts = re.split(r"(?=diff --git )", diff_text)
    out: list[IaCDiff] = []
    for chunk in parts:
        chunk = chunk.strip()
        if not chunk:
            continue
        d = collect_from_unified_diff(chunk)
        if d:
            out.append(d)
    return out
