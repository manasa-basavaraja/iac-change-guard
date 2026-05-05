from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from iac_guard.collector.diff_collector import load_diff_file
from iac_guard.model import IaCMaliciousChangeModel

LABELS = {
    "terraform_benign.diff": 0,
    "github_actions_benign.diff": 0,
    "terraform_wide_open.diff": 1,
    "k8s_privileged.diff": 1,
    "gha_exfil_style.diff": 2,
}


class DiffDataset(Dataset):
    def __init__(self, examples_dir: Path, mapping: dict[str, int]) -> None:
        self.items: list[tuple[Path, int]] = []
        for name, y in mapping.items():
            p = examples_dir / name
            if p.exists():
                self.items.append((p, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        diff = load_diff_file(path)
        if diff is None:
            raise RuntimeError(f"Could not parse diff: {path}")
        return diff, y


def train(
    epochs: int = 3,
    lr: float = 2e-5,
    out_path: Path | None = None,
) -> None:
    examples_dir = ROOT / "examples"
    ds = DiffDataset(examples_dir, LABELS)
    if len(ds) == 0:
        raise SystemExit("No labeled examples found. Add files under examples/.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IaCMaliciousChangeModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        model.train()
        for i in range(len(ds)):
            diff, y = ds[i]
            logits = model.forward_from_diff(diff)
            target = torch.tensor([y], device=device)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"epoch {ep + 1}/{epochs}  loss={total_loss / len(ds):.4f}")

    out = out_path or (ROOT / "checkpoints" / "demo.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"Saved state dict to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    train(epochs=args.epochs, lr=args.lr, out_path=args.out)


if __name__ == "__main__":
    main()
