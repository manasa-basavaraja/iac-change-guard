from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from iac_guard.collector.diff_collector import load_diff_file
from iac_guard.model import IaCMaliciousChangeModel
from iac_guard.pipeline import run_detection, should_alert


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", type=Path, required=True, help="unified diff file")
    ap.add_argument("--ckpt", type=Path, default=None, help="optional .pt weights")
    ap.add_argument("--threshold", type=float, default=0.72)
    args = ap.parse_args()

    diff = load_diff_file(args.diff)
    if diff is None:
        raise SystemExit("No IaC diff found in that file.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IaCMaliciousChangeModel().to(device)
    if args.ckpt and args.ckpt.exists():
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded weights from {args.ckpt}")
    else:
        print("No checkpoint, using random weights.")

    result = run_detection(model, diff, device=device, risk_threshold=args.threshold)
    probs = result.probs.cpu().tolist()
    print(f"file: {args.diff}")
    print(f"predicted class: {result.label}")
    print(f"class probs (benign, misconfiguration, malicious): {[round(x, 4) for x in probs]}")
    print(f"malicious probability: {result.malicious_probability:.4f}")
    print(f"alert (>= {args.threshold}): {should_alert(result, args.threshold)}")


if __name__ == "__main__":
    main()
